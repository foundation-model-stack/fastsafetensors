# SPDX-License-Identifier: Apache-2.0
"""safetensors CUDA prefetch readers, following the fst-sweep vLLM adapters.

The sharded adapter uses torch.distributed instead of vLLM's communicator.
No inference runtime is imported.
"""

import sys
from collections import deque


def require_prefetch(device):
    from safetensors import safe_open

    if not hasattr(safe_open, "prefetch"):
        raise RuntimeError(
            "safetensors prefetch requires safetensors>=0.9.0rc1; "
            "install the benchmark extra or upgrade safetensors"
        )
    if sys.platform != "linux" or device.type != "cuda":
        raise ValueError("safetensors prefetch requires Linux and a CUDA device")


def _budget(device, args):
    import torch

    free, _ = torch.cuda.mem_get_info(device)
    budget = min(
        args.prefetch_max_open_mib << 20,
        (
            args.prefetch_budget_mib << 20
            if args.prefetch_budget_mib is not None
            else free // 4
        ),
    )
    args.effective_prefetch_budget_bytes = budget
    return budget


def _can_open(count, opened_bytes, next_bytes, budget, args):
    # As in vLLM, admit at least one shard even if it exceeds the window budget.
    return count == 0 or (
        count < args.prefetch_max_open_files and opened_bytes + next_bytes <= budget
    )


def _open(path, device, args):
    from safetensors import safe_open

    return safe_open(str(path), framework="pt").prefetch(
        device=str(device), threads=args.prefetch_threads
    )


def _close(loader, device, done=None):
    import torch

    # Prefetch owns the source device allocations; wait for kernels which read
    # those views before close frees them, also when consumption stops early.
    if done is not None:
        done.synchronize()
    else:
        torch.cuda.current_stream(device).synchronize()
    loader.close()


def prefetch_iterator(files, pg, device, args):
    """Each rank reads every shard, with a bounded window of open loaders."""
    require_prefetch(device)
    ordered = list(files)
    if pg is not None and pg.size() > 1 and len(ordered) > 1:
        start = pg.rank() * len(ordered) // pg.size()
        ordered = ordered[start:] + ordered[:start]
    pending = deque(ordered)
    window = deque()
    opened_bytes = 0
    budget = _budget(device, args)

    def open_ahead():
        nonlocal opened_bytes
        while pending and _can_open(
            len(window), opened_bytes, pending[0].stat().st_size, budget, args
        ):
            path = pending.popleft()
            size = path.stat().st_size
            window.append((_open(path, device, args), size))
            opened_bytes += size

    try:
        open_ahead()
        while window:
            loader, size = window[0]
            try:
                for name, tensor in loader:
                    yield name, tensor
                    del tensor
            finally:
                try:
                    _close(loader, device)
                finally:
                    window.popleft()
                    opened_bytes -= size
            open_ahead()
    finally:
        # Attempt every close, even if one loader failed while draining its I/O.
        first_error = None
        for loader, _ in window:
            try:
                _close(loader, device)
            except Exception as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error


def _header(path):
    from safetensors import safe_open

    from fastsafetensors.frameworks import get_framework_op
    from fastsafetensors.frameworks._torch import dtype_convert
    from fastsafetensors.st_types import DType

    framework = get_framework_op("pytorch")
    with safe_open(str(path), framework="pt") as handle:
        entries = []
        for name in handle.offset_keys():
            metadata = handle.get_tensor_meta(name)
            dtype = DType(metadata.dtype)
            entries.append(
                (
                    name,
                    dtype_convert[dtype],
                    framework.get_native_shape(dtype, list(metadata.shape)),
                    metadata.data_offsets[1] - metadata.data_offsets[0],
                )
            )
    return entries


def _issue(loader, batch, owner, pg, device):
    import torch
    import torch.distributed as dist

    packed = None
    if loader is not None:
        views = [loader.take(name).flatten().view(torch.uint8) for name, *_ in batch]
        buffer = torch.cat(views)
        packed = torch.cuda.Event()
        packed.record(torch.cuda.current_stream(device))
    else:
        buffer = torch.empty(
            sum(entry[3] for entry in batch), dtype=torch.uint8, device=device
        )
    if pg is not None and pg.size() > 1:
        work = dist.broadcast(buffer, group=pg, group_src=owner, async_op=True)
        # wait orders this CUDA stream after NCCL; the event marks completion.
        work.wait()
    done = torch.cuda.Event()
    done.record(torch.cuda.current_stream(device))
    return (done, buffer, batch), packed


def _finish(issued):
    import torch

    done, buffer, batch = issued
    done.synchronize()
    offset = 0
    for name, dtype, shape, nbytes in batch:
        piece = buffer[offset : offset + nbytes]
        if offset % torch.empty((), dtype=dtype).element_size():
            piece = piece.clone()
        yield name, piece.view(dtype).view(shape)
        offset += nbytes


def sharded_prefetch_iterator(files, pg, device, args):
    """Owner rank reads each shard; packed tensor batches are broadcast."""
    require_prefetch(device)
    if (pg is None or pg.size() == 1) and getattr(
        args, "loader_profile", "generic"
    ) != "vllm":
        yield from prefetch_iterator(files, pg, device, args)
        return
    import torch

    rank, world = (pg.rank(), pg.size()) if pg is not None else (0, 1)
    owned = deque(range(rank, len(files), world))
    loaders = {}
    inflight = deque()
    opened_bytes = 0
    budget = _budget(device, args)

    def open_ahead():
        nonlocal opened_bytes
        while owned and _can_open(
            len(loaders), opened_bytes, files[owned[0]].stat().st_size, budget, args
        ):
            index = owned.popleft()
            size = files[index].stat().st_size
            loaders[index] = (_open(files[index], device, args), size)
            opened_bytes += size

    try:
        open_ahead()
        for index, path in enumerate(files):
            owner = index % world
            entries = _header(path)
            loader, size = loaders.pop(index) if owner == rank else (None, 0)
            packed = None
            try:
                batch, batch_bytes = [], 0
                for entry in entries:
                    name, dtype, shape, nbytes = entry
                    if nbytes == 0:
                        yield name, torch.empty(shape, dtype=dtype, device=device)
                        continue
                    if batch and batch_bytes + nbytes > args.prefetch_batch_mib << 20:
                        issued, packed = _issue(loader, batch, owner, pg, device)
                        inflight.append(issued)
                        batch, batch_bytes = [], 0
                        if len(inflight) >= args.prefetch_inflight:
                            yield from _finish(inflight.popleft())
                    batch.append(entry)
                    batch_bytes += nbytes
                if batch:
                    issued, packed = _issue(loader, batch, owner, pg, device)
                    inflight.append(issued)
                    while len(inflight) >= args.prefetch_inflight:
                        yield from _finish(inflight.popleft())
            except BaseException:
                # An issue may have queued new packing kernels before failing
                # to return their event. Drain the stream rather than an older
                # batch's event before releasing the engine's allocations.
                packed = None
                raise
            finally:
                if loader is not None:
                    _close(loader, device, packed)
                    opened_bytes -= size
            open_ahead()
        while inflight:
            yield from _finish(inflight.popleft())
    finally:
        first_error = None
        for done, _, _ in inflight:
            try:
                done.synchronize()
            except Exception as error:
                if first_error is None:
                    first_error = error
        for loader, _ in loaders.values():
            try:
                _close(loader, device)
            except Exception as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error
