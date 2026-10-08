# SPDX-License-Identifier: Apache-2.0

import ctypes
import gc
import json
import os
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

import pytest

from fastsafetensors import ParallelLoader, SafeTensorsFileLoader, cpp
from fastsafetensors.common import SafeTensorsMetadata
from fastsafetensors.copier.gds import GdsFileCopier
from fastsafetensors.copier.nogds import NoGdsFileCopier
from fastsafetensors.st_types import Device, DType


class DelayedReader:
    """Publish only four bytes until the test releases the remaining transfer."""

    def __init__(self):
        self.release = threading.Event()
        self.waiting = threading.Event()
        self.requests = {}
        self.prefix_calls = []

    def submit_read(self, fd, buffer, offset, length, ptr_off, track_progress=False):
        assert track_progress
        pointer = buffer.get_base_address() + ptr_off
        payload = os.pread(fd, length, offset)
        assert len(payload) == length
        ctypes.memmove(pointer, payload, min(4, length))
        request = len(self.requests) + 1
        self.requests[request] = (fd, pointer, payload)
        return request

    def complete(self, request):
        self.waiting.set()
        assert self.release.wait(10), "test failed to release the pending transfer"
        fd, pointer, payload = self.requests[request]
        os.fstat(fd)  # The descriptor must stay open until the final DMA finishes.
        ctypes.memmove(pointer, payload, len(payload))
        return pointer

    def wait_read_prefix(self, request, length):
        self.prefix_calls.append((request, length))
        if length <= 4:
            return 4
        self.complete(request)
        return len(self.requests[request][2])

    def wait_read(self, request):
        return self.complete(request)


@pytest.fixture
def delayed_load(tmp_path, monkeypatch):
    import torch
    from safetensors.torch import save_file

    path = tmp_path / "delayed.safetensors"
    save_file(
        {
            "a": torch.tensor([123], dtype=torch.int32),
            "b": torch.tensor([456, 789], dtype=torch.int32),
        },
        str(path),
    )
    # The mock uses Python's unaligned pread; native reader tests cover O_DIRECT.
    monkeypatch.setenv("FASTSAFETENSORS_NOGDS_ODIRECT", "0")
    reader = DelayedReader()
    loader = SafeTensorsFileLoader(None, "cpu", nogds=True)
    loader.copier_constructor = lambda meta, device, fw: NoGdsFileCopier(
        meta, device, reader, fw
    )
    loader.add_filenames({0: [str(path)]})
    yield loader, reader
    reader.release.set()
    loader.close()


@pytest.mark.parametrize("borrowed", [False, True])
def test_ready_tensor_delivered_while_tail_is_pending(delayed_load, borrowed):
    loader, reader = delayed_load
    buffer = loader.copy_files_to_device(allow_inflight=True, borrowed_tensors=borrowed)

    # Exercise the direct iterator used by the borrowed pipeline as well.
    def get_tensor(name):
        if borrowed:
            return next(buffer.iter_local_tensors([name]))[1]
        return buffer.get_tensor(name)

    try:
        # Creating all views must not wait for DMA, or make the unread tail ready.
        assert not reader.release.is_set()
        assert get_tensor("a").item() == 123
        with ThreadPoolExecutor(1) as pool:
            tail = pool.submit(get_tensor, "b")
            assert reader.waiting.wait(5)
            assert not tail.done()
            reader.release.set()
            assert tail.result(timeout=5).tolist() == [456, 789]
    finally:
        reader.release.set()
        buffer.close()
    if not borrowed:
        assert loader.framework.get_mem_used() > 0
        assert tail.result().tolist() == [456, 789]
    del tail  # Future retains the exported tensor until it is released.
    gc.collect()
    assert loader.framework.get_mem_used() == 0


def test_coalesced_run_waits_for_pending_source_bytes(delayed_load, monkeypatch):
    loader, reader = delayed_load
    buffer = loader.copy_files_to_device(allow_inflight=True)
    broadcast_called = threading.Event()

    def broadcast(pg, source, frames, src_rank, device):
        broadcast_called.set()
        # Capture the bytes as communication would; later DMA cannot repair them.
        return [tensor.clone() for tensor in source]

    monkeypatch.setattr(loader.framework, "broadcast_contiguous_run", broadcast)
    weights = buffer._iter_tensors(["a", "b"], 16, 2, False)
    try:
        with ThreadPoolExecutor(1) as pool:
            first = pool.submit(next, weights)
            try:
                # "a" is ready, but the same run includes pending tensor "b".
                assert reader.waiting.wait(5)
                assert not broadcast_called.is_set()
                assert not first.done()
            finally:
                reader.release.set()
            name, tensor = first.result(timeout=5)
        actual = {name: tensor, **dict(weights)}
    finally:
        reader.release.set()
        weights.close()
        buffer.close()
    assert broadcast_called.is_set()
    assert actual["a"].item() == 123
    assert actual["b"].tolist() == [456, 789]
    assert loader.framework.get_mem_used() == 0


def test_close_drains_transfer_before_freeing_buffer(delayed_load):
    loader, reader = delayed_load
    buffer = loader.copy_files_to_device(allow_inflight=True)
    with ThreadPoolExecutor(1) as pool:
        closed = pool.submit(buffer.close)
        assert reader.waiting.wait(5)
        assert not closed.done()
        assert loader.framework.get_mem_used() > 0
        reader.release.set()
        closed.result(timeout=5)
    buffer.close()  # Idempotent after inflight IO has been drained.
    assert loader.framework.get_mem_used() == 0


@pytest.mark.parametrize("target", ["cpu", "cuda:0"])
def test_native_prefix_wait_and_error(tmp_path, target):
    import torch

    if target != "cpu" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    payload = bytes(range(256)) * 4096 + b"tail"
    path = tmp_path / "data"
    path.write_bytes(payload)
    output = torch.zeros(len(payload), dtype=torch.uint8, device=target)
    buffer = cpp.gds_device_buffer(output.data_ptr(), len(payload), target != "cpu")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    reader = cpp.nogds_file_reader(False, 7, 3, target != "cpu", 0)
    try:
        request = reader.submit_read(fd, buffer, 0, len(payload), 0, True)
        # Includes an unaligned prefix, a block boundary, and the partial last block.
        for length in (0, 1, 1025, 65539, len(payload)):
            ready = reader.wait_read_prefix(request, length)
            assert length <= ready <= len(payload)
            assert output[:length].cpu().numpy().tobytes() == payload[:length]
        with pytest.raises(IndexError):
            reader.wait_read_prefix(request, len(payload) + 1)
        assert reader.wait_read(request) == output.data_ptr()
        with pytest.raises(IndexError):
            reader.wait_read_prefix(request, 1)
        plain = reader.submit_read(fd, buffer, 0, 1, 0)
        with pytest.raises(ValueError, match="not enabled"):
            reader.wait_read_prefix(plain, 1)
        assert reader.wait_read(plain)
        os.truncate(path, 0)
        failed = reader.submit_read(fd, buffer, 0, len(payload), 0, True)
        with pytest.raises(RuntimeError, match="truncated"):
            reader.wait_read_prefix(failed, len(payload))
        assert reader.wait_read(failed) == 0
    finally:
        del reader
        os.close(fd)


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("queue_size", [-1, 0, 2])
def test_pipeline_readiness_preserves_owned_outputs(input_files, overlap, queue_size):
    import torch
    from safetensors.torch import load_file

    expected = {n: t for path in input_files for n, t in load_file(path).items()}
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    with closing(
        ParallelLoader(
            None,
            input_files,
            device=device,
            nogds=True,
            bbuf_size_kb=7,
            max_threads=3,
            max_batch_bytes=4096,
            queue_size=queue_size,
            overlap_io=overlap,
            use_tqdm_on_load=False,
        )
    ) as loader:
        with closing(loader.iterate_weights()) as weights:
            actual = dict(weights)
    assert set(actual) == set(expected)
    for name in actual:
        assert torch.equal(actual[name].cpu(), expected[name])


@pytest.mark.parametrize("target", ["cpu", "cuda:0"])
def test_tensor_spanning_requests_and_conversion_fallback(tmp_path, target):
    import torch
    from safetensors.torch import save_file

    if target != "cpu" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    expected = torch.arange(5001, dtype=torch.float32)
    path = tmp_path / "split.safetensors"
    save_file({"a": expected, "empty": torch.empty(0)}, str(path))
    with closing(
        SafeTensorsFileLoader(None, target, nogds=True, bbuf_size_kb=7, max_threads=3)
    ) as loader:
        loader.add_filenames({0: [str(path)]})
        with closing(
            loader.copy_files_to_device(max_copy_block_size=4093, allow_inflight=True)
        ) as buffer:
            assert torch.equal(buffer.get_tensor("a").cpu(), expected)
            assert buffer.get_tensor("empty").numel() == 0
        with closing(
            loader.copy_files_to_device(dtype=DType.F16, allow_inflight=True)
        ) as buffer:
            assert torch.equal(buffer.get_tensor("a").cpu(), expected.half())


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize(
    "copier_type,target",
    [("nogds", "cpu"), ("nogds", "cuda:0"), ("unified", "cuda:0")],
)
def test_sparse_readiness_out_of_order_and_unread_gaps(
    tmp_path, monkeypatch, overlap, copier_type, target
):
    import torch
    from safetensors.torch import save_file

    if target != "cpu" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    monkeypatch.setenv(
        "FASTSAFETENSORS_UNIFIED_MEM", "1" if copier_type == "unified" else "0"
    )
    monkeypatch.setenv("FASTSAFETENSORS_ODIRECT", "1")
    values = {
        name: torch.arange(8, dtype=torch.int32) + 100 * index
        for index, name in enumerate("abcde")
    }
    # Gaps must exceed select_byte_ranges' 4 KiB read-coalescing allowance.
    for name in ("b", "d"):
        values[name] = torch.full((2049,), -1, dtype=torch.int32)
    values["empty"] = torch.empty(0, dtype=torch.int32)
    path = tmp_path / "sparse.safetensors"
    save_file(values, str(path))
    selected = {"a", "c", "e", "empty"}
    with closing(
        SafeTensorsFileLoader(None, target, nogds=True, bbuf_size_kb=7, max_threads=3)
    ) as loader:
        loader.set_tensor_filter(selected.__contains__)
        loader.add_filenames({0: [str(path)]})
        with closing(
            loader.copy_files_to_device(max_copy_block_size=9, allow_inflight=overlap)
        ) as buffer:
            factory = buffer.rank_loaders[0][0]
            if overlap:
                assert factory.wait_tensor is not None
                for name in ("b", "d"):
                    with pytest.raises(ValueError, match="unread bytes"):
                        factory.wait_tensor(name)
            else:
                # Blocking paths should not install per-tensor wait callbacks.
                assert factory.wait_tensor is None
            # Begin with the last range, then cross request boundaries and
            # revisit cached completion. Tensor order need not match file order.
            for name in ("e", "a", "c", "e", "empty"):
                assert torch.equal(buffer.get_tensor(name).cpu(), values[name])
        assert loader.framework.get_mem_used() == 0


def test_inflight_materialization_failure_frees_buffer(input_files, monkeypatch):
    with closing(SafeTensorsFileLoader(None, "cpu", nogds=True)) as loader:
        loader.add_filenames({0: input_files})

        def fail(self, buffer, owner=None):
            raise RuntimeError("view construction failed")

        monkeypatch.setattr(NoGdsFileCopier, "prepare_tensors", fail)
        with pytest.raises(RuntimeError, match="view construction failed"):
            loader.copy_files_to_device(allow_inflight=True)
        assert loader.framework.get_mem_used() == 0


def test_inflight_read_failure_drains_all_files(tmp_path):
    import torch
    from safetensors.torch import save_file

    paths = [tmp_path / "a.safetensors", tmp_path / "b.safetensors"]
    for index, path in enumerate(paths):
        save_file({str(index): torch.arange(10001)}, str(path))
    with closing(
        SafeTensorsFileLoader(None, "cpu", nogds=True, bbuf_size_kb=7, max_threads=3)
    ) as loader:
        loader.add_filenames({0: [str(p) for p in paths]})
        for path in paths:
            os.truncate(path, 0)
        buffer = loader.copy_files_to_device(allow_inflight=True)
        with pytest.raises(RuntimeError, match="truncated"):
            buffer.get_tensor("0")
        with pytest.raises(Exception, match="wait_nogds_read failed"):
            buffer.close()
        assert not buffer.rank_loaders
        assert loader.framework.get_mem_used() == 0


def test_iterator_close_waits_for_pending_tail(delayed_load):
    from fastsafetensors.parallel_loader import PipelineParallel

    loader, reader = delayed_load
    paths = list(loader.meta)
    pipeline = PipelineParallel(None, loader, paths, use_tqdm_on_load=False)
    iterator = pipeline.iterate_weights()
    name, tensor = next(iterator)
    assert name == "a" and tensor.item() == 123
    with ThreadPoolExecutor(1) as pool:
        closed = pool.submit(iterator.close)
        assert reader.waiting.wait(5)
        assert not closed.done()
        assert loader.framework.get_mem_used() > 0
        reader.release.set()
        closed.result(timeout=5)
    assert tensor.item() == 123  # Iterator results own their storage after close.
    assert loader.framework.get_mem_used() == 0


@pytest.mark.parametrize("fixed_allocation", [False, True])
@pytest.mark.parametrize("copier_type", ["nogds", "unified"])
def test_readiness_disjoint_ranges_and_packed_dtypes(
    tmp_path, monkeypatch, fixed_allocation, copier_type
):
    import torch

    from fastsafetensors.frameworks._torch import TorchOp

    if copier_type == "unified" and not torch.cuda.is_available():
        pytest.skip("Unified DMA needs CUDA")
    monkeypatch.setenv(
        "FASTSAFETENSORS_UNIFIED_MEM", "1" if copier_type == "unified" else "0"
    )
    monkeypatch.setenv("FASTSAFETENSORS_ODIRECT", "1")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    values = {
        "a": torch.tensor([7], dtype=torch.uint8),
        "skip": torch.arange(2049, dtype=torch.int32),
        "b": torch.arange(9, dtype=torch.float32),
        "c": torch.tensor(42, dtype=torch.int64),
    }
    if hasattr(torch, "float8_e4m3fn"):
        values["d"] = torch.arange(8).to(torch.float8_e4m3fn)
    if hasattr(torch, "float4_e2m1fn_x2"):
        values["e"] = torch.tensor([18, 52], dtype=torch.uint8).view(
            torch.float4_e2m1fn_x2
        )
    payload = bytearray()
    header = {}
    for name, tensor in values.items():
        dtype = TorchOp().wrap_tensor(tensor).dtype
        shape = list(tensor.shape)
        if dtype == DType.F4:
            shape[-1] *= 2
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        header[name] = dict(
            dtype=dtype.value,
            shape=shape,
            data_offsets=[len(payload), len(payload) + len(raw)],
        )
        payload.extend(raw)
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path = tmp_path / "mixed.safetensors"
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    with closing(
        ParallelLoader(
            None,
            [str(path)],
            device=device,
            nogds=True,
            all_local=True,
            tensor_filter=lambda n: n != "skip",
            max_batch_bytes=16 << 10,
            overlap_io=True,
            use_chunk_budget_as_allocation_size=fixed_allocation,
            use_tqdm_on_load=False,
        )
    ) as loader:
        with closing(loader.iterate_weights()) as weights:
            actual = dict(weights)
    assert set(actual) == set(values) - {"skip"}
    for name in actual:
        assert torch.equal(
            actual[name].cpu().reshape(-1).view(torch.uint8),
            values[name].reshape(-1).view(torch.uint8),
        )


def test_dma_completion_wakes_waiter_on_failure():
    # Completion changes and condition notifications must share the mutex:
    # finishing between the predicate check and wait must not lose the wakeup.
    for _ in range(32):
        completion = cpp.dma_completion(123, [123], [456])
        with ThreadPoolExecutor(1) as pool:
            waiting = pool.submit(completion.wait_range, 123, 456)
            assert completion.finish(-7) == -7
            assert waiting.result(timeout=5) == -7
        with pytest.raises(IndexError):
            completion.wait_range(122, 456)
        with pytest.raises(IndexError):
            completion.wait_range(123, 457)
    empty = cpp.dma_completion(100, [], [])
    assert empty.wait_range(100, 100) == 0
    assert empty.finish(0) == 0
    with pytest.raises(ValueError):
        cpp.dma_completion(100, [99], [101])
    with pytest.raises(ValueError):
        cpp.dma_completion(100, [100], [])
    incomplete = cpp.dma_completion(100, [100], [101])
    assert incomplete.finish(0) != 0


@pytest.mark.parametrize("queue_size", [-1, 0, 2])
@pytest.mark.parametrize("fixed_allocation", [False, True])
def test_unified_progressive_pipeline(
    tmp_path, monkeypatch, queue_size, fixed_allocation
):
    import torch
    from safetensors.torch import save_file

    if not torch.cuda.is_available():
        pytest.skip("Unified DMA needs CUDA")
    monkeypatch.setenv("FASTSAFETENSORS_UNIFIED_MEM", "1")
    monkeypatch.setenv("FASTSAFETENSORS_ODIRECT", "1")
    monkeypatch.setenv("FASTSAFETENSORS_DMA_THREADS", "3")
    expected = {
        "a": torch.tensor([123], dtype=torch.int32),
        "b": torch.arange(17 * 2**20 + 3, dtype=torch.int32).to(torch.uint8),
        "empty": torch.empty(0),
    }
    path = tmp_path / "unified.safetensors"
    save_file(expected, str(path))
    with closing(
        ParallelLoader(
            None,
            [str(path)],
            device="cuda:0",
            nogds=True,
            max_batch_bytes=32 << 20,
            queue_size=queue_size,
            use_chunk_budget_as_allocation_size=fixed_allocation,
            use_tqdm_on_load=False,
        )
    ) as loader:
        with closing(loader.iterate_weights()) as weights:
            actual = dict(weights)
    assert set(actual) == set(expected)
    for name in actual:
        assert torch.equal(actual[name].cpu(), expected[name])


def test_unified_inflight_failure_releases_buffer(input_files, monkeypatch):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("Unified DMA needs CUDA")
    monkeypatch.setenv("FASTSAFETENSORS_UNIFIED_MEM", "1")
    monkeypatch.setenv("FASTSAFETENSORS_ODIRECT", "1")

    def fail(*args):
        raise RuntimeError("DMA worker failed")

    monkeypatch.setattr(cpp, "dma_load_runs_progress", fail)
    with closing(SafeTensorsFileLoader(None, "cuda:0", nogds=True)) as loader:
        loader.add_filenames({0: input_files})
        buffer = loader.copy_files_to_device(allow_inflight=True)
        with pytest.raises(RuntimeError, match="dma_load_runs failed"):
            buffer.get_tensor(next(iter(buffer.key_to_rank_lidx)))
        with pytest.raises(RuntimeError, match="DMA worker failed"):
            buffer.close()
        assert loader.framework.get_mem_used() == 0


def test_unified_serial_dma_gate_and_close(input_files, monkeypatch):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("Unified DMA needs CUDA")
    monkeypatch.setenv("FASTSAFETENSORS_UNIFIED_MEM", "1")
    monkeypatch.setenv("FASTSAFETENSORS_ODIRECT", "1")
    release, entered = threading.Event(), threading.Event()
    active, peak, calls = 0, 0, 0
    lock = threading.Lock()

    def delayed(*args):
        nonlocal active, peak, calls
        with lock:
            active += 1
            calls += 1
            peak = max(peak, active)
        entered.set()
        assert release.wait(10)
        with lock:
            active -= 1
        return 0

    monkeypatch.setattr(cpp, "dma_load_runs_progress", delayed)
    with closing(SafeTensorsFileLoader(None, "cuda:0", nogds=True)) as loader:
        # Reuse this constructor's gate across independent inflight batches.
        loader.add_filenames({0: input_files})
        first = loader.copy_files_to_device(allow_inflight=True)
        second = loader.copy_files_to_device(allow_inflight=True)
        try:
            assert entered.wait(5)
            assert active == 1
            with ThreadPoolExecutor(1) as pool:
                closed = pool.submit(first.close)
                assert not closed.done()
                release.set()
                closed.result(timeout=5)
            second.close()
        finally:
            release.set()
            first.close()
            second.close()
        assert calls == 2 and peak == 1
        assert loader.framework.get_mem_used() == 0


def checkpoint(path, alignment=0, tail_words=2):
    payload = struct.pack("<i", 123) + struct.pack("<i", 456) * tail_words
    header = json.dumps(
        {
            "a": {"dtype": "I32", "shape": [1], "data_offsets": [0, 4]},
            "b": {
                "dtype": "I32",
                "shape": [tail_words],
                "data_offsets": [4, len(payload)],
            },
            "empty": {
                "dtype": "F32",
                "shape": [0],
                "data_offsets": [len(payload), len(payload)],
            },
        }
    ).encode()
    header += b" " * ((alignment - 8 - len(header)) % 16)
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)
    return 8 + len(header)


class DelayedGdsReader:
    def __init__(self, path, prefix):
        self.payload = path.read_bytes()
        self.prefix = prefix
        self.release = threading.Event()
        self.waiting = threading.Event()
        self.requests = {}
        self.fail = False
        self.fail_submit = False

    def submit_read(
        self, fh, buffer, offset, length, ptr_off, file_length, track_progress=False
    ):
        if self.fail_submit and self.requests:
            raise RuntimeError("submission failed")
        payload = self.payload[offset : offset + length]
        pointer = buffer.get_base_address() + ptr_off
        count = min(len(payload), max(0, self.prefix - offset))
        ctypes.memmove(pointer, payload, count)
        request = len(self.requests) + 1
        self.requests[request] = (pointer, payload, count)
        return request

    def wait_read_prefix(self, request, length):
        pointer, payload, count = self.requests[request]
        if self.fail:
            raise RuntimeError("cuFile read failed")
        if length <= count:
            return count
        self.wait_read(request)
        return len(payload)

    def wait_read(self, request):
        self.waiting.set()
        assert self.release.wait(10), "pending transfer was not released"
        pointer, payload, count = self.requests[request]
        ctypes.memmove(pointer, payload, len(payload))
        return -1 if self.fail else len(payload)


@pytest.fixture
def delayed_gds(tmp_path, framework):
    path = tmp_path / "delayed.safetensors"
    header = checkpoint(path)
    reader = DelayedGdsReader(path, header + 4)
    loader = SafeTensorsFileLoader(None, "cpu", nogds=True)
    copiers = []

    def construct(meta, device, fw):
        copier = GdsFileCopier(meta, device, reader, fw)
        copiers.append(copier)
        return copier

    loader.copier_constructor = construct
    loader.add_filenames({0: [str(path)]})
    yield loader, reader, copiers
    reader.release.set()
    loader.close()


def test_gds_ready_tensor_and_close_with_pending_tail(delayed_gds, monkeypatch):
    loader, reader, copiers = delayed_gds
    deregistered = []

    def deregister(buffer, offset):
        assert reader.release.is_set()
        deregistered.append(offset)
        return 0

    monkeypatch.setattr(cpp.gds_device_buffer, "cufile_deregister", deregister)
    buffer = loader.copy_files_to_device(allow_inflight=True)
    assert buffer.get_tensor("a").item() == 123
    assert buffer.get_tensor("empty").numel() == 0
    with ThreadPoolExecutor(1) as pool:
        tail = pool.submit(buffer.get_tensor, "b")
        assert reader.waiting.wait(5) and not tail.done()
        reader.release.set()
        assert tail.result(timeout=5).tolist() == [456, 456]
    reader.release.clear()
    reader.waiting.clear()
    with ThreadPoolExecutor(1) as pool:
        closed = pool.submit(buffer.close)
        assert reader.waiting.wait(5) and not closed.done()
        assert copiers[0].fh is not None and loader.framework.get_mem_used() > 0
        assert not deregistered
        reader.release.set()
        closed.result(timeout=5)
    assert tail.result().tolist() == [456, 456]
    del tail
    gc.collect()
    assert deregistered == [0]
    assert copiers[0].fh is None and not copiers[0].copy_reqs
    assert loader.framework.get_mem_used() == 0
    buffer.close()


def test_gds_read_error_still_drains_and_releases_buffer(delayed_gds):
    loader, reader, copiers = delayed_gds
    buffer = loader.copy_files_to_device(allow_inflight=True)
    reader.fail = True
    with pytest.raises(RuntimeError, match="read failed"):
        buffer.get_tensor("b")
    with ThreadPoolExecutor(1) as pool:
        closed = pool.submit(buffer.close)
        assert reader.waiting.wait(5) and not closed.done()
        assert loader.framework.get_mem_used() > 0
        reader.release.set()
        with pytest.raises(RuntimeError, match="read failed"):
            closed.result(timeout=5)
    assert loader.framework.get_mem_used() == 0 and copiers[0].fh is None
    buffer.close()


@pytest.mark.parametrize("conversion", [False, True])
def test_gds_alignment_and_conversion_keep_blocking_path(delayed_gds, conversion):
    loader, reader, copiers = delayed_gds
    if not conversion:
        meta = next(meta for meta, rank in loader.meta.values())
        header = checkpoint(Path(meta.src), alignment=8)
        meta2 = SafeTensorsMetadata.from_file(meta.src, loader.framework)
        loader.meta[meta.src] = (meta2, 0)
        reader.payload = Path(meta.src).read_bytes()
        reader.prefix = header + 4
    with ThreadPoolExecutor(1) as pool:
        loaded = pool.submit(
            loader.copy_files_to_device,
            allow_inflight=True,
            dtype=DType.F32 if conversion else DType.AUTO,
        )
        assert reader.waiting.wait(5) and not loaded.done()
        reader.release.set()
        buffer = loaded.result(timeout=5)
    try:
        assert buffer.get_tensor("a").item() == 123
        assert buffer.get_tensor("b").tolist() == [456, 456]
        assert not copiers[0]._request_ranges
    finally:
        buffer.close()
    assert loader.framework.get_mem_used() == 0


def test_gds_fallback_keeps_readiness(tmp_path, framework, monkeypatch):
    path = tmp_path / "fallback.safetensors"
    checkpoint(path)

    def unavailable(*args):
        raise RuntimeError("handle setup failed")

    monkeypatch.setattr(cpp, "gds_file_handle", unavailable)
    reader = cpp.gds_file_reader(2, False, 0)
    loader = SafeTensorsFileLoader(None, "cpu", nogds=True)
    loader.copier_constructor = lambda meta, device, fw: GdsFileCopier(
        meta, device, reader, fw
    )
    loader.add_filenames({0: [str(path)]})
    buffer = loader.copy_files_to_device(allow_inflight=True)
    try:
        assert buffer.get_tensor("a").item() == 123
        assert buffer.get_tensor("b").tolist() == [456, 456]
    finally:
        buffer.close()
        loader.close()
    assert framework.get_mem_used() == 0
    assert cpp.get_cpp_metrics().bounce_buffer_bytes == 0


@pytest.mark.parametrize("read_error", [False, True])
def test_gds_partial_submission_drains_before_free(tmp_path, framework, read_error):
    path = tmp_path / "submission.safetensors"
    header = checkpoint(path, tail_words=4096)
    reader = DelayedGdsReader(path, header + 4)
    reader.fail_submit = True
    reader.fail = read_error
    loader = SafeTensorsFileLoader(None, "cpu", nogds=True)
    loader.copier_constructor = lambda meta, device, fw: GdsFileCopier(
        meta, device, reader, fw
    )
    loader.add_filenames({0: [str(path)]})
    try:
        with ThreadPoolExecutor(1) as pool:
            loaded = pool.submit(
                loader.copy_files_to_device,
                allow_inflight=True,
                max_copy_block_size=4096,
            )
            assert reader.waiting.wait(5) and not loaded.done()
            assert framework.get_mem_used() > 0
            reader.release.set()
            with pytest.raises(RuntimeError, match="submission failed"):
                loaded.result(timeout=5)
    finally:
        reader.release.set()
        loader.close()
    assert framework.get_mem_used() == 0


@pytest.mark.parametrize("failed_offset", [0, 4096])
def test_gds_registration_failure_releases_previous_registrations(
    tmp_path, framework, monkeypatch, failed_offset
):
    path = tmp_path / "registration.safetensors"
    checkpoint(path, tail_words=4096)
    deregistered = []
    monkeypatch.setattr(
        cpp.gds_device_buffer,
        "cufile_register",
        lambda buf, offset, length: -1 if offset == failed_offset else 0,
    )
    monkeypatch.setattr(
        cpp.gds_device_buffer,
        "cufile_deregister",
        lambda buf, offset: deregistered.append(offset) or 0,
    )
    reader = cpp.gds_file_reader(2, False, 0)
    meta = SafeTensorsMetadata.from_file(str(path), framework)
    copier = GdsFileCopier(meta, Device.from_str("cpu"), reader, framework)
    copier.enable_tensor_readiness()
    with pytest.raises(RuntimeError, match="register_buffer failed"):
        copier.submit_io(True, 4096)
    assert deregistered == ([] if failed_offset == 0 else [0])
    assert framework.get_mem_used() == 0 and copier.fh is None


@pytest.mark.parametrize("registered", [False, True])
@pytest.mark.parametrize("target", ["cpu", "cuda:0"])
def test_native_gds_prefix_and_eof_padding(tmp_path, target, registered):
    import torch

    cuda = target != "cpu"
    if cuda and (
        not torch.cuda.is_available() or not os.path.exists("/dev/nvidia-fs0")
    ):
        pytest.skip("GDS CUDA device unavailable")
    if cuda:
        torch.cuda.set_device(target)
        assert cpp.init_gds() == 0
    payload = bytes(range(256)) * 4096 + b"tail"
    path = tmp_path / "prefix.bin"
    path.write_bytes(b"x" * 4096 + payload)
    padded = (len(payload) + 4095) // 4096 * 4096
    output = torch.zeros(padded, dtype=torch.uint8, device=target)
    buffer = cpp.gds_device_buffer(output.data_ptr(), padded, cuda)
    handle = cpp.gds_file_handle(str(path), False, cuda)
    reader = cpp.gds_file_reader(4, cuda, 0, 4096)
    if registered:
        assert buffer.cufile_register(0, padded) == 0
    try:
        request = reader.submit_read(
            handle, buffer, 4096, padded, 0, path.stat().st_size, True
        )
        for length in (0, 1, 4095, 4096, 65539, len(payload)):
            ready = reader.wait_read_prefix(request, length)
            assert length <= ready <= len(payload)
            assert output[:length].cpu().numpy().tobytes() == payload[:length]
        with pytest.raises(IndexError, match="prefix out of bounds"):
            reader.wait_read_prefix(request, len(payload) + 1)
        assert reader.wait_read(request) == len(payload)
        with pytest.raises(IndexError):
            reader.wait_read_prefix(request, 1)
        plain = reader.submit_read(handle, buffer, 4096, padded, 0, path.stat().st_size)
        with pytest.raises(ValueError, match="not enabled"):
            reader.wait_read_prefix(plain, 1)
        assert reader.wait_read(plain) == len(payload)
        os.truncate(path, 0)
        failed = reader.submit_read(
            handle, buffer, 4096, padded, 0, 4096 + len(payload), True
        )
        with pytest.raises(RuntimeError, match="read failed"):
            reader.wait_read_prefix(failed, len(payload))
        assert reader.wait_read(failed) == -1
    finally:
        del reader
        if registered:
            buffer.cufile_deregister(0)


def test_native_gds_progress_drains_on_destruction(tmp_path):
    import torch

    payload = bytes(range(256)) * 1024
    path = tmp_path / "drain.bin"
    path.write_bytes(payload)
    output = torch.zeros(len(payload), dtype=torch.uint8)
    buffer = cpp.gds_device_buffer(output.data_ptr(), len(payload), False)
    handle = cpp.gds_file_handle(str(path), False, False)
    reader = cpp.gds_file_reader(4, False, 0, 4096)
    reader.submit_read(handle, buffer, 0, len(payload), 0, len(payload), True)
    del handle, reader
    assert output.numpy().tobytes() == payload
