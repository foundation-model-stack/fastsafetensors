# SPDX-License-Identifier: Apache-2.0
"""Checkpoint selection and the shared iterator-consumption benchmark."""

import ctypes
import gc
import json
import math
import mmap
import os
import re
import socket
import statistics
import time
from contextlib import closing
from pathlib import Path

from benchmarks._sharding import partition_views


def hub_cache_dir(hf_home=None, hub_cache=None):
    if hub_cache is not None:
        return Path(hub_cache).expanduser()
    if hf_home is not None:
        return Path(hf_home).expanduser() / "hub"
    override = os.getenv("HF_HUB_CACHE") or os.getenv("HUGGINGFACE_HUB_CACHE")
    if override:
        return Path(override).expanduser()
    default = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "huggingface"
    return Path(os.getenv("HF_HOME", default)).expanduser() / "hub"


def _relative_path(value):
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Expected a relative path without '..', got {value!r}")
    return path


def resolve_input(
    input_path=None, model=None, revision="main", hf_home=None, hub_cache=None
):
    """Resolve an existing file/directory or an offline Hub snapshot."""
    if (input_path is None) == (model is None):
        raise ValueError("Specify exactly one of --input or --model")
    if input_path is not None:
        path = Path(input_path).expanduser().absolute()
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    if not re.fullmatch(r"[\w.-]+(?:/[\w.-]+)?", model):
        raise ValueError(f"Invalid model ID: {model!r}")
    repository = hub_cache_dir(hf_home, hub_cache) / (
        "models--" + model.replace("/", "--")
    )
    relative_revision = _relative_path(revision)
    snapshot = repository / "snapshots" / relative_revision
    if not snapshot.is_dir():
        ref = repository / "refs" / relative_revision
        if not ref.is_file():
            raise FileNotFoundError(
                f"Model {model!r}, revision {revision!r} is not cached in {repository}. "
                "Download it first or select a cached commit with --revision."
            )
        commit = ref.read_text().strip()
        if not re.fullmatch(r"[\w-]+", commit):
            raise ValueError(f"Invalid cached commit in {ref}")
        snapshot = repository / "snapshots" / commit
    if not snapshot.is_dir():
        raise FileNotFoundError(f"Cached snapshot is missing: {snapshot}")
    return snapshot.absolute()


def checkpoint_files(path):
    """Select only indexed weight shards, rejecting incomplete checkpoints."""
    path = Path(path)
    if path.is_dir():
        indexes = sorted(path.glob("*.safetensors.index.json"))
        if len(indexes) > 1:
            raise ValueError("Multiple indexes found; select one with --input")
        if indexes:
            path = indexes[0]
        else:
            files = list(path.glob("*.safetensors"))
    if path.name.endswith(".safetensors.index.json"):
        weight_map = json.loads(path.read_text())["weight_map"]
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Empty or invalid weight_map in {path}")
        files = [
            path.parent / _relative_path(name) for name in set(weight_map.values())
        ]
        if any(file.suffix != ".safetensors" for file in files):
            raise ValueError("The index must reference safetensors files")
    else:
        weight_map = None
        if not path.is_dir():
            if path.suffix != ".safetensors":
                raise ValueError(
                    "--input must be a directory, safetensors file or index"
                )
            files = [path]
    if not files:
        raise ValueError(f"No safetensors weights found in {path}")
    missing = [str(file) for file in files if not file.is_file()]
    if missing:
        raise FileNotFoundError(f"Checkpoint shards are missing: {missing}")
    if path.is_dir():
        # A partial, unindexed snapshot must not silently become a smaller model.
        groups = {}
        for file in files:
            match = re.fullmatch(r"(.+)-(\d+)-of-(\d+)\.safetensors", file.name)
            if match:
                prefix, number, total = match.groups()
                groups.setdefault((prefix, int(total)), set()).add(int(number))
        for (prefix, total), numbers in groups.items():
            if numbers != set(range(1, total + 1)):
                raise ValueError(
                    f"Incomplete shard set for {prefix}: expected {total} files"
                )
    files.sort(
        key=lambda p: [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", str(p))]
    )
    return files, weight_map, path.parent if weight_map is not None else None


def checkpoint_metadata(files, weight_map, index_dir):
    import torch
    from safetensors import safe_open

    from fastsafetensors import SafeTensorsMetadata
    from fastsafetensors.frameworks import get_framework_op
    from fastsafetensors.frameworks._torch import dtype_convert

    framework = get_framework_op("pytorch")
    entries = {}
    for file in files:
        # Both implementations must accept the header before measurements begin.
        with safe_open(str(file), framework="pt", device="cpu"):
            pass
        metadata = SafeTensorsMetadata.from_file(str(file), framework)
        for name, frame in metadata.tensors.items():
            if name in entries:
                raise ValueError(f"Duplicate tensor: {name}")
            dtype = dtype_convert.get(frame.dtype)
            if dtype is None:
                raise ValueError(f"PyTorch does not support {frame.dtype} ({name})")
            shape = framework.get_native_shape(frame.dtype, frame.shape)
            nbytes = frame.data_offsets[1] - frame.data_offsets[0]
            if math.prod(shape) * torch.empty((), dtype=dtype).element_size() != nbytes:
                raise ValueError(f"Unsupported storage layout for {name}")
            entries[name] = dict(
                file=file,
                offset=metadata.header_length + frame.data_offsets[0],
                dtype=dtype,
                shape=shape,
                nbytes=nbytes,
            )
    if not entries:
        raise ValueError("Checkpoint contains no tensors")
    if weight_map is not None:
        if entries.keys() != weight_map.keys() or any(
            entries[name]["file"] != index_dir / filename
            for name, filename in weight_map.items()
        ):
            raise ValueError("Checkpoint index does not match shard headers")
    return entries


def add_arguments(parser):
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--input", type=Path, help="Local file, snapshot directory or index"
    )
    inputs.add_argument(
        "--model", help="Model ID already downloaded into the Hub cache"
    )
    parser.add_argument(
        "--revision", default="main", help="Cached branch, tag or commit"
    )
    parser.add_argument(
        "--hf-home", type=Path, help="HF_HOME root; weights are under its hub directory"
    )
    parser.add_argument(
        "--hub-cache", type=Path, help="Override the Hub cache directory directly"
    )
    parser.add_argument(
        "--device", default="cuda:0", help="cuda:N or cpu; torchrun selects LOCAL_RANK"
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--readers",
        nargs="+",
        choices=[
            "fastsafetensors",
            "safetensors",
            "safetensors-prefetch",
            "safetensors-prefetch-sharded",
            "instanttensor",
        ],
        default=["fastsafetensors", "safetensors"],
    )
    parser.add_argument(
        "--cache", choices=["warm", "cold", "unchanged"], default="warm"
    )
    parser.add_argument(
        "--io",
        choices=["buffered", "direct", "auto"],
        default="auto",
        help="fastsafetensors I/O policy; auto preserves FASTSAFETENSORS_ODIRECT "
        "and follows the library default when unset",
    )
    parser.add_argument(
        "--loader-profile",
        choices=["generic", "vllm"],
        default="generic",
        help="vllm selects fst-sweep's iterator routing and ParallelLoader policy",
    )
    parser.add_argument("--enable-expert-parallel", action="store_true")
    parser.add_argument("--enable-ep-weight-filter", action="store_true")
    parser.add_argument(
        "--fastsafetensors-ep-mode",
        choices=["local", "broadcast"],
        default="local",
        help="Active EP: local reads with loader filtering (default), or unfiltered "
        "sharded reads and broadcast with consumer-side expert filtering",
    )
    parser.add_argument(
        "--num-experts", type=int, help="Override cached config.json expert count"
    )
    parser.add_argument(
        "--expert-placement", choices=["linear", "round_robin"], default="linear"
    )
    parser.add_argument(
        "--ep-global-input-scales",
        action="store_true",
        help="Copy input_scale tensors from all experts for global-scale quantization backends",
    )
    parser.add_argument(
        "--all-local",
        action="store_true",
        help="Read independently on each rank instead of sharded reads and broadcast",
    )
    parser.add_argument("--copier", choices=["nogds", "gds"], default="nogds")
    parser.add_argument(
        "--unified-memory", choices=["auto", "on", "off"], default="auto"
    )
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--bounce-mib", type=int, default=16)
    parser.add_argument(
        "--reserve-mib",
        type=int,
        help="Headroom: generic 512 MiB; vllm max(1 GiB, 5%% of free memory)",
    )
    parser.add_argument("--prefetch-threads", type=int, default=8)
    parser.add_argument("--prefetch-max-open-files", type=int, default=4)
    parser.add_argument("--prefetch-max-open-mib", type=int, default=16384)
    parser.add_argument(
        "--prefetch-budget-mib",
        type=int,
        help="Open-ahead budget; default min(16 GiB, free device memory / 4)",
    )
    parser.add_argument(
        "--prefetch-batch-mib",
        type=int,
        default=128,
        help="Sharded prefetch broadcast batch limit; a tensor is never split",
    )
    parser.add_argument("--prefetch-inflight", type=int, default=4)
    parser.add_argument(
        "--shard-map",
        type=Path,
        help="JSON mapping tensor names to dimensions; unspecified tensors are replicated",
    )
    parser.add_argument(
        "--partition-profile",
        choices=["generic", "deepseek-v4", "nemotron-h"],
        default="generic",
        help="Model profiles match fst-sweep's TP/EP slices (PP=DP=1, no SP/MTP)",
    )
    parser.add_argument(
        "--verify", choices=["sample", "full", "none"], default="sample"
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def synchronize(device):
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def storage_read_bytes():
    """Linux process IO accounting, including reads issued by worker threads."""
    try:
        with open("/proc/self/io") as handle:
            for line in handle:
                if line.startswith("read_bytes:"):
                    return int(line.split(":", 1)[1])
    except (OSError, ValueError):
        pass
    return None


def load_into_parameters(
    iterator_factory, parameters, slices, device, borrowed=False, ignored_names=()
):
    """The timed operation: construct, iterate, copy, release, synchronize.

    The factory is lazy, so loader construction is inside this function. No
    input discovery, allocation, cache preparation or byte verification is here.
    """
    io_before = storage_read_bytes()
    started = time.perf_counter()
    seen = set()
    ignored_names = set(ignored_names)
    expected_names = parameters.keys() | ignored_names
    first_tensor_s = None
    source_bytes = {"cpu": 0, "cuda": 0}
    with closing(iterator_factory()) as iterator:
        try:
            for name, tensor in iterator:
                if first_tensor_s is None:
                    first_tensor_s = time.perf_counter() - started
                if name not in expected_names or name in seen:
                    raise ValueError(f"Unexpected or duplicate tensor: {name}")
                if name not in ignored_names:
                    for source, target in partition_views(
                        tensor, parameters[name], slices[name]
                    ):
                        if source.shape != target.shape or source.dtype != target.dtype:
                            raise ValueError(f"Shape or dtype mismatch for {name}")
                        target.copy_(source)
                    del source, target
                source_bytes[tensor.device.type] += (
                    tensor.numel() * tensor.element_size()
                )
                seen.add(name)
                if borrowed and device.type == "cuda":
                    # Borrowed views may be freed/reused on the next iteration.
                    synchronize(device)
                del tensor
        finally:
            # Also protects source views when the consumer exits with an error.
            synchronize(device)
    synchronize(device)
    seconds = time.perf_counter() - started
    io_after = storage_read_bytes()
    if seen != expected_names:
        raise ValueError(f"Missing tensors: {expected_names - seen}")
    return dict(
        seconds=seconds,
        first_tensor_s=first_tensor_s,
        tensors=len(seen),
        copied_tensors=len(seen - ignored_names),
        ignored_tensors=len(seen & ignored_names),
        source_bytes=source_bytes,
        storage_read_bytes=(
            io_after - io_before
            if io_before is not None and io_after is not None
            else None
        ),
    )


def safetensors_iterator(files, keep=None):
    from safetensors import safe_open

    for file in files:
        with safe_open(str(file), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if keep is not None and not keep(name):
                    continue
                yield name, handle.get_tensor(name)


def page_residency(files):
    """Linux mincore without faulting checkpoint pages into memory."""
    libc = ctypes.CDLL(None, use_errno=True)
    mincore = getattr(libc, "mincore", None)
    if mincore is None:
        raise RuntimeError("warm/cold cache verification requires mincore (Linux)")
    mincore.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_ubyte),
    ]
    resident = total = 0
    for file in files:
        with (
            file.open("rb") as handle,
            mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_COPY) as mapping,
        ):
            pages = (len(mapping) + mmap.PAGESIZE - 1) // mmap.PAGESIZE
            vector = (ctypes.c_ubyte * pages)()
            if mincore(
                ctypes.addressof(ctypes.c_char.from_buffer(mapping)),
                len(mapping),
                vector,
            ):
                raise OSError(ctypes.get_errno(), "mincore failed")
            # mincore specifies only bit zero. Count it in C rather than
            # iterating hundreds of millions of ctypes values in Python.
            resident += (
                bytes(vector).translate(bytes(i & 1 for i in range(256))).count(1)
            )
            total += pages
    return dict(resident_pages=resident, total_pages=total)


def prepare_cache(files, mode):
    if mode == "unchanged":
        return None
    if mode == "warm":
        buffer = bytearray(16 << 20)
        for file in files:
            with file.open("rb", buffering=0) as handle:
                while handle.readinto(buffer):
                    pass
        state = page_residency(files)
        if state["resident_pages"] != state["total_pages"]:
            raise RuntimeError(
                "Checkpoint does not remain fully resident; use --cache unchanged"
            )
        return state
    if not hasattr(os, "posix_fadvise"):
        raise RuntimeError("cold cache preparation requires posix_fadvise (Linux)")
    for _ in range(20):
        for file in files:
            with file.open("rb") as handle:
                os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        state = page_residency(files)
        if state["resident_pages"] <= 32 * len(files):
            return state
        time.sleep(0.05)
    raise RuntimeError(
        f"Cold eviction failed: {state}; tmpfs is not a cold-storage test"
    )


def verify_parameters(parameters, entries, slices, mode):
    """Compare raw storage bytes in bounded pieces outside the timer."""
    import torch

    if mode == "none":
        return dict(mode=mode, checked_bytes=0)
    checked = 0
    for name, tensor in parameters.items():
        entry = entries[name]
        # Map only for reference gathering; clone only a bounded flattened run.
        if entry["nbytes"] == 0:
            continue
        with (
            entry["file"].open("rb") as handle,
            mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_COPY) as mapping,
        ):
            raw = torch.frombuffer(
                mapping,
                dtype=torch.uint8,
                count=entry["nbytes"],
                offset=entry["offset"],
            )
            for reference, destination in partition_views(
                raw.view(entry["dtype"]).reshape(entry["shape"]), tensor, slices[name]
            ):
                # A sharded column view can be strided; compare rows without making
                # a whole-model CPU copy (including for float8 dtypes).
                actual = destination.reshape(-1).view(torch.uint8)
                element_size = tensor.element_size()
                elements = (
                    min(destination.numel(), max(1, 4096 // element_size))
                    if mode == "sample"
                    else destination.numel()
                )
                stride_shape = tuple(reference.shape)
                with torch.no_grad():
                    for start in range(0, elements, (1 << 20) // element_size):
                        stop = min(elements, start + (1 << 20) // element_size)
                        indices = torch.arange(start, stop)
                        coordinates = []
                        for size in reversed(stride_shape):
                            coordinates.append(indices % size)
                            indices = indices // size
                        # Index raw bytes to support dtypes lacking advanced indexing.
                        offsets = torch.zeros(stop - start, dtype=torch.int64)
                        for coordinate, stride in zip(
                            reversed(coordinates), reference.stride()
                        ):
                            offsets += coordinate * stride
                        offsets = (offsets + reference.storage_offset()) * element_size
                        positions = offsets[:, None] + torch.arange(element_size)
                        expected = raw[positions - raw.storage_offset()].reshape(-1)
                        observed = actual[
                            start * element_size : stop * element_size
                        ].cpu()
                        if not torch.equal(observed, expected):
                            raise ValueError(
                                f"Byte verification failed for {name}, element {start}"
                            )
                        checked += observed.numel()
            del raw, reference, destination
    return dict(mode=mode, checked_bytes=checked)


def run_benchmark(args, iterator_function, benchmark_name):
    import safetensors
    import torch
    import torch.distributed as dist

    import fastsafetensors
    from benchmarks._instanttensor import (
        instanttensor_iterator,
        require_instanttensor,
    )
    from benchmarks._prefetch import (
        prefetch_iterator,
        require_prefetch,
        sharded_prefetch_iterator,
    )
    from benchmarks._sharding import (
        deepseek_v4_partition,
        nemotron_h_partition,
        partition_slice,
    )
    from benchmarks._vllm import configure, parse_expert_id

    if args.rounds < 1 or args.warmup < 0 or min(args.threads, args.bounce_mib) < 1:
        raise ValueError(
            "Rounds, threads and memory sizes must be positive; warmup may be zero"
        )
    path = resolve_input(
        args.input, args.model, args.revision, args.hf_home, args.hub_cache
    )
    files, weight_map, index_dir = checkpoint_files(path)
    entries = checkpoint_metadata(files, weight_map, index_dir)
    shard_map = json.loads(args.shard_map.read_text()) if args.shard_map else {}
    if not isinstance(shard_map, dict) or shard_map.keys() - entries.keys():
        raise ValueError("--shard-map must map checkpoint tensor names to dimensions")
    distributed = int(os.getenv("WORLD_SIZE", "1")) > 1
    rank = int(os.getenv("RANK", "0")) if distributed else 0
    world = int(os.getenv("WORLD_SIZE", "1")) if distributed else 1
    device = torch.device(args.device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only cpu and cuda devices are supported")
    if device.type == "cuda":
        if distributed:
            device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(device)
    selected, ep, routes = configure(
        args, path, entries, rank, world, device, benchmark_name
    )
    partition_ignored = set()
    if args.partition_profile != "generic":
        if args.shard_map:
            raise ValueError("--partition-profile cannot be combined with --shard-map")
        partitioner = {
            "deepseek-v4": deepseek_v4_partition,
            "nemotron-h": nemotron_h_partition,
        }[args.partition_profile]
        shard_map, partition_ignored = partitioner(
            path, entries, world, ep["filter_active"]
        )
    ignored_names = {
        name
        for name in selected
        if ep["filter_active"]
        and (expert := parse_expert_id(name)) is not None
        and expert not in args._local_expert_ids
        and not (args.ep_global_input_scales and "input_scale" in name)
    }
    ignored_names |= partition_ignored & selected.keys()
    destinations = {
        name: entry for name, entry in selected.items() if name not in ignored_names
    }
    if "instanttensor" in args.readers:
        require_instanttensor(device)
    if args.reserve_mib < 1:
        raise ValueError("Reserve must be positive")
    if any(
        route["effective_reader"].startswith("safetensors-prefetch")
        for route in routes.values()
    ):
        require_prefetch(device)
        if min(
            args.prefetch_threads,
            args.prefetch_max_open_files,
            args.prefetch_max_open_mib,
            args.prefetch_batch_mib,
            args.prefetch_inflight,
        ) <= 0 or (
            args.prefetch_budget_mib is not None and args.prefetch_budget_mib <= 0
        ):
            raise ValueError(
                "Prefetch threads, window, budget and batch sizes must be positive"
            )
    if args.borrowed_tensors and world > 1 and not getattr(args, "all_local", False):
        raise ValueError("Borrowed tensors require TP=1 or --all-local")
    if distributed:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    pg = dist.group.WORLD if distributed else None
    args._budget_pg = (
        dist.new_group(backend="gloo")
        if distributed and device.type == "cuda" and args.loader_profile == "vllm"
        else pg
    )
    environment = {
        name: os.getenv(name)
        for name in ["FASTSAFETENSORS_ODIRECT", "FASTSAFETENSORS_UNIFIED_MEM"]
    }
    for name, setting in [
        ("FASTSAFETENSORS_ODIRECT", args.io),
        ("FASTSAFETENSORS_UNIFIED_MEM", args.unified_memory),
    ]:
        if setting != "auto":
            os.environ[name] = "1" if setting in {"direct", "on"} else "0"
    report = dict(
        complete=False,
        benchmark=benchmark_name,
        model=args.model,
        revision=args.revision if args.model else None,
        resolved_revision=path.name if args.model else None,
        input=str(path),
        files=[str(file) for file in files],
        payload_bytes=sum(entry["nbytes"] for entry in entries.values()),
        tensor_count=len(entries),
        world_size=world,
        partition=dict(
            profile=args.partition_profile,
            shard_map=shard_map,
            ignored_prefixes=["mtp."] if partition_ignored else [],
        ),
        settings={
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if not key.startswith("_")
        },
        reader_routes=routes,
        ep={
            key: value
            for key, value in ep.items()
            if key not in {"rank", "local_expert_ids"}
        },
        io_environment={name: os.getenv(name) for name in environment},
        torch=torch.__version__,
        safetensors=safetensors.__version__,
        fastsafetensors=fastsafetensors.__version__,
        package=fastsafetensors.__file__,
        extension=fastsafetensors.cpp.__file__,
        timer="loader construction, iterator consumption, parameter copies, iterator/loader cleanup and device synchronization",
        warmup=[],
        samples=[],
    )

    def barrier():
        if distributed:
            dist.barrier()

    def gather(value):
        if not distributed:
            return [value]
        results = [None] * world
        dist.all_gather_object(results, value)
        return results

    try:
        slices, shapes = {}, {}
        for name, entry in destinations.items():
            shape = list(entry["shape"])
            dim = shard_map.get(name, -1)
            if ep["filter_active"] and parse_expert_id(name) is not None:
                # EP assigns whole experts rather than TP-slicing each expert.
                dim = -1
            slices[name], shapes[name] = partition_slice(shape, dim, rank, world)
        destination_bytes = sum(
            math.prod(shapes[name])
            * torch.empty((), dtype=entry["dtype"]).element_size()
            for name, entry in destinations.items()
        )
        if device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(device)
            if destination_bytes + (args.reserve_mib << 20) >= free:
                raise ValueError(
                    "Destination weights plus reserve do not fit; use TP with --shard-map or a smaller model"
                )
        parameters = {
            name: torch.empty(shapes[name], dtype=entry["dtype"], device=device)
            for name, entry in destinations.items()
        }
        synchronize(device)
        report["destinations"] = gather(
            dict(
                rank=rank,
                hostname=socket.gethostname(),
                device=str(device),
                bytes=destination_bytes,
                local_expert_ids=ep["local_expert_ids"],
                selected_tensors=len(selected),
                selected_payload_bytes=sum(
                    entry["nbytes"] for entry in selected.values()
                ),
                skipped_tensors=len(entries) - len(selected),
                copied_tensors=len(destinations),
                ignored_tensors=len(ignored_names),
            )
        )
        for repeat in range(-args.warmup, args.rounds):
            readers = list(args.readers)
            if repeat >= 0 and repeat % 2:
                readers.reverse()
            for reader in readers:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                for parameter in parameters.values():
                    parameter.reshape(-1).view(torch.uint8).fill_(0xA5)
                synchronize(device)
                barrier()
                cache_before = prepare_cache(files, args.cache)
                barrier()
                telemetry = []
                instanttensor_settings = []
                route = routes[reader]
                effective_reader = route["effective_reader"]
                reader_filter = (
                    args._fastsafetensors_tensor_filter
                    if effective_reader == "fastsafetensors"
                    else (
                        args._tensor_filter
                        if effective_reader == "safetensors"
                        else None
                    )
                )
                reader_entries = entries if reader_filter is None else selected
                reader_ignored = reader_entries.keys() - parameters.keys()
                if effective_reader == "safetensors":
                    factory = lambda: safetensors_iterator(files, args._tensor_filter)
                elif effective_reader == "safetensors-prefetch":
                    factory = lambda: prefetch_iterator(files, pg, device, args)
                elif effective_reader == "safetensors-prefetch-sharded":
                    factory = lambda: sharded_prefetch_iterator(files, pg, device, args)
                elif effective_reader == "instanttensor":
                    factory = lambda: instanttensor_iterator(
                        files, pg, device, instanttensor_settings
                    )
                else:
                    factory = lambda: iterator_function(
                        files, pg, device, args, telemetry
                    )
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                    baseline = torch.cuda.memory_allocated(device)
                    baseline_reserved = torch.cuda.memory_reserved(device)
                measured = load_into_parameters(
                    factory,
                    parameters,
                    slices,
                    device,
                    args.borrowed_tensors and reader == "fastsafetensors",
                    reader_ignored,
                )
                if device.type == "cuda":
                    measured["peak_extra_allocated_bytes"] = (
                        torch.cuda.max_memory_allocated(device) - baseline
                    )
                    measured["peak_extra_reserved_bytes"] = (
                        torch.cuda.max_memory_reserved(device) - baseline_reserved
                    )
                measured.update(
                    rank=rank,
                    cache_before=cache_before,
                    copiers=sorted(set(telemetry)),
                    reader_tensor_filter_active=reader_filter is not None,
                    **route,
                )
                if instanttensor_settings:
                    measured["instanttensor"] = instanttensor_settings[0]
                if reader == "fastsafetensors":
                    measured["all_local"] = args.all_local
                    measured["borrowed_tensors"] = args.borrowed_tensors
                if benchmark_name == "parallel" and reader == "fastsafetensors":
                    measured["memory_budget_bytes"] = args.effective_memory_budget_bytes
                    measured["batch_bytes"] = args.effective_batch_bytes
                    measured["loading_passes"] = args.effective_loading_passes
                    measured["use_chunk_budget_as_allocation_size"] = (
                        args.effective_chunk_budget_as_allocation_size
                    )
                expected_bytes = sum(
                    entry["nbytes"] for entry in reader_entries.values()
                )
                if (
                    measured["tensors"] != len(reader_entries)
                    or sum(measured["source_bytes"].values()) != expected_bytes
                ):
                    raise ValueError(
                        "Iterator must deliver exactly the reader's expected tensors and bytes"
                    )
                if effective_reader.startswith("safetensors-prefetch"):
                    if measured["source_bytes"]["cuda"] != report["payload_bytes"]:
                        raise ValueError(
                            "Prefetch must deliver all checkpoint tensors on CUDA"
                        )
                    measured["prefetch_budget_bytes"] = (
                        args.effective_prefetch_budget_bytes
                    )
                measured["selected_payload_bytes"] = expected_bytes
                if effective_reader == "instanttensor":
                    if measured["source_bytes"]["cuda"] != expected_bytes:
                        raise ValueError(
                            "InstantTensor must deliver the full checkpoint on CUDA"
                        )
                    if measured["instanttensor"]["total_tensor_size"] != expected_bytes:
                        raise ValueError(
                            "InstantTensor payload does not match checkpoint headers"
                        )
                # A fast rank must not reread reference files while another
                # rank is still timing its loader: that can contend for disk
                # bandwidth and populate pages during the measured operation.
                barrier()
                measured["verification"] = verify_parameters(
                    parameters, destinations, slices, args.verify
                )
                ranks = gather(measured)
                if rank == 0:
                    seconds = max(row["seconds"] for row in ranks)
                    sample = dict(
                        reader=reader,
                        **route,
                        repeat=repeat,
                        max_seconds=seconds,
                        payload_gbps=report["payload_bytes"] / seconds / 1e9,
                        ranks=ranks,
                    )
                    report["warmup" if repeat < 0 else "samples"].append(sample)
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                    print(json.dumps(sample), flush=True)
                barrier()
        if rank == 0:
            report["summary"] = {
                reader: dict(
                    median_seconds=statistics.median(
                        sample["max_seconds"]
                        for sample in report["samples"]
                        if sample["reader"] == reader
                    )
                )
                for reader in args.readers
            }
            report["complete"] = True
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    except Exception as error:
        if rank == 0:
            report["error"] = f"{type(error).__name__}: {error}"
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        raise
    finally:
        for name, value in environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        if distributed:
            dist.destroy_process_group()
