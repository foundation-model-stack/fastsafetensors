# SPDX-License-Identifier: Apache-2.0
"""Writing large tensor sets fast, as a set of safetensors shards.

Buffered writes into one file serialize on its inode (a couple of GiB/s on
common filesystems), while separate files write in parallel, so the tensors are
spread over shards balanced by size. Each shard is sized up front and mmapped;
the caller's fill function writes every tensor's bytes straight into the maps
(e.g. a device backend encoding its native format), then each completed shard
is published under its final name.

    entries = [WriteEntry("w", DType.F16, [4, 8], 64, source=tensor), ...]
    save_sharded(entries, "/cache/model")   # model-00001-of-00016.safetensors, ...
"""

import json
import mmap
import os
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from .frameworks import TensorBase, get_framework_op
from .st_types import DTYPE_SIZES, DType

if TYPE_CHECKING:
    from .common import SafeTensorsMetadata

KEY_SHARD_INDEX = "fst.shard.index"
KEY_SHARD_COUNT = "fst.shard.count"
KEY_SHARD_SET = "fst.shard.set"


class ShardSetError(ValueError):
    """Checkpoint files do not form one complete, ordered shard set."""


def validate_shard_set(metas: Sequence["SafeTensorsMetadata"]) -> None:
    """Check that all shards of one write are present in index order."""
    if not metas:
        raise ShardSetError("no shards")
    ids = set()
    for i, meta in enumerate(metas):
        md = meta.metadata if isinstance(meta.metadata, dict) else {}
        try:
            index, count = int(md[KEY_SHARD_INDEX]), int(md[KEY_SHARD_COUNT])
        except (KeyError, TypeError, ValueError):
            raise ShardSetError(f"{meta.src}: not a shard of a written set")
        if index != i or count != len(metas):
            raise ShardSetError(
                f"{meta.src}: shard {index} of {count}, expected {i} of {len(metas)}"
            )
        ids.add(md.get(KEY_SHARD_SET))
    if len(ids) != 1 or None in ids:
        raise ShardSetError(f"shards were written separately: {sorted(map(str, ids))}")


@dataclass
class WriteEntry:
    """One tensor to write: its on-disk dtype/shape, size, and what fills it."""

    name: str
    dtype: DType
    shape: List[int]
    nbytes: int
    # Opaque to the saver; handed back to the fill function.
    source: Any = None


@dataclass
class ShardPlan:
    path: str
    entries: List[WriteEntry]
    header: bytes
    # name -> [begin, end) absolute offsets in the file
    offsets: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    total_bytes: int = 0


# Called once with every (buffer, entry) of a write, largest first. The callback
# must fill the planned byte ranges before returning and must not retain buffers.
FillFn = Callable[[List[Tuple[memoryview, WriteEntry]]], None]
ShardMetadataFn = Callable[[Sequence[WriteEntry]], Mapping[str, str]]


def _wrap_tensor(tensor: Any, framework: str) -> TensorBase:
    if isinstance(tensor, TensorBase):
        return tensor
    return get_framework_op(framework).wrap_tensor(tensor)


def plain_entry(name: str, tensor: Any, *, framework: str = "pytorch") -> WriteEntry:
    """Describe a native or wrapped tensor, preserving its source object."""
    wrapped = _wrap_tensor(tensor, framework)
    return WriteEntry(
        name, wrapped.dtype, wrapped.get_shape(), wrapped.get_nbytes(), tensor
    )


def shard_paths(base: str, num_shards: int) -> List[str]:
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    return [
        f"{base}-{i + 1:05d}-of-{num_shards:05d}.safetensors" for i in range(num_shards)
    ]


def _check_entry(e: WriteEntry) -> None:
    if e.dtype == DType.AUTO:
        raise ValueError(f"{e.name}: AUTO is not an on-disk dtype")
    if any(d < 0 for d in e.shape):
        raise ValueError(f"{e.name}: shape must be nonnegative: {e.shape}")
    if e.dtype == DType.F4 and (not e.shape or e.shape[-1] % 2):
        raise ValueError(f"{e.name}: F4 requires an even last dimension")
    numel = 1
    for d in e.shape:
        numel *= d
    expected = numel * DTYPE_SIZES[e.dtype]
    if expected != e.nbytes:
        raise ValueError(
            f"{e.name}: {e.dtype.value}{e.shape} is {expected} bytes, not {e.nbytes}"
        )


def plan_shards(
    entries: Sequence[WriteEntry],
    paths: Sequence[str],
    *,
    metadata: Optional[Mapping[str, str]] = None,
    shard0_metadata: Optional[Mapping[str, str]] = None,
    shard_metadata: Optional[ShardMetadataFn] = None,
    header_align: int = 8,
) -> List[ShardPlan]:
    """Spread ``entries`` over ``paths`` (largest first into the emptiest shard).

    ``metadata`` goes into every shard, ``shard0_metadata`` into the first only
    (e.g. an index too large to repeat). ``shard_metadata`` adds metadata
    computed from each shard's entries, before any files are created.
    """
    if not paths or len(set(map(os.path.abspath, paths))) != len(paths):
        raise ValueError("paths must be nonempty and unique")
    if header_align < 8 or header_align % 8:
        raise ValueError("header_align must be a positive multiple of 8")
    names = set()
    for e in entries:
        if e.name in names or e.name == "__metadata__":
            raise ValueError(f"duplicate or reserved tensor name: {e.name}")
        names.add(e.name)
        _check_entry(e)
    for given in (metadata or {}, shard0_metadata or {}):
        for k, v in given.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise ValueError(f"__metadata__ entries must be strings: {k!r}")

    members: List[List[WriteEntry]] = [[] for _ in paths]
    loads = [0] * len(paths)
    for e in sorted(entries, key=lambda e: -e.nbytes):
        i = loads.index(min(loads))
        members[i].append(e)
        loads[i] += e.nbytes

    set_id = uuid.uuid4().hex
    plans = []
    for i, (path, shard) in enumerate(zip(paths, members)):
        header: Dict[str, Any] = {}
        offset = 0
        for e in shard:
            header[e.name] = {
                "dtype": e.dtype.value,
                "shape": list(e.shape),
                "data_offsets": [offset, offset + e.nbytes],
            }
            offset += e.nbytes
        md: Dict[str, str] = dict(metadata or {})
        if i == 0:
            md.update(shard0_metadata or {})
        md.update(
            {
                KEY_SHARD_INDEX: str(i),
                KEY_SHARD_COUNT: str(len(paths)),
                KEY_SHARD_SET: set_id,
            }
        )
        if shard_metadata is not None:
            extra = shard_metadata(tuple(shard))
            for k, v in extra.items():
                if not isinstance(k, str) or not isinstance(v, str):
                    raise ValueError(f"__metadata__ entries must be strings: {k!r}")
                if k in {KEY_SHARD_INDEX, KEY_SHARD_COUNT, KEY_SHARD_SET}:
                    raise ValueError(f"shard metadata key is reserved: {k}")
            md.update(extra)
        header["__metadata__"] = md
        header_bytes = json.dumps(header, separators=(",", ":")).encode()
        header_bytes += b" " * (-len(header_bytes) % header_align)
        data_start = 8 + len(header_bytes)
        plan = ShardPlan(
            path=path,
            entries=shard,
            header=len(header_bytes).to_bytes(8, "little") + header_bytes,
            total_bytes=data_start + offset,
        )
        for e in shard:
            b, end = header[e.name]["data_offsets"]
            plan.offsets[e.name] = (data_start + b, data_start + end)
        plans.append(plan)
    return plans


def write_shards(
    plans: Sequence[ShardPlan],
    fill: FillFn,
    *,
    min_free_bytes: int = 0,
    fsync: bool = False,
) -> List[str]:
    """Create every planned shard and have ``fill`` write the tensors' bytes.

    After every shard is complete, each temporary file is renamed to its final
    path. Publication is atomic per file. A failed fill removes temporary files.
    """
    total = sum(p.total_bytes for p in plans)
    directories = {os.path.dirname(os.path.abspath(p.path)) for p in plans}
    for d in directories:
        os.makedirs(d, exist_ok=True)
        free = shutil.disk_usage(d).free
        if free < total + min_free_bytes:
            raise OSError(
                f"not enough space in {d}: {total / 2**30:.1f} GiB needed, "
                f"{free / 2**30:.1f} GiB free"
            )

    write_id = uuid.uuid4().hex
    tmps = [f"{p.path}.tmp.{write_id}" for p in plans]
    fds: List[int] = []
    maps: List[mmap.mmap] = []
    views: List[memoryview] = []
    work: List[Tuple[memoryview, WriteEntry]] = []

    def release_views() -> None:
        # Every view into a map must be gone before the map can close.
        for buf, _ in work:
            buf.release()
        work.clear()
        for view in views:
            view.release()
        views.clear()

    try:
        for tmp, plan in zip(tmps, plans):
            fd = os.open(tmp, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o644)
            fds.append(fd)
            os.ftruncate(fd, plan.total_bytes)
            mm = mmap.mmap(fd, plan.total_bytes, access=mmap.ACCESS_WRITE)
            maps.append(mm)
            mm[: len(plan.header)] = plan.header
            view = memoryview(mm)
            views.append(view)
            for e in plan.entries:
                b, end = plan.offsets[e.name]
                work.append((view[b:end], e))
        work.sort(key=lambda w: -w[1].nbytes)
        fill(work)
        release_views()
        for mm in maps:
            if fsync:
                mm.flush()
            mm.close()
        maps.clear()
        if fsync:
            for fd in fds:
                os.fsync(fd)
        while fds:
            os.close(fds.pop())
        for tmp, plan in zip(tmps, plans):
            os.replace(tmp, plan.path)
    except BaseException:
        try:
            release_views()
        except BufferError:
            pass  # the fill still holds a view; do not mask the original error
        for mm in maps:
            try:
                mm.close()
            except BufferError:
                pass  # the fill kept a view alive; the map goes with it
        for tmp in tmps:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
        raise
    finally:
        while fds:
            os.close(fds.pop())
    if fsync and os.name != "nt":
        for d in directories:
            dfd = os.open(d, os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
    return [p.path for p in plans]


def per_entry_fill(
    fn: Callable[[memoryview, WriteEntry], None], num_threads: int = 8
) -> FillFn:
    """A FillFn running ``fn(buffer, entry)`` for each entry on ``num_threads`` threads."""
    if num_threads < 1:
        raise ValueError("num_threads must be positive")

    def fill(work: List[Tuple[memoryview, WriteEntry]]) -> None:
        if num_threads == 1 or len(work) < 2:
            for buf, e in work:
                fn(buf, e)
            return
        with ThreadPoolExecutor(num_threads) as pool:
            for f in [pool.submit(fn, buf, e) for buf, e in work]:
                f.result()

    return fill


def tensor_fill(num_threads: int = 8, *, framework: str = "pytorch") -> FillFn:
    """A FillFn writing each entry's ``source``, a framework tensor, as-is."""
    return per_entry_fill(
        lambda buf, e: _wrap_tensor(e.source, framework).copy_to_buffer(buf),
        num_threads,
    )


def save_sharded(
    entries: Sequence[WriteEntry],
    base: str,
    *,
    num_shards: int = 16,
    fill: Optional[FillFn] = None,
    framework: str = "pytorch",
    min_free_bytes: int = 0,
    fsync: bool = False,
    metadata: Optional[Mapping[str, str]] = None,
    shard0_metadata: Optional[Mapping[str, str]] = None,
    shard_metadata: Optional[ShardMetadataFn] = None,
    header_align: int = 8,
) -> List[str]:
    """Plan and write ``entries`` as ``{base}-0000i-of-0000N.safetensors``.

    ``fill`` defaults to writing each entry's ``source`` framework tensor.
    Custom encoders receive entries and buffers with the planned output sizes.
    The saver treats entry sources and application metadata as opaque.
    """
    plans = plan_shards(
        entries,
        shard_paths(base, num_shards),
        metadata=metadata,
        shard0_metadata=shard0_metadata,
        shard_metadata=shard_metadata,
        header_align=header_align,
    )
    return write_shards(
        plans,
        fill if fill is not None else tensor_fill(framework=framework),
        min_free_bytes=min_free_bytes,
        fsync=fsync,
    )


class ParallelSaver:
    """Reusable settings for synchronous sharded checkpoint writes.

    ``fill`` optionally writes every entry using a caller-provided encoder.
    ``framework`` selects the adapter for native tensor objects; callers do
    not need to wrap them. Existing TensorBase wrappers are also accepted.
    Custom callbacks receive entry sources unchanged.
    ``num_threads`` controls framework tensor writes;
    a custom callback manages its own parallelism. Callbacks must complete
    their writes before returning and must not retain buffers. File resources
    are released within each save, so no explicit close is needed.
    """

    def __init__(
        self,
        *,
        num_shards: int = 16,
        framework: str = "pytorch",
        num_threads: int = 8,
        shard_metadata: Optional[ShardMetadataFn] = None,
        fill: Optional[FillFn] = None,
        min_free_bytes: int = 0,
        fsync: bool = False,
    ) -> None:
        if num_shards < 1:
            raise ValueError("num_shards must be positive")
        if num_threads < 1:
            raise ValueError("num_threads must be positive")
        self.num_shards = num_shards
        self.framework = framework
        self.num_threads = num_threads
        self.shard_metadata = shard_metadata
        self.fill = fill
        self.min_free_bytes = min_free_bytes
        self.fsync = fsync

    def save(
        self,
        tensors: Mapping[str, Any],
        base: str,
        *,
        metadata: Optional[Mapping[str, str]] = None,
        shard0_metadata: Optional[Mapping[str, str]] = None,
        header_align: int = 8,
    ) -> List[str]:
        """Save native tensors from the selected framework, or existing wrappers.

        Custom callbacks receive the original tensor objects in entry sources.
        """
        return self.save_entries(
            [
                plain_entry(name, tensor, framework=self.framework)
                for name, tensor in tensors.items()
            ],
            base,
            metadata=metadata,
            shard0_metadata=shard0_metadata,
            header_align=header_align,
        )

    def save_entries(
        self,
        entries: Sequence[WriteEntry],
        base: str,
        *,
        metadata: Optional[Mapping[str, str]] = None,
        shard0_metadata: Optional[Mapping[str, str]] = None,
        header_align: int = 8,
    ) -> List[str]:
        """Save entries whose on-disk dtype, shape, and byte sizes are known."""
        fill = self.fill
        if fill is None:
            fill = tensor_fill(self.num_threads, framework=self.framework)
        return save_sharded(
            entries,
            base,
            num_shards=self.num_shards,
            fill=fill,
            min_free_bytes=self.min_free_bytes,
            fsync=self.fsync,
            metadata=metadata,
            shard0_metadata=shard0_metadata,
            shard_metadata=self.shard_metadata,
            header_align=header_align,
        )
