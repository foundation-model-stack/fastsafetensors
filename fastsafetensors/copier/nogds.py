# SPDX-License-Identifier: Apache-2.0

import errno
import os
import sys
from bisect import bisect_right
from operator import itemgetter
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from .. import cpp as fstcpp
from ..common import (
    ODIRECT_HINT,
    SafeTensorsMetadata,
    get_device_numa_node,
    init_logger,
    is_gpu_found,
    is_odirect_enabled,
    resolve_runtime_lib_name,
)
from ..frameworks import FrameworkOpBase, TensorBase
from ..st_types import Device, DeviceType, DType
from .base import (
    CopierInterface,
    validated_byte_ranges,
    validated_chunk_allocation_size,
)
from .registry import CopierConstructFunc, register_copier_constructor

if TYPE_CHECKING:
    from ..allocation import SharedDeviceAllocation

_request_end = itemgetter(2)
logger = init_logger(__name__)
_warned_odirect_rejected = False


def _open_checkpoint(path: str, flags: int, direct: bool) -> int:
    """Open ``path`` for reading, with O_DIRECT when ``direct`` is set.

    The filesystem policy in ``is_odirect_enabled`` cannot know whether this
    particular file accepts direct I/O; tmpfs before Linux 6.6 and some FUSE
    filesystems refuse it at open with EINVAL. Fall back to a buffered
    descriptor in that case, as the gds and unified copiers do, instead of
    failing a load that worked before O_DIRECT became the default.
    """
    global _warned_odirect_rejected
    o_direct = getattr(os, "O_DIRECT", 0)
    if direct and o_direct:
        try:
            return os.open(path, flags | o_direct, 0o644)
        except OSError as error:
            if error.errno != errno.EINVAL:
                raise
            if not _warned_odirect_rejected:
                _warned_odirect_rejected = True
                logger.warning(
                    "filesystem rejected O_DIRECT for %s: using buffered reads",
                    path,
                )
    return os.open(path, flags, 0o644)


def _fd_is_odirect(fd: int) -> bool:
    o_direct = getattr(os, "O_DIRECT", 0)
    if not o_direct or sys.platform == "win32":
        return False
    import fcntl

    try:
        return bool(fcntl.fcntl(fd, fcntl.F_GETFL) & o_direct)
    except OSError:
        return False


class NoGdsFileCopier(CopierInterface):
    def __init__(
        self,
        metadata: SafeTensorsMetadata,
        device: Device,
        reader: fstcpp.nogds_file_reader,
        framework: FrameworkOpBase,
    ):
        self.framework = framework
        self.metadata = metadata
        self.reader = reader
        flags = os.O_RDONLY
        # On Windows, O_RDONLY defaults to text mode which translates \r\n
        # and stops at 0x1A (Ctrl+Z), corrupting binary tensor data.
        if sys.platform == "win32" and hasattr(os, "O_BINARY"):
            flags |= os.O_BINARY
        self.fd = _open_checkpoint(
            metadata.src,
            flags,
            is_odirect_enabled(metadata.src, buffered_ram_fs=True),
        )
        if self.fd < 0:
            raise Exception(
                f"NoGdsFileCopier.__init__: failed to open, file={metadata.src}"
            )
        self.device = device
        self.reqs: List[int] = []
        self.byte_ranges: Optional[List[Tuple[int, int]]] = None
        self._chunk_names: Optional[Set[str]] = None
        self._chunk_allocation_size: Optional[int] = None
        self._base_off = metadata.header_length
        self._readiness = False
        self._request_ranges: List[Tuple[int, int, int]] = []
        self._ready_prefix: Dict[int, int] = {}

    def enable_tensor_readiness(self) -> bool:
        self._readiness = True
        return True

    def prepare_tensors(
        self,
        gbuf: fstcpp.gds_device_buffer,
        owner: Optional["SharedDeviceAllocation"] = None,
    ) -> Dict[str, TensorBase]:
        # AUTO view construction does not access tensor contents. Online dtype
        # conversion still uses blocking wait_io, since it reads/writes bytes.
        return self.metadata._get_tensors(
            gbuf, self.device, self._base_off, names=self._chunk_names, owner=owner
        )

    def wait_tensor(self, name: str) -> None:
        frame = self.metadata.tensors[name]
        start = self.metadata.header_length + frame.data_offsets[0]
        end = self.metadata.header_length + frame.data_offsets[1]
        if start == end:
            return
        covered = start
        # Requests are ordered and disjoint. Skip earlier requests without
        # rescanning them for every tensor, including out-of-order access.
        index = bisect_right(self._request_ranges, start, key=_request_end)
        while covered < end and index < len(self._request_ranges):
            req, lo, hi = self._request_ranges[index]
            if lo > covered:
                break
            needed = min(end, hi) - lo
            if self._ready_prefix.get(req, 0) < needed:
                self._ready_prefix[req] = self.reader.wait_read_prefix(req, needed)
            covered = min(end, hi)
            index += 1
        if covered != end:
            raise ValueError(f"tensor {name} includes unread bytes")

    def set_byte_ranges(self, byte_ranges: Optional[List[Tuple[int, int]]]) -> None:
        """Restrict reads to these ``[start, end)`` absolute file-offset runs.

        Bytes outside the given runs are not read; their regions of the device
        buffer are left uninitialized, so the corresponding tensors must not be
        requested. ``None`` (the default) reads the whole data section. Build
        runs with ``SafeTensorsMetadata.select_byte_ranges``.
        """
        self.byte_ranges = validated_byte_ranges(self.metadata, byte_ranges)

    def set_chunk(
        self,
        byte_ranges: List[Tuple[int, int]],
        names: Set[str],
        allocation_size: Optional[int] = None,
    ) -> None:
        """Load ``names`` into a compact or fixed-size device buffer.

        Unlike ``set_byte_ranges`` (which still allocates the whole data section
        and leaves it sparsely filled), this allocates only
        ``max_end - min_start`` by default, or ``allocation_size`` when set.
        """
        checked_ranges = validated_byte_ranges(self.metadata, byte_ranges)
        assert checked_ranges is not None
        self.byte_ranges = checked_ranges
        self._chunk_names = names
        self._chunk_allocation_size = validated_chunk_allocation_size(
            checked_ranges, allocation_size
        )

    @classmethod
    def chunk_transient_multiplier(cls, paths: List[str]) -> int:
        """Per in-flight-chunk transient cost, as a multiple of chunk span: 1.

        Reads land in the reader's fixed pool of host bounce buffers, so the
        only device-side allocation that scales with a chunk is the chunk
        buffer itself.
        """
        return 1

    @classmethod
    def fixed_device_overhead(cls, paths: List[str]) -> int:
        """The reader's bounce buffers are host memory on discrete GPUs."""
        return 0

    def submit_io(
        self, use_buf_register: bool, max_copy_block_size: int
    ) -> fstcpp.gds_device_buffer:
        if max_copy_block_size <= 0:
            raise ValueError("max_copy_block_size must be positive")
        header_length = self.metadata.header_length
        # Default to a single run spanning the whole data section, which
        # reproduces the original full-file read.
        runs = self.byte_ranges
        if runs is None:
            runs = [(header_length, self.metadata.size_bytes)]
        if self._chunk_names is not None:
            # Compact chunk: allocate only the runs' span and map gbuf[0] to the
            # first run's start, so peak memory tracks the chunk, not the shard.
            base_off = min(s for s, _ in runs)
            chunk_span = max(e for _, e in runs) - base_off
            alloc_length = self._chunk_allocation_size or chunk_span
        else:
            base_off = header_length
            alloc_length = self.metadata.size_bytes - header_length
        self._base_off = base_off
        gbuf = self.framework.alloc_tensor_memory(alloc_length, self.device)
        try:
            for start, end in runs:
                count = start
                while count < end:
                    l = end - count
                    if max_copy_block_size < l:
                        l = max_copy_block_size
                    if self._readiness:
                        req = self.reader.submit_read(
                            self.fd, gbuf, count, l, count - base_off, True
                        )
                    else:
                        req = self.reader.submit_read(
                            self.fd, gbuf, count, l, count - base_off
                        )
                    if req < 0:
                        raise Exception(
                            f"submit_io: submit_nogds_read failed, err={req}"
                        )
                    self.reqs.append(req)
                    if self._readiness:
                        self._request_ranges.append((req, count, count + l))
                    count += l
        except BaseException:
            try:
                self.finish_io()
            finally:
                self.framework.free_tensor_memory(gbuf, self.device)
            raise
        return gbuf

    def wait_io(
        self,
        gbuf: fstcpp.gds_device_buffer,
        dtype: DType = DType.AUTO,
        noalign: bool = False,
        owner: Optional["SharedDeviceAllocation"] = None,
    ) -> Dict[str, TensorBase]:
        self.finish_io()
        return self.metadata._get_tensors(
            gbuf,
            self.device,
            self._base_off,
            dtype=dtype,
            names=self._chunk_names,
            owner=owner,
        )

    def finish_io(self) -> None:
        # Drain every request before closing the fd so no in-flight read can
        # observe a closed descriptor, then report failures.
        failed = []
        for req in self.reqs:
            count = self.reader.wait_read(req)
            if count == 0:
                failed.append(req)
        self.reqs.clear()
        self._request_ranges.clear()
        self._ready_prefix.clear()
        # Check the live flag, not the open-time choice: the native reader
        # clears O_DIRECT on the descriptor when direct reads are rejected.
        direct = self.fd >= 0 and len(failed) > 0 and _fd_is_odirect(self.fd)
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1
        if len(failed) > 0:
            hint = f" (reads used O_DIRECT; {ODIRECT_HINT})" if direct else ""
            raise Exception(f"wait_io: wait_nogds_read failed, reqs={failed}{hint}")


_loaded_library = False


def load_library_func(framework=None):
    global _loaded_library
    if _loaded_library:
        return

    lib = resolve_runtime_lib_name(framework)
    fstcpp.load_library_functions(lib)
    if lib and not is_gpu_found():
        # The framework hinted a specific vendor's runtime but loading it found
        # no GPU. A GPU-built framework only reports a vendor when it sees a
        # device, so this is a real mismatch (wrong/missing runtime for that
        # vendor).
        raise Exception(
            f"[FAIL] framework hinted GPU runtime '{lib}' but no GPU was found "
            "after loading it (runtime/devices for that vendor not present)"
        )
    _loaded_library = True


@register_copier_constructor("nogds", NoGdsFileCopier)
def new_nogds_file_copier(
    device: Device,
    bbuf_size_kb: int = 16 * 1024,
    max_threads: int = 16,
    **kwargs,
) -> CopierConstructFunc:
    load_library_func(kwargs.get("framework"))
    device_is_not_cpu = device.type != DeviceType.CPU
    if device_is_not_cpu and not is_gpu_found():
        raise Exception(
            "[FAIL] GPU runtime library not found (expected libcudart.so, libamdhip64.so, or cudart64_XX.dll)"
        )

    device_id = device.index if device.index is not None else 0
    numa_node = (
        get_device_numa_node(device_id)
        if device_is_not_cpu and kwargs.get("set_numa", True)
        else None
    )
    nogds_reader = fstcpp.nogds_file_reader(
        False,
        bbuf_size_kb,
        max_threads,
        device_is_not_cpu,
        device_id,
        numa_node if numa_node is not None else -1,
    )

    def construct_nogds_copier(
        metadata: SafeTensorsMetadata,
        device: Device,
        framework: FrameworkOpBase,
    ) -> CopierInterface:
        return NoGdsFileCopier(metadata, device, nogds_reader, framework)

    return construct_nogds_copier
