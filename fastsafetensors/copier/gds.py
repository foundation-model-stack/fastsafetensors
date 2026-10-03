# SPDX-License-Identifier: Apache-2.0

import os
import platform
import warnings
from bisect import bisect_right
from operator import itemgetter
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from .. import cpp as fstcpp
from ..common import SafeTensorsMetadata, init_logger, is_gpu_found
from ..frameworks import FrameworkOpBase, TensorBase
from ..st_types import Device, DeviceType, DType
from .base import (
    CopierInterface,
    validated_byte_ranges,
    validated_chunk_allocation_size,
)
from .nogds import load_library_func, new_nogds_file_copier
from .registry import CopierConstructFunc, register_copier_constructor

if TYPE_CHECKING:
    from ..allocation import SharedDeviceAllocation

logger = init_logger(__name__)
_request_end = itemgetter(2)

_warned_gds_fallback = False


class GdsFileCopier(CopierInterface):
    def __init__(
        self,
        metadata: SafeTensorsMetadata,
        device: Device,
        reader: fstcpp.gds_file_reader,
        framework: FrameworkOpBase,
        fallback_cache: Optional[List[CopierConstructFunc]] = None,
    ):
        self.framework = framework
        self.metadata = metadata
        self.device = device
        self.reader = reader
        self.gbuf: Optional[fstcpp.gds_device_buffer] = None
        self.fh: Optional[fstcpp.gds_file_handle] = None
        self.copy_reqs: List[int] = []
        self.aligned_length = 0
        self._readiness = False
        self._request_ranges: List[Tuple[int, int, int]] = []
        self._ready_prefix: Dict[int, int] = {}
        self._registered_offsets: List[int] = []
        self._fallback: Optional[CopierInterface] = None
        self.byte_ranges: Optional[List[Tuple[int, int]]] = None
        self._chunk_names: Optional[Set[str]] = None
        self._chunk_allocation_size: Optional[int] = None
        # One-slot cell shared by all copiers from the same factory, so a
        # broken-GDS host builds a single nogds fallback reader (and its
        # pinned bounce buffer) per loader instead of one per file.
        self._fallback_cache = fallback_cache
        cuda_ver = framework.get_cuda_ver()
        if cuda_ver and cuda_ver != "0.0":
            # Parse version string (e.g., "cuda-12.1" or "hip-5.7.0")
            # Extract the numeric part after the platform prefix
            ver_parts = cuda_ver.split("-", 1)
            if len(ver_parts) == 2:
                cudavers = list(map(int, ver_parts[1].split(".")))
                # CUDA 12.2 (GDS version 1.7) introduces support for non O_DIRECT file descriptors
                # Compatible with CUDA 11.x
                # Only applies to CUDA platform (not ROCm/HIP)
                if ver_parts[0] == "cuda":
                    self.o_direct = not (
                        cudavers[0] > 12 or (cudavers[0] == 12 and cudavers[1] >= 2)
                    )
                else:
                    # ROCm/HIP platform, use O_DIRECT
                    self.o_direct = True
            else:
                # Fallback if format is unexpected
                self.o_direct = True
        else:
            # No GPU platform detected, use O_DIRECT
            self.o_direct = True

    def set_o_direct(self, enable: bool):
        self.o_direct = enable

    def set_byte_ranges(self, byte_ranges: Optional[List[Tuple[int, int]]]) -> None:
        self.byte_ranges = validated_byte_ranges(self.metadata, byte_ranges)

    def set_chunk(
        self,
        byte_ranges: List[Tuple[int, int]],
        names: Set[str],
        allocation_size: Optional[int] = None,
    ) -> None:
        """Read selected runs into a compact, optionally budget-sized buffer.

        CUDA reads pad ranges and the device pointer to I/O alignment. The
        planner charges that padding separately from the payload budget.
        """
        checked = validated_byte_ranges(self.metadata, byte_ranges)
        assert checked is not None
        self.byte_ranges = checked
        self._chunk_names = names
        self._chunk_allocation_size = validated_chunk_allocation_size(
            checked, allocation_size
        )

    @classmethod
    def chunk_transient_multiplier(cls, paths: List[str]) -> int:
        # Only the destination scales with the chunk span. cuFile's internal
        # bounce cache is process-wide and charged as a fixed cost below.
        return 1

    @classmethod
    def chunk_device_overhead(cls, paths: List[str]) -> int:
        # Prefix, suffix, and alignment of the allocator's device pointer.
        # A CUDA allocation need not be 4 KiB aligned. CPU/fallback reads
        # may not need this reservation, but planning remains conservative.
        return 3 * (fstcpp.get_alignment_size() - 1)

    @classmethod
    def fixed_device_overhead(cls, paths: List[str]) -> int:
        return fstcpp.gds_device_cache_size()

    def enable_tensor_readiness(self) -> bool:
        # Relocating misaligned bytes in place can race outstanding DMA. Those
        # whole-file loads retain the blocking alignment repair. Partial
        # reads expose views at their final offsets, including I/O padding.
        self._readiness = self.metadata.aligned or self.byte_ranges is not None
        return self._readiness

    def prepare_tensors(
        self,
        gbuf: fstcpp.gds_device_buffer,
        owner: Optional["SharedDeviceAllocation"] = None,
    ) -> Dict[str, TensorBase]:
        if self._fallback is not None:
            return self._fallback.prepare_tensors(gbuf, owner=owner)
        return self.metadata._get_tensors(
            gbuf,
            self.device,
            self.aligned_offset,
            names=self._chunk_names,
            owner=owner,
        )

    def wait_tensor(self, name: str) -> None:
        if self._fallback is not None:
            self._fallback.wait_tensor(name)
            return
        frame = self.metadata.tensors[name]
        start = self.metadata.header_length + frame.data_offsets[0]
        end = self.metadata.header_length + frame.data_offsets[1]
        if self._chunk_names is not None and name not in self._chunk_names:
            raise ValueError(f"tensor {name} includes unread bytes")
        if start == end:
            return
        if self.byte_ranges is not None:
            # Aligned I/O may fetch adjacent unselected bytes. Those bytes do
            # not turn a skipped tensor into a requested tensor.
            selected = start
            index = bisect_right(self.byte_ranges, start, key=itemgetter(1))
            while selected < end and index < len(self.byte_ranges):
                lo, hi = self.byte_ranges[index]
                if lo > selected:
                    break
                selected = min(end, hi)
                index += 1
            if selected != end:
                raise ValueError(f"tensor {name} includes unread bytes")
        index = bisect_right(self._request_ranges, start, key=_request_end)
        covered = start
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

    def submit_io(
        self, use_buf_register: bool, max_copy_block_size: int
    ) -> fstcpp.gds_device_buffer:
        if max_copy_block_size <= 0:
            raise ValueError("max_copy_block_size must be positive")
        dev_is_cuda = (
            self.device.type == DeviceType.CUDA or self.device.type == DeviceType.GPU
        )
        ALIGN: int = fstcpp.get_alignment_size()
        partial = self.byte_ranges is not None or self._chunk_names is not None
        try:
            self.fh = fstcpp.gds_file_handle(
                # CPU pread cannot handle exact unaligned runs with O_DIRECT.
                self.metadata.src,
                self.o_direct and (dev_is_cuda or not partial),
                dev_is_cuda,
            )
        except RuntimeError as e:
            # cuFile can probe as available yet fail at I/O time: handle
            # registration errors on compat-mode hosts or unsupported
            # filesystems (e.g. overlayfs), or open(O_DIRECT) rejections.
            # Downgrade this copier to the nogds bounce path instead of
            # failing, so consumers don't each need their own gds->nogds
            # retry. Deliberately limited to file-handle setup: failures in
            # already-submitted reads stay fatal (falling back mid-cycle
            # would re-read earlier data).
            global _warned_gds_fallback
            if not _warned_gds_fallback:
                _warned_gds_fallback = True
                # str(e): keeping the exception object in the log record would
                # retain its traceback (and this frame's locals) via any
                # record-capturing handler.
                logger.warning(
                    "GDS file-handle setup failed (%s); "
                    "falling back to the nogds copier",
                    str(e),
                )
            if self._fallback_cache is not None:
                if not self._fallback_cache:
                    self._fallback_cache.append(
                        new_nogds_file_copier(self.device, framework=self.framework)
                    )
                self._fallback = self._fallback_cache[0](
                    self.metadata, self.device, self.framework
                )
            else:
                # direct construction (no factory): reader lives only for this
                # file's submit/wait cycle and is released in wait_io
                self._fallback = new_nogds_file_copier(
                    self.device, framework=self.framework
                )(self.metadata, self.device, self.framework)
            if self._chunk_names is not None:
                assert self.byte_ranges is not None
                self._fallback.set_chunk(
                    self.byte_ranges, self._chunk_names, self._chunk_allocation_size
                )
            else:
                self._fallback.set_byte_ranges(self.byte_ranges)
            if self._readiness:
                if not self._fallback.enable_tensor_readiness():
                    raise RuntimeError("GDS fallback does not support tensor readiness")
            return self._fallback.submit_io(use_buf_register, max_copy_block_size)
        if partial:
            runs = self.byte_ranges or []
            aligned_offset = self.metadata.header_length
            if self._chunk_names is not None:
                if runs:
                    aligned_offset = runs[0][0]
                span = runs[-1][1] - aligned_offset if runs else 0
            else:
                span = self.metadata.size_bytes - aligned_offset
            aligned_length = self._chunk_allocation_size or span
        else:
            offset = self.metadata.header_length
            length = self.metadata.size_bytes - offset
            aligned_offset = offset - offset % ALIGN
            aligned_length = (length + offset % ALIGN + ALIGN - 1) // ALIGN * ALIGN
            runs = [(aligned_offset, aligned_offset + aligned_length)]

        align_partial = (
            partial and dev_is_cuda and bool(runs) and max_copy_block_size >= ALIGN
        )
        if align_partial:
            aligned_offset -= aligned_offset % ALIGN
            aligned_length += self.chunk_device_overhead([self.metadata.src])
            padded: List[Tuple[int, int]] = []
            for start, end in runs:
                lo = start - start % ALIGN
                hi = (end + ALIGN - 1) // ALIGN * ALIGN
                # Neighboring runs can share a boundary page. Submit it once
                # so readiness never races a second DMA to the same bytes.
                if padded and lo <= padded[-1][1]:
                    padded[-1] = (padded[-1][0], max(padded[-1][1], hi))
                else:
                    padded.append((lo, hi))
            runs = padded
            max_copy_block_size = max_copy_block_size // ALIGN * ALIGN

        gbuf = self.framework.alloc_tensor_memory(aligned_length, self.device)
        self.gbuf = gbuf
        if align_partial:
            # Keep ownership at the allocator's original base for free().
            # Shift only the I/O destinations and the tensor-view mapping.
            aligned_offset -= (-gbuf.get_base_address()) % ALIGN
        self.aligned_offset = aligned_offset
        self.aligned_length = aligned_length
        try:
            for start, end in runs:
                count = start
                while count < end:
                    req_len = min(end - count, max_copy_block_size)
                    ptr_off = count - aligned_offset
                    # Each native request uses this exact pointer as its cuFile
                    # base. Register that base, not a different enclosing run.
                    # Exact unaligned reads (e.g. very small block limits)
                    # still use cuFile's cache and remain unregistered.
                    if use_buf_register and (
                        not partial
                        or (
                            count % ALIGN == 0
                            and req_len % ALIGN == 0
                            and (gbuf.get_base_address() + ptr_off) % ALIGN == 0
                        )
                    ):
                        if gbuf.cufile_register(ptr_off, req_len) < 0:
                            raise RuntimeError(
                                f"submit_io: register_buffer failed, offset={ptr_off}, length={req_len}"
                            )
                        self._registered_offsets.append(ptr_off)
                    args = (
                        self.fh,
                        gbuf,
                        count,
                        req_len,
                        ptr_off,
                        self.metadata.size_bytes,
                    )
                    req = (
                        self.reader.submit_read(*args, True)
                        if self._readiness
                        else self.reader.submit_read(*args)
                    )
                    if req < 0:
                        raise RuntimeError(
                            f"submit_io: submit_gds_read failed, err={req}"
                        )
                    self.copy_reqs.append(req)
                    if self._readiness:
                        self._request_ranges.append(
                            (req, count, min(count + req_len, self.metadata.size_bytes))
                        )
                    count += req_len
        except BaseException:
            try:
                self.finish_io()
            except Exception:
                pass  # Preserve the submission error after draining all requests.
            finally:
                self.framework.free_tensor_memory(gbuf, self.device)
                self.gbuf = None
            raise
        return gbuf

    def finish_io(self) -> None:
        if self._fallback is not None:
            try:
                self._fallback.finish_io()
            finally:
                self._fallback = None
            return
        error = None
        for req in self.copy_reqs:
            try:
                if self.reader.wait_read(req) < 0:
                    raise RuntimeError(f"wait_io: wait_gds_read failed, request={req}")
            except Exception as exc:
                if error is None:
                    error = exc
        self.copy_reqs.clear()
        self._request_ranges.clear()
        self._ready_prefix.clear()
        for offset in self._registered_offsets:
            try:
                if self.gbuf is not None and self.gbuf.cufile_deregister(offset) < 0:
                    raise RuntimeError("wait_io: deregister_buffer failed")
            except Exception as exc:
                if error is None:
                    error = exc
        self._registered_offsets.clear()
        self.fh = None
        if error is not None:
            raise error

    def wait_io(
        self,
        gbuf: fstcpp.gds_device_buffer,
        dtype: DType = DType.AUTO,
        noalign: bool = False,
        owner: Optional["SharedDeviceAllocation"] = None,
    ) -> Dict[str, TensorBase]:
        if self._fallback is not None:
            tensors = self._fallback.wait_io(
                gbuf, dtype=dtype, noalign=noalign, owner=owner
            )
            # Drop the fallback copier so its bounce-buffer reader is freed.
            self._fallback = None
            return tensors
        self.finish_io()
        if (
            not noalign
            and not self.metadata.aligned
            and self.aligned_length > 0
            and self.byte_ranges is None
            and self._chunk_names is None
        ):
            misaligned_bytes = (
                self.metadata.header_length % self.framework.get_device_ptr_align()
            )
            length = 1024 * 1024 * 1024
            tmp_gbuf = self.framework.alloc_tensor_memory(length, self.device)
            count = 0
            while count + misaligned_bytes < self.aligned_length:
                l = self.aligned_length - misaligned_bytes - count
                if l > length:
                    l = length
                logger.debug(
                    "wait_io: fix misalignment, src=0x%x, misaligned_bytes=%d, count=%d, tmp=0x%x",
                    gbuf.get_base_address(),
                    misaligned_bytes,
                    count,
                    tmp_gbuf.get_base_address(),
                )
                gbuf.memmove(count, misaligned_bytes + count, tmp_gbuf, l)
                count += l
            self.framework.free_tensor_memory(tmp_gbuf, self.device)
            self.aligned_offset += misaligned_bytes
        return self.metadata._get_tensors(
            gbuf,
            self.device,
            self.aligned_offset,
            dtype=dtype,
            names=self._chunk_names,
            owner=owner,
        )


_inited_gds = False


def init_gds(framework: Optional[FrameworkOpBase] = None):
    load_library_func(framework)
    global _inited_gds
    if not _inited_gds:
        if fstcpp.init_gds() != 0:
            raise Exception(f"[FAIL] init_gds()")
        _inited_gds = True


@register_copier_constructor("gds", GdsFileCopier)
def new_gds_file_copier(
    device: Device,
    bbuf_size_kb: int = 16 * 1024,
    max_threads: int = 16,
    device_memory_budget: Optional[int] = None,
    **kwargs,
) -> CopierConstructFunc:
    framework = kwargs.get("framework")
    # Capability checks depend on symbols resolved by load_library_func().
    load_library_func(framework)
    is_hip = fstcpp.is_hip_found()

    # On NVIDIA Linux hosts, check for GDS device nodes before init_gds(), which
    # invokes cuFileDriverOpen(). On hosts where the nvidia-fs kernel module
    # is loaded but /dev/nvidia-fs* device nodes are missing (common in
    # containers without device mapping), cuFileDriverOpen()'s error path
    # closes the process's stdin (fd 0) — an NVIDIA libcufile bug. This
    # corrupts subsequent subprocess calls (e.g., nvcc JIT compilation in
    # DeepGEMM). Windows and macOS never have this device node, so the check
    # is Linux-only to avoid spurious warnings and skipping init_gds on
    # platforms where the cuFile codepath is never reached. AMD hipFile does
    # not use /dev/nvidia-fs0 and must not be gated on this NVIDIA-only node.
    gds_device_available = True
    if platform.system() == "Linux" and not is_hip:
        gds_device_available = os.path.exists("/dev/nvidia-fs0")
        if not gds_device_available:
            warnings.warn(
                "GDS device node /dev/nvidia-fs0 not found; "
                "falling back to nogds copier to avoid cuFileDriverOpen() "
                "corrupting fd 0.",
                UserWarning,
            )

    device_is_not_cpu = device.type != DeviceType.CPU
    if device_is_not_cpu and not is_gpu_found():
        raise Exception(
            "[FAIL] GPU runtime library not found (expected libcudart.so, libamdhip64.so, or cudart64_XX.dll)"
        )
    nogds = False
    if not gds_device_available:
        # Skip init_gds() entirely — calling it on a half-configured host
        # triggers the close(0) bug regardless of device type (CPU or GPU).
        nogds = True
    elif device_is_not_cpu:
        gds_supported = fstcpp.is_gds_supported(
            device.index if device.index is not None else 0
        )
        if gds_supported < 0:
            raise Exception(f"is_gds_supported({device.index}) failed")
        if not fstcpp.is_cufile_found():
            # Windows does not have cuFile, do not warning about it
            if platform.system() != "Windows":
                warnings.warn(
                    "libcufile.so does not exist but nogds is False. use nogds=True",
                    UserWarning,
                )
            nogds = True
        elif gds_supported == 0:
            warnings.warn(
                "GDS is not supported in this platform but nogds is False. use nogds=True",
                UserWarning,
            )
            nogds = True

    if (
        not nogds
        and device_memory_budget is not None
        and is_hip
        and not fstcpp.is_hipfile_memory_budget_supported()
    ):
        # Select the actual copier before planning: its transient and fixed
        # costs must be used with the original budget, including on UMA hosts.
        warnings.warn(
            "AMD hipFile memory accounting is unavailable for this version; "
            "falling back to nogds while preserving device_memory_budget.",
            UserWarning,
        )
        nogds = True

    if gds_device_available and not nogds:
        init_gds(framework)

    device_id = device.index if device.index is not None else 0
    if nogds:
        # Prefer unified copier on systems with shared CPU/GPU memory
        from .unified import is_unified_memory_system, new_unified_copier

        if device_is_not_cpu and is_unified_memory_system(framework):
            return new_unified_copier(device, framework=framework)
        return new_nogds_file_copier(
            device, bbuf_size_kb, max_threads, framework=framework
        )

    reader = fstcpp.gds_file_reader(max_threads, device_is_not_cpu, device_id)

    fallback_cache: List[CopierConstructFunc] = []

    def construct_copier(
        metadata: SafeTensorsMetadata,
        device: Device,
        framework: FrameworkOpBase,
    ) -> CopierInterface:
        return GdsFileCopier(
            metadata, device, reader, framework, fallback_cache=fallback_cache
        )

    return construct_copier
