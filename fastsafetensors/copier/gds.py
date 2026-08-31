# SPDX-License-Identifier: Apache-2.0

import os
import platform
import warnings
from typing import Dict, List, Optional

from .. import cpp as fstcpp
from ..common import SafeTensorsMetadata, init_logger, is_gpu_found
from ..frameworks import FrameworkOpBase, TensorBase
from ..st_types import Device, DeviceType, DType
from .base import CopierInterface
from .nogds import load_library_func, new_nogds_file_copier
from .registry import CopierConstructFunc, register_copier_constructor

logger = init_logger(__name__)

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
        self.gbuf = None
        self.fh: Optional[fstcpp.gds_file_handle] = None
        self.copy_reqs: Dict[int, int] = {}
        self.aligned_length = 0
        self._fallback: Optional[CopierInterface] = None
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

    def submit_io(
        self, use_buf_register: bool, max_copy_block_size: int
    ) -> fstcpp.gds_device_buffer:
        dev_is_cuda = (
            self.device.type == DeviceType.CUDA or self.device.type == DeviceType.GPU
        )
        ALIGN: int = fstcpp.get_alignment_size()
        try:
            self.fh = fstcpp.gds_file_handle(
                self.metadata.src, self.o_direct, dev_is_cuda
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
            return self._fallback.submit_io(use_buf_register, max_copy_block_size)
        offset = self.metadata.header_length
        length = self.metadata.size_bytes - self.metadata.header_length
        head_bytes = offset % ALIGN
        tail_bytes = (length + head_bytes) % ALIGN
        if tail_bytes > 0:
            tail_bytes = ALIGN - tail_bytes
            aligned_length = length + head_bytes + tail_bytes
        else:
            aligned_length = length + head_bytes
        aligned_offset = offset - head_bytes

        gbuf = self.framework.alloc_tensor_memory(aligned_length, self.device)
        if use_buf_register:
            count = 0
            while count < aligned_length:
                req_len = aligned_length - count
                if req_len > max_copy_block_size:
                    req_len = max_copy_block_size
                if gbuf.cufile_register(count, req_len) < 0:
                    raise Exception(
                        "submit_io: register_buffer failed, ptr=0x{:x}, count={}, len={}".format(
                            gbuf.get_base_address(), count, req_len
                        )
                    )
                count += req_len

        count = 0
        while count < aligned_length:
            req_len = aligned_length - count
            if req_len > max_copy_block_size:
                req_len = max_copy_block_size
            # TODO: pass timeout so that wait_copy_tensors can recognize too slow pread()
            req = self.reader.submit_read(
                self.fh,
                gbuf,
                aligned_offset + count,
                req_len,
                count,
                self.metadata.size_bytes,
            )
            if req < 0:
                raise Exception(f"submit_io: submit_gds_read failed, err={req}")
            self.copy_reqs[req] = -1 if not use_buf_register else count
            count += req_len
        self.aligned_offset = aligned_offset
        self.aligned_length = aligned_length
        return gbuf

    def wait_io(
        self,
        gbuf: fstcpp.gds_device_buffer,
        dtype: DType = DType.AUTO,
        noalign: bool = False,
    ) -> Dict[str, TensorBase]:
        if self._fallback is not None:
            tensors = self._fallback.wait_io(gbuf, dtype=dtype, noalign=noalign)
            # Drop the fallback copier so its bounce-buffer reader is freed.
            self._fallback = None
            return tensors
        failed = []
        for req, c in sorted(self.copy_reqs.items(), key=lambda x: x[0]):
            count = self.reader.wait_read(req)
            if count < 0:
                failed.append(req)
            if c != -1:
                gbuf.cufile_deregister(c)
        if self.fh is not None:
            del self.fh
            self.fh = None
        if len(failed) > 0:
            raise Exception(
                f"wait_io: wait_gds_read failed, failed={failed}, reqs={self.copy_reqs}"
            )
        self.copy_reqs = {}
        if not noalign and not self.metadata.aligned and self.aligned_length > 0:
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
        return self.metadata.get_tensors(
            gbuf, self.device, self.aligned_offset, dtype=dtype
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
    **kwargs,
) -> CopierConstructFunc:
    # On Linux, check for GDS device nodes before calling init_gds(), which
    # invokes cuFileDriverOpen(). On hosts where the nvidia-fs kernel module
    # is loaded but /dev/nvidia-fs* device nodes are missing (common in
    # containers without device mapping), cuFileDriverOpen()'s error path
    # closes the process's stdin (fd 0) — an NVIDIA libcufile bug. This
    # corrupts subsequent subprocess calls (e.g., nvcc JIT compilation in
    # DeepGEMM). Windows and macOS never have this device node, so the check
    # is Linux-only to avoid spurious warnings and skipping init_gds on
    # platforms where the cuFile codepath is never reached.
    gds_device_available = True
    if platform.system() == "Linux":
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

    if gds_device_available and not nogds:
        init_gds(kwargs.get("framework"))

    device_id = device.index if device.index is not None else 0
    if nogds:
        # Prefer unified copier on systems with shared CPU/GPU memory
        from .unified import is_unified_memory_system, new_unified_copier

        if device_is_not_cpu and is_unified_memory_system(kwargs.get("framework")):
            return new_unified_copier(device, framework=kwargs.get("framework"))
        return new_nogds_file_copier(
            device, bbuf_size_kb, max_threads, framework=kwargs.get("framework")
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
