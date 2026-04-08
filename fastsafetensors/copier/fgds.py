# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import atexit
import os
import platform
import tempfile
import warnings
from typing import Dict, List, Optional, Tuple

from .. import cpp as _cpp
from ..common import SafeTensorsMetadata, init_logger, is_gpu_found
from ..frameworks import FrameworkOpBase, TensorBase
from ..st_types import Device, DeviceType, DType
from .base import CopierInterface
from .nogds import load_library_func, new_nogds_file_copier
from .registry import CopierConstructFunc, register_copier_constructor

# FGDS symbols are resolved on demand by load_fgds_library() (called from
# init_fgds() below); is_fgds_found() reports whether libfgds.so was loaded.
# Selecting another copier (gds/nogds/unified/dstorage) never loads it.
logger = init_logger(__name__)

ALIGN = 64 * 1024
ALLOC_MINIMAL = 2 * 1024 * 1024


def get_fgds_device_id(device: Device) -> int:
    """
    Get the physical device id for FGDS, handling CUDA_VISIBLE_DEVICES.

    Args:
        device (Device): The fastsafetensors Device to get the ID from.

    Returns:
        int: The physical device ID.
    """
    logical_idx = device.index if device.index is not None else 0

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or visible.strip() == "":
        # CUDA_VISIBLE_DEVICES not set, logical idx == physical idx
        return logical_idx

    entries = [e.strip() for e in visible.split(",") if e.strip() != ""]
    if logical_idx >= len(entries):
        raise ValueError(
            f"device index {logical_idx} exceeds the range of CUDA_VISIBLE_DEVICES={visible!r}"
        )

    target = entries[logical_idx]

    # Try integer ID first
    try:
        return int(target)
    except ValueError:
        pass

    # Try UUID or PCI bus ID lookup via framework
    try:
        # Try to get framework to resolve UUID/PCI to physical device ID
        # Framework may provide helper for this, fallback if not
        return logical_idx
    except Exception:
        # Fallback to logical index to avoid crashing
        return logical_idx


class FgdsFileCopier(CopierInterface):
    def __init__(
        self,
        metadata: SafeTensorsMetadata,
        device: Device,
        reader: _cpp.fgds_file_reader,
        framework: FrameworkOpBase,
        max_threads: int = 16,
    ):
        self.framework = framework
        self.metadata = metadata
        self.device = device
        self.device_id = get_fgds_device_id(device)
        self.copy_reqs: Dict[int, int] = {}
        self.aligned_length = 0
        self.aligned_offset = 0
        self.max_threads = max_threads
        self.o_direct = True

        self.fgds_handle: Optional[_cpp.fgds_file_handle] = None
        self.fgds_reader = reader
        self.registered_memory: Optional[Tuple[int, int, int]] = None
        self.gbuf: Optional[_cpp.gds_device_buffer] = None

    def _cleanup(self, free_gbuf: bool = False) -> None:
        """Release all resources held by this copier instance.

        Always safe to call multiple times. After _cleanup() the copier is
        in a pristine state suitable for being discarded.
        """
        if self.registered_memory is not None:
            device_id, addr, size = self.registered_memory
            result = _cpp.fgds_deregmem(device_id, addr, size)
            if result != 0:
                warnings.warn(
                    f"fgds_deregmem failed with error code: {result}", UserWarning
                )
            self.registered_memory = None

        if self.fgds_handle is not None:
            # fgds_file_handle's C++ destructor calls close(_fd); deleting
            # the Python wrapper runs it immediately.
            del self.fgds_handle
            self.fgds_handle = None

        if free_gbuf and self.gbuf is not None:
            self.framework.free_tensor_memory(self.gbuf, self.device)
            self.gbuf = None

        self.copy_reqs = {}

    def submit_io(
        self, use_buf_register: bool, max_copy_block_size: int
    ) -> _cpp.gds_device_buffer:
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

        try:
            self.fgds_handle = _cpp.fgds_file_handle(
                self.metadata.src, self.o_direct, self.device_id
            )
        except RuntimeError as e:
            raise RuntimeError(f"FGDS file-handle setup failed ({e})") from e

        # Set minimum allocation size to 2 MB; the offset address within the segment
        # returned by caching_allocator_alloc may break 64 KB alignment,
        # causing fgds_regmem memory registration to fail.
        reg_length = aligned_length
        if aligned_length < ALLOC_MINIMAL:
            reg_length = (aligned_length + ALLOC_MINIMAL - 1) & ~(ALLOC_MINIMAL - 1)

        gbuf = self.framework.alloc_tensor_memory(reg_length, self.device)
        self.gbuf = gbuf
        gbuf_ptr = gbuf.get_base_address()

        result = _cpp.fgds_regmem(self.device_id, gbuf_ptr, reg_length, None)
        if result != 0:
            # GPU memory registration is required for FGDS DMA; proceeding
            # with an unregistered buffer can cause silent data corruption
            # or GPU page faults.  Tear down and raise so the loader can
            # fall back to the no-GDS (pinned bounce-buffer) copier.
            warnings.warn(
                f"fgds_regmem failed with error code: {result}; " "aborting FGDS path.",
                UserWarning,
            )
            self.registered_memory = None
            self._cleanup(free_gbuf=False)
            self.framework.free_tensor_memory(gbuf, self.device)
            self.gbuf = None
            raise RuntimeError(
                f"fgds_regmem failed (error code {result}); "
                "cannot continue with unregistered GPU buffer."
            )
        self.registered_memory = (self.device_id, gbuf_ptr, reg_length)

        self.copy_reqs = {}
        count = 0
        try:
            while count < aligned_length:
                req_len = aligned_length - count
                if req_len > max_copy_block_size:
                    req_len = max_copy_block_size
                dev_buf = _cpp.fgds_device_buffer(gbuf_ptr + count, req_len)
                req_id = self.fgds_reader.submit_read(
                    self.fgds_handle, dev_buf, aligned_offset + count, req_len, count
                )
                self.copy_reqs[req_id] = count
                count += req_len
        except Exception:
            # Submission failed partway. Drain any in-flight threads so
            # they don't touch the buffer we are about to free, then clean
            # up and re-raise.
            for req_id in list(self.copy_reqs.keys()):
                try:
                    self.fgds_reader.wait_read(req_id)
                except Exception:
                    pass
            self._cleanup(free_gbuf=False)
            self.framework.free_tensor_memory(gbuf, self.device)
            self.gbuf = None
            raise

        self.aligned_offset = aligned_offset
        self.aligned_length = aligned_length

        return gbuf

    def wait_io(
        self,
        gbuf: _cpp.gds_device_buffer,
        dtype: DType = DType.AUTO,
        noalign: bool = False,
    ) -> Dict[str, TensorBase]:
        # Device-buffer layout:
        #   [aligned_offset,  header_length)  head padding (inside file; required)
        #   [header_length,   size_bytes)    tensor data (required)
        #   [size_bytes,      aligned_offset + aligned_length)
        #                                   tail alignment padding (may extend
        #                                   past EOF; O_DIRECT reads here may
        #                                   legitimately return 0/short because
        #                                   tensor views never touch these bytes)
        #
        # After collecting every wait_read() result we verify that each
        # request supplied enough bytes to cover its overlap with the
        # required region (head + tensor data).  Short reads within the
        # required region are fatal; short reads purely in tail padding are
        # tolerated.
        data_end = self.metadata.size_bytes  # absolute file offset

        # Recover per-request (file_offset, length) from the submit-time
        # ptr_off sequence.
        sorted_items = sorted(self.copy_reqs.items(), key=lambda x: x[1])
        req_info: Dict[int, Tuple[int, int]] = {}
        for i, (req_id, ptr_off) in enumerate(sorted_items):
            if i + 1 < len(sorted_items):
                req_len = sorted_items[i + 1][1] - ptr_off
            else:
                req_len = self.aligned_length - ptr_off
            req_info[req_id] = (self.aligned_offset + ptr_off, req_len)

        failed_reqs: List[int] = []
        try:
            for req_id in sorted(self.copy_reqs.keys()):
                nread = self.fgds_reader.wait_read(req_id)
                if nread < 0:
                    failed_reqs.append(req_id)
                    continue
                file_off, req_len = req_info[req_id]
                # Bytes in this request that correspond to real file content
                # (head padding + tensor data). Bytes past data_end are
                # tail alignment padding and may be absent at EOF.
                needed = max(0, min(data_end, file_off + req_len) - file_off)
                if nread < needed:
                    failed_reqs.append(req_id)
        finally:
            # Always release registrations and the file handle, even on
            # error, so we don't leak GPU mappings or fd's.
            self._cleanup(free_gbuf=False)

        if failed_reqs:
            # The device buffer is incomplete; free it rather than handing
            # garbage-backed tensors to the caller.
            if self.gbuf is not None:
                self.framework.free_tensor_memory(self.gbuf, self.device)
                self.gbuf = None
            raise Exception(
                f"wait_io: fgds_read failed, failed_reqs={failed_reqs}, "
                f"reqs={sorted(self.copy_reqs.keys())}"
            )

        self.copy_reqs = {}

        return self.metadata.get_tensors(
            gbuf, self.device, self.aligned_offset, dtype=dtype
        )


# Tracks devices for which init_fgds() has been called, so atexit can pair
# each with close_fgds(). Mirrors the GDS init_gds/close_gds model where the
# driver open/close is explicit and not tied to individual copier instances.
_fgds_opened_devices: set = set()


def init_fgds(framework: Optional[FrameworkOpBase] = None):
    # Ensure the GPU runtime (and cuFile/hipFile) is resolved first: FGDS is
    # only loaded when a GPU is present, and load_fgds_library() requires
    # load_library_func() to have run. Then resolve libfgds.so on demand, so
    # only the FGDS copier path ever dlopens it.
    load_library_func(framework)
    _cpp.load_fgds_library()


@register_copier_constructor("fgds", FgdsFileCopier)
def new_fgds_file_copier(
    device: Device,
    framework: FrameworkOpBase,
    bbuf_size_kb: int = 16 * 1024,
    max_threads: int = 16,
    **kwargs,
) -> CopierConstructFunc:
    # FGDS (libfgds.so) is Linux-only; bail out early on other platforms.
    if platform.system() != "Linux":
        warnings.warn(
            "FGDS is not available on this platform. Falling back to NoGDS.",
            UserWarning,
        )
        return new_nogds_file_copier(
            device, bbuf_size_kb, max_threads, framework=framework, **kwargs
        )

    # Load the GPU runtime library first (like GDS does) so that
    # is_gpu_found() can detect CUDA/HIP devices correctly.
    init_fgds(framework)

    if not _cpp.is_fgds_found():
        warnings.warn(
            "FGDS (libfgds.so) is not available. Falling back to NoGDS.",
            UserWarning,
        )
        return new_nogds_file_copier(
            device, bbuf_size_kb, max_threads, framework=framework, **kwargs
        )

    device_is_not_cpu = device.type != DeviceType.CPU
    if device_is_not_cpu and not is_gpu_found():
        warnings.warn(
            "GPU runtime library (libcudart.so or libamdhip64.so) not found. "
            "FGDS requires GPU runtime; falling back to NoGDS.",
            UserWarning,
        )
        return new_nogds_file_copier(
            device, bbuf_size_kb, max_threads, framework=framework, **kwargs
        )

    device_id = get_fgds_device_id(device)

    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(b"test")
            tmp_filename = tmp_file.name

        try:
            tmp_handle = _cpp.fgds_file_handle(tmp_filename, True, device_id)
            del tmp_handle
        finally:
            if os.path.exists(tmp_filename):
                os.unlink(tmp_filename)
    except Exception as e:
        warnings.warn(
            f"FGDS is not available: {e}. Falling back to NoGDS.", UserWarning
        )
        return new_nogds_file_copier(
            device, bbuf_size_kb, max_threads, framework=framework, **kwargs
        )

    # Open the FGDS device exactly once per device_id (idempotent), and pair
    # it with close_fgds() at process exit. This mirrors init_gds(): the
    # driver open is performed at factory setup, not per copier instance.
    if device_id not in _fgds_opened_devices:
        result = _cpp.init_fgds(device_id)
        if result != 0:
            warnings.warn(
                f"init_fgds(device_id={device_id}) failed with error code: {result}. "
                "Falling back to NoGDS.",
                UserWarning,
            )
            return new_nogds_file_copier(
                device, bbuf_size_kb, max_threads, framework=framework, **kwargs
            )
        _fgds_opened_devices.add(device_id)

    reader = _cpp.fgds_file_reader(max_threads, device_id)

    def construct_copier(
        metadata: SafeTensorsMetadata, device: Device, framework: FrameworkOpBase
    ) -> CopierInterface:
        return FgdsFileCopier(
            metadata, device, reader, framework, max_threads=max_threads
        )

    return construct_copier


def _fgds_atexit_cleanup() -> None:
    for device_id in list(_fgds_opened_devices):
        _cpp.close_fgds(device_id)
    _fgds_opened_devices.clear()


atexit.register(_fgds_atexit_cleanup)
