# SPDX-License-Identifier: Apache-2.0

import os
import sys
from typing import Dict, List, Optional, Tuple

from .. import cpp as fstcpp
from ..common import SafeTensorsMetadata, is_gpu_found, resolve_cudart_lib_name
from ..frameworks import FrameworkOpBase, TensorBase
from ..st_types import Device, DeviceType, DType
from .base import CopierInterface
from .registry import CopierConstructFunc, register_copier_constructor


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
        self.fd = os.open(metadata.src, flags, 0o644)
        if self.fd < 0:
            raise Exception(
                f"NoGdsFileCopier.__init__: failed to open, file={metadata.src}"
            )
        self.device = device
        self.reqs: List[int] = []
        self.byte_ranges: Optional[List[Tuple[int, int]]] = None

    def set_byte_ranges(self, byte_ranges: Optional[List[Tuple[int, int]]]) -> None:
        """Restrict reads to these ``[start, end)`` absolute file-offset runs.

        Bytes outside the given runs are not read; their regions of the device
        buffer are left uninitialized, so the corresponding tensors must not be
        requested. ``None`` (the default) reads the whole data section. Build
        runs with ``SafeTensorsMetadata.select_byte_ranges``.
        """
        self.byte_ranges = byte_ranges

    def submit_io(
        self, use_buf_register: bool, max_copy_block_size: int
    ) -> fstcpp.gds_device_buffer:
        header_length = self.metadata.header_length
        total_length = self.metadata.size_bytes - header_length
        gbuf = self.framework.alloc_tensor_memory(total_length, self.device)
        # Default to a single run spanning the whole data section, which
        # reproduces the original full-file read.
        runs = self.byte_ranges
        if runs is None:
            runs = [(header_length, self.metadata.size_bytes)]
        for start, end in runs:
            count = start
            while count < end:
                l = end - count
                if max_copy_block_size < l:
                    l = max_copy_block_size
                req = self.reader.submit_read(
                    self.fd, gbuf, count, l, count - header_length
                )
                if req < 0:
                    raise Exception(f"submit_io: submit_nogds_read failed, err={req}")
                self.reqs.append(req)
                count += l
        return gbuf

    def wait_io(
        self,
        gbuf: fstcpp.gds_device_buffer,
        dtype: DType = DType.AUTO,
        noalign: bool = False,
    ) -> Dict[str, TensorBase]:
        for req in self.reqs:
            count = self.reader.wait_read(req)
            if count < 0:
                raise Exception(f"wait_io: wait_nogds_read failed, req={req}")
        if self.fd > 0:
            os.close(self.fd)
            self.fd = 0
        return self.metadata.get_tensors(
            gbuf, self.device, self.metadata.header_length, dtype=dtype
        )


_loaded_library = False


def load_library_func():
    global _loaded_library
    if not _loaded_library:
        cudart_lib = resolve_cudart_lib_name()
        fstcpp.load_library_functions(cudart_lib)
        _loaded_library = True


@register_copier_constructor("nogds")
def new_nogds_file_copier(
    device: Device,
    bbuf_size_kb: int = 16 * 1024,
    max_threads: int = 16,
    **kwargs,
) -> CopierConstructFunc:
    load_library_func()
    device_is_not_cpu = device.type != DeviceType.CPU
    if device_is_not_cpu and not is_gpu_found():
        raise Exception(
            "[FAIL] GPU runtime library not found (expected libcudart.so, libamdhip64.so, or cudart64_XX.dll)"
        )

    device_id = device.index if device.index is not None else 0
    nogds_reader = fstcpp.nogds_file_reader(
        False, bbuf_size_kb, max_threads, device_is_not_cpu, device_id
    )

    def construct_nogds_copier(
        metadata: SafeTensorsMetadata,
        device: Device,
        framework: FrameworkOpBase,
    ) -> CopierInterface:
        return NoGdsFileCopier(metadata, device, nogds_reader, framework)

    return construct_nogds_copier
