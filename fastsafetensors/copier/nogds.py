# SPDX-License-Identifier: Apache-2.0

import os
import sys
from typing import Dict, List

from .. import cpp as fstcpp
from ..common import (
    SafeTensorsMetadata,
    is_gpu_found,
    resolve_runtime_lib_name,
)
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

    def submit_io(
        self, use_buf_register: bool, max_copy_block_size: int
    ) -> fstcpp.gds_device_buffer:
        total_length = self.metadata.size_bytes - self.metadata.header_length
        gbuf = self.framework.alloc_tensor_memory(total_length, self.device)
        count = 0
        while count < total_length:
            l = total_length - count
            if max_copy_block_size < l:
                l = max_copy_block_size
            req = self.reader.submit_read(
                self.fd, gbuf, self.metadata.header_length + count, l, count
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
        # Drain every request before closing the fd so no in-flight read can
        # observe a closed descriptor, then report failures.
        failed = []
        for req in self.reqs:
            count = self.reader.wait_read(req)
            if count < 0:
                failed.append(req)
        if self.fd > 0:
            os.close(self.fd)
            self.fd = 0
        if len(failed) > 0:
            raise Exception(f"wait_io: wait_nogds_read failed, reqs={failed}")
        return self.metadata.get_tensors(
            gbuf, self.device, self.metadata.header_length, dtype=dtype
        )


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


@register_copier_constructor("nogds")
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
