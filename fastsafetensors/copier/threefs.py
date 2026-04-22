# SPDX-License-Identifier: Apache-2.0

from typing import Dict

from fastsafetensors import cpp as fstcpp
from fastsafetensors.common import SafeTensorsMetadata, init_logger
from fastsafetensors.copier.base import CopierInterface
from fastsafetensors.copier.registry import (
    CopierConstructFunc,
    register_copier_constructor,
)
from fastsafetensors.frameworks import FrameworkOpBase, TensorBase
from fastsafetensors.st_types import Device, DType

logger = init_logger(__name__)


class ThreeFSFileCopier(CopierInterface):
    def __init__(
        self,
        metadata: SafeTensorsMetadata,
        device: Device,
        reader,  # duck-typed: must have read_chunked(path, dev_ptr, file_offset, total_length, chunk_size) -> int
        framework: FrameworkOpBase,
    ):
        self.framework = framework
        self.metadata = metadata
        self.device = device
        self.reader = reader

    def submit_io(
        self, use_buf_register: bool, max_copy_block_size: int
    ) -> fstcpp.gds_device_buffer:
        offset = self.metadata.header_length
        length = self.metadata.size_bytes - self.metadata.header_length

        gbuf = self.framework.alloc_tensor_memory(length, self.device)

        logger.info(
            f"Reading {length} bytes from {self.metadata.src} using chunked read"
        )

        total_read = self.reader.read_chunked(
            path=self.metadata.src,
            dev_ptr=gbuf.get_base_address(),
            file_offset=offset,
            total_length=length,
            chunk_size=max_copy_block_size if max_copy_block_size > 0 else 0,
        )

        if total_read != length:
            raise Exception(
                f"ThreeFSFileCopier.submit_io: incomplete read, "
                f"expected={length}, actual={total_read}"
            )

        logger.info(f"Successfully read {total_read} bytes")

        return gbuf

    def wait_io(
        self,
        gbuf: fstcpp.gds_device_buffer,
        dtype: DType = DType.AUTO,
        noalign: bool = False,
    ) -> Dict[str, TensorBase]:
        # read_chunked is synchronous; data is fully read in submit_io
        return self.metadata.get_tensors(
            gbuf, self.device, self.metadata.header_length, dtype=dtype
        )


@register_copier_constructor("3fs")
def new_threefs_file_copier(
    device: Device,
    mount_point: str,
    entries: int = 64,
    io_depth: int = 0,
    buffer_size: int = 64 * 1024 * 1024,
    **kwargs,
) -> CopierConstructFunc:
    from fastsafetensor_3fs_reader import ThreeFSFileReader

    reader = ThreeFSFileReader(
        mount_point=mount_point,
        entries=entries,
        io_depth=io_depth,
        buffer_size=buffer_size,
    )

    def construct_copier(
        metadata: SafeTensorsMetadata,
        device: Device,
        framework: FrameworkOpBase,
    ) -> CopierInterface:
        return ThreeFSFileCopier(metadata, device, reader, framework)

    construct_copier.reader = reader  # type: ignore[attr-defined]

    return construct_copier
