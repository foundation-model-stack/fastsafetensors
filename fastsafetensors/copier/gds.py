# Copyright 2024 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Optional

from .. import cpp as fstcpp
from ..common import SafeTensorsMetadata
from ..frameworks import FrameworkOpBase, TensorBase
from ..st_types import Device, DeviceType, DType


class GdsFileCopier:
    def __init__(
        self,
        metadata: SafeTensorsMetadata,
        device: Device,
        reader: fstcpp.gds_file_reader,
        framework: FrameworkOpBase,
        debug_log: bool = False,
    ):
        self.framework = framework
        self.metadata = metadata
        self.device = device
        self.reader = reader
        self.debug_log = debug_log
        self.gbuf = None
        self.fh: Optional[fstcpp.gds_file_handle] = None
        self.copy_reqs: Dict[int, int] = {}
        self.aligned_length = 0
        cudavers = list(map(int, framework.get_cuda_ver().split(".")))
        # CUDA 12.2 (GDS version 1.7) introduces support for non O_DIRECT file descriptors
        # Compatible with CUDA 11.x
        self.o_direct = not (
            cudavers[0] > 12 or (cudavers[0] == 12 and cudavers[1] >= 2)
        )

    def set_o_direct(self, enable: bool):
        self.o_direct = enable

    def submit_io(
        self, use_buf_register: bool, max_copy_block_size: int
    ) -> fstcpp.gds_device_buffer:
        dev_is_cuda = (
            self.device.type == DeviceType.CUDA or self.device.type == DeviceType.GPU
        )
        ALIGN: int = fstcpp.get_alignment_size()
        self.fh = fstcpp.gds_file_handle(self.metadata.src, self.o_direct, dev_is_cuda)
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
                if self.debug_log:
                    print(
                        "wait_io: fix misalignment, src=0x{:x}, misaligned_bytes={}, count={}, tmp=0x{:x}".format(
                            gbuf.get_base_address(),
                            misaligned_bytes,
                            count,
                            tmp_gbuf.get_base_address(),
                        )
                    )
                gbuf.memmove(count, misaligned_bytes + count, tmp_gbuf, l)
                count += l
            self.framework.free_tensor_memory(tmp_gbuf, self.device)
            self.aligned_offset += misaligned_bytes
        return self.metadata.get_tensors(
            gbuf, self.device, self.aligned_offset, dtype=dtype
        )
