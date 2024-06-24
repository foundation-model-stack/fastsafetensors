# Copyright 2024 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0

import torch
import os
from .. import cpp as fstcpp
from typing import Dict
from ..common import alloc_tensor_memory, SafeTensorsMetadata, ALIGN, CUDA_PTR_ALIGN

class NoGdsFileCopier:
    def __init__(self, metadata: SafeTensorsMetadata, device: torch.device, reader: fstcpp.nogds_file_reader, debug_log: bool=False):
        self.metadata = metadata
        self.reader = reader
        self.fd = os.open(metadata.src, os.O_RDONLY, 0o644)
        if self.fd < 0:
            raise Exception(f"NoGdsFileCopier.__init__: failed to open, file={metadata.src}")
        self.device = device
        self.debug_log = debug_log
        self.reqs = []

    def submit_io(self, use_buf_register: bool, max_copy_block_size: int)->fstcpp.gds_device_buffer:
        total_length = self.metadata.size_bytes - self.metadata.header_length
        gbuf = alloc_tensor_memory(total_length)
        count = 0
        while count < total_length:
            l = total_length - count
            if max_copy_block_size < l:
                l = max_copy_block_size
            req = self.reader.submit_read(self.fd, gbuf, self.metadata.header_length + count, l, count)
            if req < 0:
                raise Exception(f"submit_io: submit_nogds_read failed, err={req}")
            self.reqs.append(req)
            count += l
        return gbuf

    def wait_io(self, gbuf: fstcpp.gds_device_buffer, dtype: torch.dtype=None)->Dict[str, torch.Tensor]:
        for req in self.reqs:
            count = self.reader.wait_read(req)
            if count < 0:
                raise Exception(f"wait_io: wait_nogds_read failed, req={req}")
        if self.fd > 0:
            os.close(self.fd)
            self.fd = 0
        return self.metadata.get_tensors(gbuf, self.device, self.metadata.header_length, dtype=dtype)
