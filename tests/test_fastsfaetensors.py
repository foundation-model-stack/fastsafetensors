# Copyright 2024- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import torch
from safetensors import safe_open
from typing import Dict, Tuple
from fastsafetensors.dlpack import from_cuda_buffer
from fastsafetensors import SafeTensorsFileLoader, SingleGroup, SafeTensorsMetadata
from fastsafetensors.copier.gds import GdsFileCopier
from fastsafetensors.copier.nogds import NoGdsFileCopier
from fastsafetensors import cpp as fstcpp

def run_gds_file_read(input_file: str)->Tuple[SafeTensorsMetadata, fstcpp.gds_device_buffer]:
    fh = fstcpp.gds_file_handle(input_file)
    meta = SafeTensorsMetadata.from_file(input_file)
    gbuf = fstcpp.gds_device_buffer(((meta.size_bytes + 4096 - 1) // 4096) * 4096)
    reader = fstcpp.gds_file_reader(4)
    reqs = []
    c = 0
    while c < meta.size_bytes:
        l = meta.size_bytes // 4
        if c + l > meta.size_bytes:
            l = meta.size_bytes - c
        if l % 4096 > 0:
            l = l - (l % 4096) + 4096
        req = reader.submit_read(fh, gbuf, c, l, c, meta.size_bytes)
        assert req > 0
        reqs.append(req)
        c += l
    for req in reqs:
        assert reader.wait_read(req) > 0
    return (meta, gbuf)

def test_load_metadata_and_dlpack(fstcpp_log, input_files):
    print("test_load_metadata_and_dlpack")
    assert len(input_files) > 0
    reader = fstcpp.gds_file_reader(4)
    for input_file in input_files:
        expected_tensors: Dict[str, torch.Tensor] = {}
        with safe_open(input_file, framework="pt") as f:
            for k in f.keys():
                expected_tensors[k] = f.get_tensor(k)
        meta, gbuf = run_gds_file_read(input_file)
        assert meta.header_length > 0
        assert meta.size_bytes > 0
        assert len(meta.tensors) > 0
        printed = False
        for name, actual_meta in sorted(meta.tensors.items(), key=lambda x:x[0]):
            dst_dev_ptr = gbuf.get_base_address() + meta.header_length + actual_meta.data_offsets[0]
            actual = torch.from_dlpack(from_cuda_buffer(dst_dev_ptr, actual_meta.shape, actual_meta.strides, actual_meta.dtype, torch.device("cpu")))
            exp = expected_tensors[name]
            assert torch.all(exp.eq(actual))
            if not printed:
                print(actual_meta.__repr__())
                printed = True

def test_set_debug_log():
    fstcpp.set_debug_log(False)
    assert True

def test_get_alignment_size():
    assert fstcpp.get_alignment_size() == 4096

def test_init_gds(fstcpp_log):
    assert fstcpp.init_gds(16 * 1024, 64 * 1024 * 1024) == 0

def test_close_gds(fstcpp_log):
    assert fstcpp.close_gds() == 0

def test_get_device_pci_bus(fstcpp_log):
    bus = fstcpp.get_device_pci_bus(0)
    if fstcpp.is_stub():
        assert bus ==  "0000:00:00:00.00"
    else:
        print(f"bus for cuda:0: {bus}")
        assert len(bus) > 0

def test_set_numa_node(fstcpp_log):
    assert fstcpp.set_numa_node(0) == 0

def test_alloc_gds_buffer(fstcpp_log):
    print("test_alloc_gds_buffer")
    gbuf = fstcpp.gds_device_buffer(1024)
    addr = gbuf.get_base_address()
    assert addr != 0

def test_cufile_register_deregister(fstcpp_log):
    print("test_cufile_register_deregister")
    gbuf = fstcpp.gds_device_buffer(1024)
    assert gbuf.cufile_register(0, 256) == 0
    assert gbuf.cufile_register(256, 1024-256) == 0
    assert gbuf.cufile_deregister(0) == 0
    assert gbuf.cufile_deregister(256) == 0

def test_memmove(fstcpp_log):
    print("test_memmove")
    gbuf = fstcpp.gds_device_buffer(1024)
    tmp = fstcpp.gds_device_buffer(1024)
    assert gbuf.memmove(0, 12, tmp, 1024) == 0

def test_nogds_file_reader(fstcpp_log, input_files):
    print("test_nogds_file_reader")
    fd = os.open(input_files[0], os.O_RDONLY, 0o644)
    s = os.fstat(fd)
    assert fd > 0
    gbuf = fstcpp.gds_device_buffer(s.st_size)
    reader = fstcpp.nogds_file_reader(False, 256 * 1024, 4)
    step = s.st_size // 4
    reqs = []
    off = 0
    while off < s.st_size:
        l = step
        if off + l > s.st_size:
            l = s.st_size - off
        if 256 * 1024 * 1024 < l:
            l = 256 * 1024 * 1024
        req = reader.submit_read(fd, gbuf, off, l, off)
        assert req > 0
        reqs.append(req)
        off += l
    for req in reqs:
        assert reader.wait_read(req) > 0
    os.close(fd)

def test_NoGdsFileCopier(fstcpp_log, input_files):
    print("test_NoGdsFileCopier")
    meta = SafeTensorsMetadata.from_file(input_files[0])
    reader = fstcpp.nogds_file_reader(False, 256 * 1024, 4)
    copier = NoGdsFileCopier(meta, torch.device("cpu"), reader, True)
    gbuf = copier.submit_io(False, 10 * 1024 * 1024 * 1024)
    tensors = copier.wait_io(gbuf, None)
    with safe_open(input_files[0], framework="pt") as f:
        for key in tensors.keys():
            assert torch.all(f.get_tensor(key).eq(tensors[key]))

def test_GdsFileCopier(fstcpp_log, input_files):
    print("test_GdsFileCopier")
    meta = SafeTensorsMetadata.from_file(input_files[0])
    reader = fstcpp.gds_file_reader(4)
    copier = GdsFileCopier(meta, torch.device("cpu"), reader, True)
    gbuf = copier.submit_io(False, 10 * 1024 * 1024 * 1024)
    tensors = copier.wait_io(gbuf, None)
    with safe_open(input_files[0], framework="pt") as f:
        for key in tensors.keys():
            assert torch.all(f.get_tensor(key).eq(tensors[key]))

def test_SafeTensorsFileLoader(fstcpp_log, input_files):
    loader = SafeTensorsFileLoader(SingleGroup(), torch.device("cpu"), nogds=False, debug_log=True)
    loader.add_filenames({0: input_files})
    bufs = loader.copy_files_to_device(dtype=torch.float16, use_buf_register=True, max_copy_block_size=256*1024*1024)
    key_dims = {key: -1 for key in loader.get_keys()}
    tensors = bufs.as_dict(key_dims)
    last_key = ""
    last_shape: torch.Size = None
    with safe_open(input_files[0], framework="pt") as f:
        for key in tensors.keys():
            exp = f.get_tensor(key).to(dtype=torch.float16)
            assert torch.all(exp.eq(bufs.get_tensor(key)))
            last_key = key
            last_shape = exp.shape
    if last_key != "":
        assert bufs.get_filename(last_key) == input_files[0]
        assert bufs.get_shape(last_key) == last_shape
        assert loader.get_shape(last_key) == last_shape
    assert bufs.get_filename("aaaaaaaaaaaaa") == None
    bufs.close()
    loader.close()


def test_SafeTensorsFileLoaderNoGds(fstcpp_log, input_files):
    loader = SafeTensorsFileLoader(SingleGroup(), torch.device("cpu"), nogds=True, debug_log=True)
    loader.add_filenames({0: input_files})
    bufs = loader.copy_files_to_device()
    key_dims = {key: -1 for key in loader.get_keys()}
    tensors = bufs.as_dict(key_dims)
    with safe_open(input_files[0], framework="pt") as f:
        for key in tensors.keys():
            assert torch.all(f.get_tensor(key).eq(tensors[key]))
    loader.close()