# Copyright 2024- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from typing import Dict, Tuple
from fastsafetensors.dlpack import from_cuda_buffer
from fastsafetensors import SafeTensorsFileLoader, SingleGroup, SafeTensorsMetadata, fastsafe_open
from fastsafetensors.copier.gds import GdsFileCopier
from fastsafetensors.copier.nogds import NoGdsFileCopier
from fastsafetensors.common import alloc_tensor_memory, free_tensor_memory, need_workaround_dtypes, paddle_loaded
from fastsafetensors import cpp as fstcpp
if paddle_loaded:
    import paddle
    from safetensors.paddle import save_file as paddle_save_file

def get_and_check_device(framework="pytorch"):
    dev_is_gpu = fstcpp.is_cuda_found()
    if framework == "pytorch" or framework == "pt":
        device = torch.device("cuda:0" if dev_is_gpu else "cpu")
    elif paddle_loaded and framework == "paddle":
        device = "gpu:0" if dev_is_gpu else "cpu"
    else:
        raise NotImplementedError(f"Do not support framework: {framework}")
    return device, dev_is_gpu

def run_nogds_file_read(input_file: str, framework="pytorch")->Tuple[SafeTensorsMetadata, fstcpp.gds_device_buffer]:
    fd = os.open(input_file, os.O_RDONLY, 0o644)
    meta = SafeTensorsMetadata.from_file(input_file, framework=framework)
    size = meta.size_bytes - meta.header_length
    device, dev_is_gpu = get_and_check_device(framework)
    gbuf = alloc_tensor_memory(size, device, framework=framework)
    reader = fstcpp.nogds_file_reader(False, 20 * 1024, 1, dev_is_gpu)
    req = reader.submit_read(fd, gbuf, meta.header_length, size, 0)
    assert req > 0
    assert reader.wait_read(req) >= 0
    os.close(fd)
    return (meta, gbuf)

def test_load_metadata_and_dlpack(fstcpp_log, input_files, framework="pytorch"):
    print("test_load_metadata_and_dlpack")
    assert len(input_files) > 0
    device, _ = get_and_check_device(framework)
    for input_file in input_files:
        expected_tensors: Dict[str, torch.Tensor] = {}
        with safe_open(input_file, framework="pt") as f:
            for k in f.keys():
                expected_tensors[k] = f.get_tensor(k)
                if framework == "pytorch":
                    expected_tensors[k] = expected_tensors[k].to(device=device)
                elif framework == "paddle":
                    expected_tensors[k] = paddle.to_tensor(expected_tensors[k].numpy(), place=device)
        meta, gbuf = run_nogds_file_read(input_file, framework=framework)
        assert meta.header_length > 0
        assert meta.size_bytes > 0
        assert len(meta.tensors) > 0
        printed = False
        for name, actual_meta in sorted(meta.tensors.items(), key=lambda x:x[0]):
            dst_dev_ptr = gbuf.get_base_address() + actual_meta.data_offsets[0]
            if actual_meta.dtype in need_workaround_dtypes:
                wdtype = need_workaround_dtypes[actual_meta.dtype]
                cu_buf = from_cuda_buffer(dst_dev_ptr, actual_meta.shape, actual_meta.strides, wdtype, device)
                if framework == "pytorch":
                    actual = torch.from_dlpack(cu_buf).view(actual_meta.dtype)
                elif framework == "paddle":
                    actual = paddle.utils.dlpack.from_dlpack(cu_buf).view(actual_meta.dtype)
            else:
                cu_buf = from_cuda_buffer(dst_dev_ptr, actual_meta.shape, actual_meta.strides, actual_meta.dtype, device)
                if framework == "pytorch":
                    actual = torch.from_dlpack(cu_buf)
                elif framework == "paddle":
                    actual = paddle.utils.dlpack.from_dlpack(cu_buf)
            exp = expected_tensors[name]
            if framework == "pytorch":
                assert torch.all(exp.eq(actual))
            elif framework == "paddle":
                assert paddle.all(exp.equal(actual))
            if not printed:
                print(actual_meta.__repr__())
                printed = True

def test_load_metadata_and_dlpack_for_paddle(fstcpp_log, input_files):
    if paddle_loaded:
        test_load_metadata_and_dlpack(fstcpp_log, input_files, "paddle")

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
    if not fstcpp.is_cuda_found():
        assert bus ==  ""
    else:
        print(f"bus for cuda:0: {bus}")
        assert len(bus) > 0

def test_set_numa_node(fstcpp_log):
    assert fstcpp.set_numa_node(0) == 0

def test_alloc_gds_buffer(fstcpp_log, framework="pytorch"):
    print("test_alloc_gds_buffer")
    device, _ = get_and_check_device(framework)
    gbuf = alloc_tensor_memory(1024, device, framework=framework)
    addr = gbuf.get_base_address()
    assert addr != 0

def test_alloc_gds_buffer_for_paddle(fstcpp_log):
    if paddle_loaded:
        test_alloc_gds_buffer(fstcpp_log, "paddle")

def test_cufile_register_deregister(fstcpp_log, framework="pytorch"):
    print("test_cufile_register_deregister")
    device, _ = get_and_check_device(framework)
    gbuf = alloc_tensor_memory(1024, device, framework=framework)
    assert gbuf.cufile_register(0, 256) == 0
    assert gbuf.cufile_register(256, 1024-256) == 0
    assert gbuf.cufile_deregister(0) == 0
    assert gbuf.cufile_deregister(256) == 0

def test_cufile_register_deregister_for_paddle(fstcpp_log):
    if paddle_loaded:
        test_alloc_gds_buffer(fstcpp_log, "paddle")

def test_memmove(fstcpp_log , framework="pytorch"):
    print("test_memmove")
    device, _ = get_and_check_device(framework)
    gbuf = alloc_tensor_memory(1024, device, framework=framework)
    tmp = alloc_tensor_memory(1024, device, framework=framework)
    assert gbuf.memmove(0, 12, tmp, 256*3) == 0
    # Confuse about this test : gbuf.memmove(0, 12, tmp, 1024)
    # I think this test should start copying a section of 1024 memory 
    # from the position of gbuf+12 to the position of gbuf+0. 
    # However, this piece of memory itself is only 1024. 
    # After offsetting by 12, there is no 1024 left in the remaining memory. 
    # This part really puzzles me. So I change the moving size to 256*3 (<1024)

def test_memmove_for_paddle(fstcpp_log):
    if paddle_loaded:
        test_memmove(fstcpp_log, "paddle")

def test_nogds_file_reader(fstcpp_log, input_files, framework="pytorch"):
    print("test_nogds_file_reader")
    fd = os.open(input_files[0], os.O_RDONLY, 0o644)
    s = os.fstat(fd)
    assert fd > 0
    device, dev_is_gpu = get_and_check_device(framework)
    gbuf = alloc_tensor_memory(s.st_size, device, framework=framework)
    reader = fstcpp.nogds_file_reader(False, 256 * 1024, 4, dev_is_gpu)
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

def test_nogds_file_reader_for_paddle(fstcpp_log, input_files):
    if paddle_loaded:
        test_nogds_file_reader(fstcpp_log, input_files, "paddle")

def test_NoGdsFileCopier(fstcpp_log, input_files, framework="pytorch"):
    print("test_NoGdsFileCopier")
    meta = SafeTensorsMetadata.from_file(input_files[0], framework)
    device, dev_is_gpu = get_and_check_device(framework)
    reader = fstcpp.nogds_file_reader(False, 256 * 1024, 4, dev_is_gpu)
    copier = NoGdsFileCopier(meta, device, reader, True)
    gbuf = copier.submit_io(False, 10 * 1024 * 1024 * 1024)
    tensors = copier.wait_io(gbuf, None)
    with safe_open(input_files[0], framework="pt") as f:
        for key in tensors.keys():
            if framework == "pytorch":
                assert torch.all(f.get_tensor(key).to(device=device).eq(tensors[key]))
            elif framework == "paddle":
                assert paddle.all(paddle.to_tensor(f.get_tensor(key).numpy(), place=device).equal(tensors[key]))
    free_tensor_memory(gbuf, device, framework)

def test_NoGdsFileCopier_for_paddle(fstcpp_log, input_files):
    if paddle_loaded:
        test_NoGdsFileCopier(fstcpp_log, input_files,"paddle")

def test_GdsFileCopier(fstcpp_log, input_files, framework="pytorch"):
    print("test_GdsFileCopier")
    if not fstcpp.is_cufile_found():
        pytest.skip("cufile.so is not found")
        return
    meta = SafeTensorsMetadata.from_file(input_files[0], framework=framework)
    device, dev_is_gpu = get_and_check_device(framework)
    reader = fstcpp.gds_file_reader(4, dev_is_gpu)
    copier = GdsFileCopier(meta, device, reader, True)
    gbuf = copier.submit_io(False, 10 * 1024 * 1024 * 1024)
    tensors = copier.wait_io(gbuf, None)
    with safe_open(input_files[0], framework="pt") as f:
        for key in tensors.keys():
            if framework == "torch":
                assert torch.all(f.get_tensor(key).to(device=device).eq(tensors[key]))
            elif framework == "paddle":
                assert paddle.all(paddle.to_tensor(f.get_tensor(key).numpy(), place=device).equal(tensors[key]))
    free_tensor_memory(gbuf, device, framework=framework)

def test_GdsFileCopier_for_paddle(fstcpp_log, input_files):
    if paddle_loaded:
        test_GdsFileCopier(fstcpp_log, input_files, "paddle")

def test_SafeTensorsFileLoader(fstcpp_log, input_files, framework="pytorch"):
    device, _ = get_and_check_device(framework)
    if framework == "pytorch":
        data_type = torch.float16
    elif framework == "paddle":
        # There are some lack of accuracy in paddle.float16 (about 1e-4)
        data_type = paddle.float32
    else:
        raise NotImplementedError(f"Do not support the framework: {framework}")
    loader = SafeTensorsFileLoader(SingleGroup(), device, nogds=False, debug_log=True, framework=framework)
    loader.add_filenames({0: input_files})
    bufs = loader.copy_files_to_device(dtype=data_type, use_buf_register=True, max_copy_block_size=256*1024*1024)
    key_dims = {key: -1 for key in loader.get_keys()}
    tensors = bufs.as_dict(key_dims)
    last_key = ""
    last_shape: torch.Size = None
    with safe_open(input_files[0], framework="pt") as f:
        for key in tensors.keys():
            if framework == "pytorch":
                exp = f.get_tensor(key).to(device=device, dtype=data_type)
                assert torch.all(exp.eq(bufs.get_tensor(key)))
            elif framework == "paddle":
                exp = paddle.to_tensor(f.get_tensor(key).numpy(), place=device, dtype=data_type)
                assert paddle.all(exp.equal(bufs.get_tensor(key)))
            last_key = key
            last_shape = exp.shape
    if last_key != "":
        assert bufs.get_filename(last_key) == input_files[0]
        assert bufs.get_shape(last_key) == last_shape
        assert loader.get_shape(last_key) == last_shape
    assert bufs.get_filename("aaaaaaaaaaaaa") == None
    bufs.close()
    loader.close()

def test_SafeTensorsFileLoader_for_paddle(fstcpp_log, input_files):
    if paddle_loaded:
        test_SafeTensorsFileLoader(fstcpp_log, input_files,"paddle")

def test_SafeTensorsFileLoaderNoGds(fstcpp_log, input_files, framework="pytorch"):
    device, _ = get_and_check_device(framework)
    loader = SafeTensorsFileLoader(SingleGroup(), device, nogds=True, debug_log=True, framework=framework)
    loader.add_filenames({0: input_files})
    bufs = loader.copy_files_to_device()
    key_dims = {key: -1 for key in loader.get_keys()}
    tensors = bufs.as_dict(key_dims)
    with safe_open(input_files[0], framework="pt") as f:
        for key in tensors.keys():
            if framework == "pytorch":
                assert torch.all(f.get_tensor(key).to(device=device).eq(tensors[key]))
            elif framework == "paddle":
                assert paddle.all(paddle.to_tensor(f.get_tensor(key).numpy(), place=device).equal(tensors[key]))
    bufs.close()
    loader.close()

def test_SafeTensorsFileLoaderNoGds_for_paddle(fstcpp_log, input_files):
    if paddle_loaded:
        test_SafeTensorsFileLoaderNoGds(fstcpp_log, input_files, "paddle")

def test_fastsafe_open(fstcpp_log, input_files, framework="pt"):
    device, _ = get_and_check_device(framework)
    def weight_iterator():
        with fastsafe_open(input_files, pg=SingleGroup(), device=device, nogds=True, debug_log=True, framework=framework) as f:
            for k in f.get_keys():
                t = f.get_tensor(k)
                yield k, t
    tensors = {}
    with safe_open(input_files[0], framework="pt") as f:
        for key in f.keys():
            if framework == "pt":
                tensors[key] = f.get_tensor(key).to(device=device)
            elif framework == "paddle":
                tensors[key] = paddle.to_tensor(f.get_tensor(key).numpy(), place=device)
    for k, t in weight_iterator():
        if framework == "pt":
            assert torch.all(tensors[k].eq(t))
        elif framework == "paddle":
            assert paddle.all(tensors[k].equal(t))

def test_fastsafe_open_for_paddle(fstcpp_log, input_files):
    if paddle_loaded:
        test_fastsafe_open(fstcpp_log, input_files, "paddle")

def _test_type(tmp_dir, dtype, device):
    filename = os.path.join(tmp_dir, f"a.safetensors")
    t0 = torch.randn((8, 16), dtype=torch.float32).to(dtype=dtype)
    save_file({f"a": t0}, filename, metadata={"fst": "sample"})
    with fastsafe_open(filenames=[filename], nogds=True, device=device, debug_log=True) as f:
        for key in f.get_keys():
            t1 = f.get_tensor(key).clone().detach()
    with safe_open(filename, framework='pt', device=device) as f:
        for key in f.keys():
            t2 = f.get_tensor(key)
    assert torch.all(t2.eq(t1))

def _test_type_for_paddle(tmp_dir, dtype, device):
    filename = os.path.join(tmp_dir, f"a.safetensors")
    t0 = paddle.randn((8, 16), dtype=paddle.float32).to(dtype=dtype)
    paddle_save_file({f"a": t0}, filename, metadata={"fst": "sample"})
    with fastsafe_open(filenames=[filename], nogds=True, device=device, debug_log=True, framework="paddle") as f:
        for key in f.get_keys():
            t1 = f.get_tensor(key).clone().detach()
    with safe_open(filename, framework='pt') as f:
        for key in f.keys():
            t2 = paddle.to_tensor(f.get_tensor(key).numpy(), place=device)
    assert paddle.all(t2.equal(t1))

def test_int8(fstcpp_log, tmp_dir):
    _test_type(tmp_dir, torch.int8, "cuda:0" if fstcpp.is_cuda_found() else "cpu")
    if paddle_loaded:
        _test_type_for_paddle(tmp_dir, paddle.int8, "gpu:0" if fstcpp.is_cuda_found() else "cpu")

def test_float8_e5m2(fstcpp_log, tmp_dir):
    _test_type(tmp_dir, torch.float8_e5m2, "cuda:0" if fstcpp.is_cuda_found() else "cpu")
    if paddle_loaded:
        _test_type_for_paddle(tmp_dir, paddle.float8_e5m2, "gpu:0" if fstcpp.is_cuda_found() else "cpu")

def test_float8_e4m3fn(fstcpp_log, tmp_dir):
    _test_type(tmp_dir, torch.float8_e4m3fn, "cuda:0" if fstcpp.is_cuda_found() else "cpu")
    if paddle_loaded:
        _test_type_for_paddle(tmp_dir, paddle.float8_e4m3fn, "gpu:0" if fstcpp.is_cuda_found() else "cpu")
