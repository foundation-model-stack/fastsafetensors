# Copyright 2024- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0

import os
from collections import OrderedDict
from typing import List, Tuple, Dict, Any

import pytest
from fastsafetensors import SafeTensorsFileLoader, SafeTensorsMetadata
from fastsafetensors.frameworks import FrameworkOpBase
from fastsafetensors import cpp as fstcpp
from fastsafetensors import fastsafe_open
from fastsafetensors.copier.gds import GdsFileCopier
from fastsafetensors.copier.nogds import NoGdsFileCopier
from fastsafetensors.dlpack import from_cuda_buffer
from fastsafetensors.st_types import DType, Device, DeviceType

def load_safetensors_file(filename: str, device: Device, framework: FrameworkOpBase) -> Dict[str, Any]:
    if framework.get_name() == "pytorch":
        from safetensors.torch import load_file
    elif framework.get_name() == "paddle":
        from safetensors.paddle import load_file
    else:
        raise Exception(f"unkown framework: {framework.get_name()}")
    return load_file(filename, device.as_str())

def save_safetensors_file(tensors: Dict[str, Any], filename: str, metadata: Dict[str, str], framework: FrameworkOpBase) -> None:
    if framework.get_name() == "pytorch":
        from safetensors.torch import save_file
    elif framework.get_name() == "paddle":
        from safetensors.paddle import save_file
    else:
        raise Exception(f"unkown framework: {framework.get_name()}")
    save_file(tensors, filename, metadata)

def get_and_check_device(framework: FrameworkOpBase):
    dev_is_gpu = fstcpp.is_cuda_found()
    device = "cpu"
    if dev_is_gpu:
        if framework.get_name() == "pytorch":
            device = "cuda:0"
        elif framework.get_name() == "paddle":
            device = "gpu:0"
    return Device.from_str(device), dev_is_gpu


def run_nogds_file_read(
    input_file: str,
    framework: FrameworkOpBase,
) -> Tuple[SafeTensorsMetadata, fstcpp.gds_device_buffer]:
    fd = os.open(input_file, os.O_RDONLY, 0o644)
    meta = SafeTensorsMetadata.from_file(input_file, framework)
    size = meta.size_bytes - meta.header_length
    device, dev_is_gpu = get_and_check_device(framework)
    gbuf = framework.alloc_tensor_memory(size, device)
    reader = fstcpp.nogds_file_reader(False, 20 * 1024, 1, dev_is_gpu)
    req = reader.submit_read(fd, gbuf, meta.header_length, size, 0)
    assert req > 0
    assert reader.wait_read(req) >= 0
    os.close(fd)
    return (meta, gbuf)

def test_device(fstcpp_log) -> None:
    print("test_device")
    with pytest.raises(ValueError, match="Unknown device type: aaaa"):
        Device.from_str("aaaa:0")
    with pytest.raises(ValueError, match="Invalid index: -xxx"):
        Device.from_str("cpu:-xxx")
    with pytest.raises(ValueError, match="Unknown device type: aaa"):
        Device.from_str("aaa")
    cuda = Device.from_str("cuda:4")
    assert cuda.type == DeviceType.CUDA
    assert cuda.index == 4
    cpu = Device(DeviceType.CPU, None)
    assert cpu.type == DeviceType.CPU and cpu.index == None

def test_framework(fstcpp_log) -> None:
    print("test_framework")
    try:
        from fastsafetensors.frameworks._torch import TorchOp
        torch = TorchOp()
        t = torch.get_empty_tensor([1], DType.F16, Device.from_str("cpu"))
        from fastsafetensors.frameworks._paddle import PaddleOp
        paddle = PaddleOp()
        t2 = paddle.get_empty_tensor([1], DType.F16, Device.from_str("cpu"))
        with pytest.raises(Exception):
            torch.is_equal(t, t2.get_raw())
        with pytest.raises(Exception):
            paddle.is_equal(t2, t.get_raw())
        assert torch.get_cuda_ver() == paddle.get_cuda_ver()
    except:
        pytest.skip("test_framework requires installing all frameworks")

def test_load_metadata_and_dlpack(fstcpp_log, input_files, framework) -> None:
    print("test_load_metadata_and_dlpack")
    assert len(input_files) > 0
    device, _ = get_and_check_device(framework)
    for input_file in input_files:
        expected_tensors = load_safetensors_file(input_files[0], device, framework)
        meta, gbuf = run_nogds_file_read(input_file, framework)
        assert meta.header_length > 0
        assert meta.size_bytes > 0
        assert len(meta.tensors) > 0
        printed = False
        for name, actual_meta in sorted(meta.tensors.items(), key=lambda x: x[0]):
            dst_dev_ptr = gbuf.get_base_address() + actual_meta.data_offsets[0]
            wdtype = framework.as_workaround_dtype(actual_meta.dtype)
            cu_buf = from_cuda_buffer(
                dst_dev_ptr, actual_meta.shape, actual_meta.strides, wdtype, device
            )
            actual = framework.from_dlpack(cu_buf, device, wdtype)
            if wdtype != actual_meta.dtype:
                actual = actual.view(actual_meta.dtype)
            exp = expected_tensors[name]
            assert framework.is_equal(actual, exp)
            if not printed:
                print(actual_meta.__repr__())
                printed = True


def test_set_debug_log() -> None:
    fstcpp.set_debug_log(False)
    assert True


def test_get_alignment_size() -> None:
    assert fstcpp.get_alignment_size() == 4096


def test_init_gds(fstcpp_log) -> None:
    assert fstcpp.init_gds() == 0


def test_close_gds(fstcpp_log) -> None:
    assert fstcpp.close_gds() == 0


def test_get_device_pci_bus(fstcpp_log) -> None:
    bus = fstcpp.get_device_pci_bus(0)
    if not fstcpp.is_cuda_found():
        assert bus == ""
    else:
        print(f"bus for cuda:0: {bus}")
        assert len(bus) > 0


def test_set_numa_node(fstcpp_log) -> None:
    assert fstcpp.set_numa_node(0) == 0


def test_alloc_gds_buffer(fstcpp_log, framework) -> None:
    print("test_alloc_gds_buffer")
    device, _ = get_and_check_device(framework)
    gbuf = framework.alloc_tensor_memory(1024, device)
    addr = gbuf.get_base_address()
    assert addr != 0


def test_cufile_register_deregister(fstcpp_log, framework) -> None:
    print("test_cufile_register_deregister")
    device, _ = get_and_check_device(framework)
    gbuf = framework.alloc_tensor_memory(1024, device)
    assert gbuf.cufile_register(0, 256) == 0
    assert gbuf.cufile_register(256, 1024 - 256) == 0
    assert gbuf.cufile_deregister(0) == 0
    assert gbuf.cufile_deregister(256) == 0


def test_memmove(fstcpp_log, framework) -> None:
    print("test_memmove")
    device, _ = get_and_check_device(framework)
    gbuf = framework.alloc_tensor_memory(1024, device)
    tmp = framework.alloc_tensor_memory(1024, device)
    assert gbuf.memmove(0, 12, tmp, 256 * 3) == 0
    # Confuse about this test : gbuf.memmove(0, 12, tmp, 1024)
    # I think this test should start copying a section of 1024 memory
    # from the position of gbuf+12 to the position of gbuf+0.
    # However, this piece of memory itself is only 1024.
    # After offsetting by 12, there is no 1024 left in the remaining memory.
    # This part really puzzles me. So I change the moving size to 256*3 (<1024)


def test_nogds_file_reader(fstcpp_log, input_files, framework) -> None:
    print("test_nogds_file_reader")
    fd = os.open(input_files[0], os.O_RDONLY, 0o644)
    s = os.fstat(fd)
    assert fd > 0
    device, dev_is_gpu = get_and_check_device(framework)
    gbuf = framework.alloc_tensor_memory(s.st_size, device)
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


def test_NoGdsFileCopier(fstcpp_log, input_files, framework) -> None:
    print("test_NoGdsFileCopier")
    meta = SafeTensorsMetadata.from_file(input_files[0], framework)
    device, dev_is_gpu = get_and_check_device(framework)
    reader = fstcpp.nogds_file_reader(False, 256 * 1024, 4, dev_is_gpu)
    copier = NoGdsFileCopier(meta, device, reader, framework, True)
    gbuf = copier.submit_io(False, 10 * 1024 * 1024 * 1024)
    tensors = copier.wait_io(gbuf)
    for key, exp in load_safetensors_file(input_files[0], device, framework).items():
        actual = tensors[key]
        assert framework.is_equal(actual, exp)
    framework.free_tensor_memory(gbuf, device)


def test_GdsFileCopier(fstcpp_log, input_files, framework) -> None:
    print("test_GdsFileCopier")
    if not fstcpp.is_cufile_found():
        pytest.skip("cufile.so is not found")
        return
    meta = SafeTensorsMetadata.from_file(input_files[0], framework)
    device, dev_is_gpu = get_and_check_device(framework)
    reader = fstcpp.gds_file_reader(4, dev_is_gpu)
    copier = GdsFileCopier(meta, device, reader, framework, True)
    gbuf = copier.submit_io(False, 10 * 1024 * 1024 * 1024)
    tensors = copier.wait_io(gbuf)
    for key, exp in load_safetensors_file(input_files[0], device, framework).items():
        actual = tensors[key]
        assert framework.is_equal(actual, exp)
    framework.free_tensor_memory(gbuf, device)


def test_SafeTensorsFileLoader(fstcpp_log, input_files, framework) -> None:
    device, _ = get_and_check_device(framework)
    if framework.get_name() == "pytorch":
        import torch
        data_type = DType.F16
        data_type_real = torch.float16
    elif framework.get_name() == "paddle":
        # There are some lack of accuracy in paddle.float16 (about 1e-4) in cpu.
        import paddle
        data_type = DType.F32
        data_type_real = paddle.float32
    else:
        raise NotImplementedError(
            f"Do not support the framework: {framework.get_name()}"
        )
    loader = SafeTensorsFileLoader(device=device.as_str(), framework=framework.get_name(), nogds=False, debug_log=True)
    loader.add_filenames({0: input_files})
    bufs = loader.copy_files_to_device(
        dtype=data_type, use_buf_register=True, max_copy_block_size=256 * 1024 * 1024
    )
    last_key = ""
    last_shape: List[int] = []
    for key, exp in load_safetensors_file(input_files[0], device, framework).items():
        exp = exp.to(dtype=data_type_real)
        actual = bufs.get_tensor(key)
        assert framework.is_equal(actual, exp)
        last_key = key
        last_shape = list(exp.shape)
    if last_key != "":
        assert bufs.get_filename(last_key) == input_files[0]
        assert bufs.get_shape(last_key) == last_shape
        assert loader.get_shape(last_key) == last_shape
    assert bufs.get_filename("aaaaaaaaaaaaa") == ""
    bufs.close()
    loader.close()


def test_SafeTensorsFileLoaderNoGds(fstcpp_log, input_files, framework) -> None:
    device, _ = get_and_check_device(framework)
    loader = SafeTensorsFileLoader(device=device.as_str(), framework=framework.get_name(), nogds=True, debug_log=True)
    loader.add_filenames({0: input_files})
    bufs = loader.copy_files_to_device()
    key_dims = OrderedDict({key: -1 for key in loader.get_keys()})
    tensors = bufs.as_dict(key_dims)
    for key, exp in load_safetensors_file(input_files[0], device, framework).items():
        actual = tensors[key]
        assert framework.is_equal(actual, exp)
    bufs.close()
    loader.close()


def test_fastsafe_open(fstcpp_log, input_files, framework) -> None:
    device, _ = get_and_check_device(framework)

    def weight_iterator():
        with fastsafe_open(
            input_files,
            device=device.as_str(),
            nogds=True,
            debug_log=True,
            framework=framework.get_name(),
        ) as f:
            for k in f.keys():
                t = f.get_tensor_wrapped(k)
                yield k, t

    tensors = load_safetensors_file(input_files[0], device, framework)
    for k, t in weight_iterator():
        assert framework.is_equal(t, tensors[k])

    with fastsafe_open(input_files[0], device=device.as_str(), nogds=True, framework=framework.get_name()) as f:
        for filename in f.metadata().keys():
            assert filename in input_files

    with fastsafe_open({0: input_files}, device=device.as_str(), nogds=True, framework=framework.get_name()) as f:
        for k in f.keys():
            t = f.get_tensor(k)
            if framework.get_name() == "pytorch":
                import torch
                assert isinstance(t, torch.Tensor)
            elif framework.get_name() == "paddle":
                import paddle
                assert isinstance(t, paddle.Tensor)
            break

def _test_type(tmp_dir, dtype: DType, device: Device, framework: FrameworkOpBase) -> None:
    filename = os.path.join(tmp_dir, f"a.safetensors")
    t0 = framework.randn((8, 16), device=device, dtype=DType.F32).to(dtype=dtype)
    save_safetensors_file({f"a": t0.get_raw()}, filename, {"fst": "sample"}, framework)
    t2 = load_safetensors_file(filename, device, framework)
    with fastsafe_open(
        filenames=[filename], nogds=True, device=device.as_str(), framework=framework.get_name(), debug_log=True
    ) as f:
        for key in f.keys():
            t1 = f.get_tensor_wrapped(key).clone().detach()
            assert framework.is_equal(t1, t2[key])


def test_int8(fstcpp_log, tmp_dir, framework) -> None:
    if not framework.support_fp8():
        pytest.skip("FP8 is not supported")
        return
    device, _ = get_and_check_device(framework)
    _test_type(tmp_dir, DType.I8, device, framework)

def test_float8_e5m2(fstcpp_log, tmp_dir, framework) -> None:
    if not framework.support_fp8():
        pytest.skip("FP8 is not supported")
        return
    device, _ = get_and_check_device(framework)
    _test_type(tmp_dir, DType.F8_E5M2, device, framework)

def test_float8_e4m3fn(fstcpp_log, tmp_dir, framework) -> None:
    if not framework.support_fp8():
        pytest.skip("FP8 is not supported")
        return
    device, _ = get_and_check_device(framework)
    _test_type(tmp_dir, DType.F8_E4M3, device, framework)
