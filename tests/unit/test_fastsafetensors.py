# SPDX-License-Identifier: Apache-2.0

import itertools
import os
import sys
from collections import OrderedDict
from typing import Any, Dict, List, Tuple

import pytest

from fastsafetensors import (
    SafeTensorsFileLoader,
    SafeTensorsMetadata,
    SingleGroup,
    TensorFrame,
)
from fastsafetensors import cpp as fstcpp
from fastsafetensors import (
    fastsafe_open,
)
from fastsafetensors.common import get_device_numa_node, is_gpu_found
from fastsafetensors.copier.gds import GdsFileCopier
from fastsafetensors.copier.nogds import NoGdsFileCopier
from fastsafetensors.copier.unified import UnifiedMemCopier, is_unified_memory_system
from fastsafetensors.dlpack import from_cuda_buffer
from fastsafetensors.frameworks import FrameworkOpBase
from fastsafetensors.st_types import Device, DeviceType, DType


def load_safetensors_file(
    filename: str,
    device: Device,
    framework: FrameworkOpBase,
    to_dtype: DType = DType.AUTO,
) -> Dict[str, Any]:
    if framework.get_name() == "pytorch":
        from safetensors.torch import load_file
    elif framework.get_name() == "paddle":
        from safetensors.paddle import load_file
    else:
        raise Exception(f"unkown framework: {framework.get_name()}")
    d = load_file(filename, device.as_str())
    if to_dtype != DType.AUTO:
        if framework.get_name() == "pytorch":
            from fastsafetensors.frameworks._torch import dtype_convert
        elif framework.get_name() == "paddle":
            from fastsafetensors.frameworks._paddle import dtype_convert

        for k, t in d.items():
            d[k] = t.to(dtype=dtype_convert[to_dtype])
    return d


def save_safetensors_file(
    tensors: Dict[str, Any],
    filename: str,
    metadata: Dict[str, str],
    framework: FrameworkOpBase,
) -> None:
    if framework.get_name() == "pytorch":
        from safetensors.torch import save_file
    elif framework.get_name() == "paddle":
        from safetensors.paddle import save_file
    else:
        raise Exception(f"unkown framework: {framework.get_name()}")
    save_file(tensors, filename, metadata)


def get_and_check_device(framework: FrameworkOpBase):
    dev_is_gpu = is_gpu_found()
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
    flags = os.O_RDONLY
    if sys.platform == "win32" and hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    fd = os.open(input_file, flags, 0o644)
    meta = SafeTensorsMetadata.from_file(input_file, framework)
    size = meta.size_bytes - meta.header_length
    device, dev_is_gpu = get_and_check_device(framework)
    gbuf = framework.alloc_tensor_memory(size, device)
    reader = fstcpp.nogds_file_reader(
        False, 20 * 1024, 1, dev_is_gpu, device.index or 0
    )
    req = reader.submit_read(fd, gbuf, meta.header_length, size, 0)
    assert req > 0
    assert reader.wait_read(req) >= 0
    os.close(fd)
    return (meta, gbuf)


def test_device(fstcpp_log) -> None:
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


def test_framework(fstcpp_log, framework) -> None:
    t = framework.get_empty_tensor([1], DType.F16, Device.from_str("cpu"))
    with pytest.raises(Exception):
        framework.is_equal(t, [float(0.0)])
    with pytest.raises(Exception):
        framework.get_process_group(int(0))
    # Test that get_cuda_ver() returns a string with platform prefix
    cuda_ver = framework.get_cuda_ver()
    assert isinstance(cuda_ver, str)
    # Should be "hip-X.Y.Z", "cuda-X.Y", or "0.0"
    assert (
        cuda_ver.startswith("hip-") or cuda_ver.startswith("cuda-") or cuda_ver == "0.0"
    )

    # Verify it matches what torch reports
    if framework.get_name() == "pytorch":
        import torch

        if torch.cuda.is_available():
            if hasattr(torch.version, "hip") and torch.version.hip:
                assert cuda_ver.startswith("hip-")
                assert str(torch.version.hip) in cuda_ver
            else:
                assert cuda_ver.startswith("cuda-")
                assert str(torch.version.cuda) in cuda_ver
        else:
            assert cuda_ver == "0.0"


def test_get_framework_fail(fstcpp_log) -> None:
    from fastsafetensors.frameworks import get_framework_op

    with pytest.raises(Exception, match="Unknown framework name"):
        get_framework_op("aaaaa")


def make_header_bytes(s: str):
    header = s.encode("utf-8")
    n = len(header)
    return n.to_bytes(8, byteorder="little", signed=False) + header


def test_from_buffer_header_too_small(framework):
    with pytest.raises(Exception, match="HeaderTooSmall"):
        SafeTensorsMetadata.from_buffer(
            buf=0, buffer_len=4, filename="testfile", framework=framework
        )


def test_from_buffer_header_too_large(monkeypatch, framework):
    def fake_read_buffer(buf, size):
        return (100_000_001).to_bytes(8, "little")

    monkeypatch.setattr(fstcpp, "read_buffer", fake_read_buffer)

    with pytest.raises(Exception, match="HeaderTooLarge"):
        SafeTensorsMetadata.from_buffer(
            buf=0, buffer_len=1024, filename="testfile", framework=framework
        )


def test_from_buffer_invalid_header_length(monkeypatch, framework):
    def fake_read_buffer(buf, size):
        return (100).to_bytes(8, "little")

    monkeypatch.setattr(fstcpp, "read_buffer", fake_read_buffer)

    with pytest.raises(Exception, match="InvalidHeaderLength"):
        SafeTensorsMetadata.from_buffer(
            buf=0, buffer_len=50, filename="testfile", framework=framework
        )


def test_from_buffer_success(monkeypatch, framework):
    json_str = '{"__metadata__": {"data_offsets": [0, 123]}}'
    header_bytes = make_header_bytes(json_str)
    buf_data = header_bytes

    def fake_read_buffer(buf, size):
        return buf_data[buf : buf + size]

    monkeypatch.setattr(fstcpp, "read_buffer", fake_read_buffer)

    meta = SafeTensorsMetadata.from_buffer(
        buf=0, buffer_len=len(buf_data), filename="goodfile", framework=framework
    )
    assert isinstance(meta, SafeTensorsMetadata)


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
        framework.free_tensor_memory(gbuf, device)
        assert framework.get_mem_used() == 0
        assert fstcpp.get_cpp_metrics().bounce_buffer_bytes == 0


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
    if not is_gpu_found():
        assert bus == ""
    else:
        print(f"bus for gpu:0: {bus}")
        assert len(bus) > 0


def test_set_numa_node(fstcpp_log) -> None:
    assert fstcpp.set_numa_node(0) == 0


def test_alloc_gds_buffer(fstcpp_log, framework) -> None:
    print("test_alloc_gds_buffer")
    device, _ = get_and_check_device(framework)
    gbuf = framework.alloc_tensor_memory(1024, device)
    addr = gbuf.get_base_address()
    assert addr != 0
    framework.free_tensor_memory(gbuf, device)


def test_cufile_register_deregister(fstcpp_log, framework) -> None:
    print("test_cufile_register_deregister")
    device, _ = get_and_check_device(framework)
    gbuf = framework.alloc_tensor_memory(1024, device)
    assert gbuf.cufile_register(0, 256) == 0
    assert gbuf.cufile_register(256, 1024 - 256) == 0
    assert gbuf.cufile_deregister(0) == 0
    assert gbuf.cufile_deregister(256) == 0
    framework.free_tensor_memory(gbuf, device)


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
    framework.free_tensor_memory(gbuf, device)
    framework.free_tensor_memory(tmp, device)
    assert framework.get_mem_used() == 0


def test_nogds_file_reader(fstcpp_log, input_files, framework) -> None:
    print("test_nogds_file_reader")
    flags = os.O_RDONLY
    if sys.platform == "win32" and hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    fd = os.open(input_files[0], flags, 0o644)
    s = os.fstat(fd)
    assert fd > 0
    device, dev_is_gpu = get_and_check_device(framework)
    gbuf = framework.alloc_tensor_memory(s.st_size, device)
    reader = fstcpp.nogds_file_reader(
        False, 256 * 1024, 4, dev_is_gpu, device.index or 0
    )
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
    framework.free_tensor_memory(gbuf, device)
    del reader
    assert framework.get_mem_used() == 0
    assert fstcpp.get_cpp_metrics().bounce_buffer_bytes == 0


def test_NoGdsFileCopier(fstcpp_log, input_files, framework) -> None:
    print("test_NoGdsFileCopier")
    meta = SafeTensorsMetadata.from_file(input_files[0], framework)
    device, dev_is_gpu = get_and_check_device(framework)
    reader = fstcpp.nogds_file_reader(
        False, 256 * 1024, 4, dev_is_gpu, device.index or 0
    )
    copier = NoGdsFileCopier(meta, device, reader, framework)
    gbuf = copier.submit_io(False, 10 * 1024 * 1024 * 1024)
    tensors = copier.wait_io(gbuf)
    for key, exp in load_safetensors_file(input_files[0], device, framework).items():
        actual = tensors[key]
        assert framework.is_equal(actual, exp)
    framework.free_tensor_memory(gbuf, device)
    del copier
    del reader
    assert framework.get_mem_used() == 0
    assert fstcpp.get_cpp_metrics().bounce_buffer_bytes == 0


def test_GdsFileCopier(fstcpp_log, input_files, framework) -> None:
    print("test_GdsFileCopier")
    meta = SafeTensorsMetadata.from_file(input_files[0], framework)
    device, dev_is_gpu = get_and_check_device(framework)
    reader = fstcpp.gds_file_reader(4, dev_is_gpu, device.index or 0)
    copier = GdsFileCopier(meta, device, reader, framework)
    gbuf = copier.submit_io(False, 10 * 1024 * 1024 * 1024)
    tensors = copier.wait_io(gbuf)
    for key, exp in load_safetensors_file(input_files[0], device, framework).items():
        actual = tensors[key]
        assert framework.is_equal(actual, exp)
    framework.free_tensor_memory(gbuf, device)
    del reader
    assert framework.get_mem_used() == 0
    assert fstcpp.get_cpp_metrics().bounce_buffer_bytes == 0


def _skip_if_not_pytorch(framework: FrameworkOpBase) -> None:
    if framework.get_name() != "pytorch":
        pytest.skip(
            "UnifiedMemCopier requires a framework implementing mmap_file_pinned"
        )


def test_UnifiedMemCopier(fstcpp_log, input_files, framework, monkeypatch) -> None:
    print("test_UnifiedMemCopier")
    _skip_if_not_pytorch(framework)
    import ctypes

    import torch

    meta = SafeTensorsMetadata.from_file(input_files[0], framework)
    device, dev_is_gpu = get_and_check_device(framework)

    if not dev_is_gpu:
        # Stand in for the CUDA-only primitives so the flow can run on CPU CI.
        monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)

        def fake_memcpy_h2d_async(dst, src, size):
            ctypes.memmove(dst, src, size)
            return 0

        monkeypatch.setattr(fstcpp, "memcpy_h2d_async", fake_memcpy_h2d_async)
        monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    copier = UnifiedMemCopier(meta, device, framework)
    gbuf = copier.submit_io(False, 10 * 1024 * 1024 * 1024)
    tensors = copier.wait_io(gbuf)
    for key, exp in load_safetensors_file(input_files[0], device, framework).items():
        actual = tensors[key]
        assert framework.is_equal(actual, exp)
    # Lifecycle: pinned mmap reference released in wait_io
    assert copier._pinned is None
    framework.free_tensor_memory(gbuf, device)
    assert framework.get_mem_used() == 0
    assert fstcpp.get_cpp_metrics().bounce_buffer_bytes == 0


def test_UnifiedMemCopier_cuda_error(
    fstcpp_log, input_files, framework, monkeypatch
) -> None:
    print("test_UnifiedMemCopier_cuda_error")
    _skip_if_not_pytorch(framework)
    import torch

    meta = SafeTensorsMetadata.from_file(input_files[0], framework)
    device, _ = get_and_check_device(framework)

    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    monkeypatch.setattr(fstcpp, "memcpy_h2d_async", lambda dst, src, size: 99)

    copier = UnifiedMemCopier(meta, device, framework)
    with pytest.raises(RuntimeError, match="99"):
        copier.submit_io(False, 10 * 1024 * 1024 * 1024)
    # gbuf must be freed and the pinned mmap ref released on error
    assert framework.get_mem_used() == 0
    assert copier._pinned is None


@pytest.mark.parametrize(
    "env,device_name,expected",
    [
        ("1", "", True),
        ("0", "NVIDIA GB10 Tegra Blackwell", False),
        (None, "", False),  # no device available
        (None, "NVIDIA A100-SXM4", False),
        (None, "NVIDIA GB10 Tegra Blackwell", True),
    ],
)
def test_is_unified_memory_system(
    monkeypatch, framework, env, device_name, expected
) -> None:
    if env is None:
        monkeypatch.delenv("FASTSAFETENSORS_UNIFIED_MEM", raising=False)
    else:
        monkeypatch.setenv("FASTSAFETENSORS_UNIFIED_MEM", env)
    monkeypatch.setattr(framework, "get_device_name", lambda idx: device_name)
    assert is_unified_memory_system(framework) is expected
    if env is None:
        # without a framework, only the environment override can enable it
        assert is_unified_memory_system(None) is False


@pytest.mark.parametrize(
    "device_str,nogds,unified_env,expected_type",
    [
        ("cuda:0", True, "1", "unified"),  # opt-in on GPU
        ("cuda:0", True, "0", "nogds"),  # opt-out on GPU
        ("cpu", True, "1", "nogds"),  # CPU device skips unified even with env
        ("cuda:0", False, "1", "gds"),  # nogds=False always picks gds
    ],
)
def test_SafeTensorsFileLoader_copier_selection(
    framework, monkeypatch, device_str, nogds, unified_env, expected_type
) -> None:
    _skip_if_not_pytorch(framework)
    import fastsafetensors.loader as loader_mod

    monkeypatch.setenv("FASTSAFETENSORS_UNIFIED_MEM", unified_env)

    captured = {}

    def spy_create_copier_constructor(copier_type, device, **kwargs):
        captured["copier_type"] = copier_type
        return lambda metadata, device, framework: None

    monkeypatch.setattr(
        loader_mod, "create_copier_constructor", spy_create_copier_constructor
    )

    SafeTensorsFileLoader(
        pg=SingleGroup(),
        device=device_str,
        framework=framework.get_name(),
        nogds=nogds,
    )
    assert captured["copier_type"] == expected_type


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
    loader = SafeTensorsFileLoader(
        pg=SingleGroup(),
        device=device.as_str(),
        framework=framework.get_name(),
        nogds=False,
        debug_log=True,
    )
    loader.add_filenames({0: input_files})
    bufs = loader.copy_files_to_device(
        dtype=data_type, use_buf_register=True, max_copy_block_size=256 * 1024 * 1024
    )
    last_key = ""
    last_shape: List[int] = []
    for key, exp in load_safetensors_file(input_files[0], device, framework).items():
        exp = exp.to(dtype=data_type_real)
        actual = bufs.get_tensor_wrapped(key)
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
    assert framework.get_mem_used() == 0
    assert fstcpp.get_cpp_metrics().bounce_buffer_bytes == 0


def test_SafeTensorsFileLoaderNoGds(fstcpp_log, input_files, framework) -> None:
    device, _ = get_and_check_device(framework)
    loader = SafeTensorsFileLoader(
        pg=SingleGroup(),
        device=device.as_str(),
        framework=framework.get_name(),
        nogds=True,
        debug_log=True,
    )
    loader.add_filenames({0: input_files})
    bufs = loader.copy_files_to_device()
    key_dims = OrderedDict({key: -1 for key in loader.get_keys()})
    tensors = bufs.as_dict(key_dims)
    for key, exp in load_safetensors_file(input_files[0], device, framework).items():
        actual = tensors[key]
        assert framework.is_equal(actual, exp)
    bufs.close()
    loader.close()
    assert framework.get_mem_used() == 0
    assert fstcpp.get_cpp_metrics().bounce_buffer_bytes == 0


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

    with fastsafe_open(
        input_files[0],
        device=device.as_str(),
        nogds=True,
        framework=framework.get_name(),
    ) as f:
        for filename in f.metadata().keys():
            assert filename in input_files

    with fastsafe_open(
        {0: input_files},
        device=device.as_str(),
        nogds=True,
        framework=framework.get_name(),
    ) as f:
        for k in f.keys():
            t = f.get_tensor(k)
            if framework.get_name() == "pytorch":
                import torch

                assert isinstance(t, torch.Tensor)
            elif framework.get_name() == "paddle":
                import paddle

                assert isinstance(t, paddle.Tensor)
            break
    assert framework.get_mem_used() == 0
    assert fstcpp.get_cpp_metrics().bounce_buffer_bytes == 0


def _test_type(
    tmp_dir,
    dtype: DType,
    device: Device,
    framework: FrameworkOpBase,
    to_dtype: DType = DType.AUTO,
) -> None:
    filename = os.path.join(tmp_dir, f"a.safetensors")
    t0 = framework.randn((8, 16), device=device, dtype=DType.F32).to(dtype=dtype)
    if to_dtype is not DType.AUTO:
        t0 = t0.to(dtype=to_dtype)
    save_safetensors_file({f"a": t0.get_raw()}, filename, {"fst": "sample"}, framework)
    t2 = load_safetensors_file(filename, device, framework, to_dtype=to_dtype)
    with fastsafe_open(
        filenames=[filename],
        nogds=True,
        device=device.as_str(),
        framework=framework.get_name(),
        debug_log=True,
    ) as f:
        for key in f.keys():
            t1 = f.get_tensor_wrapped(key).clone().detach()
            assert framework.is_equal(t1, t2[key])
    assert framework.get_mem_used() == 0
    assert fstcpp.get_cpp_metrics().bounce_buffer_bytes == 0


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


def test_float8_e4m3fn_to_int8(fstcpp_log, tmp_dir, framework) -> None:
    if not framework.support_fp8():
        pytest.skip("FP8 is not supported")
        return
    device, _ = get_and_check_device(framework)
    _test_type(tmp_dir, DType.F8_E4M3, device, framework, DType.I8)


def test_float4_e2m1fn_x2(fstcpp_log, tmp_dir, framework) -> None:
    """Test bit-exact round-trip for F4 (torch.float4_e2m1fn_x2).

    F4 is a packed FP4 format (two 4-bit values per byte, dtype string "F4" in
    safetensors) used for expert weight matrices in models like DeepSeek V4-Flash.

    The safetensors shape is in FP4-element count while torch.float4_e2m1fn_x2
    counts packed byte-pairs, so the PyTorch shape has half as many elements in
    the last dimension.  fastsafetensors handles the shape conversion internally.
    """
    if framework.get_name() != "pytorch":
        pytest.skip("F4 is only available in PyTorch")
        return
    import torch

    if not hasattr(torch, "float4_e2m1fn_x2"):
        pytest.skip("torch.float4_e2m1fn_x2 requires PyTorch 2.10+")
        return
    device, _ = get_and_check_device(framework)
    filename = os.path.join(tmp_dir, "f4.safetensors")
    # F4 tensors cannot be created via randn/cast; create via uint8 view.
    # Shape [8, 16] in FP4-element terms = shape [8, 8] in float4_e2m1fn_x2.
    u8 = torch.arange(64, dtype=torch.uint8, device=device.as_str()).reshape(8, 8)
    t0 = u8.view(torch.float4_e2m1fn_x2)
    save_safetensors_file({"a": t0}, filename, {"fst": "sample"}, framework)
    t_ref = load_safetensors_file(filename, device, framework)
    with fastsafe_open(
        filenames=[filename],
        nogds=True,
        device=device.as_str(),
        framework=framework.get_name(),
        debug_log=True,
    ) as f:
        for key in f.keys():
            t1 = f.get_tensor_wrapped(key).clone().detach()
            assert framework.is_equal(t1, t_ref[key])
    assert framework.get_mem_used() == 0
    assert fstcpp.get_cpp_metrics().bounce_buffer_bytes == 0


def test_float4_native_shape_and_slices(framework) -> None:
    if framework.get_name() != "pytorch":
        pytest.skip("F4 is only available in PyTorch")
        return

    assert framework.get_native_shape(DType.F4, [8, 16]) == [8, 8]
    assert framework.get_native_shape(DType.F4, [16]) == [8]
    assert framework.get_storage_shape(DType.F4, [8, 16], [16, 1]) == (
        [64],
        [1],
    )
    assert framework.get_native_slices(
        DType.F4, [8, 16], (slice(None, None, None), slice(4, 12, 1))
    ) == (slice(None, None, None), slice(2, 6, 1))
    implicit_last_dim = framework.get_native_slices(
        DType.F4, [8, 16], (slice(2, 6, 1),)
    )
    assert implicit_last_dim == (slice(2, 6, 1),)

    with pytest.raises(ValueError):
        framework.get_native_shape(DType.F4, [8, 15])
    with pytest.raises(ValueError):
        framework.get_storage_shape(DType.F4, [8, 15], [15, 1])
    with pytest.raises(ValueError):
        framework.get_native_slices(
            DType.F4, [8, 16], (slice(None, None, None), slice(3, 12, 1))
        )


def test_float8_e8m0fnu(fstcpp_log, tmp_dir, framework) -> None:
    """Test bit-exact round-trip for F8_E8M0 (torch.float8_e8m0fnu).

    F8_E8M0 is an unsigned 8-bit exponent-only format used as per-tile
    quantization scales in models like DeepSeek V4-Flash.  It has no mantissa
    bits, so ordinary randn -> cast is safe for creating test tensors.
    """
    if framework.get_name() != "pytorch":
        pytest.skip("F8_E8M0 is only available in PyTorch")
        return
    import torch

    if not hasattr(torch, "float8_e8m0fnu"):
        pytest.skip("torch.float8_e8m0fnu requires PyTorch 2.5+")
        return
    device, _ = get_and_check_device(framework)
    _test_type(tmp_dir, DType.F8_E8M0, device, framework)


def test_cpp_metrics(fstcpp_log, framework) -> None:
    device, _ = get_and_check_device(framework)
    exp_length = 0
    assert framework.get_mem_used() == exp_length

    gbuf = framework.alloc_tensor_memory(128, device)
    exp_length += 128
    assert framework.get_mem_used() == exp_length

    framework.free_tensor_memory(gbuf, device)
    exp_length -= 128
    assert framework.get_mem_used() == exp_length

    gbuf2 = framework.alloc_tensor_memory(1024, device)
    exp_length += 1024
    assert framework.get_mem_used() == exp_length

    gbuf3 = framework.alloc_tensor_memory(128, device)
    exp_length += 128
    assert framework.get_mem_used() == exp_length

    framework.free_tensor_memory(gbuf2, device)
    exp_length -= 1024
    assert framework.get_mem_used() == exp_length

    framework.free_tensor_memory(gbuf3, device)
    exp_length -= 128
    assert framework.get_mem_used() == exp_length

    assert exp_length == 0
    assert framework.get_mem_used() == 0


def test_tensor_frame_getitem() -> None:
    # Regression test: TensorFrame.__getitem__ must follow Python sequence
    # semantics. It used to add an extra +1 when converting negative indices
    # (frame[-1] addressed past the end of the dimension), undercount strided
    # slices by using floor instead of ceiling division, and raise NameError
    # for an empty tuple index.
    n, m = 5, 6

    def make_frame() -> TensorFrame:
        return TensorFrame(DType.F32, [n, m], [0, n * m * 4], [m, 1], [0, 0], False)

    ref = list(range(n))

    # integer indices: resulting offset must be the Python-normalized index
    for i in range(-n, n):
        got = make_frame()[i]
        assert got.shape == [1, m], f"frame[{i}]: shape={got.shape}"
        assert got.offsets[0] == ref[i], f"frame[{i}]: offsets={got.offsets}"
    for i in (n, -n - 1):
        with pytest.raises(IndexError):
            make_frame()[i]

    # slices: row offsets must match list slicing for every bound/step combo
    bounds = [None, -n - 2, -n, -2, -1, 0, 1, n - 2, n - 1, n, n + 2]
    steps = [None, 1, 2, 3, -1, -2]
    for start, stop, step in itertools.product(bounds, bounds, steps):
        sl = slice(start, stop, step)
        expected = ref[sl]
        got = make_frame()[sl]
        if len(expected) == 0:
            assert got.shape == [], f"frame[{sl}]: shape={got.shape}"
            continue
        assert got.shape[1] == m, f"frame[{sl}]: shape={got.shape}"
        row_step = got.strides[0] // m
        actual = [got.offsets[0] + k * row_step for k in range(got.shape[0])]
        assert actual == expected, f"frame[{sl}]: {actual} != {expected}"

    with pytest.raises(ValueError):
        make_frame()[::0]

    # empty tuple index returns the frame as-is instead of raising NameError
    got = make_frame()[()]
    assert got.shape == [n, m]
    assert got.offsets == [0, 0]
    assert got.strides == [m, 1]

    # multi-dimensional indexing
    got = make_frame()[1:-1, ::2]
    assert got.shape == [n - 2, m // 2]
    assert got.offsets == [1, 0]
    assert got.strides == [m, 2]


def test_get_multi_cols_multi_file_auto_free(fstcpp_log, tmp_dir, framework) -> None:
    # Regression test: when get_multi_cols() spans multiple files with
    # auto_mem_delete enabled, the per-tensor accounting must look up each
    # tensor's own file. It used to index rank_loaders with a leftover loop
    # variable, comparing against the wrong file's tensor count and freeing
    # the wrong device buffer.
    device, _ = get_and_check_device(framework)
    file_a = os.path.join(tmp_dir, "multicols_a.safetensors")
    file_b = os.path.join(tmp_dir, "multicols_b.safetensors")
    a0 = framework.randn((4, 8), device=device, dtype=DType.F32)
    b0 = framework.randn((4, 8), device=device, dtype=DType.F32)
    b1 = framework.randn((4, 8), device=device, dtype=DType.F32)
    save_safetensors_file({"a0": a0.get_raw()}, file_a, {"fst": "a"}, framework)
    save_safetensors_file(
        {"b0": b0.get_raw(), "b1": b1.get_raw()}, file_b, {"fst": "b"}, framework
    )

    loader = SafeTensorsFileLoader(
        SingleGroup(), device.as_str(), nogds=True, framework=framework.get_name()
    )
    try:
        loader.add_filenames({0: [file_a, file_b]})
        fb = loader.copy_files_to_device()
        # force the multi-rank auto-free accounting on a single-process group
        fb.auto_mem_delete = True

        out = fb.get_multi_cols(["a0", "b0"], dim=0)
        assert framework.is_equal(out[0:4], a0.get_raw())
        assert framework.is_equal(out[4:8], b0.get_raw())

        # file_a (lidx 0) is fully instantiated and must be freed; file_b
        # (lidx 1) still holds b1 and must keep its device buffer
        assert fb.rank_loaders[0][0].gbuf is None
        assert fb.rank_loaders[0][1].gbuf is not None

        out2 = fb.get_multi_cols(["b1"], dim=0)
        assert framework.is_equal(out2[0:4], b1.get_raw())
        assert fb.rank_loaders[0][1].gbuf is None

        fb.close()
    finally:
        loader.close()
    assert framework.get_mem_used() == 0


def test_as_dict_partial_request_close_frees_buffers(
    fstcpp_log, tmp_dir, framework
) -> None:
    # Regression test: as_dict() used to drop rank_loaders entirely when
    # auto_mem_delete was enabled, so files whose tensors were not all
    # requested kept their device buffers allocated forever (close() had
    # nothing left to free).
    device, _ = get_and_check_device(framework)
    filename = os.path.join(tmp_dir, "asdict_partial.safetensors")
    a0 = framework.randn((4, 8), device=device, dtype=DType.F32)
    a1 = framework.randn((4, 8), device=device, dtype=DType.F32)
    save_safetensors_file(
        {"a0": a0.get_raw(), "a1": a1.get_raw()}, filename, {"fst": "a"}, framework
    )

    loader = SafeTensorsFileLoader(
        SingleGroup(), device.as_str(), nogds=True, framework=framework.get_name()
    )
    try:
        loader.add_filenames({0: [filename]})
        fb = loader.copy_files_to_device()
        # force the multi-rank auto-free accounting on a single-process group
        fb.auto_mem_delete = True

        # request only a0; a1 keeps the file's buffer alive until close()
        tensors = fb.as_dict(OrderedDict([("a0", -1)]))
        assert framework.is_equal(tensors["a0"], a0.get_raw())
        assert fb.rank_loaders[0][0].gbuf is not None

        fb.close()
    finally:
        loader.close()
    assert framework.get_mem_used() == 0


def test_from_fd_short_reads(monkeypatch, tmp_dir, framework) -> None:
    # Regression test: from_fd() used to assume a single os.read() returns
    # all requested bytes; a short read truncated the header JSON.
    device, _ = get_and_check_device(framework)
    filename = os.path.join(tmp_dir, "short_read.safetensors")
    a0 = framework.randn((4, 8), device=device, dtype=DType.F32)
    save_safetensors_file({"a0": a0.get_raw()}, filename, {"fst": "a"}, framework)

    real_read = os.read

    def dribbling_read(fd: int, length: int) -> bytes:
        return real_read(fd, min(length, 3))

    monkeypatch.setattr(os, "read", dribbling_read)
    flags = os.O_RDONLY
    if sys.platform == "win32" and hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    fd = os.open(filename, flags, 0o644)
    try:
        meta = SafeTensorsMetadata.from_fd(fd, filename, framework)
    finally:
        os.close(fd)
    assert "a0" in meta.tensors
    assert meta.tensors["a0"].shape == [4, 8]


def test_no_module_level_torch_import_outside_frameworks() -> None:
    # Policy: framework-specific imports live behind the frameworks
    # abstraction. Only the torch backend (frameworks/_torch.py) may import
    # torch at module level; copiers and core modules must go through
    # FrameworkOpBase.
    import ast

    import fastsafetensors

    pkg_dir = os.path.dirname(fastsafetensors.__file__)
    allowed = {
        os.path.join("frameworks", "_torch.py"),
    }

    def imports_torch(stmts) -> bool:
        for node in stmts:
            if isinstance(node, ast.Import):
                if any(a.name.split(".")[0] == "torch" for a in node.names):
                    return True
            elif isinstance(node, ast.ImportFrom):
                if node.module is not None and node.module.split(".")[0] == "torch":
                    return True
            elif isinstance(node, ast.Try):
                if imports_torch(node.body):
                    return True
        return False

    violations = []
    for root, _, files in os.walk(pkg_dir):
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            rel = os.path.relpath(path, pkg_dir)
            if rel in allowed:
                continue
            with open(path) as f:
                tree = ast.parse(f.read(), filename=path)
            if imports_torch(tree.body):
                violations.append(rel)
    assert violations == [], f"module-level torch import found in: {violations}"
