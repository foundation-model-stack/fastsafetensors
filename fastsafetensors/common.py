# Copyright 2024 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch

from .st_types import STDevice, STDeviceType, STDType, STEnv, get_dtype_size

try:
    import paddle
    from paddle.framework import core as paddle_core

    paddle_loaded = True
except:
    paddle_loaded = False
if paddle_loaded:
    import numpy

import json
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from . import cpp as fstcpp
from .dlpack import from_cuda_buffer

ALIGN: int = fstcpp.get_alignment_size()
CUDA_PTR_ALIGN: int = 16

try:
    CUDA_VER = str(torch.version.cuda if not paddle_loaded else paddle.version.cuda())
except:
    CUDA_VER = "0.0"

dtype_convert: Dict[STEnv, Dict[STDType, Any]] = {
    STEnv.Pytorch: {
        STDType.BOOL: torch.bool,
        STDType.U8: torch.uint8,
        STDType.I8: torch.int8,
        STDType.F8_E5M2: torch.float8_e5m2,
        STDType.F8_E4M3: torch.float8_e4m3fn,
        STDType.I16: torch.int16,
        STDType.U16: torch.int16,
        STDType.I32: torch.int32,
        STDType.U32: torch.int32,
        STDType.I64: torch.int64,
        STDType.U64: torch.int64,
        STDType.F16: torch.float16,
        STDType.BF16: torch.bfloat16,
        STDType.F32: torch.float32,
        STDType.F64: torch.float64,
    }
}

if paddle_loaded:
    dtype_convert[STEnv.Paddle] = {
        STDType.BOOL: paddle.bool,
        STDType.I8: paddle.uint8,
        STDType.I8: paddle.int8,
        STDType.F8_E5M2: paddle.float8_e5m2,
        STDType.F8_E4M3: paddle.float8_e4m3fn,
        STDType.I16: paddle.int16,
        STDType.U16: paddle.bfloat16,
        STDType.I32: paddle.int32,
        STDType.U32: paddle.int32,
        STDType.I64: paddle.int64,
        STDType.U64: paddle.int64,
        STDType.F16: paddle.float16,
        STDType.BF16: paddle.bfloat16,
        STDType.F32: paddle.float32,
        STDType.F64: paddle.float64,
    }

need_workaround_dtypes: Dict[STDType, STDType] = {
    STDType.F8_E5M2: STDType.I8,
    STDType.F8_E4M3: STDType.I8,
}


def get_device_numa_node(device: Optional[int]):
    if device is None:
        return
    pci_addr = fstcpp.get_device_pci_bus(device)
    if pci_addr == "":
        # raise Exception(f"get_device_numa_node, get_device_pci_bus failed, device={device}")
        return
    bus_addr = ":".join(pci_addr.split(":")[:2]).lower()
    syspath = f"/sys/class/pci_bus/{bus_addr}/device/numa_node"
    if not os.path.exists(syspath):
        return 0
    with open(syspath) as f:
        return int(f.read().strip())


def alloc_tensor_memory(
    length: int, dev: STDevice, framework: STEnv = STEnv.Pytorch
) -> fstcpp.gds_device_buffer:
    dev_is_gpu = True
    if framework == STEnv.Pytorch and dev.type == STDeviceType.CUDA:
        rbuf = torch.cuda.caching_allocator_alloc(length)
    elif paddle_loaded and framework == STEnv.Paddle and dev.type == STDeviceType.GPU:
        rbuf = fstcpp.gpu_malloc(length)
    else:
        dev_is_gpu = False
        rbuf = fstcpp.cpu_malloc(length)
    return fstcpp.gds_device_buffer(rbuf, length, dev_is_gpu)


def free_tensor_memory(
    gbuf: fstcpp.gds_device_buffer, dev: STDevice, framework: STEnv = STEnv.Pytorch
):
    if framework == STEnv.Pytorch and dev.type == STDeviceType.CUDA:
        torch.cuda.caching_allocator_delete(gbuf.get_base_address())
    elif paddle_loaded and framework == STEnv.Paddle and dev.type == STDeviceType.GPU:
        fstcpp.gpu_free(gbuf.get_base_address())
    else:
        fstcpp.cpu_free(gbuf.get_base_address())


class SafeTensorsMetadata:
    def __init__(
        self,
        string: str,
        header_length: int,
        size_bytes: int,
        src: str = "",
        keep_orig_dict: bool = False,
        framework: STEnv = STEnv.Pytorch,
    ):
        self.src = src
        self.framework = framework
        ser = json.loads(string, object_pairs_hook=OrderedDict)
        self.metadata = ser.get("__metadata__", "")
        if self.metadata:
            del ser["__metadata__"]
        self.tensors: Dict[str, TensorFrame] = {}
        self.header_length = header_length
        self.aligned = header_length % CUDA_PTR_ALIGN == 0
        if keep_orig_dict:
            self.ser = ser

        start = 0
        for _, (k, buffer) in enumerate(
            sorted(ser.items(), key=lambda x: x[1]["data_offsets"][0])
        ):
            t: TensorFrame = TensorFrame.from_buffer(buffer)
            self.tensors[k] = t
            # validation
            s, e = t.data_offsets
            if s != start or e < s:
                raise Exception(
                    f"validate(tensor {k}): InvalidOffset s={s}, start={start}, e={e}, src={src}"
                )
            # if (header_length + s) % CUDA_PTR_ALIGN > 0:
            #    print(f"[WARNING] misaligned tensor is detected at {header_length + s}. this will cause cuda pointer alignment errors later.")
            start = e
            nelements = 1
            for sh in t.shape:
                nelements *= sh
            real_dtype = dtype_convert[self.framework][t.dtype]
            if self.framework == STEnv.Pytorch:
                t_dtype_size = real_dtype.itemsize
            elif paddle_loaded and self.framework == STEnv.Paddle:
                t_dtype_size = paddle_core.size_of_dtype(real_dtype)
            nbytes = nelements * t_dtype_size
            if (e - s) != nbytes:
                raise Exception(
                    f"validate(tensor {k}): TensorInvalidInfo, e-s={e-s}, nbytes={nbytes}, src={src}"
                )
        self.size_bytes = size_bytes
        if start + header_length != size_bytes:
            raise Exception(
                f"MetadataIncompleteBuffer, src={src}, start={start}, header_length={header_length}, size_bytes={size_bytes}"
            )

    @classmethod
    def from_buffer(self, buf: int, buffer_len: int, filename: str):
        if buffer_len < 8:
            raise Exception(
                f"from_buffer: HeaderTooSmall, filename={filename}, buffer_len={buffer_len}"
            )
        arr = fstcpp.read_buffer(buf, 8)
        n = int.from_bytes(arr, byteorder="little", signed=False)
        if n > 100000000:
            raise Exception(
                f"from_buffer: HeaderTooLarge, n={n}, filename={filename}, buffer_len={buffer_len}"
            )
        if n > buffer_len - 8:
            raise Exception(
                f"from_buffer: InvalidHeaderLength, n={n}, filename={filename}, buffer_len={buffer_len}"
            )
        string = fstcpp.read_buffer(buf + 8, n).decode("utf-8")
        # Assert the string starts with {
        # NOTE: Add when we move to 0.4.0
        # if string.startswith('{'):
        #    raise Exception(f"{filename}: InvalidHeaderStart")
        return SafeTensorsMetadata(string, n + 8, buffer_len)

    @classmethod
    def from_fd(
        self,
        fd: int,
        filename: str,
        keep_orig_dict: bool = False,
        framework: STEnv = STEnv.Pytorch,
    ):
        status = os.fstat(fd)
        buffer_len = status.st_size
        if buffer_len < 8:
            raise Exception(f"{filename}: HeaderTooSmall, buffer_len={buffer_len}")
        arr = os.read(fd, 8)
        n = int.from_bytes(arr, byteorder="little", signed=False)
        if n > 100000000:
            raise Exception(
                f"{filename}: HeaderTooLarge, n={n}, buffer_len={buffer_len}"
            )
        if n > buffer_len - 8:
            raise Exception(
                f"{filename}: InvalidHeaderLength, n={n}, buffer_len={buffer_len}"
            )
        string = os.read(fd, n).decode("utf-8")
        # Assert the string starts with {
        # NOTE: Add when we move to 0.4.0
        # if string.startswith('{'):
        #    raise Exception(f"{filename}: InvalidHeaderStart")
        return SafeTensorsMetadata(
            string,
            n + 8,
            buffer_len,
            filename,
            keep_orig_dict=keep_orig_dict,
            framework=framework,
        )

    @classmethod
    def from_file(self, filename: str, framework: STEnv = STEnv.Pytorch):
        fd = os.open(filename, os.O_RDONLY, 0o644)
        ret = self.from_fd(fd, filename, keep_orig_dict=False, framework=framework)
        os.close(fd)
        return ret

    def get_tensors(
        self,
        gbuf: fstcpp.gds_device_buffer,
        device: STDevice,
        copy_start_offset: int,
        dtype: STDType = STDType.AUTO,
    ) -> Dict[str, torch.Tensor]:
        ret = {}
        converter = dtype_convert[self.framework]
        for tensor_name, t in self.tensors.items():
            dst_dev_ptr = (
                gbuf.get_base_address()
                + self.header_length
                + t.data_offsets[0]
                - copy_start_offset
            )
            disk_dtype = t.dtype
            if disk_dtype in need_workaround_dtypes:
                disk_dtype = need_workaround_dtypes[disk_dtype]
            dl_tensor = from_cuda_buffer(
                dst_dev_ptr,
                t.shape,
                t.strides,
                disk_dtype,
                device,
            )
            real_disk_dtype = converter[t.dtype]
            if self.framework == STEnv.Pytorch:
                t2 = torch.from_dlpack(dl_tensor)
                if disk_dtype != t.dtype:
                    t2 = t2.view(real_disk_dtype)
            elif self.framework == STEnv.Paddle:
                t2 = paddle.utils.dlpack.from_dlpack(dl_tensor)
                if disk_dtype != t.dtype:
                    t2_np = t2.numpy().view(numpy.int8)
                    t2 = paddle.to_tensor(t2_np, dtype=real_disk_dtype)
            else:
                raise Exception(f"framework is not supported: {self.framework}")

            if dtype != STDType.AUTO and dtype != t.dtype:
                if get_dtype_size(dtype) > get_dtype_size(t.dtype):
                    raise Exception(
                        f"Online type conversion to larger sizes is not supported ({t.dtype} -> {dtype})"
                    )
                real_req_dtype = converter[dtype]
                t3 = t2.to(dtype=real_req_dtype)
                conv_dtype: STDType = dtype
                if conv_dtype in need_workaround_dtypes:
                    conv_dtype = need_workaround_dtypes[conv_dtype]
                dl_tensor = from_cuda_buffer(
                    dst_dev_ptr,
                    t.shape,
                    t.strides,
                    conv_dtype,
                    device,
                )
                if self.framework == STEnv.Pytorch:
                    t2 = torch.from_dlpack(dl_tensor)
                    if dtype != conv_dtype:
                        t2 = t2.view(real_req_dtype)
                    t2.copy_(t3)
                elif self.framework == STEnv.Paddle:
                    t2 = paddle.utils.dlpack.from_dlpack(dl_tensor)
                    if dtype != conv_dtype:
                        x_np = t2.numpy().view(numpy.int8)
                        t2 = paddle.to_tensor(x_np, dtype=real_req_dtype)
                    paddle.assign(t3, output=t2)
                else:
                    raise Exception(f"framework is not supported: {self.framework}")
                self.tensors[tensor_name].dtype = dtype
            ret[tensor_name] = t2
        return ret

    def __repr__(self) -> str:
        return str({"__metadata__": self.metadata, "tensors": self.tensors})


@dataclass
class TensorFrame:
    dtype: STDType
    shape: List[int]
    data_offsets: List[int]
    strides: List[int]
    offsets: List[int]
    sliced: bool

    @classmethod
    def from_buffer(self, entry: OrderedDict[str, List[int]]):
        shape = entry["shape"]
        data_offsets = list(entry["data_offsets"])
        strides = []
        offsets = []
        for i in range(0, len(shape)):
            s = 1
            for j in range(i + 1, len(shape)):
                s *= shape[j]
            strides.append(s)
            offsets.append(0)
        return TensorFrame(
            STDType(entry["dtype"]), shape, data_offsets, strides, offsets, False
        )

    def __repr__(self) -> str:
        return str(
            {
                "dtype": self.dtype,
                "shape": self.shape,
                "data_offsets": self.data_offsets,
            }
        )

    # TODO: reduce dim if isinstance(_val, int) == True
    def __getitem__(self, _val) -> "TensorFrame":
        val: Tuple = ()
        if isinstance(_val, slice) or isinstance(_val, int):
            val = (_val,)
        elif isinstance(_val, tuple):
            val = _val
        else:
            raise Exception(f"[BUG] Unsupported index type for DiskTensor: {_val}")
        if len(val) > len(self.shape):
            raise Exception(
                f"[BUG] tried to get too large slice {_val} from {self.shape}"
            )
        shape: List[int] = []
        strides: List[int] = []
        offsets: List[int] = []
        for dim in range(0, len(val)):
            if isinstance(val[dim], int):
                start = val[dim]
                if start >= self.shape[dim] or start < -self.shape[dim]:
                    raise IndexError(
                        f"[BUG] tried to access index {start} at dim={dim} for shape={self.shape}"
                    )
                if start < 0:
                    start = self.shape[dim] + start + 1
                stop = start + 1
                step = 1
                length = 1
            elif isinstance(val[dim], slice):
                start = val[dim].start
                if start is None:
                    start = 0
                if start >= self.shape[dim] or start < -self.shape[dim]:
                    start = self.shape[dim]
                if start < 0:
                    start = self.shape[dim] + start + 1
                stop = val[dim].stop
                if stop is None or stop >= self.shape[dim] or stop < -self.shape[dim]:
                    stop = self.shape[dim]
                if stop < 0:
                    stop = self.shape[dim] + stop + 1
                step = val[dim].step
                if step is None:
                    step = 1
                if step == 0:
                    raise ValueError(f"[BUG] slice step cannot be zero")
                length = stop - start
                if (
                    length == 0
                    or (length < 0 and step > 0)
                    or (length > 0 and step < 0)
                ):
                    return TensorFrame(self.dtype, [], self.data_offsets, [], [], False)
                if length < 0 and step < 0:
                    length *= -1
            else:
                raise Exception(
                    f"[BUG] Unsupported index type for DiskTensor: {_val} at dim={dim}"
                )
            offsets.append(self.offsets[dim] + start)
            strides.append(self.strides[dim] * step)
            shape.append(length // (step if step > 0 else -step))
        for rdim in range(dim + 1, len(self.shape)):
            offsets.append(self.offsets[rdim])
            strides.append(self.strides[rdim])
            shape.append(self.shape[rdim])
        return TensorFrame(self.dtype, shape, self.data_offsets, strides, offsets, True)
