# Copyright 2024 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0

import torch
import os
import json
from collections import OrderedDict
from . import cpp as fstcpp
from typing import List, Dict, Tuple
from .dlpack import from_cuda_buffer

class SingleGroup:
    def size(self):
        return 1
    def rank(self):
        return 0

ALIGN: int = fstcpp.get_alignment_size()
CUDA_PTR_ALIGN: int = 16

framework_index = {
    "pytorch": 1,
}
dtype_convert = {
    'BOOL': (1, torch.bool), 'U8': (1, torch.uint8), 'I8': (1, torch.int8), 'F8_E5M2': (1, torch.float), 'F8_E4M3': (1, torch.int8),
    'I16': (2, torch.int16), 'U16': (2, torch.int16), 'I32': (4, torch.int32), 'U32': (4, torch.int32), 'I64': (8, torch.int64), 'U64': (8, torch.int64),
    'F16': (2, torch.float16), 'BF16': (2, torch.bfloat16), 'F32': (4, torch.float32), 'F64': (8, torch.float64)
}

def str_to_dtype(dtype_str: str, framework: str="pytorch")->torch.dtype:
    if framework != "pytorch":
        raise NotImplementedError(f"str_to_dtype: Not implemented for other frameworks than pytorch")
    if dtype_str not in dtype_convert:
        raise ValueError(f"str_to_dtype: Not supported dtype: {dtype_str}")
    return dtype_convert[dtype_str][framework_index[framework]]

def get_device_numa_node(device: int):
    if device is None:
        return
    pci_addr = fstcpp.get_device_pci_bus(device)
    if pci_addr == "":
        raise Exception(f"get_device_numa_node, get_device_pci_bus failed, device={device}")
    bus_addr = ':'.join(pci_addr.split(":")[:2]).lower()
    with open(f"/sys/class/pci_bus/{bus_addr}/device/numa_node") as f:
        return int(f.read().strip())

def alloc_tensor_memory(length: int)->fstcpp.gds_device_buffer:
    if torch.cuda.is_available():
        rbuf = torch.cuda.caching_allocator_alloc(length)
    else:
        rbuf = fstcpp.cpu_malloc(length)
    return fstcpp.gds_device_buffer(rbuf, length)

def free_tensor_memory(gbuf: fstcpp.gds_device_buffer):
    if torch.cuda.is_available():
        rbuf = torch.cuda.caching_allocator_delete(ptr)
    else:
        rbuf = fstcpp.cpu_free(ptr)
    return rbuf


class SafeTensorsMetadata:
    def __init__(self, string: str, header_length: int, size_bytes: int, src: str="", keep_orig_dict: bool=False):
        self.src = src
        ser = json.loads(string, object_pairs_hook=OrderedDict)
        self.metadata = ser['__metadata__']
        del(ser['__metadata__'])
        self.tensors: Dict[str, TensorFrame] = {}
        self.header_length = header_length
        self.aligned = header_length % CUDA_PTR_ALIGN == 0
        if keep_orig_dict:
            self.ser = ser

        start = 0
        for _, (k, buffer) in enumerate(sorted(ser.items(), key=lambda x: x[1]['data_offsets'][0])):
            t = TensorFrame.from_buffer(buffer)
            self.tensors[k] = t
            # validation
            s, e = t.data_offsets
            if s != start or e < s:
                raise Exception(f"validate(tensor {k}): InvalidOffset s={s}, start={start}, e={e}, src={src}")
            #if (header_length + s) % CUDA_PTR_ALIGN > 0:
            #    print(f"[WARNING] misaligned tensor is detected at {header_length + s}. this will cause cuda pointer alignment errors later.")
            start = e
            nelements = 1
            for sh in t.shape:
                nelements *= sh
            nbytes = nelements * t.dtype.itemsize
            if (e - s) != nbytes:
                raise Exception(f"validate(tensor {k}): TensorInvalidInfo, e-s={e-s}, nbytes={nbytes}, src={src}")
        self.size_bytes = size_bytes
        if start + header_length != size_bytes:
            raise Exception(f"MetadataIncompleteBuffer, src={src}, start={start}, header_length={header_length}, size_bytes={size_bytes}")

    @classmethod
    def from_buffer(self, buf: int, buffer_len: int, filename: str):
        if buffer_len < 8:
            raise Exception(f"from_buffer: HeaderTooSmall, filename={filename}, buffer_len={buffer_len}")
        arr = fstcpp.read_buffer(buf, 8)
        n = int.from_bytes(arr, byteorder='little', signed=False)
        if n > 100000000:
            raise Exception(f"from_buffer: HeaderTooLarge, n={n}, filename={filename}, buffer_len={buffer_len}")
        if n > buffer_len - 8:
            raise Exception(f"from_buffer: InvalidHeaderLength, n={n}, filename={filename}, buffer_len={buffer_len}")
        string = fstcpp.read_buffer(buf+8, n).decode('utf-8')
        # Assert the string starts with {
        # NOTE: Add when we move to 0.4.0
        #if string.startswith('{'):
        #    raise Exception(f"{filename}: InvalidHeaderStart")
        return SafeTensorsMetadata(string, n + 8, buffer_len)

    @classmethod
    def from_fd(self, fd: int, filename: str, keep_orig_dict: bool=False):
        status = os.fstat(fd)
        buffer_len = status.st_size
        if buffer_len < 8:
            raise Exception(f"{filename}: HeaderTooSmall, buffer_len={buffer_len}")
        arr = os.read(fd, 8)
        n = int.from_bytes(arr, byteorder='little', signed=False)
        if n > 100000000:
            raise Exception(f"{filename}: HeaderTooLarge, n={n}, buffer_len={buffer_len}")
        if n > buffer_len - 8:
            raise Exception(f"{filename}: InvalidHeaderLength, n={n}, buffer_len={buffer_len}")
        string = os.read(fd, n).decode('utf-8')
        # Assert the string starts with {
        # NOTE: Add when we move to 0.4.0
        #if string.startswith('{'):
        #    raise Exception(f"{filename}: InvalidHeaderStart")
        return SafeTensorsMetadata(string, n + 8, buffer_len, filename, keep_orig_dict=keep_orig_dict)

    @classmethod
    def from_file(self, filename: str):
        fd = os.open(filename, os.O_RDONLY, 0o644)
        ret = self.from_fd(fd, filename, keep_orig_dict=False)
        os.close(fd)
        return ret

    def get_tensors(self, gbuf: fstcpp.gds_device_buffer, device: torch.device, copy_start_offset: int, dtype: torch.dtype=None) -> Dict[str, torch.Tensor]:
        ret = {}
        for tensor_name, t in self.tensors.items():
            dst_dev_ptr = gbuf.get_base_address() + self.header_length + t.data_offsets[0]-copy_start_offset
            t2 = torch.from_dlpack(from_cuda_buffer(dst_dev_ptr, t.shape, t.strides, t.dtype, device))
            if dtype is not None and dtype != t.dtype:
                if dtype.itemsize > t.dtype.itemsize:
                    raise Exception(f"Online type conversion to larger sizes is not supported ({t.dtype} -> {dtype})")
                t3 = t2.to(dtype=dtype)
                t2 = torch.from_dlpack(from_cuda_buffer(dst_dev_ptr, t.shape, t.strides, dtype, device))
                t2.copy_(t3)
                self.tensors[tensor_name].dtype = dtype
            ret[tensor_name] = t2
        return ret

    def __repr__(self)->str:
        return str({"__metadata__": self.metadata, "tensors": self.tensors})

class TensorFrame:
    def __init__(self, dtype: torch.dtype, shape: torch.Size, data_offsets: List[int], strides: List[int], offsets: List[int], sliced: bool):
        self.dtype = dtype
        self.shape = shape
        self.data_offsets = data_offsets
        self.strides = strides
        self.offsets = offsets
        self.sliced = sliced

    @classmethod
    def from_buffer(self, entry: OrderedDict[str, List[int]], framework:str="pytorch"):
        dtype = str_to_dtype(entry['dtype'])
        shape = torch.Size(entry['shape'])
        data_offsets = list(entry['data_offsets'])
        strides = []
        offsets = []
        for i in range(0, len(shape)):
            s = 1
            for j in range(i+1, len(shape)):
                s *= shape[j]
            strides.append(s)
            offsets.append(0)
        return TensorFrame(dtype, shape, data_offsets, strides, offsets, False)

    def __repr__(self)->str:
        return str({
            "dtype": self.dtype, "shape": self.shape, "data_offsets": self.data_offsets,
        })

    # TODO: reduce dim if isinstance(_val, int) == True
    def __getitem__(self, _val):
        val: Tuple = ()
        if isinstance(_val, slice) or isinstance(_val, int):
            val = (_val,)
        elif isinstance(_val, Tuple):
            val = _val
        else:
            raise Exception(f"[BUG] Unsupported index type for DiskTensor: {_val}")
        if len(val) > len(self.shape):
            raise Exception(f"[BUG] tried to get too large slice {_val} from {self.shape}")
        shape: List[int] = []
        strides: List[int] = []
        offsets: List[int] = []
        for dim in range(0, len(val)):
            if isinstance(val[dim], int):
                start = val[dim]
                if start >= self.shape[dim] or start < -self.shape[dim]:
                    raise IndexError(f"[BUG] tried to access index {start} at dim={dim} for shape={self.shape}")
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
                if length == 0 or (length < 0 and step > 0) or (length > 0 and step < 0):
                    return TensorFrame(self.dtype, torch.Size([]), self.data_offsets, (), ())
                if length < 0 and step < 0:
                    length *= -1
            else:
                raise Exception(f"[BUG] Unsupported index type for DiskTensor: {_val} at dim={dim}")
            offsets.append(self.offsets[dim] + start)
            strides.append(self.strides[dim] * step)
            shape.append(length // (step if step > 0 else -step))
        for rdim in range(dim + 1, len(self.shape)):
            offsets.append(self.offsets[rdim])
            strides.append(self.strides[rdim])
            shape.append(self.shape[rdim])
        return TensorFrame(self.dtype, torch.Size(shape), self.data_offsets, strides, offsets, True)
