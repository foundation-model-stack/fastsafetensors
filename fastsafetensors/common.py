# Copyright 2024 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0

import json
import os
import platform
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from . import cpp as fstcpp
from .dlpack import from_cuda_buffer
from .frameworks import FrameworkOpBase, TensorBase
from .st_types import Device, DType


def get_device_numa_node(device: Optional[int]) -> Optional[int]:
    if device is None or platform.system() != "Linux":
        return None
    pci_addr = fstcpp.get_device_pci_bus(device)
    if pci_addr == "":
        return None
    bus_addr = ":".join(pci_addr.split(":")[:2]).lower()
    syspath = f"/sys/class/pci_bus/{bus_addr}/device/numa_node"
    if not os.path.exists(syspath):
        return None
    with open(syspath) as f:
        return int(f.read().strip())


# keep this for compatibility
class SingleGroup:
    def size(self):
        return 1

    def rank(self):
        return 0


class SafeTensorsMetadata:
    def __init__(
        self,
        string: str,
        header_length: int,
        size_bytes: int,
        framework: FrameworkOpBase,
        src: str = "",
        keep_orig_dict: bool = False,
    ):
        self.src = src
        self.framework = framework
        ser = json.loads(string, object_pairs_hook=OrderedDict)
        self.metadata = ser.get("__metadata__", "")
        if self.metadata:
            del ser["__metadata__"]
        self.tensors: Dict[str, TensorFrame] = {}
        self.header_length = header_length
        self.aligned = header_length % framework.get_device_ptr_align() == 0
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
            nbytes = nelements * framework.get_dtype_size(t.dtype)
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
    def from_buffer(
        self, buf: int, buffer_len: int, filename: str, framework: FrameworkOpBase
    ):
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
        return SafeTensorsMetadata(string, n + 8, buffer_len, framework)

    @classmethod
    def from_fd(
        self,
        fd: int,
        filename: str,
        framework: FrameworkOpBase,
        keep_orig_dict: bool = False,
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
            framework,
            filename,
            keep_orig_dict=keep_orig_dict,
        )

    @classmethod
    def from_file(self, filename: str, framework: FrameworkOpBase):
        fd = os.open(filename, os.O_RDONLY, 0o644)
        ret = self.from_fd(fd, filename, framework=framework, keep_orig_dict=False)
        os.close(fd)
        return ret

    def get_tensors(
        self,
        gbuf: fstcpp.gds_device_buffer,
        device: Device,
        copy_start_offset: int,
        dtype: DType = DType.AUTO,
    ) -> Dict[str, TensorBase]:
        ret = {}
        for tensor_name, t in self.tensors.items():
            dst_dev_ptr = (
                gbuf.get_base_address()
                + self.header_length
                + t.data_offsets[0]
                - copy_start_offset
            )
            disk_dtype = self.framework.as_workaround_dtype(t.dtype)
            dl_tensor = from_cuda_buffer(
                dst_dev_ptr,
                t.shape,
                t.strides,
                disk_dtype,
                device,
            )
            t2 = self.framework.from_dlpack(dl_tensor, device, disk_dtype)
            if disk_dtype != t.dtype:
                t2 = t2.view(t.dtype)

            if dtype != DType.AUTO and dtype != t.dtype:
                if self.framework.get_dtype_size(dtype) > self.framework.get_dtype_size(
                    t.dtype
                ):
                    raise Exception(
                        f"Online type conversion to larger sizes is not supported ({t.dtype} -> {dtype})"
                    )
                t3 = t2.to(dtype=dtype)
                conv_dtype: DType = self.framework.as_workaround_dtype(dtype)
                dl_tensor = from_cuda_buffer(
                    dst_dev_ptr,
                    t.shape,
                    t.strides,
                    conv_dtype,
                    device,
                )
                t2 = self.framework.from_dlpack(dl_tensor, device, conv_dtype)
                if dtype != conv_dtype:
                    t2 = t2.view(dtype)
                self.framework.copy_tensor(t2, t3)
                self.tensors[tensor_name].dtype = dtype
            ret[tensor_name] = t2
        return ret

    def __repr__(self) -> str:
        return str({"__metadata__": self.metadata, "tensors": self.tensors})


@dataclass
class TensorFrame:
    dtype: DType
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
            DType(entry["dtype"]), shape, data_offsets, strides, offsets, False
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
