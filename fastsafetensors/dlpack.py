# SPDX-License-Identifier: Apache-2.0
# Copyright 2017 by Contributors
# Copyright 2024 IBM Inc. All rights reserved
# modified apps/numpy_dlpack/dlpack/dlpack.py and apps/numpy_dlpack/dlpack/from_numpy.py in https://github.com/dmlc/dlpack
# to add from_cuda_buffer()

import ctypes
import torch
from typing import List

_c_str_dltensor = b"dltensor"

class DLDevice(ctypes.Structure):
    def __init__(self, device: torch.device):
        self.device_type = self.TYPE_MAP[device.type]
        self.device_id = 0
        if device.index:
            self.device_id = device.index

    kDLCPU = 1
    kDLCUDA = 2
    _fields_ = [
        ("device_type", ctypes.c_int),
        ("device_id", ctypes.c_int),
    ]
    TYPE_MAP= {
        "cpu": kDLCPU,
        "cuda": kDLCUDA,
    }


class DLDataTypeCode(ctypes.c_uint8):
    kDLInt = 0
    kDLUInt = 1
    kDLFloat = 2
    kDLBfloat = 4

    def __str__(self):
        return {
            self.kDLInt: "int",
            self.kDLUInt: "uint",
            self.kDLFloat: "float",
            self.kDLBfloat: "bfloat",
        }[self.value]


class DLDataType(ctypes.Structure):
    _fields_ = [
        ("type_code", DLDataTypeCode),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]
    TYPE_MAP = {
        torch.bool: (1, 1, 1),
        torch.int16: (0, 16, 1),
        torch.int32: (0, 32, 1),
        torch.int: (0, 32, 1),
        torch.int64: (0, 64, 1),
        torch.uint8: (1, 8, 1),
        torch.float16: (2, 16, 1),
        torch.float32: (2, 32, 1),
        torch.float64: (2, 64, 1),
        torch.bfloat16: (4, 16, 1),
    }


class DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", DLDevice),
        ("ndim", ctypes.c_int),
        ("dtype", DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]

    @property
    def itemsize(self):
        return self.dtype.lanes * self.dtype.bits // 8

    @property
    def __array_interface__(self):
        shape = tuple(self.shape[dim] for dim in range(self.ndim))
        if self.strides:
            strides = tuple(
                self.strides[dim] * self.itemsize for dim in range(self.ndim)
            )
        else:
            # Array is compact, make it numpy compatible.
            strides = []
            for i, s in enumerate(shape):
                cumulative = 1
                for e in range(i + 1, self.ndim):
                    cumulative *= shape[e]
                strides.append(cumulative * self.itemsize)
            strides = tuple(strides)
        typestr = "|" + str(self.dtype.type_code)[0] + str(self.itemsize)
        return dict(
            version=3,
            shape=shape,
            strides=strides,
            data=(self.data, True),
            offset=self.byte_offset,
            typestr=typestr,
        )


class DLManagedTensor(ctypes.Structure):
    _fields_ = [
        ("dl_tensor", DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.CFUNCTYPE(None, ctypes.c_void_p)),
    ]

    @property
    def __array_interface__(self):
        return self.dl_tensor.__array_interface__


ctypes.pythonapi.PyMem_RawMalloc.restype = ctypes.c_void_p
ctypes.pythonapi.PyMem_RawFree.argtypes = [ctypes.c_void_p]

ctypes.pythonapi.PyCapsule_New.restype=ctypes.py_object
ctypes.pythonapi.PyCapsule_New.argtypes=[ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]


class _Holder:
    def __init__(self, shape: List[int], strides: List[int]):
        self.shape = (ctypes.c_int64*len(shape))(*shape)
        self.strides = (ctypes.c_int64*len(strides))(*strides)

    def _as_manager_ctx(self) -> ctypes.c_void_p:
        py_obj = ctypes.py_object(self)
        py_obj_ptr = ctypes.pointer(py_obj)
        ctypes.pythonapi.Py_IncRef(py_obj)
        ctypes.pythonapi.Py_IncRef(ctypes.py_object(py_obj_ptr))
        return ctypes.cast(py_obj_ptr, ctypes.c_void_p)

@ctypes.CFUNCTYPE(None, ctypes.c_void_p)
def _numpy_cuda_buffer_deleter(handle: ctypes.c_void_p) -> None:
    """A function to deallocate the memory of a cuda buffer."""
    dl_managed_tensor = DLManagedTensor.from_address(handle)
    py_obj_ptr = ctypes.cast(
        dl_managed_tensor.manager_ctx, ctypes.POINTER(ctypes.py_object)
    )
    py_obj = py_obj_ptr.contents
    ctypes.pythonapi.Py_DecRef(py_obj)
    ctypes.pythonapi.Py_DecRef(ctypes.py_object(py_obj_ptr))
    ctypes.pythonapi.PyMem_RawFree(handle)

@ctypes.CFUNCTYPE(None, ctypes.c_void_p)
def _numpy_pycapsule_deleter(handle: ctypes.c_void_p) -> None:
    """A function to deallocate a pycapsule that wraps a cuda buffer."""
    pycapsule: ctypes.py_object = ctypes.cast(handle, ctypes.py_object)
    if ctypes.pythonapi.PyCapsule_IsValid(pycapsule, _c_str_dltensor):
        dl_managed_tensor = ctypes.pythonapi.PyCapsule_GetPointer(
            pycapsule, _c_str_dltensor
        )
        _numpy_cuda_buffer_deleter(dl_managed_tensor)
        ctypes.pythonapi.PyCapsule_SetDestructor(pycapsule, None)

def from_cuda_buffer(dev_ptr: int, shape: List[int], strides: List[int], dtype: torch.dtype, device: torch.device):
    holder = _Holder(shape, strides)
    size = ctypes.c_size_t(ctypes.sizeof(DLManagedTensor))
    dl_managed_tensor = DLManagedTensor.from_address(
        ctypes.pythonapi.PyMem_RawMalloc(size)
    )
    dl_managed_tensor.dl_tensor.data = dev_ptr
    dl_managed_tensor.dl_tensor.device = DLDevice(device)
    dl_managed_tensor.dl_tensor.ndim = len(holder.shape)
    dl_managed_tensor.dl_tensor.dtype = DLDataType.TYPE_MAP[dtype]
    dl_managed_tensor.dl_tensor.shape = holder.shape
    dl_managed_tensor.dl_tensor.strides = holder.strides
    dl_managed_tensor.dl_tensor.byte_offset = 0
    dl_managed_tensor.manager_ctx = holder._as_manager_ctx()
    dl_managed_tensor.deleter = _numpy_cuda_buffer_deleter
    pycapsule = ctypes.pythonapi.PyCapsule_New(
        ctypes.byref(dl_managed_tensor),
        _c_str_dltensor,
        _numpy_pycapsule_deleter,
    )
    return pycapsule