# SPDX-License-Identifier: Apache-2.0
# Copyright 2017 by Contributors
# Copyright 2024 IBM Inc. All rights reserved
# modified apps/numpy_dlpack/dlpack/dlpack.py and apps/numpy_dlpack/dlpack/from_numpy.py in https://github.com/dmlc/dlpack
# to add from_cuda_buffer()

import ctypes
from typing import Dict, List, Union

from .st_types import Device, DeviceType, DType

_c_str_dltensor = b"dltensor"


class DLDevice(ctypes.Structure):
    def __init__(self, dev: Device):
        self.device_type = self.DeviceToDL[dev.type]
        self.device_id = dev.index if dev.index is not None else 0

    kDLCPU = 1
    kDLCUDA = 2
    _fields_ = [
        ("device_type", ctypes.c_int),
        ("device_id", ctypes.c_int),
    ]

    DeviceToDL = {
        DeviceType.CPU: kDLCPU,
        DeviceType.CUDA: kDLCUDA,
        DeviceType.GPU: kDLCUDA,
    }


class c_DLDataType(ctypes.Structure):
    def __init__(self, dtype: DType):
        (self.type_code, self.bits, self.lanes) = self.STDataToDL[dtype]

    kDLInt = 0
    kDLUInt = 1
    kDLFloat = 2
    kDLBfloat = 4
    kDLBool = 6
    _fields_ = [
        ("type_code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]

    STDataToDL: Dict[DType, tuple[int, int, int]] = {
        DType.BOOL: (kDLBool, 8, 1),
        DType.I8: (kDLInt, 8, 1),
        DType.I16: (kDLInt, 16, 1),
        DType.I32: (kDLInt, 32, 1),
        DType.I64: (kDLInt, 64, 1),
        DType.U8: (kDLUInt, 8, 1),
        DType.U16: (kDLUInt, 16, 1),
        DType.U32: (kDLUInt, 32, 1),
        DType.U64: (kDLUInt, 64, 1),
        DType.F16: (kDLFloat, 16, 1),
        DType.F32: (kDLFloat, 32, 1),
        DType.F64: (kDLFloat, 64, 1),
        DType.BF16: (kDLBfloat, 16, 1),
    }


class _Holder:
    def __init__(self, shape: List[int], strides: List[int]):
        self.shape = (ctypes.c_int64 * len(shape))(*shape)
        self.strides = (ctypes.c_int64 * len(strides))(*strides)

    def _as_manager_ctx(self) -> ctypes.c_void_p:
        py_obj = ctypes.py_object(self)
        py_obj_ptr = ctypes.pointer(py_obj)
        ctypes.pythonapi.Py_IncRef(py_obj)
        ctypes.pythonapi.Py_IncRef(ctypes.py_object(py_obj_ptr))
        return ctypes.cast(py_obj_ptr, ctypes.c_void_p)


class DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", DLDevice),
        ("ndim", ctypes.c_int),
        ("dtype", c_DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]

    def __init__(self, dev_ptr: int, dev: Device, dtype: DType, holder: _Holder):
        self.data = dev_ptr
        self.device = DLDevice(dev)
        self.ndim = len(holder.shape)
        self.dtype = c_DLDataType(dtype)
        self.shape = holder.shape
        self.strides = holder.strides
        self.byte_offset = 0

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

    def as_py(
        self,
        dev_ptr: int,
        shape: List[int],
        strides: List[int],
        dtype: DType,
        dev: Device,
    ):
        holder = _Holder(shape, strides)
        self.dl_tensor = DLTensor(dev_ptr, dev, dtype, holder)
        self.manager_ctx = holder._as_manager_ctx()
        self.deleter = _numpy_buffer_deleter
        return ctypes.pythonapi.PyCapsule_New(
            ctypes.byref(self),
            _c_str_dltensor,
            _numpy_pycapsule_deleter,
        )

    @property
    def __array_interface__(self):
        return self.dl_tensor.__array_interface__


ctypes.pythonapi.PyMem_RawMalloc.restype = ctypes.c_void_p
ctypes.pythonapi.PyMem_RawFree.argtypes = [ctypes.c_void_p]

ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object
ctypes.pythonapi.PyCapsule_New.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.c_void_p,
]


@ctypes.CFUNCTYPE(None, ctypes.c_void_p)
def _numpy_buffer_deleter(handle: Union[int, ctypes.c_void_p]) -> None:
    """A function to deallocate the memory of a cuda buffer."""
    if isinstance(handle, int):
        dl_managed_tensor = DLManagedTensor.from_address(handle)
    elif isinstance(handle, ctypes.c_void_p):
        dl_managed_tensor = DLManagedTensor.from_address(
            handle.value if handle.value else 0
        )
    else:
        raise Exception("invalid type of handle!")
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
        _numpy_buffer_deleter(dl_managed_tensor)
        ctypes.pythonapi.PyCapsule_SetDestructor(pycapsule, None)


def from_cuda_buffer(
    dev_ptr: int, shape: List[int], strides: List[int], dtype: DType, dev: Device
):
    size = ctypes.c_size_t(ctypes.sizeof(DLManagedTensor))
    dl_managed_tensor = DLManagedTensor.from_address(
        ctypes.pythonapi.PyMem_RawMalloc(size)
    )
    return dl_managed_tensor.as_py(dev_ptr, shape, strides, dtype, dev)
