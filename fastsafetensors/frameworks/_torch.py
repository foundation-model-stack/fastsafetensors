# Copyright 2025 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0

try:
    import torch
    import torch.distributed as dist
except ImportError as e:
    raise ImportError("could not import torch. Please install it.") from e

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..common import SingleGroup
from ..cpp import cpu_free, cpu_malloc, gds_device_buffer
from ..st_types import Device, DeviceType, DType
from . import FrameworkOpBase, ProcessGroupBase, TensorBase

dtype_convert: Dict[DType, Any] = {
    DType.BOOL: torch.bool,
    DType.U8: torch.uint8,
    DType.I8: torch.int8,
    DType.I16: torch.int16,
    DType.I32: torch.int32,
    DType.I64: torch.int64,
    DType.F16: torch.float16,
    DType.BF16: torch.bfloat16,
    DType.F32: torch.float32,
    DType.F64: torch.float64,
}
need_workaround_dtypes: Dict[DType, DType] = {
    DType.F8_E5M2: DType.I8,
    DType.F8_E4M3: DType.I8,
}

if hasattr(torch, "float8_e5m2"):
    dtype_convert[DType.F8_E5M2] = torch.float8_e5m2
if hasattr(torch, "float8_e4m3fn"):
    dtype_convert[DType.F8_E4M3] = torch.float8_e4m3fn
if hasattr(torch, "uint16"):
    dtype_convert[DType.U16] = torch.uint16
if hasattr(torch, "uint32"):
    dtype_convert[DType.U32] = torch.uint32
if hasattr(torch, "uint64"):
    dtype_convert[DType.U64] = torch.uint64


@dataclass
class TorchTensor(TensorBase):
    real_tensor: torch.Tensor

    def get_raw(self) -> torch.Tensor:
        return self.real_tensor

    def contiguous(self) -> "TorchTensor":
        return TorchTensor(self.device, self.dtype, self.real_tensor.contiguous())

    def to(
        self,
        device: Optional[Device] = None,
        dtype: DType = DType.AUTO,
    ) -> "TorchTensor":
        to_dev: Optional[str] = None
        if device is not None and self.device != device:
            to_dev = device.as_str()
        else:
            device = self.device
        to_dtype: Optional[torch.dtype] = None
        if dtype != DType.AUTO and (dtype != self.dtype):
            to_dtype = dtype_convert[dtype]
        else:
            dtype = self.dtype
        if to_dev is not None or to_dtype is not None:
            return TorchTensor(
                device, dtype, self.real_tensor.to(device=to_dev, dtype=to_dtype)
            )
        return self

    def clone(self) -> "TorchTensor":
        return TorchTensor(self.device, self.dtype, self.real_tensor.clone())

    def detach(self) -> "TorchTensor":
        return TorchTensor(self.device, self.dtype, self.real_tensor.detach())

    def view(self, dtype: DType) -> "TorchTensor":
        t2 = self.real_tensor.view(dtype_convert[dtype])
        return TorchTensor(self.device, dtype, t2)

    def __getitem__(self, _val) -> "TorchTensor":
        return TorchTensor(self.device, self.dtype, self.real_tensor[_val])


@dataclass
class TorchProcessGroup(ProcessGroupBase[TorchTensor]):
    real_pg: Optional[dist.ProcessGroup]

    def size(self) -> int:
        return self.real_pg.size() if self.real_pg else 1

    def rank(self) -> int:
        return self.real_pg.rank() if self.real_pg else 0

    def broadcast(self, dst: TorchTensor, rank: int) -> None:
        if self.real_pg:
            dist.broadcast(dst.real_tensor, rank, group=self.real_pg)

    def scatter(
        self,
        dst: TorchTensor,
        scatter_list: List[TorchTensor],
        src: int,
    ) -> None:
        if self.real_pg:
            sl = [t.real_tensor for t in scatter_list]
            dist.scatter(dst.real_tensor, scatter_list=sl, src=src, group=self.real_pg)

    def send(
        self,
        t: TorchTensor,
        dst_rank: int,
        tag: int,
    ):
        if self.real_pg:
            dist.send(t.real_tensor, dst_rank, group=self.real_pg, tag=tag)

    def recv(
        self,
        t: TorchTensor,
        src_rank: int,
        tag: int,
    ):
        if self.real_pg:
            dist.recv(t.real_tensor, src_rank, group=self.real_pg, tag=tag)


class TorchOp(FrameworkOpBase[TorchTensor, TorchProcessGroup]):
    def get_name(self) -> str:
        return "pytorch"

    def get_device(self, device: str, pg: TorchProcessGroup) -> Device:
        dev = torch.device(device)
        return Device(DeviceType(dev.type), dev.index)

    def set_device(self, device: Device) -> None:
        if device.type != DeviceType.CPU:
            torch.cuda.set_device(device.as_str())

    def alloc_tensor_memory(self, length: int, dev: Device) -> gds_device_buffer:
        if dev.type == DeviceType.CUDA:
            rbuf = torch.cuda.caching_allocator_alloc(length)
        else:
            rbuf = cpu_malloc(length)
        return gds_device_buffer(rbuf, length, dev.type == DeviceType.CUDA)

    def free_tensor_memory(self, gbuf: gds_device_buffer, dev: Device):
        if dev.type == DeviceType.CUDA:
            torch.cuda.caching_allocator_delete(gbuf.get_base_address())
        else:
            cpu_free(gbuf.get_base_address())

    def get_empty_tensor(
        self, shape: List[int], dtype: DType, device: Device
    ) -> TorchTensor:
        dst = torch.empty(
            size=shape, dtype=dtype_convert[dtype], device=device.as_str()
        )
        return TorchTensor(device, dtype, dst)

    def concat_tensors(self, tensors: List[TorchTensor], dim: int) -> TorchTensor:
        ts = [tensor.real_tensor for tensor in tensors]
        return TorchTensor(tensors[0].device, tensors[0].dtype, torch.cat(ts, dim=dim))

    def get_dtype_size(self, dtype: DType) -> int:
        return dtype_convert[dtype].itemsize

    def from_dlpack(self, dl_tensor: Any, device: Device, dtype: DType) -> TorchTensor:
        t = torch.from_dlpack(dl_tensor)
        return TorchTensor(device, dtype, t)

    def copy_tensor(self, dst: TorchTensor, src: TorchTensor):
        dst.real_tensor.copy_(src.real_tensor)

    def get_cuda_ver(self) -> str:
        if torch.cuda.is_available():
            return str(torch.version.cuda)
        return "0.0"

    def get_device_ptr_align(self) -> int:
        CUDA_PTR_ALIGN: int = 16
        return CUDA_PTR_ALIGN

    def as_workaround_dtype(self, dtype: DType) -> DType:
        if dtype in need_workaround_dtypes:
            return need_workaround_dtypes[dtype]
        return dtype

    def get_process_group(self, pg: Optional[Any]) -> TorchProcessGroup:
        if pg is not None:
            if isinstance(pg, SingleGroup):
                pg = None
            elif not isinstance(pg, dist.ProcessGroup):
                raise Exception(
                    "pg must be an instance of torch.disributed.ProcessGroup"
                )
        return TorchProcessGroup(pg)

    # for testing
    def is_equal(self, wrapped: TorchTensor, real: Any) -> bool:
        if isinstance(real, torch.Tensor):
            return bool(torch.all(wrapped.real_tensor.eq(real)))
        raise Exception("real is not torch.Tensor")

    def randn(self, s: tuple, device: Device, dtype: DType) -> TorchTensor:
        return TorchTensor(
            device,
            dtype,
            torch.randn(s, device=device.as_str(), dtype=dtype_convert[dtype]),
        )

    def support_fp8(self) -> bool:
        return DType.F8_E5M2 in dtype_convert
