# Copyright 2025 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0

try:
    import paddle
    import paddle.distributed as pdist
    from paddle.distributed.communication.group import Group
    from paddle.framework import core as paddle_core
except ImportError as e:
    raise ImportError(
        "could not import paddle, paddle_core, or numpy. Please install them."
    ) from e

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..common import SingleGroup
from ..cpp import cpu_free, cpu_malloc, gds_device_buffer, gpu_free, gpu_malloc
from ..st_types import Device, DeviceType, DType
from . import FrameworkOpBase, ProcessGroupBase, TensorBase

dtype_convert: Dict[DType, Any] = {
    DType.BOOL: paddle.bool,
    DType.I8: paddle.uint8,
    DType.I8: paddle.int8,
    DType.I16: paddle.int16,
    DType.U16: paddle.bfloat16,
    DType.I32: paddle.int32,
    DType.U32: paddle.int32,
    DType.I64: paddle.int64,
    DType.U64: paddle.int64,
    DType.F16: paddle.float16,
    DType.BF16: paddle.bfloat16,
    DType.F32: paddle.float32,
    DType.F64: paddle.float64,
}
need_workaround_dtypes: Dict[DType, DType] = {
    DType.F8_E5M2: DType.I8,
    DType.F8_E4M3: DType.I8,
}

if hasattr(paddle, "float8_e5m2"):
    dtype_convert[DType.F8_E5M2] = paddle.float8_e5m2
if hasattr(paddle, "float8_e4m3fn"):
    dtype_convert[DType.F8_E4M3] = paddle.float8_e4m3fn


@dataclass
class PaddleTensor(TensorBase):
    real_tensor: paddle.Tensor

    def get_raw(self) -> paddle.Tensor:
        return self.real_tensor

    def contiguous(self) -> "PaddleTensor":
        return PaddleTensor(self.device, self.dtype, self.real_tensor.contiguous())

    def to(
        self,
        device: Optional[Device] = None,
        dtype: DType = DType.AUTO,
    ) -> "PaddleTensor":
        to_dev: Optional[str] = None
        if device is not None and self.device != device:
            to_dev = device.as_str()
        else:
            device = self.device
        to_dtype: Optional[paddle.dtype] = None
        if dtype != DType.AUTO and (dtype != self.dtype):
            to_dtype = dtype_convert[dtype]
        else:
            dtype = self.dtype
        if to_dev is not None or to_dtype is not None:
            return PaddleTensor(
                device, dtype, self.real_tensor.to(device=to_dev, dtype=to_dtype)
            )
        return self

    def clone(self) -> "PaddleTensor":
        return PaddleTensor(self.device, self.dtype, self.real_tensor.clone())

    def detach(self) -> "PaddleTensor":
        return PaddleTensor(self.device, self.dtype, self.real_tensor.detach())

    def view(self, dtype: DType) -> "PaddleTensor":
        t2 = self.real_tensor.view(dtype_convert[dtype])
        return PaddleTensor(self.device, dtype, t2)

    def __getitem__(self, _val) -> "PaddleTensor":
        return PaddleTensor(self.device, self.dtype, self.real_tensor[_val])


@dataclass
class PaddleProcessGroup(ProcessGroupBase[PaddleTensor]):
    real_pg: Optional[Group]

    def size(self) -> int:
        return self.real_pg.process_group.size() if self.real_pg else 1

    def rank(self) -> int:
        return self.real_pg.process_group.rank() if self.real_pg else 0

    def broadcast(self, dst: PaddleTensor, rank: int) -> None:
        if self.real_pg:
            pdist.broadcast(dst.real_tensor, rank, group=self.real_pg)

    def scatter(
        self,
        dst: PaddleTensor,
        scatter_list: List[PaddleTensor],
        src: int,
    ) -> None:
        if self.real_pg:
            sl = [t.real_tensor for t in scatter_list]
            pdist.scatter(
                dst.real_tensor,
                tensor_list=sl,
                src=src,
                group=self.real_pg,
            )

    def send(
        self,
        t: PaddleTensor,
        dst_rank: int,
        tag: int,
    ) -> None:
        if self.real_pg:
            pdist.send(t.real_tensor, dst_rank, group=self.real_pg)

    def recv(
        self,
        t: PaddleTensor,
        src_rank: int,
        tag: int,
    ) -> None:
        if self.real_pg:
            pdist.recv(t.real_tensor, src_rank, group=self.real_pg)


class PaddleOp(FrameworkOpBase[PaddleTensor, PaddleProcessGroup]):
    def get_name(self) -> str:
        return "paddle"

    def get_device(self, device: str, pg: PaddleProcessGroup) -> Device:
        dev_index: Optional[int] = None
        try:
            dev_split = device.split(":", 1)
            dev_type = DeviceType(dev_split[0].lower())
            if dev_type != DeviceType.CPU:
                dev_index = 0
                if len(dev_split) > 1:
                    dev_index = int(dev_split[1])
        except ValueError:
            raise ValueError(f"Unknown device: {device}")

        if paddle.device.cuda.device_count() > 0 and pg.real_pg is not None:
            # For single (gpu:x, gpu)
            # gpu:x, like gpu:0, gpu:1, ...
            # For distributed
            # The gpu determines the current rank
            # rank0 use gpu:0, rank1 use gpu:1
            dev_index = pg.rank() % paddle.device.cuda.device_count()
        return Device(dev_type, dev_index)

    def set_device(self, device: Device) -> None:
        if device.type != DeviceType.CPU:
            paddle.set_device(device.as_str())

    def alloc_tensor_memory(self, length: int, dev: Device) -> gds_device_buffer:
        if dev.type == DeviceType.GPU:
            rbuf = gpu_malloc(length)
        else:
            rbuf = cpu_malloc(length)
        return gds_device_buffer(rbuf, length, dev.type == DeviceType.GPU)

    def free_tensor_memory(self, gbuf: gds_device_buffer, dev: Device) -> None:
        if dev.type == DeviceType.GPU:
            gpu_free(gbuf.get_base_address())
        else:
            cpu_free(gbuf.get_base_address())

    def get_empty_tensor(
        self, shape: List[int], dtype: DType, device: Device
    ) -> PaddleTensor:
        dst = paddle.to_tensor(
            paddle.empty(shape=shape, dtype=dtype_convert[dtype]),
            place=device.as_str(),
        )
        return PaddleTensor(device, dtype, dst)

    def concat_tensors(self, tensors: List[PaddleTensor], dim) -> PaddleTensor:
        ts = [tensor.real_tensor for tensor in tensors]
        return PaddleTensor(
            tensors[0].device, tensors[0].dtype, paddle.concat(ts, axis=dim)
        )

    def get_dtype_size(self, dtype: DType) -> int:
        return paddle_core.size_of_dtype(dtype_convert[dtype])

    def from_dlpack(self, dl_tensor: Any, device: Device, dtype: DType) -> PaddleTensor:
        return PaddleTensor(device, dtype, paddle.utils.dlpack.from_dlpack(dl_tensor))

    def copy_tensor(self, dst: PaddleTensor, src: PaddleTensor) -> None:
        paddle.assign(src.real_tensor, output=dst.real_tensor)
        dst.dtype = src.dtype
        dst.device = src.device

    def get_cuda_ver(self) -> str:
        return (
            str(paddle.version.cuda())
            if paddle.device.is_compiled_with_cuda()
            else "0.0"
        )

    def get_device_ptr_align(self) -> int:
        CUDA_PTR_ALIGN: int = 16
        return CUDA_PTR_ALIGN

    def as_workaround_dtype(self, dtype: DType) -> DType:
        if dtype in need_workaround_dtypes:
            return need_workaround_dtypes[dtype]
        return dtype

    def get_process_group(self, pg: Optional[Any]) -> PaddleProcessGroup:
        if pg is not None:
            if isinstance(pg, SingleGroup):
                pg = None
            elif not isinstance(pg, Group):
                raise Exception(
                    "pg must be an instance of paddle.distributed.communication.group.Group"
                )
        return PaddleProcessGroup(pg)

    # for testing
    def is_equal(self, wrapped: PaddleTensor, real: Any) -> bool:
        if isinstance(real, paddle.Tensor):
            return paddle.all(wrapped.real_tensor.equal(real))
        raise Exception("real is not paddle.Tensor")

    def randn(self, s: tuple, device: Device, dtype: DType) -> PaddleTensor:
        return PaddleTensor(
            device,
            dtype,
            paddle.randn(s, dtype=dtype_convert[dtype]).to(device=device.as_str()),
        )

    def support_fp8(self) -> bool:
        return DType.F8_E5M2 in dtype_convert
