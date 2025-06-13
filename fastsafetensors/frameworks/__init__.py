# Copyright 2025 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, List, Optional, TypeVar

from ..cpp import gds_device_buffer
from ..st_types import Device, DType


@dataclass
class TensorBase:
    device: Device
    dtype: DType

    @abstractmethod
    def get_raw(self) -> Any:
        pass

    @abstractmethod
    def contiguous(self) -> "TensorBase":
        pass

    @abstractmethod
    def to(
        self,
        device: Optional[Device] = None,
        dtype: DType = DType.AUTO,
    ) -> "TensorBase":
        pass

    @abstractmethod
    def clone(self) -> "TensorBase":
        pass

    @abstractmethod
    def detach(self) -> "TensorBase":
        pass

    @abstractmethod
    def view(self, dtype: DType) -> "TensorBase":
        pass

    @abstractmethod
    def __getitem__(self, _val) -> "TensorBase":
        pass


T = TypeVar("T", bound=TensorBase)


class ProcessGroupBase(ABC, Generic[T]):
    @abstractmethod
    def is_single(self) -> bool:
        pass

    @abstractmethod
    def size(self) -> int:
        pass

    @abstractmethod
    def rank(self) -> int:
        pass

    @abstractmethod
    def broadcast(self, dst: T, rank: int) -> None:
        pass

    @abstractmethod
    def scatter(
        self,
        dst: T,
        scatter_list: List[T],
        src: int,
    ) -> None:
        pass

    @abstractmethod
    def send(
        self,
        t: T,
        dst_rank: int,
        tag: int,
    ) -> None:
        pass

    @abstractmethod
    def recv(
        self,
        t: T,
        src_rank: int,
        tag: int,
    ) -> None:
        pass


K = TypeVar("K", bound=ProcessGroupBase)


class FrameworkOpBase(ABC, Generic[T, K]):
    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_device(self, device: str, pg: K) -> Device:
        pass

    @abstractmethod
    def set_device(self, device: Device) -> None:
        pass

    @abstractmethod
    def alloc_tensor_memory(self, length: int, dev: Device) -> gds_device_buffer:
        pass

    @abstractmethod
    def free_tensor_memory(self, gbuf: gds_device_buffer, dev: Device) -> None:
        pass

    @abstractmethod
    def get_empty_tensor(self, shape: List[int], dtype: DType, device: Device) -> T:
        pass

    @abstractmethod
    def concat_tensors(self, tensors: List[T], dim: int) -> T:
        pass

    @abstractmethod
    def copy_tensor(self, dst: T, src: T) -> None:
        pass

    @abstractmethod
    def get_dtype_size(self, dtype: DType) -> int:
        pass

    @abstractmethod
    def from_dlpack(self, dl_tensor: Any, device: Device, dtype: DType) -> T:
        pass

    @abstractmethod
    def get_cuda_ver(self) -> str:
        pass

    @abstractmethod
    def get_device_ptr_align(self) -> int:
        pass

    @abstractmethod
    def as_workaround_dtype(self, dtype: DType) -> DType:
        pass

    @abstractmethod
    def get_process_group(self, pg: Optional[Any]) -> ProcessGroupBase:
        pass

    @abstractmethod
    def is_equal(self, wrapped: T, real: Any) -> bool:
        pass

    @abstractmethod
    def randn(self, s: tuple, dtype: DType) -> T:
        pass

@dataclass
class NoTensor(TensorBase):
    def get_raw(self) -> Any:
        raise NotImplementedError("call init_framework_op()")

    def contiguous(self) -> "NoTensor":
        raise NotImplementedError("call init_framework_op()")

    def to(
        self,
        device: Optional[Device] = None,
        dtype: DType = DType.AUTO,
    ) -> "NoTensor":
        raise NotImplementedError("call init_framework_op()")

    def clone(self) -> "NoTensor":
        raise NotImplementedError("call init_framework_op()")

    def detach(self) -> "NoTensor":
        raise NotImplementedError("call init_framework_op()")

    def view(self, dtype: DType) -> "NoTensor":
        raise NotImplementedError("call init_framework_op()")

    def __getitem__(self, _val) -> "NoTensor":
        raise NotImplementedError("call init_framework_op()")


class NoProcessGroup(ProcessGroupBase[NoTensor]):
    def is_single(self) -> bool:
        raise NotImplementedError("call init_framework_op()")

    def size(self) -> int:
        raise NotImplementedError("call init_framework_op()")

    def rank(self) -> int:
        raise NotImplementedError("call init_framework_op()")

    def broadcast(self, dst: T, rank: int) -> None:
        raise NotImplementedError("call init_framework_op()")

    def scatter(
        self,
        dst: T,
        scatter_list: List[T],
        src: int,
    ) -> None:
        raise NotImplementedError("call init_framework_op()")

    def send(
        self,
        t: T,
        dst_rank: int,
        tag: int,
    ) -> None:
        raise NotImplementedError("call init_framework_op()")

    def recv(
        self,
        t: T,
        src_rank: int,
        tag: int,
    ) -> None:
        raise NotImplementedError("call init_framework_op()")


class NoOp(FrameworkOpBase[NoTensor, NoProcessGroup]):
    def get_name(self) -> str:
        raise NotImplementedError("call init_framework_op()")
        
    def get_device(self, device: str, pg: NoProcessGroup) -> Device:
        raise NotImplementedError("call init_framework_op()")

    def set_device(self, device: Device) -> None:
        raise NotImplementedError("call init_framework_op()")

    def alloc_tensor_memory(self, length: int, dev: Device) -> gds_device_buffer:
        raise NotImplementedError("call init_framework_op()")

    def free_tensor_memory(self, gbuf: gds_device_buffer, dev: Device) -> None:
        raise NotImplementedError("call init_framework_op()")

    def get_empty_tensor(
        self, shape: List[int], dtype: DType, device: Device
    ) -> NoTensor:
        raise NotImplementedError("call init_framework_op()")

    def concat_tensors(self, tensors: List[NoTensor], dim: int) -> NoTensor:
        raise NotImplementedError("call init_framework_op()")

    def copy_tensor(self, dst: NoTensor, src: NoTensor) -> None:
        raise NotImplementedError("call init_framework_op()")

    def get_dtype_size(self, dtype: DType) -> int:
        raise NotImplementedError("call init_framework_op()")

    def from_dlpack(self, dl_tensor: Any, device: Device, dtype: DType) -> NoTensor:
        raise NotImplementedError("call init_framework_op()")

    def get_cuda_ver(self) -> str:
        raise NotImplementedError("call init_framework_op()")

    def get_device_ptr_align(self) -> int:
        raise NotImplementedError("call init_framework_op()")

    def as_workaround_dtype(self, dtype: DType) -> DType:
        raise NotImplementedError("call init_framework_op()")

    def broadcast(self, dst: NoTensor, rank: int, group: NoProcessGroup):
        raise NotImplementedError("call init_framework_op()")

    def scatter(
        self,
        dst: NoTensor,
        scatter_list: List[NoTensor],
        src: int,
        group: NoProcessGroup,
    ):
        raise NotImplementedError("call init_framework_op()")

    def send(
        self,
        t: NoTensor,
        dst_rank: int,
        group: NoProcessGroup,
        tag: int,
    ):
        raise NotImplementedError("call init_framework_op()")

    def recv(
        self,
        t: NoTensor,
        src_rank: int,
        group: NoProcessGroup,
        tag: int,
    ):
        raise NotImplementedError("call init_framework_op()")

    def get_process_group(self, pg: Optional[Any]) -> NoProcessGroup:
        raise NotImplementedError("call init_framework_op()")

    def is_equal(self, one: NoTensor, two: NoTensor) -> bool:
        raise NotImplementedError("call init_framework_op()")

    def randn(self, s: tuple, dtype: DType) -> NoTensor:
        raise NotImplementedError("call init_framework_op()")

nop: FrameworkOpBase = NoOp()
FRAMEWORK: FrameworkOpBase = nop


def init_framework_op(name: str):
    global nop
    global FRAMEWORK
    if FrameworkOpBase is not nop:
        return
    if name == "pt" or name == "pytorch":
        from .torch import TorchOp

        FRAMEWORK = TorchOp()
    elif name == "paddle":
        from ._paddle import PaddleOp

        FRAMEWORK = PaddleOp()
    raise Exception(f"Unknown framework name: {name}")
