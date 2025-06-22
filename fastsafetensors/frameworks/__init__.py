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
    def randn(self, s: tuple, device: Device, dtype: DType) -> T:
        pass

    @abstractmethod
    def support_fp8(self) -> bool:
        pass


def get_framework_op(name: str) -> FrameworkOpBase:
    if name == "pt" or name == "pytorch" or name == "torch":
        from ._torch import TorchOp

        return TorchOp()
    elif name == "paddle":
        from ._paddle import PaddleOp

        return PaddleOp()
    else:
        raise Exception(f"Unknown framework name: {name}")
