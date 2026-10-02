# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class DeviceType(Enum):
    CPU = "cpu"
    CUDA = "cuda"
    GPU = "gpu"


@dataclass(frozen=True)
class Device:
    type: DeviceType = DeviceType.CPU
    index: Optional[int] = None

    def as_str(self) -> str:
        if self.index is None:
            return self.type.value
        return f"{self.type.value}:{self.index}"

    @classmethod
    def from_str(cls, s: str) -> "Device":
        if ":" in s:
            type_str, index_str = s.split(":", 1)
            try:
                dev_type = DeviceType(type_str.lower())
            except ValueError:
                raise ValueError(f"Unknown device type: {type_str}")
            try:
                index = int(index_str)
            except ValueError:
                raise ValueError(f"Invalid index: {index_str}")
            return cls(type=dev_type, index=index)
        else:
            try:
                dev_type = DeviceType(s.lower())
            except ValueError:
                raise ValueError(f"Unknown device type: {s}")
            return cls(type=dev_type, index=None)


class DType(Enum):
    BOOL = "BOOL"
    I8 = "I8"
    I16 = "I16"
    I32 = "I32"
    I64 = "I64"
    U8 = "U8"
    U16 = "U16"
    U32 = "U32"
    U64 = "U64"
    F16 = "F16"
    F32 = "F32"
    F64 = "F64"
    BF16 = "BF16"
    F8_E5M2 = "F8_E5M2"
    F8_E4M3 = "F8_E4M3"
    F8_E8M0 = "F8_E8M0"
    F4 = "F4"
    AUTO = "AUTO"


# Bytes per logical safetensors element; F4 packs two elements into one byte.
DTYPE_SIZES = {
    DType.BOOL: 1,
    DType.U8: 1,
    DType.I8: 1,
    DType.F8_E5M2: 1,
    DType.F8_E4M3: 1,
    DType.F8_E8M0: 1,
    DType.I16: 2,
    DType.U16: 2,
    DType.F16: 2,
    DType.BF16: 2,
    DType.I32: 4,
    DType.U32: 4,
    DType.F32: 4,
    DType.I64: 8,
    DType.U64: 8,
    DType.F64: 8,
    DType.F4: 0.5,
}
