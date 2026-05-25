# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from .. import cpp as fstcpp
from ..frameworks import TensorBase
from ..st_types import DType


class CopierInterface(ABC):
    def set_byte_ranges(self, byte_ranges: Optional[List[Tuple[int, int]]]) -> None:
        """Restrict reads to these ``[start, end)`` absolute file-offset runs.

        The default implementation ignores the runs and reads the whole file, so
        the byte-range filter is a correct no-op on copiers that don't implement
        partial reads. Range-capable copiers (``nogds``, ``unified``) override
        this to read only the given runs, leaving the rest of the device buffer
        uninitialized (so skipped tensors must not be requested). Build runs with
        ``SafeTensorsMetadata.select_byte_ranges``; ``None`` means full read.
        """
        return

    @abstractmethod
    def submit_io(
        self, use_buf_register: bool, max_copy_block_size: int
    ) -> fstcpp.gds_device_buffer:
        pass

    @abstractmethod
    def wait_io(
        self,
        gbuf: fstcpp.gds_device_buffer,
        dtype: DType = DType.AUTO,
        noalign: bool = False,
    ) -> Dict[str, TensorBase]:
        pass


class DummyDeviceBuffer(fstcpp.gds_device_buffer):
    def __init__(self):
        super().__init__(0, 0, False)
