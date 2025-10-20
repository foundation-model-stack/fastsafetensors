from typing import Dict, Protocol

from .. import cpp as fstcpp
from ..frameworks import TensorBase
from ..st_types import DType


class CopierInterface(Protocol):
    def submit_io(
        self, use_buf_register: bool, max_copy_block_size: int
    ) -> fstcpp.gds_device_buffer:
        pass

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
