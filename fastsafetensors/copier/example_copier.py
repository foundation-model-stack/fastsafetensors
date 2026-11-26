# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any, Dict, Optional

from .. import cpp as fstcpp
from ..common import SafeTensorsMetadata
from ..frameworks import FrameworkOpBase, TensorBase
from ..st_types import Device, DType
from .base import CopierInterface, DummyDeviceBuffer
from .registry import CopierConstructFunc, register_copier_constructor

if TYPE_CHECKING:
    from ..allocation import SharedDeviceAllocation


class ExampleCopier(CopierInterface):
    def __init__(
        self,
        metadata: SafeTensorsMetadata,
        device: Device,
        reader,
        framework: FrameworkOpBase,
    ):
        self.metadata = metadata

    def submit_io(
        self, use_buf_register: bool, max_copy_block_size: int
    ) -> fstcpp.gds_device_buffer:
        return DummyDeviceBuffer()

    def wait_io(
        self,
        gbuf: fstcpp.gds_device_buffer,
        dtype: DType = DType.AUTO,
        noalign: bool = False,
        owner: Optional["SharedDeviceAllocation"] = None,
    ) -> Dict[str, TensorBase]:
        # get tensor
        res: Dict[str, TensorBase] = {}
        return res


def load_library_example(device: Device, **kwargs):
    pass


def init_and_check_example(device: Device, **kwargs) -> bool:
    return True


@register_copier_constructor(
    "example",
    ExampleCopier,
    load_library=load_library_example,
    init_and_check=init_and_check_example,
)
def new_example_file_copier(
    device: Device,
    bbuf_size_kb: Optional[int] = None,
    max_threads: int = 16,
    **kwargs,
) -> CopierConstructFunc:
    # reader = example_reader()
    reader: Any = {}

    def construct_copier(
        metadata: SafeTensorsMetadata,
        device: Device,
        framework: FrameworkOpBase,
    ) -> CopierInterface:
        return ExampleCopier(metadata, device, reader, framework)

    return construct_copier
