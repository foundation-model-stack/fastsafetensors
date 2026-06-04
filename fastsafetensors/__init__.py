# SPDX-License-Identifier: Apache-2.0

from importlib.metadata import version

__version__ = version(__name__)

from .auto_loader import AutoLoader
from .common import (
    SafeTensorsMetadata,
    SingleGroup,
    TensorFrame,
    get_device_numa_node,
)
from .config import LoaderConfig, load_config
from .ep_slice import (
    expert_parallel_filter,
    expert_parallel_filter_from_env,
    owned_expert_range,
)
from .file_buffer import FilesBufferOnDevice
from .loader import SafeTensorsFileLoader, fastsafe_open
from .parallel_loader import ParallelLoader
