# Copyright 2024 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0

from importlib.metadata import version

__version__ = version(__name__)

from .common import (
    SafeTensorsMetadata,
    SingleGroup,
    TensorFrame,
    get_device_numa_node,
)
from .file_buffer import FilesBufferOnDevice
from .loader import BaseSafeTensorsFileLoader, SafeTensorsFileLoader, fastsafe_open
from .parallel_loader import ParallelLoader
