# Copyright 2024 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0

try:
    from importlib.metadata import version
except ImportError:  # Python <3.8
    from importlib_metadata import version

__version__ = version(__name__)

from .common import SafeTensorsMetadata, SingleGroup, TensorFrame, get_device_numa_node
from .file_buffer import FilesBufferOnDevice
from .loader import SafeTensorsFileLoader, fastsafe_open
