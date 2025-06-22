# Copyright 2024 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0

from .common import SafeTensorsMetadata, SingleGroup, TensorFrame, get_device_numa_node
from .file_buffer import FilesBufferOnDevice
from .loader import SafeTensorsFileLoader, fastsafe_open

loaded_nvidia: bool = False
if not loaded_nvidia:
    from . import cpp

    cpp.load_nvidia_functions()
    if cpp.init_gds() != 0:
        raise Exception(f"[FAIL] init_gds()")
    loaded_nvidia = True
