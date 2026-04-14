# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any, Dict

import pytest

# Import fixtures from parent conftest so they are available in this directory
from conftest import dev_init, input_files  # noqa: F401

from fastsafetensors import SingleGroup
from fastsafetensors import cpp as fstcpp
from fastsafetensors.common import is_gpu_found
from fastsafetensors.cpp import load_library_functions
from fastsafetensors.frameworks import FrameworkOpBase, get_framework_op
from fastsafetensors.st_types import Device

try:
    import fastsafetensor_3fs_reader  # noqa: F401

    _HAS_3FS = True
except ImportError:
    _HAS_3FS = False

requires_3fs = pytest.mark.skipif(
    not _HAS_3FS,
    reason="fastsafetensor_3fs_reader not available (set FASTSAFETENSORS_BACKEND=mock or install 3FS)",
)

load_library_functions()
FRAMEWORK = get_framework_op(os.getenv("TEST_FASTSAFETENSORS_FRAMEWORK", "please set"))


def get_device(framework: FrameworkOpBase):
    dev_is_gpu = is_gpu_found()
    device = "cpu"
    if dev_is_gpu:
        if framework.get_name() == "pytorch":
            device = "cuda:0"
        elif framework.get_name() == "paddle":
            device = "gpu:0"
    return Device.from_str(device), dev_is_gpu


def load_safetensors_file(
    filename: str,
    device: Device,
    framework: FrameworkOpBase,
) -> Dict[str, Any]:
    if framework.get_name() == "pytorch":
        from safetensors.torch import load_file
    elif framework.get_name() == "paddle":
        from safetensors.paddle import load_file
    else:
        raise Exception(f"unknown framework: {framework.get_name()}")
    return load_file(filename, device.as_str())


def tensors_equal(actual: Any, expected: Any, framework: FrameworkOpBase) -> bool:
    """Compare raw tensors (torch.Tensor / paddle.Tensor) for equality."""
    if framework.get_name() == "pytorch":
        import torch

        return bool(torch.all(actual.eq(expected)))
    elif framework.get_name() == "paddle":
        import paddle

        return bool(paddle.all(actual == expected))
    else:
        raise Exception(f"unknown framework: {framework.get_name()}")


@pytest.fixture(scope="session")
def framework() -> FrameworkOpBase:
    return FRAMEWORK


@pytest.fixture(scope="function")
def fstcpp_log() -> None:
    fstcpp.set_debug_log(True)
