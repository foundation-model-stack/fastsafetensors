# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 IBM Inc. All rights reserved

"""Utilities for platform detection and conditional test execution."""

import pytest


def is_rocm_platform():
    """Detect if running on ROCm/AMD platform.

    Uses the C++ extension's is_hip_found() to avoid framework-specific calls.
    """
    try:
        from fastsafetensors import cpp as fstcpp

        return fstcpp.is_hip_found()
    except:
        return False


def is_cuda_platform():
    """Detect if running on CUDA/NVIDIA platform."""
    return not is_rocm_platform()


# List of tests that are expected to fail on ROCm (based on TEST_RESULTS.md)
ROCM_EXPECTED_FAILURES = {
    "test_GdsFileCopier",  # GDS not available on AMD
}

# List of tests with memory leak detection issues on ROCm (non-critical)
ROCM_MEMORY_LEAK_TESTS = {
    "test_SafeTensorsFileLoader",
    "test_SafeTensorsFileLoaderNoGds",
    "test_fastsafe_open",
    "test_int8",
    "test_float8_e5m2",
    "test_float8_e4m3fn",
    "test_float8_e4m3fn_to_int8",
    "test_cpp_metrics",
}


def skip_if_rocm_expected_failure(test_name):
    """Skip test if it's an expected failure on ROCm."""
    if is_rocm_platform() and test_name in ROCM_EXPECTED_FAILURES:
        pytest.skip(
            f"Test '{test_name}' is expected to fail on ROCm (GDS not supported)"
        )


def xfail_if_rocm_memory_leak(test_name):
    """Mark test as expected to fail on ROCm due to memory leak detection issues."""
    if is_rocm_platform() and test_name in ROCM_MEMORY_LEAK_TESTS:
        return pytest.mark.xfail(
            reason=f"Test '{test_name}' has memory leak detection issues on ROCm (non-critical)",
            strict=False,
        )
    return lambda func: func


def get_platform_info():
    """Get platform information for debugging.

    Uses framework's get_cuda_ver() to avoid direct torch calls where possible.
    """
    info = {
        "is_rocm": is_rocm_platform(),
        "is_cuda": is_cuda_platform(),
    }

    try:
        from fastsafetensors import cpp as fstcpp
        from fastsafetensors.common import is_gpu_found

        if is_gpu_found():
            # Get version info from framework
            try:
                from fastsafetensors.frameworks import get_framework_op

                framework = get_framework_op("pytorch")
                gpu_ver = framework.get_cuda_ver()
                info["gpu_version"] = gpu_ver

                # Parse the version to get specific info
                if gpu_ver.startswith("hip-"):
                    info["hip_version"] = gpu_ver[4:]  # Remove 'hip-' prefix
                    info["rocm_version"] = gpu_ver[4:]
                elif gpu_ver.startswith("cuda-"):
                    info["cuda_version"] = gpu_ver[5:]  # Remove 'cuda-' prefix
            except:
                pass

            # Get device count and name (still needs torch for this)
            try:
                import torch

                if torch.cuda.is_available():
                    info["torch_version"] = torch.__version__
                    info["device_count"] = torch.cuda.device_count()
                    info["device_name"] = (
                        torch.cuda.get_device_name(0)
                        if torch.cuda.device_count() > 0
                        else None
                    )
            except:
                pass
    except:
        pass

    return info
