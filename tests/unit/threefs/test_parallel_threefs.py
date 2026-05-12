# SPDX-License-Identifier: Apache-2.0

import os
from typing import List

import pytest
from threefs.conftest import (
    get_device,
    load_safetensors_file,
    tensors_equal,
)

from fastsafetensors import SingleGroup
from fastsafetensors import cpp as fstcpp
from fastsafetensors.frameworks import FrameworkOpBase
from fastsafetensors.st_types import Device

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
EXAMPLES_DIR = os.path.join(REPO_ROOT, "examples")


@pytest.fixture(scope="module")
def example_files(framework) -> List[str]:
    if framework.get_name() == "pytorch":
        files = [
            os.path.join(EXAMPLES_DIR, "a.safetensors"),
            os.path.join(EXAMPLES_DIR, "b.safetensors"),
        ]
    elif framework.get_name() == "paddle":
        files = [
            os.path.join(EXAMPLES_DIR, "a_paddle.safetensors"),
            os.path.join(EXAMPLES_DIR, "b_paddle.safetensors"),
        ]
    else:
        raise Exception(f"unknown framework: {framework.get_name()}")
    for f in files:
        if not os.path.exists(f):
            pytest.skip(f"Example file not found: {f}")
    return files


def test_parallel_single_file(fstcpp_log, example_files, framework):
    from fastsafetensors.threefs_loader import ParallelThreeFSLoader

    device, _ = get_device(framework)

    loader = ParallelThreeFSLoader(
        pg=SingleGroup(),
        hf_weights_files=[example_files[0]],
        device=device.as_str(),
        debug_log=True,
        framework=framework.get_name(),
    )

    expected = load_safetensors_file(example_files[0], device, framework)
    loaded = {}
    for key, tensor in loader.iterate_weights():
        loaded[key] = tensor

    assert set(loaded.keys()) == set(expected.keys())
    for key, exp in expected.items():
        assert tensors_equal(loaded[key], exp, framework), f"Tensor mismatch: {key}"

    loader.close()
    assert framework.get_mem_used() == 0


def test_parallel_multiple_files(fstcpp_log, example_files, framework):
    from fastsafetensors.threefs_loader import ParallelThreeFSLoader

    device, _ = get_device(framework)

    loader = ParallelThreeFSLoader(
        pg=SingleGroup(),
        hf_weights_files=example_files,
        device=device.as_str(),
        debug_log=True,
        framework=framework.get_name(),
    )

    all_expected = {}
    for f in example_files:
        all_expected.update(load_safetensors_file(f, device, framework))

    loaded = {}
    for key, tensor in loader.iterate_weights():
        loaded[key] = tensor

    assert set(loaded.keys()) == set(all_expected.keys())
    for key, exp in all_expected.items():
        actual = loaded[key]
        # Verify shape, dtype, and values
        if framework.get_name() == "pytorch":
            assert list(actual.shape) == list(exp.shape), f"Shape mismatch: {key}"
            assert actual.dtype == exp.dtype, f"Dtype mismatch: {key}"
        elif framework.get_name() == "paddle":
            assert actual.shape == exp.shape, f"Shape mismatch: {key}"
            assert actual.dtype == exp.dtype, f"Dtype mismatch: {key}"
        assert tensors_equal(actual, exp, framework), f"Value mismatch: {key}"

    loader.close()
    assert framework.get_mem_used() == 0


def test_parallel_close_and_memory_release(fstcpp_log, example_files, framework):
    from fastsafetensors.threefs_loader import ParallelThreeFSLoader

    device, _ = get_device(framework)

    loader = ParallelThreeFSLoader(
        pg=SingleGroup(),
        hf_weights_files=example_files,
        device=device.as_str(),
        debug_log=True,
        framework=framework.get_name(),
    )

    count = 0
    for key, tensor in loader.iterate_weights():
        assert tensor is not None
        count += 1
    assert count > 0

    loader.close()
    assert framework.get_mem_used() == 0
    assert fstcpp.get_cpp_metrics().bounce_buffer_bytes == 0


def test_parallel_close_without_iterate(fstcpp_log, example_files, framework):
    from fastsafetensors.threefs_loader import ParallelThreeFSLoader

    device, _ = get_device(framework)

    loader = ParallelThreeFSLoader(
        pg=SingleGroup(),
        hf_weights_files=example_files,
        device=device.as_str(),
        debug_log=True,
        framework=framework.get_name(),
    )

    loader.close()
    assert framework.get_mem_used() == 0
