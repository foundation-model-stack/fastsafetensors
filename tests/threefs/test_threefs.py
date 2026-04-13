# SPDX-License-Identifier: Apache-2.0

import os

import pytest
from threefs.conftest import get_device, load_safetensors_file

from fastsafetensors import SafeTensorsMetadata, SingleGroup
from fastsafetensors.copier import create_copier_constructor


def test_threefs_copier_registered(fstcpp_log, framework):
    device, _ = get_device(framework)
    copier_fn = create_copier_constructor(
        copier_type="3fs",
        device=device,
        mount_point="/tmp",
    )
    assert copier_fn is not None


def test_ThreeFSFileCopier(fstcpp_log, input_files, framework):
    from fastsafetensors.copier.threefs import (
        ThreeFSFileCopier,
        new_threefs_file_copier,
    )

    device, _ = get_device(framework)
    input_file = input_files[0]
    mount_point = os.path.dirname(input_file)

    copier_fn = new_threefs_file_copier(device=device, mount_point=mount_point)
    meta = SafeTensorsMetadata.from_file(input_file, framework)
    copier = copier_fn(meta, device, framework)
    assert isinstance(copier, ThreeFSFileCopier)

    gbuf = copier.submit_io(False, 16 * 1024 * 1024 * 1024)
    tensors = copier.wait_io(gbuf)

    expected = load_safetensors_file(input_file, device, framework)
    for key, exp in expected.items():
        assert key in tensors, f"Missing tensor key: {key}"
        assert framework.is_equal(tensors[key], exp), f"Tensor mismatch for key: {key}"

    framework.free_tensor_memory(gbuf, device)
    assert framework.get_mem_used() == 0


def test_ThreeFSLoader(fstcpp_log, input_files, framework):
    from fastsafetensors.threefs_loader import ThreeFSLoader

    device, _ = get_device(framework)
    input_file = input_files[0]
    mount_point = os.path.dirname(input_file)

    loader = ThreeFSLoader(
        pg=SingleGroup(),
        device=device.as_str(),
        mount_point=mount_point,
        debug_log=True,
        framework=framework.get_name(),
    )
    loader.add_filenames({0: [input_file]})
    bufs = loader.copy_files_to_device()

    expected = load_safetensors_file(input_file, device, framework)
    for key, exp in expected.items():
        actual = bufs.get_tensor_wrapped(key)
        assert framework.is_equal(actual, exp), f"Tensor mismatch for key: {key}"

    bufs.close()
    loader.close()
    assert framework.get_mem_used() == 0


def test_ThreeFSLoader_multiple_files(fstcpp_log, input_files, framework):
    from fastsafetensors.threefs_loader import ThreeFSLoader

    device, _ = get_device(framework)
    mount_point = os.path.dirname(os.path.commonpath(input_files))

    loader = ThreeFSLoader(
        pg=SingleGroup(),
        device=device.as_str(),
        mount_point=mount_point,
        debug_log=True,
        framework=framework.get_name(),
    )
    loader.add_filenames({0: input_files})
    bufs = loader.copy_files_to_device()

    keys = loader.get_keys()
    assert len(keys) > 0

    for test_file in input_files:
        expected = load_safetensors_file(test_file, device, framework)
        for key, exp in expected.items():
            actual = bufs.get_tensor_wrapped(key)
            assert framework.is_equal(actual, exp), f"Tensor mismatch for key: {key}"

    bufs.close()
    loader.close()
    assert framework.get_mem_used() == 0
