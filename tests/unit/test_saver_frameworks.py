# SPDX-License-Identifier: Apache-2.0
"""Exercise the shared saver contract with the selected tensor framework."""

import pytest

from fastsafetensors import ParallelSaver, SafeTensorsMetadata, fastsafe_open
from fastsafetensors.st_types import Device, DType


@pytest.mark.parametrize(
    "dtype", [DType.F16, DType.F32, DType.BF16, DType.I32, DType.I64, DType.BOOL]
)
@pytest.mark.parametrize("wrapped_input", [False, True])
def test_save_framework_tensors(tmp_path, framework, dtype, wrapped_input):
    sources = {
        "matrix": framework.randn((4, 6), Device(), DType.F32)
        .to(dtype=dtype)
        .get_raw(),
        "scalar": framework.randn((), Device(), DType.F32).to(dtype=dtype).get_raw(),
        "empty": framework.get_empty_tensor([0, 3], dtype, Device()).get_raw(),
    }
    tensors = {name: framework.wrap_tensor(source) for name, source in sources.items()}
    for name, tensor in tensors.items():
        assert tensor.get_raw() is sources[name]
        assert tensor.dtype == dtype
    paths = ParallelSaver(
        num_shards=2,
        num_threads=2,
        framework="unused" if wrapped_input else framework.get_name(),
    ).save(tensors if wrapped_input else sources, str(tmp_path / "model"))
    frames = {
        name: frame
        for path in paths
        for name, frame in SafeTensorsMetadata.from_file(
            path, framework
        ).tensors.items()
    }
    assert set(frames) == set(tensors)
    for name, tensor in tensors.items():
        assert frames[name].dtype == dtype
        assert frames[name].shape == tensor.get_shape()
        assert (
            frames[name].data_offsets[1] - frames[name].data_offsets[0]
            == tensor.get_nbytes()
        )
    with fastsafe_open(
        paths, framework=framework.get_name(), device="cpu", nogds=True
    ) as checkpoint:
        for name, tensor in tensors.items():
            assert framework.is_equal(
                checkpoint.get_tensor_wrapped(name), tensor.get_raw()
            )


def test_save_strided_framework_tensor(tmp_path, framework):
    source = framework.randn((4, 6), Device(), DType.F32).get_raw()
    if framework.get_name() == "pytorch":
        source = source.t()
    else:
        source = source.transpose([1, 0])
    assert not source.is_contiguous()
    paths = ParallelSaver(num_shards=1, framework=framework.get_name()).save(
        {"w": source}, str(tmp_path / "model")
    )
    with fastsafe_open(
        paths, framework=framework.get_name(), device="cpu", nogds=True
    ) as checkpoint:
        assert framework.is_equal(checkpoint.get_tensor_wrapped("w"), source)
