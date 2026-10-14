# SPDX-License-Identifier: Apache-2.0

"""Chunk boundaries, empty selections, and cleanup before I/O submission."""

import json
import os
import struct
from contextlib import closing

import pytest

from fastsafetensors import ParallelLoader, ParallelSaver, SafeTensorsFileLoader, cpp
from fastsafetensors.copier.gds import GdsFileCopier
from fastsafetensors.copier.nogds import NoGdsFileCopier
from fastsafetensors.copier.unified import UnifiedMemCopier
from fastsafetensors.st_types import Device, DType


# Sample the IO/allocation paths without multiplying every independent option.
@pytest.mark.parametrize("target", ["cpu", "cuda:0"])
@pytest.mark.parametrize("selection", ["empty_shard", "only_empty"])
@pytest.mark.parametrize(
    "copier_type,mode,fixed,overlap",
    [
        ("nogds", "max", False, False),
        ("nogds", "budget", True, True),
        ("gds", "max", True, True),
        ("gds", "budget", False, False),
        ("unified", "max", False, True),
        ("unified", "both", True, False),
    ],
)
def test_empty_chunk_preserves_tensors_without_reads(
    tmp_path,
    framework,
    monkeypatch,
    copier_type,
    selection,
    mode,
    fixed,
    overlap,
    target,
):
    if target != "cpu":
        if framework.get_name() != "pytorch":
            pytest.skip("CUDA fixtures use PyTorch")
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        if copier_type == "gds" and not os.path.exists("/dev/nvidia-fs0"):
            pytest.skip("native GDS unavailable")
    payload_size = 8192 if selection != "empty_shard" else 0
    tensors = {"a": {"dtype": "F32", "shape": [0, 3], "data_offsets": [0, 0]}}
    if payload_size:
        tensors["middle"] = {
            "dtype": "F32",
            "shape": [2048],
            "data_offsets": [0, payload_size],
        }
    tensors["z"] = {
        "dtype": "F32",
        "shape": [0, 3],
        "data_offsets": [payload_size, payload_size],
    }
    encoded = json.dumps(tensors).encode()
    encoded += b" " * (-len(encoded) % 8)
    path = tmp_path / "model.safetensors"
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(payload_size))
    paths = [str(path)]
    submitted = []

    class NoReads:
        def submit_read(self, *args):
            submitted.append(args)
            raise AssertionError("empty selection must not submit reads")

    copier_class = {
        "nogds": NoGdsFileCopier,
        "gds": GdsFileCopier,
        "unified": UnifiedMemCopier,
    }[copier_type]

    def construct(meta, device, fw):
        if copier_type == "unified":
            return UnifiedMemCopier(meta, device, fw)
        return copier_class(meta, device, NoReads(), fw)

    construct.copier_class = copier_class
    monkeypatch.setattr(
        "fastsafetensors.loader.create_copier_constructor", lambda **kw: construct
    )
    monkeypatch.setattr(
        cpp, "dma_load_runs", lambda *args: pytest.fail("empty DMA submission")
    )
    monkeypatch.setattr(
        cpp, "dma_load_runs_progress", lambda *args: pytest.fail("empty DMA submission")
    )
    before = framework.get_mem_used()
    options = {
        "max_batch_bytes": 256 if mode in ("max", "both") else None,
        "device_memory_budget": 1 << 30 if mode in ("budget", "both") else None,
    }
    with closing(
        ParallelLoader(
            None,
            paths,
            device=target,
            framework=framework.get_name(),
            nogds=True,
            tensor_filter=lambda name: name != "middle",
            use_chunk_budget_as_allocation_size=fixed,
            overlap_io=overlap,
            use_tqdm_on_load=False,
            **options,
        )
    ) as loader:
        with closing(loader.iterate_weights()) as weights:
            actual = dict(weights)
    assert set(actual) == {"a", "z"}
    for tensor in actual.values():
        wrapped = framework.wrap_tensor(tensor)
        assert wrapped.get_shape() == [0, 3]
        assert wrapped.dtype == DType.F32
    assert submitted == []
    assert framework.get_mem_used() == before


@pytest.mark.parametrize("dtype", [DType.F16, DType.F64])
def test_empty_tensor_conversion_uses_the_same_size_rules(tmp_path, framework, dtype):
    empty = framework.get_empty_tensor([0, 3], DType.F32, Device())
    paths = ParallelSaver(num_shards=1, framework=framework.get_name()).save(
        {"empty": empty}, str(tmp_path / "model")
    )
    before = framework.get_mem_used()
    with closing(
        SafeTensorsFileLoader(None, "cpu", nogds=True, framework=framework.get_name())
    ) as loader:
        loader.add_filenames({0: paths})
        if dtype == DType.F64:
            with pytest.raises(Exception, match="conversion to larger sizes"):
                loader.copy_files_to_device(dtype=dtype)
        else:
            with closing(loader.copy_files_to_device(dtype=dtype)) as buffer:
                tensor = buffer.get_tensor_wrapped("empty")
                assert tensor.get_shape() == [0, 3]
                assert tensor.dtype == dtype
                assert (
                    next(iter(loader.meta.values()))[0].tensors["empty"].dtype == dtype
                )
    assert framework.get_mem_used() == before


def test_filter_selecting_nothing_yields_no_tensors(input_files, framework):
    with closing(
        ParallelLoader(
            None,
            input_files,
            device="cpu",
            framework=framework.get_name(),
            nogds=True,
            tensor_filter=lambda name: False,
            max_batch_bytes=256,
            use_tqdm_on_load=False,
        )
    ) as loader:
        with closing(loader.iterate_weights()) as weights:
            assert list(weights) == []


def test_chunk_configuration_failure_closes_open_file(
    input_files, framework, monkeypatch
):
    loader = SafeTensorsFileLoader(
        None, "cpu", nogds=True, framework=framework.get_name()
    )
    loader.add_filenames({0: [input_files[0]]})
    loader._set_chunk_plan({input_files[0]: ({"missing"}, [(0, 1)], None)})
    original = loader.copier_constructor
    copiers = []

    def construct(*args):
        copier = original(*args)
        copiers.append(copier)
        return copier

    monkeypatch.setattr(loader, "copier_constructor", construct)
    with closing(loader):
        with pytest.raises(ValueError, match="data section"):
            loader.copy_files_to_device()
    assert len(copiers) == 1
    assert copiers[0].fd == -1
