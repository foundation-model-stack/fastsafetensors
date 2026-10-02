# SPDX-License-Identifier: Apache-2.0
"""Backend-independent saver helpers, without a device runtime."""

import os
import subprocess
import sys
import textwrap

import pytest
import torch
from safetensors import safe_open

from fastsafetensors import SafeTensorsMetadata
from fastsafetensors.frameworks import get_framework_op
from fastsafetensors.frameworks._torch import dtype_convert, fst_dtype
from fastsafetensors.saver import (
    ParallelSaver,
    ShardSetError,
    WriteEntry,
    plain_entry,
    save_sharded,
    validate_shard_set,
)
from fastsafetensors.st_types import DType


@pytest.mark.parametrize("use_saver", [False, True])
def test_custom_encoder_with_opaque_sources_and_metadata(tmp_path, use_saver):
    sources = {"a": {"value": 7}, "b": {"value": 13}}
    entries = [
        WriteEntry(name, DType.U8, [5], 5, source) for name, source in sources.items()
    ]

    def fill(work):
        for buffer, entry in work:
            assert entry.source is sources[entry.name]
            buffer[:] = bytes([entry.source["value"]]) * entry.nbytes

    def metadata(shard):
        return {"application.entries": ",".join(e.name for e in shard)}

    if use_saver:
        save = ParallelSaver(
            num_shards=3, fill=fill, shard_metadata=metadata, framework="unused"
        ).save_entries
        options = {}
    else:
        save = save_sharded
        options = {"num_shards": 3, "fill": fill, "shard_metadata": metadata}
    paths = save(
        entries, str(tmp_path / "encoded"), metadata={"format": "opaque/1"}, **options
    )
    seen = set()
    for path in paths:
        with safe_open(path, framework="pt") as checkpoint:
            names = list(checkpoint.keys())
            assert checkpoint.metadata()["format"] == "opaque/1"
            assert checkpoint.metadata()["application.entries"] == ",".join(names)
            for name in names:
                assert (
                    checkpoint.get_tensor(name).tolist() == [sources[name]["value"]] * 5
                )
                seen.add(name)
    assert seen == sources.keys()


@pytest.mark.parametrize("use_saver", [False, True])
def test_native_tensor_sources_reach_custom_encoder(tmp_path, use_saver):
    tensors = {
        "matrix": torch.arange(12.0).reshape(3, 4).t(),
        "vector": torch.ones(3),
    }
    seen = set()

    def fill(work):
        for buffer, entry in work:
            assert entry.source is tensors[entry.name]
            buffer[:] = entry.source.contiguous().numpy().tobytes()
            seen.add(entry.name)

    def metadata(shard):
        for entry in shard:
            assert entry.source is tensors[entry.name]
        return {"application": "native-tensors"}

    if use_saver:
        paths = ParallelSaver(num_shards=2, fill=fill, shard_metadata=metadata).save(
            tensors, str(tmp_path / "model")
        )
    else:
        entries = [plain_entry(name, tensor) for name, tensor in tensors.items()]
        assert all(entry.source is tensors[entry.name] for entry in entries)
        paths = save_sharded(
            entries,
            str(tmp_path / "model"),
            num_shards=2,
            fill=fill,
            shard_metadata=metadata,
        )
    assert seen == tensors.keys()
    for path in paths:
        with safe_open(path, framework="pt") as checkpoint:
            for name in checkpoint.keys():
                torch.testing.assert_close(checkpoint.get_tensor(name), tensors[name])


@pytest.mark.parametrize(
    "metadata", [{"layout": {}}, {1: "value"}, {"fst.shard.index": "9"}]
)
def test_shard_metadata_is_checked_before_creating_files(tmp_path, metadata):
    saver = ParallelSaver(shard_metadata=lambda entries: metadata)
    with pytest.raises(ValueError, match="strings|reserved"):
        saver.save(
            {"w": torch.ones(2)},
            str(tmp_path / "missing" / "model"),
        )
    assert list(tmp_path.iterdir()) == []


def test_custom_fill_failure_cleans_up_files(tmp_path):
    def fill(work):
        work[0][0][:] = b"abcd"
        raise RuntimeError("encoder failed")

    saver = ParallelSaver(num_shards=2, fill=fill)
    with pytest.raises(RuntimeError, match="encoder failed"):
        saver.save_entries(
            [WriteEntry("w", DType.U8, [4], 4, object())],
            str(tmp_path / "model"),
        )
    assert list(tmp_path.iterdir()) == []


def test_saver_module_import_is_lazy():
    code = """
        import builtins
        import json
        import tempfile
        import sys
        real_import = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "paddle", "torch_spyre"}:
                raise AssertionError("unexpected optional dependency: " + name)
            return real_import(name, *args, **kwargs)
        builtins.__import__ = guarded
        import fastsafetensors.saver
        from fastsafetensors import ParallelSaver
        from fastsafetensors.frameworks import TensorBase
        from fastsafetensors.st_types import Device, DType
        from typing import Any, get_type_hints, Mapping

        class ByteTensor(TensorBase):
            def get_raw(self):
                return b"hello"
            def get_shape(self):
                return [5]
            def get_nbytes(self):
                return 5
            def copy_to_buffer(self, buffer):
                buffer[:] = self.get_raw()

        assert get_type_hints(ParallelSaver.save)["tensors"] == Mapping[str, Any]
        with tempfile.TemporaryDirectory() as directory:
            paths = ParallelSaver(num_shards=2).save(
                {"w": ByteTensor(Device(), DType.U8)}, directory + "/model"
            )
            with open(paths[0], "rb") as file:
                header = json.loads(file.read(int.from_bytes(file.read(8), "little")))
                assert header["w"]["dtype"] == "U8"
                assert header["w"]["shape"] == [5]
                assert file.read() == b"hello"
        assert "torch" not in sys.modules
        assert "paddle" not in sys.modules
        assert "torch_spyre" not in sys.modules
    """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        env={**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_plain_checkpoint_without_torch_spyre(tmp_path):
    code = """
        import builtins
        import sys
        real_import = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name == "torch_spyre" or name.startswith("torch_spyre."):
                raise AssertionError("unexpected torch-spyre dependency")
            return real_import(name, *args, **kwargs)
        builtins.__import__ = guarded

        import torch
        from safetensors import safe_open
        from fastsafetensors.saver import plain_entry
        from fastsafetensors.saver import save_sharded

        tensors = {
            "matrix": torch.arange(12, dtype=torch.float32).reshape(3, 4).t(),
            "empty": torch.empty(0, dtype=torch.float16),
            "scalar": torch.tensor(3, dtype=torch.int64),
        }
        paths = save_sharded(
            [plain_entry(n, t) for n, t in tensors.items()],
            sys.argv[1], num_shards=2,
        )
        got = {}
        for path in paths:
            with safe_open(path, framework="pt", device="cpu") as f:
                got.update({name: f.get_tensor(name) for name in f.keys()})
        for name, actual in got.items():
            torch.testing.assert_close(actual, tensors[name])
        assert not any(n == "torch_spyre" or n.startswith("torch_spyre.") for n in sys.modules)
    """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code), str(tmp_path / "plain")],
        env={**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("dtype", list(dtype_convert))
def test_dtype_round_trip(dtype):
    assert fst_dtype(dtype_convert[dtype]) == dtype


def test_parallel_saver_saves_tensor_mapping(tmp_path):
    from fastsafetensors import ParallelSaver

    saver = ParallelSaver(num_shards=2, num_threads=2, fsync=True)
    tensors = {
        "matrix": torch.arange(12, dtype=torch.float32).reshape(3, 4).t(),
        "empty": torch.empty(0),
        "scalar": torch.tensor(3),
    }
    for version in ("first", "second"):
        paths = saver.save(
            tensors,
            str(tmp_path / version),
            metadata={"version": version},
            shard0_metadata={"index": "only-in-0"},
        )
        assert len(paths) == 2
        restored = {}
        for i, path in enumerate(paths):
            with safe_open(path, framework="pt", device="cpu") as f:
                assert f.metadata()["version"] == version
                assert ("index" in f.metadata()) == (i == 0)
                restored.update({name: f.get_tensor(name) for name in f.keys()})
        assert restored.keys() == tensors.keys()
        for name, tensor in tensors.items():
            torch.testing.assert_close(restored[name], tensor)


def test_metadata_rejects_incomplete_shard_set(tmp_path):
    paths = save_sharded(
        [plain_entry("w", torch.ones(2))],
        str(tmp_path / "shards"),
        num_shards=2,
    )
    with pytest.raises(ShardSetError, match="expected"):
        validate_shard_set(
            [SafeTensorsMetadata.from_file(paths[0], get_framework_op("pytorch"))]
        )


@pytest.mark.skipif(not hasattr(torch, "float4_e2m1fn_x2"), reason="F4 unavailable")
@pytest.mark.parametrize("shape", [(4, 8), (0, 4)])
def test_packed_float4_round_trip(tmp_path, shape):
    from fastsafetensors import fastsafe_open

    raw = torch.arange(shape[0] * shape[1], dtype=torch.uint8).reshape(shape)
    tensor = raw.view(torch.float4_e2m1fn_x2)
    paths = ParallelSaver(num_shards=1).save({"w": tensor}, str(tmp_path / "f4"))
    meta = SafeTensorsMetadata.from_file(paths[0], get_framework_op("pytorch"))
    assert meta.tensors["w"].shape == [shape[0], shape[1] * 2]
    with safe_open(paths[0], framework="pt") as checkpoint:
        restored = checkpoint.get_tensor("w")
        assert restored.shape == tensor.shape
        assert torch.equal(restored.view(torch.uint8), raw)
    with fastsafe_open(paths, device="cpu", nogds=True) as checkpoint:
        restored = checkpoint.get_tensor("w")
        assert restored.shape == tensor.shape
        assert torch.equal(restored.view(torch.uint8), raw)
