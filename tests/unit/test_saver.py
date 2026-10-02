# SPDX-License-Identifier: Apache-2.0
"""Sharded safetensors writes and generic shard-set validation."""

import json
import os

import pytest

from fastsafetensors.common import SafeTensorsMetadata
from fastsafetensors.saver import (
    ShardSetError,
    WriteEntry,
    per_entry_fill,
    plain_entry,
    plan_shards,
    save_sharded,
    shard_paths,
    validate_shard_set,
)
from fastsafetensors.st_types import DType

torch = pytest.importorskip("torch")


def _tensors():
    g = torch.Generator().manual_seed(0)
    return {
        "big": torch.randn(64, 64, generator=g).half(),
        "mid": torch.randn(32, 16, generator=g),
        "small": torch.arange(7, dtype=torch.int64),
        "strided": torch.randn(8, 4, generator=g).t(),
        "empty": torch.empty(0, 3),
    }


def test_plan_balances_largest_first(tmp_path):
    entries = [
        WriteEntry(f"t{i}", DType.U8, [n], n) for i, n in enumerate([10, 9, 8, 1, 1, 1])
    ]
    plans = plan_shards(entries, shard_paths(str(tmp_path / "m"), 3))
    loads = sorted(sum(e.nbytes for e in p.entries) for p in plans)
    assert loads == [10, 10, 10]


def test_save_sharded_round_trips_through_safetensors(tmp_path, framework):
    from safetensors import safe_open

    tensors = _tensors()
    paths = save_sharded(
        [plain_entry(n, t) for n, t in tensors.items()],
        str(tmp_path / "model"),
        num_shards=3,
        metadata={"who": "test"},
        shard0_metadata={"index": "only-in-0"},
    )
    assert [os.path.basename(p) for p in paths] == [
        f"model-{i:05d}-of-00003.safetensors" for i in (1, 2, 3)
    ]
    assert sorted(os.listdir(tmp_path)) == sorted(os.path.basename(p) for p in paths)

    got = {}
    for i, path in enumerate(paths):
        with safe_open(path, framework="pt") as f:
            md = f.metadata()
            assert md["who"] == "test"
            assert ("index" in md) == (i == 0)
            for k in f.keys():
                got[k] = f.get_tensor(k)
    assert got.keys() == tensors.keys()
    for name, t in tensors.items():
        assert torch.equal(got[name], t.contiguous()), name
    validate_shard_set([SafeTensorsMetadata.from_file(p, framework) for p in paths])


def test_tensor_fill_matches_save_file(tmp_path):
    from safetensors.torch import save_file

    tensors = {k: v.contiguous() for k, v in _tensors().items()}
    ours = save_sharded(
        [plain_entry(n, t) for n, t in tensors.items()],
        str(tmp_path / "a"),
        num_shards=1,
    )
    save_file(tensors, str(tmp_path / "ref.safetensors"))
    with open(ours[0], "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
        data = f.read()
    for name, t in tensors.items():
        b, e = header[name]["data_offsets"]
        assert data[b:e] == t.numpy().tobytes(), name


def test_failed_fill_leaves_nothing(tmp_path):
    def boom(work):
        raise RuntimeError("fill failed")

    with pytest.raises(RuntimeError, match="fill failed"):
        save_sharded(
            [plain_entry(n, t) for n, t in _tensors().items()],
            str(tmp_path / "m"),
            num_shards=2,
            fill=boom,
        )
    assert os.listdir(tmp_path) == []


def test_not_enough_space(tmp_path):
    with pytest.raises(OSError, match="not enough space"):
        save_sharded(
            [plain_entry("t", torch.zeros(4))],
            str(tmp_path / "m"),
            num_shards=1,
            min_free_bytes=1 << 62,
        )
    assert os.listdir(tmp_path) == []


def test_entry_size_is_checked(tmp_path):
    with pytest.raises(ValueError, match="bytes"):
        plan_shards(
            [WriteEntry("t", DType.F16, [4], 7)], shard_paths(str(tmp_path / "m"), 1)
        )


def test_shards_of_different_writes_are_rejected(tmp_path, framework):
    a = save_sharded(
        [plain_entry("a", torch.ones(3))],
        str(tmp_path / "a"),
        num_shards=2,
    )
    b = save_sharded(
        [plain_entry("b", torch.ones(3))],
        str(tmp_path / "b"),
        num_shards=2,
    )
    with pytest.raises(ShardSetError, match="separately"):
        validate_shard_set(
            [
                SafeTensorsMetadata.from_file(a[0], framework),
                SafeTensorsMetadata.from_file(b[1], framework),
            ]
        )
    with pytest.raises(ShardSetError, match="expected 0"):
        validate_shard_set(
            [
                SafeTensorsMetadata.from_file(a[1], framework),
                SafeTensorsMetadata.from_file(a[0], framework),
            ]
        )


def test_per_entry_fill_runs_every_entry(tmp_path):
    seen = []
    fill = per_entry_fill(
        lambda buf, e: (
            seen.append(e.name),
            buf.__setitem__(slice(None), bytes(len(buf))),
        ),
        4,
    )
    save_sharded(
        [WriteEntry(f"t{i}", DType.U8, [3], 3) for i in range(5)],
        str(tmp_path / "m"),
        num_shards=2,
        fill=fill,
    )
    assert sorted(seen) == [f"t{i}" for i in range(5)]


@pytest.mark.parametrize(
    "entry,match",
    [
        (WriteEntry("w", DType.AUTO, [1], 1), "AUTO"),
        (WriteEntry("w", DType.U8, [-2, -2], 4), "nonnegative"),
        (WriteEntry("w", DType.F4, [2, 3], 3), "even"),
        (WriteEntry("w", DType.F4, [2, 4], 8), "bytes"),
    ],
)
def test_invalid_entries_are_rejected_before_creating_files(tmp_path, entry, match):
    with pytest.raises(ValueError, match=match):
        save_sharded([entry], str(tmp_path / "m"), num_shards=1)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("num_shards", [0, -1])
def test_invalid_shard_count(tmp_path, num_shards):
    with pytest.raises(ValueError, match="num_shards"):
        save_sharded([], str(tmp_path / "m"), num_shards=num_shards)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("header_align", [0, -8, 7, 12])
def test_invalid_alignment(tmp_path, header_align):
    with pytest.raises(ValueError, match="header_align"):
        save_sharded([], str(tmp_path / "m"), header_align=header_align)
    assert list(tmp_path.iterdir()) == []


def test_empty_or_duplicate_paths(tmp_path):
    path = str(tmp_path / "m.safetensors")
    for paths in ([], [path, path], [path, str(tmp_path / "." / "m.safetensors")]):
        with pytest.raises(ValueError, match="paths"):
            plan_shards([], paths)


def test_zero_byte_entries_reach_custom_fill(tmp_path):
    from safetensors import safe_open

    seen = []

    def fill(work):
        seen.extend((e.name, len(buf)) for buf, e in work)

    paths = save_sharded(
        [plain_entry("empty", torch.empty(0, 3))],
        str(tmp_path / "m"),
        num_shards=2,
        fill=fill,
    )
    assert seen == [("empty", 0)]
    with safe_open(paths[0], framework="pt") as checkpoint:
        assert checkpoint.get_tensor("empty").shape == (0, 3)


def test_concurrent_saves_use_distinct_temporary_files(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from safetensors import safe_open

    barrier = Barrier(2)

    def save(value):
        def fill(work):
            for buf, _ in work:
                buf[:] = bytes([value]) * len(buf)
            barrier.wait(timeout=10)

        return save_sharded(
            [WriteEntry("w", DType.U8, [4], 4)],
            str(tmp_path / "m"),
            num_shards=1,
            fill=fill,
        )

    with ThreadPoolExecutor(2) as pool:
        a, b = pool.submit(save, 11), pool.submit(save, 22)
        assert a.result() == b.result()
    assert len(list(tmp_path.iterdir())) == 1
    with safe_open(
        str(tmp_path / "m-00001-of-00001.safetensors"), framework="pt"
    ) as checkpoint:
        assert checkpoint.get_tensor("w").tolist() in ([11] * 4, [22] * 4)
