# SPDX-License-Identifier: Apache-2.0
"""Tests for read-time dim-0 row slicing (SafeTensorsFileLoader.set_slice).

The frame math is pure Python (no GPU / C extension needed). The end-to-end
tests reuse the fixture helpers and the nogds copier to prove that loading
only the selected rows yields byte-identical data for those rows, that the
compact-chunk path bounds the device buffer to the selected span, and that
the pipeline planner sees the narrowed frames (issue #127).
"""

from contextlib import closing

import pytest

try:
    import torch
except ImportError:  # paddle CI: torch-only tests self-skip below
    torch = None  # type: ignore[assignment]

from fastsafetensors import (
    SafeTensorsFileLoader,
    SafeTensorsMetadata,
    SingleGroup,
    TensorFrame,
)
from fastsafetensors.copier.nogds import NoGdsFileCopier
from fastsafetensors.parallel_loader import ParallelLoader, PipelineParallel
from fastsafetensors.st_types import DType

# Reuse helpers from the main test module (tests/unit is on sys.path via conftest).
from test_fastsafetensors import load_safetensors_file, save_safetensors_file

_requires_torch = pytest.mark.skipif(
    torch is None, reason="row-slice data checks need torch tensors"
)
_requires_cuda = pytest.mark.skipif(
    torch is None or not torch.cuda.is_available(),
    reason="device slice loading requires a CUDA device",
)


def _pick_2d(meta, min_rows=4):
    for name, frame in meta.tensors.items():
        if len(frame.shape) == 2 and frame.shape[0] >= min_rows:
            return name, frame
    pytest.skip("fixture has no 2D tensor with enough rows")


# ---- with_tensor_slices frame math ----


def test_with_tensor_slices_partitions_rows(input_files, framework):
    meta = SafeTensorsMetadata.from_file(input_files[0], framework)
    name, frame = _pick_2d(meta)
    rows = frame.shape[0]
    a, b = rows // 3, 2 * rows // 3
    parts = []
    for selection in (slice(0, a), slice(a, b), slice(b, None)):
        d = meta.with_tensor_slices({name: selection})
        nf = d.tensors[name]
        assert nf.shape[1:] == frame.shape[1:]
        parts.append(nf)
        # other tensors untouched (same object)
        for other, of in d.tensors.items():
            if other != name:
                assert of is meta.tensors[other]
    # rows partition exactly, byte ranges are adjacent and cover the original
    assert sum(p.shape[0] for p in parts) == rows
    assert parts[0].data_offsets[0] == frame.data_offsets[0]
    assert parts[-1].data_offsets[1] == frame.data_offsets[1]
    for first, second in zip(parts, parts[1:]):
        assert first.data_offsets[1] == second.data_offsets[0]


def test_with_tensor_slices_bounds(input_files, framework):
    meta = SafeTensorsMetadata.from_file(input_files[0], framework)
    name, frame = _pick_2d(meta)
    rows = frame.shape[0]
    nbytes = frame.data_offsets[1] - frame.data_offsets[0]
    row_bytes = nbytes // rows
    s, _ = frame.data_offsets
    # negative start counts from the end; open stop runs to the last row
    d = meta.with_tensor_slices({name: slice(-2, None)})
    assert d.tensors[name].shape[0] == 2
    assert d.tensors[name].data_offsets == [
        s + (rows - 2) * row_bytes,
        s + rows * row_bytes,
    ]
    # a full-range selection keeps the original span
    d = meta.with_tensor_slices({name: slice(None, None)})
    assert d.tensors[name].data_offsets == frame.data_offsets
    assert d.tensors[name].shape == frame.shape
    # out-of-range stops clamp like Python sequences
    d = meta.with_tensor_slices({name: slice(1, rows + 10)})
    assert d.tensors[name].shape[0] == rows - 1
    # an empty selection narrows to a zero-row, zero-byte tensor
    d = meta.with_tensor_slices({name: slice(2, 2)})
    assert d.tensors[name].shape[0] == 0
    assert d.tensors[name].data_offsets == [s + 2 * row_bytes] * 2
    # a reversed range (stop < start) is empty too, not a negative
    # dimension with reversed byte offsets
    d = meta.with_tensor_slices({name: slice(5, 2)})
    assert d.tensors[name].shape[0] == 0
    assert d.tensors[name].data_offsets == [s + 5 * row_bytes] * 2


def test_with_tensor_slices_passthrough(input_files, framework):
    meta = SafeTensorsMetadata.from_file(input_files[0], framework)
    # names absent from this metadata are ignored (they live in other files)
    d = meta.with_tensor_slices({"no-such-tensor": slice(0, 1)})
    assert d is meta
    # a slice that matches nothing in the dict is likewise a no-op
    assert d.tensors == meta.tensors


def test_narrow_tensor_frame_validation(framework):
    # scalar tensor: no shape[0] to slice
    scalar = TensorFrame(DType.U8, [], [0, 1], [], [], False)
    with pytest.raises(ValueError, match="scalar"):
        SafeTensorsMetadata._narrow_tensor_frame("t", scalar, slice(0, 1))
    # non-unit step is not a contiguous byte range
    frame = TensorFrame(DType.U8, [4, 2], [0, 8], [2, 1], [0, 0], False)
    with pytest.raises(ValueError, match="step must be 1"):
        SafeTensorsMetadata._narrow_tensor_frame("t", frame, slice(0, 4, 2))
    # rows not byte-addressable (packed sub-byte layout)
    odd = TensorFrame(DType.U8, [3, 2], [0, 5], [2, 1], [0, 0], False)
    with pytest.raises(ValueError, match="byte-addressable"):
        SafeTensorsMetadata._narrow_tensor_frame("t", odd, slice(0, 2))


# ---- set_slice API on the loader ----


def test_set_slice_before_and_after_registration(input_files, framework):
    meta = SafeTensorsMetadata.from_file(input_files[0], framework)
    name, frame = _pick_2d(meta)
    rows = frame.shape[0]
    row_bytes = (frame.data_offsets[1] - frame.data_offsets[0]) // rows
    for order in ("before", "after"):
        loader = SafeTensorsFileLoader(
            None, "cpu", nogds=True, framework=framework.get_name()
        )
        if order == "after":
            loader.add_filenames({0: input_files})
            loader.set_slice(name, slice(1, rows - 1))
        else:
            loader.set_slice(name, slice(1, rows - 1))
            loader.add_filenames({0: input_files})
        assert loader.get_shape(name) == [rows - 2] + frame.shape[1:]
        loaded_meta, _ = loader.meta[input_files[0]]
        loaded = loaded_meta.tensors[name]
        assert loaded.data_offsets == [
            frame.data_offsets[0] + row_bytes,
            frame.data_offsets[1] - row_bytes,
        ]
        loader.close()


def test_set_slice_clear_restores_full_frame(input_files, framework):
    # set_slice(name, None) clears a recorded slice (mirrors
    # set_tensor_filter(None)): the pristine frame comes back and the file
    # loads through the normal path again.
    meta = SafeTensorsMetadata.from_file(input_files[0], framework)
    name, frame = _pick_2d(meta)
    loader = SafeTensorsFileLoader(
        None, "cpu", nogds=True, framework=framework.get_name()
    )
    loader.add_filenames({0: input_files})
    loader.set_slice(name, slice(0, 2))
    assert loader.get_shape(name)[0] == 2
    loader.set_slice(name, None)
    assert loader.get_shape(name) == frame.shape
    with closing(loader.copy_files_to_device()) as bufs:
        assert list(bufs.get_tensor(name).shape) == frame.shape
    loader.close()


@_requires_torch
def test_set_slice_rejected_selection_not_recorded(tmp_path, framework):
    """A slice that cannot be applied is not remembered.

    Without the cleanup, one rejected set_slice (scalar tensor, packed
    rows) poisons every later set_slice, add_filenames, or copy on the
    same file.
    """
    scalar = torch.tensor(1.0)
    weight = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4)
    path = str(tmp_path / "scalar.safetensors")
    save_safetensors_file(
        {"scalar": scalar, "engram.embed.weight": weight},
        path,
        {"format": "pt"},
        framework,
    )
    with closing(
        SafeTensorsFileLoader(None, "cpu", nogds=True, framework=framework.get_name())
    ) as loader:
        loader.add_filenames({0: [path]})
        with pytest.raises(ValueError, match="scalar"):
            loader.set_slice("scalar", slice(0, 1))
        # the rejected slice did not stay recorded: a valid slice and a
        # copy on the same file work
        loader.set_slice("engram.embed.weight", slice(1, 5))
        with closing(loader.copy_files_to_device()) as bufs:
            assert torch.equal(bufs.get_tensor("engram.embed.weight"), weight[1:5])


@_requires_torch
def test_set_slice_compact_span_and_multi_runs(tmp_path, framework, monkeypatch):
    """Two sliced tensors split by a >merge_gap hole load as two runs into
    one compact buffer spanning them (the documented allocation cost:
    unselected rows between kept runs are allocated, not read)."""
    weight = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4)  # 128 B
    scale = torch.arange(8, dtype=torch.float32)  # 32 B
    filler = torch.zeros(16 * 1024, dtype=torch.float32)  # 64 KiB hole
    path = str(tmp_path / "engram-gap.safetensors")
    save_safetensors_file(
        {
            "engram.embed.weight": weight,
            "filler": filler,
            "engram.embed.scale": scale,
        },
        path,
        {"format": "pt"},
        framework,
    )
    recorded = []
    original = NoGdsFileCopier.set_chunk

    def chunk_spy(self, byte_ranges, names, allocation_size=None):
        recorded.append((tuple(byte_ranges), set(names), allocation_size))
        return original(self, byte_ranges, names, allocation_size)

    monkeypatch.setattr(NoGdsFileCopier, "set_chunk", chunk_spy)
    allocs = []
    orig_alloc = framework.alloc_tensor_memory

    def alloc_spy(length, device):
        allocs.append(length)
        return orig_alloc(length, device)

    monkeypatch.setattr(framework, "alloc_tensor_memory", alloc_spy)
    with closing(
        SafeTensorsFileLoader(None, "cpu", nogds=True, framework=framework.get_name())
    ) as loader:
        loader.add_filenames({0: [path]})
        loader.set_tensor_filter(lambda name: name != "filler")
        loader.set_slice("engram.embed.weight", slice(0, 2))
        with closing(loader.copy_files_to_device()) as bufs:
            assert torch.equal(bufs.get_tensor("engram.embed.weight"), weight[:2])
    assert len(recorded) == 1
    ranges, names, allocation_size = recorded[0]
    assert names == {"engram.embed.weight", "engram.embed.scale"}
    # the 64 KiB filtered hole exceeds merge_gap: the runs stay split
    assert len(ranges) == 2
    # the compact buffer spans min_start..max_end of the kept runs, hole
    # included, and the allocation size is left to the copier's default
    assert allocs and max(allocs) == ranges[-1][1] - ranges[0][0]
    assert allocation_size is None


@_requires_torch
def test_parallel_loader_tensor_slices_across_shards(tmp_path, framework):
    """A slice target living in a later shard is not rejected while an
    earlier shard's batch copies: name validation covers all input files
    (the pipeline pre-seeds the loader's metadata cache)."""
    a = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4)
    b = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4)
    f1 = str(tmp_path / "shard-1.safetensors")
    f2 = str(tmp_path / "shard-2.safetensors")
    save_safetensors_file({"a": a}, f1, {"format": "pt"}, framework)
    save_safetensors_file({"b": b}, f2, {"format": "pt"}, framework)
    with closing(
        ParallelLoader(
            None,
            [f1, f2],
            device="cpu",
            framework=framework.get_name(),
            nogds=True,
            tensor_slices={"b": slice(2, 6)},
            use_tqdm_on_load=False,
        )
    ) as loader:
        with closing(loader.iterate_weights()) as weights:
            actual = dict(weights)
    assert torch.equal(actual["a"], a)
    assert torch.equal(actual["b"], b[2:6])


def test_set_slice_reapplication_is_idempotent(input_files, framework):
    # narrowing a second time must start from the pristine frame, not the
    # already-narrowed one (no slice-of-slice)
    meta = SafeTensorsMetadata.from_file(input_files[0], framework)
    name, frame = _pick_2d(meta)
    loader = SafeTensorsFileLoader(
        None, "cpu", nogds=True, framework=framework.get_name()
    )
    loader.add_filenames({0: input_files})
    loader.set_slice(name, slice(0, 2))
    assert loader.get_shape(name)[0] == 2
    loader.set_slice(name, slice(0, 4))
    assert loader.get_shape(name)[0] == 4
    narrowed, _ = loader.meta[input_files[0]]
    frame_after = narrowed.tensors[name]
    assert frame_after.data_offsets[0] == frame.data_offsets[0]
    loader.close()


def test_set_slice_validation(input_files, framework):
    meta = SafeTensorsMetadata.from_file(input_files[0], framework)
    name, frame = _pick_2d(meta)
    loader = SafeTensorsFileLoader(
        None, "cpu", nogds=True, framework=framework.get_name()
    )
    loader.add_filenames({0: input_files})
    with pytest.raises(TypeError, match="must be a slice"):
        loader.set_slice(name, (0, 2))
    with pytest.raises(ValueError, match="step must be 1"):
        loader.set_slice(name, slice(0, 2, 2))
    # a name present in no registered file fails at copy time
    loader.set_slice("no-such-tensor", slice(0, 2))
    with pytest.raises(ValueError, match="no registered tensor"):
        loader.copy_files_to_device()
    loader.close()

    # multi-rank loader group: a broadcast cannot deliver per-rank rows
    loader = SafeTensorsFileLoader(
        None, "cpu", nogds=True, framework=framework.get_name()
    )

    class _TwoRankGroup(SingleGroup):
        def size(self):
            return 2

    loader.pg = _TwoRankGroup()
    with pytest.raises(ValueError, match="single-process loader group"):
        loader.set_slice(name, slice(0, 2))
    loader.close()


def test_parallel_loader_tensor_slices_requires_single_process(
    input_files, framework
):
    loader = SafeTensorsFileLoader(
        None, "cpu", nogds=True, framework=framework.get_name()
    )

    class _TwoRankGroup(SingleGroup):
        def size(self):
            return 2

    with pytest.raises(ValueError, match="single-process loader group"):
        PipelineParallel(
            _TwoRankGroup(), loader, input_files, tensor_slices={"x": slice(0, 1)}
        )
    loader.close()


# ---- end-to-end: sliced load == full load narrowed ----


def _engram_fixture(tmp_path, framework):
    """Two dim-0-shardable 'Engram' tensors plus an unrelated one."""
    assert torch is not None
    weight = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4)
    scale = torch.arange(8, dtype=torch.float32)
    other = torch.arange(2 * 3, dtype=torch.float32).reshape(2, 3)
    path = str(tmp_path / "engram.safetensors")
    save_safetensors_file(
        {"engram.embed.weight": weight, "engram.embed.scale": scale, "other": other},
        path,
        {"format": "pt"},
        framework,
    )
    return path, {"engram.embed.weight": weight, "engram.embed.scale": scale}, other


@_requires_torch
def test_set_slice_loads_selected_rows(tmp_path, framework):
    path, reference, _ = _engram_fixture(tmp_path, framework)
    with closing(
        SafeTensorsFileLoader(None, "cpu", nogds=True, framework=framework.get_name())
    ) as loader:
        loader.add_filenames({0: [path]})
        loader.set_slice("engram.embed.weight", slice(3, 7))
        with closing(loader.copy_files_to_device()) as bufs:
            got = bufs.get_tensor("engram.embed.weight")
            assert list(got.shape) == [4, 4]
            assert torch.equal(got, reference["engram.embed.weight"][3:7])
            # unsliced tensors load in full
            assert torch.equal(bufs.get_tensor("engram.embed.scale"), reference["engram.embed.scale"])


@_requires_cuda
def test_set_slice_on_cuda_device(tmp_path, framework):
    """Slices load into a bounded device buffer (unified/nogds copier)."""
    from test_fastsafetensors import get_and_check_device

    device, _ = get_and_check_device(framework)
    path, reference, _ = _engram_fixture(tmp_path, framework)
    with closing(
        SafeTensorsFileLoader(
            None, device.as_str(), nogds=True, framework=framework.get_name()
        )
    ) as loader:
        loader.add_filenames({0: [path]})
        loader.set_slice("engram.embed.weight", slice(2, 6))
        with closing(loader.copy_files_to_device()) as bufs:
            got = bufs.get_tensor("engram.embed.weight")
            assert list(got.shape) == [4, 4]
            assert torch.equal(
                got.to("cpu"), reference["engram.embed.weight"][2:6]
            )


@_requires_torch
def test_set_slice_engram_pattern_from_issue(tmp_path, framework):
    """Takeshi's sketch: filter to the Engram keys, slice both, get both."""
    path, reference, _ = _engram_fixture(tmp_path, framework)
    names = {"engram.embed.weight", "engram.embed.scale"}
    with closing(
        SafeTensorsFileLoader(None, "cpu", nogds=True, framework=framework.get_name())
    ) as loader:
        loader.add_filenames({0: [path]})
        loader.set_tensor_filter(lambda name: name in names)
        loader.set_slice("engram.embed.weight", slice(2, 6))
        loader.set_slice("engram.embed.scale", slice(1, 5))
        with closing(loader.copy_files_to_device()) as bufs:
            weight = bufs.get_tensor("engram.embed.weight")
            scale = bufs.get_tensor("engram.embed.scale")
            assert torch.equal(weight, reference["engram.embed.weight"][2:6])
            assert torch.equal(scale, reference["engram.embed.scale"][1:5])
            with pytest.raises(ValueError):
                bufs.get_tensor("other")


@_requires_torch
def test_set_slice_compact_chunk_bounds_buffer(tmp_path, framework, monkeypatch):
    """A sliced file loads through set_chunk with the kept tensors' ranges."""
    path, reference, _ = _engram_fixture(tmp_path, framework)
    recorded = []
    original = NoGdsFileCopier.set_chunk

    def spy(self, byte_ranges, names, allocation_size=None):
        recorded.append((tuple(byte_ranges), set(names), allocation_size))
        return original(self, byte_ranges, names, allocation_size)

    monkeypatch.setattr(NoGdsFileCopier, "set_chunk", spy)
    with closing(
        SafeTensorsFileLoader(None, "cpu", nogds=True, framework=framework.get_name())
    ) as loader:
        loader.add_filenames({0: [path]})
        loader.set_tensor_filter(lambda name: name != "other")
        loader.set_slice("engram.embed.weight", slice(0, 4))
        # snapshot before close(): reset() clears loader.meta
        meta = loader.meta[path][0]
        with closing(loader.copy_files_to_device()) as bufs:
            weight = bufs.get_tensor("engram.embed.weight")
            assert torch.equal(weight, reference["engram.embed.weight"][:4])
    assert len(recorded) == 1
    ranges, names, allocation_size = recorded[0]
    # only the kept tensors' (narrowed) byte spans are requested, and the
    # allocation is left to the copier's compact default (None)
    expected = {n for n in meta.tensors if n != "other"}
    assert names == expected
    assert ranges == tuple(meta.select_byte_ranges(lambda n: n != "other"))
    assert allocation_size is None


@_requires_torch
def test_parallel_loader_tensor_slices(tmp_path, framework):
    path, reference, _ = _engram_fixture(tmp_path, framework)
    with closing(
        ParallelLoader(
            None,
            [path],
            device="cpu",
            framework=framework.get_name(),
            nogds=True,
            tensor_slices={"engram.embed.weight": slice(1, 6)},
            use_tqdm_on_load=False,
        )
    ) as loader:
        with closing(loader.iterate_weights()) as weights:
            actual = dict(weights)
    assert set(actual) == {"engram.embed.weight", "engram.embed.scale", "other"}
    assert torch.equal(actual["engram.embed.weight"], reference["engram.embed.weight"][1:6])
    assert torch.equal(actual["engram.embed.scale"], reference["engram.embed.scale"])


@_requires_torch
def test_parallel_loader_tensor_slices_through_chunk_planning(tmp_path, framework):
    """The planner must see narrowed frames: a budget below the full tensor
    but above the slice plans fine (plan_chunks would otherwise refuse)."""
    path, reference, _ = _engram_fixture(tmp_path, framework)
    # full engram.embed.weight is 8*4*4 = 128 bytes; the slice(0, 2) span is 32
    with closing(
        ParallelLoader(
            None,
            [path],
            device="cpu",
            framework=framework.get_name(),
            nogds=True,
            tensor_slices={"engram.embed.weight": slice(0, 2)},
            max_batch_bytes=64,
            use_tqdm_on_load=False,
        )
    ) as loader:
        with closing(loader.iterate_weights()) as weights:
            actual = dict(weights)
    assert torch.equal(actual["engram.embed.weight"], reference["engram.embed.weight"][:2])
