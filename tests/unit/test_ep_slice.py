# SPDX-License-Identifier: Apache-2.0
"""Tests for the sub-file byte-range read primitive and the EP-slice demonstrator.

The expert-range math is pure Python (no GPU / C extension needed). The
partial-read tests reuse the gpt2 fixture and the nogds copier to prove that
loading only a selected subset of tensors yields byte-identical data for the
kept tensors while skipping the rest.
"""
import pytest
import torch

from fastsafetensors import SafeTensorsMetadata
from fastsafetensors import cpp as fstcpp
from fastsafetensors.copier.nogds import NoGdsFileCopier
from fastsafetensors.copier.unified import new_unified_copier
from fastsafetensors.ep_slice import (
    expert_parallel_filter,
    owned_expert_range,
)

# The unified copier (mmap → pin_memory → cudaMemcpyAsync) needs a CUDA device;
# skip its partial-read tests on CPU-only runners.
_requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="unified copier requires a CUDA device"
)

# Reuse helpers from the main test module (tests/unit is on sys.path via conftest).
from test_fastsafetensors import get_and_check_device, load_safetensors_file

# ---- pure-Python EP range math (contiguous-block "linear" assignment) ----


def test_owned_expert_range_even():
    assert owned_expert_range(256, 2, 0) == (0, 128)
    assert owned_expert_range(256, 2, 1) == (128, 256)


def test_owned_expert_range_remainder():
    # remainder goes to the lowest-numbered ranks
    assert owned_expert_range(10, 3, 0) == (0, 4)
    assert owned_expert_range(10, 3, 1) == (4, 7)
    assert owned_expert_range(10, 3, 2) == (7, 10)
    # whole owned set tiles the expert space with no gaps/overlaps
    covered = []
    for r in range(4):
        lo, hi = owned_expert_range(13, 4, r)
        covered.extend(range(lo, hi))
    assert covered == list(range(13))


def test_owned_expert_range_invalid():
    with pytest.raises(ValueError):
        owned_expert_range(8, 0, 0)
    with pytest.raises(ValueError):
        owned_expert_range(8, 2, 2)


def test_expert_parallel_filter_keeps_nonexpert_and_owned():
    keep = expert_parallel_filter(num_experts=256, ep_size=2, ep_rank=0)
    # non-expert tensors are kept on every rank
    assert keep("model.embed_tokens.weight") is True
    assert keep("model.layers.0.self_attn.q_proj.weight") is True
    # owned vs unowned routed experts
    assert keep("model.layers.0.mlp.experts.5.w1.weight") is True
    assert keep("model.layers.0.mlp.experts.200.w1.weight") is False
    # DeepSeek-style "ffn.experts" naming also matches the default pattern
    assert keep("model.layers.3.ffn.experts.10.gate_proj.weight") is True
    assert keep("model.layers.3.ffn.experts.130.gate_proj.weight") is False


# ---- byte-range selection + partial read (uses the gpt2 fixture) ----


def _keep_every_other(meta: SafeTensorsMetadata):
    """A non-EP predicate exercising the primitive on a model without experts:
    keep every other tensor by sorted name (so kept tensors are non-adjacent and
    produce multiple, non-mergeable runs)."""
    kept = set(sorted(meta.tensors.keys())[::2])
    return lambda name: name in kept


def test_select_byte_ranges_all_equals_full(input_files, framework):
    meta = SafeTensorsMetadata.from_file(input_files[0], framework)
    ranges = meta.select_byte_ranges(lambda name: True)
    # contiguous tensors with no large gaps coalesce into one run that begins at
    # the data section and never exceeds the file size
    assert len(ranges) == 1
    assert ranges[0][0] == meta.header_length
    assert ranges[0][1] <= meta.size_bytes


def test_select_byte_ranges_covers_only_kept(input_files, framework):
    meta = SafeTensorsMetadata.from_file(input_files[0], framework)
    keep = _keep_every_other(meta)
    ranges = meta.select_byte_ranges(keep)
    # sorted, non-overlapping
    for (a_lo, a_hi), (b_lo, b_hi) in zip(ranges, ranges[1:]):
        assert a_hi <= b_lo
    # every kept tensor is fully covered by some run
    for name, fr in meta.tensors.items():
        if not keep(name):
            continue
        s = meta.header_length + fr.data_offsets[0]
        e = meta.header_length + fr.data_offsets[1]
        assert any(lo <= s and e <= hi for lo, hi in ranges), name


def test_nogds_partial_read_byte_identical(fstcpp_log, input_files, framework):
    device, dev_is_gpu = get_and_check_device(framework)
    meta = SafeTensorsMetadata.from_file(input_files[0], framework)
    keep = _keep_every_other(meta)
    ranges = meta.select_byte_ranges(keep)

    reader = fstcpp.nogds_file_reader(
        False, 256 * 1024, 4, dev_is_gpu, device.index or 0
    )
    copier = NoGdsFileCopier(meta, device, reader, framework)
    copier.set_byte_ranges(ranges)
    gbuf = copier.submit_io(False, 10 * 1024 * 1024 * 1024)
    tensors = copier.wait_io(gbuf)

    ref = load_safetensors_file(input_files[0], device, framework)
    kept_names = [n for n in meta.tensors if keep(n)]
    assert kept_names, "fixture should have at least one kept tensor"
    for name in kept_names:
        assert framework.is_equal(tensors[name], ref[name]), name

    framework.free_tensor_memory(gbuf, device)
    del copier
    del reader
    assert framework.get_mem_used() == 0


def test_nogds_full_read_unchanged(fstcpp_log, input_files, framework):
    """set_byte_ranges(None) must reproduce the original full-file load exactly."""
    device, dev_is_gpu = get_and_check_device(framework)
    meta = SafeTensorsMetadata.from_file(input_files[0], framework)
    reader = fstcpp.nogds_file_reader(
        False, 256 * 1024, 4, dev_is_gpu, device.index or 0
    )
    copier = NoGdsFileCopier(meta, device, reader, framework)
    copier.set_byte_ranges(None)  # explicit default
    gbuf = copier.submit_io(False, 10 * 1024 * 1024 * 1024)
    tensors = copier.wait_io(gbuf)
    for key, exp in load_safetensors_file(input_files[0], device, framework).items():
        assert framework.is_equal(tensors[key], exp), key
    framework.free_tensor_memory(gbuf, device)
    del copier
    del reader
    assert framework.get_mem_used() == 0


# ---- same partial-read guarantees for the unified-memory copier ----


@_requires_cuda
def test_unified_partial_read_byte_identical(fstcpp_log, input_files, framework):
    device, dev_is_gpu = get_and_check_device(framework)
    if not dev_is_gpu:
        pytest.skip("unified copier targets a GPU device")
    meta = SafeTensorsMetadata.from_file(input_files[0], framework)
    keep = _keep_every_other(meta)
    ranges = meta.select_byte_ranges(keep)

    # factory path loads the CUDA fn pointers (load_library_func); constructing
    # UnifiedMemCopier directly would leave memcpy_h2d_async unbound.
    copier = new_unified_copier(device)(meta, device, framework)
    copier.set_byte_ranges(ranges)
    gbuf = copier.submit_io(False, 10 * 1024 * 1024 * 1024)
    tensors = copier.wait_io(gbuf)

    ref = load_safetensors_file(input_files[0], device, framework)
    kept_names = [n for n in meta.tensors if keep(n)]
    assert kept_names, "fixture should have at least one kept tensor"
    for name in kept_names:
        assert framework.is_equal(tensors[name], ref[name]), name

    framework.free_tensor_memory(gbuf, device)
    del copier
    assert framework.get_mem_used() == 0


@_requires_cuda
def test_unified_full_read_unchanged(fstcpp_log, input_files, framework):
    """unified set_byte_ranges(None) must reproduce the full-file load exactly."""
    device, dev_is_gpu = get_and_check_device(framework)
    if not dev_is_gpu:
        pytest.skip("unified copier targets a GPU device")
    meta = SafeTensorsMetadata.from_file(input_files[0], framework)
    copier = new_unified_copier(device)(meta, device, framework)
    copier.set_byte_ranges(None)  # explicit default
    gbuf = copier.submit_io(False, 10 * 1024 * 1024 * 1024)
    tensors = copier.wait_io(gbuf)
    for key, exp in load_safetensors_file(input_files[0], device, framework).items():
        assert framework.is_equal(tensors[key], exp), key
    framework.free_tensor_memory(gbuf, device)
    del copier
    assert framework.get_mem_used() == 0
