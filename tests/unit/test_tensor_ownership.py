# SPDX-License-Identifier: Apache-2.0

"""Shared allocation references and owning/borrowed storage lifetimes."""

import gc

import pytest

from fastsafetensors import (
    SafeTensorsFileLoader,
    live_allocation_bytes,
    live_allocation_count,
)
from fastsafetensors.allocation import SharedDeviceAllocation
from fastsafetensors.st_types import Device


class _FakeGbuf:
    def __init__(self, addr: int, length: int):
        self._addr = addr
        self._length = length

    def get_base_address(self) -> int:
        return self._addr

    def get_length(self) -> int:
        return self._length


class _RecordingFramework:
    def __init__(self):
        self.freed = []

    def free_tensor_memory(self, gbuf, device):
        self.freed.append(gbuf)


def test_allocation_frees_exactly_once_after_last_reference():
    fw = _RecordingFramework()
    gbuf = _FakeGbuf(0x1000, 128)
    alloc = SharedDeviceAllocation(gbuf, fw, Device.from_str("cpu"))
    assert alloc.refcount() == 1 and alloc.live

    alloc.acquire()  # simulate an exported tensor
    alloc.acquire()  # and another
    assert alloc.refcount() == 3

    alloc.release()  # one tensor gone
    alloc.release()  # buffer-side (factory) gone
    assert fw.freed == [] and alloc.live  # last tensor still holds it

    alloc.release()  # final reference
    assert fw.freed == [gbuf] and not alloc.live

    # Idempotent past zero: extra releases must not double-free.
    alloc.release()
    alloc.release()
    assert fw.freed == [gbuf]


def test_allocation_acquire_after_free_raises():
    fw = _RecordingFramework()
    alloc = SharedDeviceAllocation(_FakeGbuf(0x2000, 64), fw, Device.from_str("cpu"))
    alloc.release()
    assert not alloc.live
    with pytest.raises(RuntimeError):
        alloc.acquire()


def test_allocation_non_owning_never_frees():
    fw = _RecordingFramework()
    alloc = SharedDeviceAllocation(
        _FakeGbuf(0x3000, 64), fw, Device.from_str("cpu"), owns_memory=False
    )
    alloc.release()
    assert fw.freed == []  # DummyDeviceBuffer-style: nothing to free


@pytest.mark.parametrize("overlap", [False, True])
def test_tensor_view_and_clone_lifetimes(open_tensor_buffer, framework, overlap):
    before_memory = framework.get_mem_used()
    before_count, before_bytes = live_allocation_count(), live_allocation_bytes()
    fb = open_tensor_buffer(allow_inflight=overlap)
    key = list(fb.key_to_rank_lidx)[0]
    tensor = fb.get_tensor_wrapped(key)
    again = fb.get_tensor_wrapped(key)
    assert framework.is_equal(tensor, again.get_raw())
    assert key in list(fb.key_to_rank_lidx)
    independent = tensor.clone()
    view = tensor.get_raw().reshape([-1])
    expected = independent.get_raw().reshape([-1])
    allocation_bytes = fb.rank_loaders[0][0].gbuf.get_length()
    assert live_allocation_count() == before_count + 1
    assert live_allocation_bytes() == before_bytes + allocation_bytes

    fb.close()
    fb.close()
    assert framework.is_equal(tensor, independent.get_raw())
    del tensor, again
    gc.collect()
    assert framework.is_equal(framework.wrap_tensor(view), expected)
    assert live_allocation_count() == before_count + 1
    assert live_allocation_bytes() == before_bytes + allocation_bytes

    del view
    gc.collect()
    assert framework.get_mem_used() == before_memory
    assert live_allocation_count() == before_count
    assert live_allocation_bytes() == before_bytes
    # The independent clone does not retain the loader allocation.
    assert framework.is_equal(
        independent, expected.reshape(independent.get_raw().shape)
    )


@pytest.fixture
def mixed_tensors(tmp_path, framework):
    if framework.get_name() != "pytorch":
        pytest.skip("shared typed storage checks use PyTorch")
    import torch
    from safetensors.torch import save_file

    tensors = {
        "bytes": torch.arange(1024, dtype=torch.int32).to(torch.uint8),
        "bf16": torch.arange(512, dtype=torch.float32).to(torch.bfloat16),
    }
    path = tmp_path / "mixed.safetensors"
    save_file(tensors, str(path))
    return str(path), tensors


@pytest.mark.parametrize(
    "device,borrowed,overlap",
    [
        ("cpu", False, False),
        ("cuda:0", False, True),
        ("cpu", True, True),
        ("cuda:0", True, False),
    ],
)
def test_retained_small_view_pins_only_owning_allocation(
    mixed_tensors, device, borrowed, overlap
):
    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    path, expected = mixed_tensors
    before_count, before_bytes = live_allocation_count(), live_allocation_bytes()
    loader = SafeTensorsFileLoader(None, device, nogds=True)
    loader.add_filenames({0: [path]})
    fb = loader.copy_files_to_device(borrowed_tensors=borrowed, allow_inflight=overlap)
    factory = fb.rank_loaders[0][0]
    allocation_bytes = factory.gbuf.get_length()
    assert allocation_bytes > 16
    assert live_allocation_count() == before_count + 1
    assert live_allocation_bytes() == before_bytes + allocation_bytes
    acquire = fb.get_tensor
    byte_tensor, bf16_tensor = acquire("bytes"), acquire("bf16")
    # Finish CUDA reads of borrowed storage before closing the buffer.
    assert torch.equal(byte_tensor.cpu(), expected["bytes"])
    assert torch.equal(bf16_tensor.cpu(), expected["bf16"])
    view = byte_tensor[:16]
    fb.close()
    loader.close()
    del byte_tensor, bf16_tensor, acquire
    gc.collect()
    if borrowed:
        # Keep the Python view alive to prove it holds no allocation owner.
        # Never dereference its released data.
        assert live_allocation_count() == before_count
        assert live_allocation_bytes() == before_bytes
    else:
        assert torch.equal(view.cpu(), expected["bytes"][:16])
        assert live_allocation_count() == before_count + 1
        # A 16-byte view retains the entire original allocation.
        assert live_allocation_bytes() == before_bytes + allocation_bytes
    del view
    gc.collect()
    assert live_allocation_count() == before_count
    assert live_allocation_bytes() == before_bytes


def test_ownership_mode_is_selected_per_copy(mixed_tensors):
    import torch

    path, expected = mixed_tensors
    before = live_allocation_count()
    loader = SafeTensorsFileLoader(None, "cpu", nogds=True)
    loader.add_filenames({0: [path]})
    held = []
    for borrowed in [False, True]:
        fb = loader.copy_files_to_device(borrowed_tensors=borrowed)
        tensor = fb.get_tensor("bytes")
        assert torch.equal(tensor, expected["bytes"])
        if not borrowed:
            held.append(tensor)
        fb.close()
        del tensor
        assert live_allocation_count() == before + len(held)
    loader.close()
    assert torch.equal(held[0], expected["bytes"])
    held.clear()
    gc.collect()
    assert live_allocation_count() == before
