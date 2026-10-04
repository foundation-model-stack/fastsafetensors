# SPDX-License-Identifier: Apache-2.0

from contextlib import closing
from unittest.mock import MagicMock

import pytest

from fastsafetensors import (
    ParallelLoader,
    SingleGroup,
    live_allocation_bytes,
    live_allocation_count,
)
from fastsafetensors.file_buffer import FilesBufferOnDevice
from fastsafetensors.loader import BaseSafeTensorsFileLoader
from fastsafetensors.parallel_loader import FileBatch, PipelineParallel


@pytest.mark.parametrize("pipeline_size,loader_size", [(2, 2), (1, 2), (2, 1)])
def test_borrowed_tensors_reject_broadcast_groups(pipeline_size, loader_size):
    pg = MagicMock()
    pg.size.return_value = pipeline_size
    loader = MagicMock()
    loader.pg.size.return_value = loader_size
    with pytest.raises(ValueError, match="single-process loader group"):
        PipelineParallel(
            pg, loader, [], accumulate_resident=False, borrowed_tensors=True
        )


def test_borrowed_tensors_reject_resident_accumulation():
    loader = MagicMock()
    loader.pg = SingleGroup()
    with pytest.raises(ValueError, match="accumulate_resident=False"):
        PipelineParallel(None, loader, [], borrowed_tensors=True)


def test_borrowed_tensors_allow_all_local_loading(input_files, framework):
    # A distributed application may opt into independent per-rank loading.
    pg = MagicMock()
    pg.size.return_value = 2
    loader = ParallelLoader(
        pg,
        input_files,
        device="cpu",
        framework=framework.get_name(),
        nogds=True,
        all_local=True,
        accumulate_resident=False,
        borrowed_tensors=True,
    )
    try:
        assert loader.loader.pg.size() == 1
        assert not loader.need_clone
    finally:
        loader.close()


# Cover blocking/overlapped IO and each queue path without a Cartesian product.
@pytest.mark.parametrize(
    "device,queue_size,early_exit,overlap",
    [
        ("cpu", -1, False, False),
        ("cpu", 0, True, True),
        ("cpu", 2, False, True),
        ("cuda:0", 0, True, True),
        ("cuda:0", 2, False, True),
        ("cuda:0", 2, True, False),
    ],
)
@pytest.mark.parametrize("borrowed", [False, True])
def test_tensor_storage_and_cleanup(
    input_files,
    framework,
    monkeypatch,
    device,
    queue_size,
    borrowed,
    early_exit,
    overlap,
):
    if framework.get_name() != "pytorch":
        pytest.skip("storage identity check uses PyTorch")
    import torch
    from safetensors.torch import load_file

    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    expected = load_file(input_files[0])
    largest = max(t.numel() * t.element_size() for t in expected.values())
    source_ptrs = {}
    get_tensor = FilesBufferOnDevice.get_tensor

    def record_source(self, name, *args, **kwargs):
        tensor = get_tensor(self, name, *args, **kwargs)
        source_ptrs[name] = tensor.data_ptr()
        return tensor

    iter_local = FilesBufferOnDevice.iter_local_tensors

    def record_local(self, names):
        for name, tensor in iter_local(self, names):
            source_ptrs[name] = tensor.data_ptr()
            yield name, tensor

    monkeypatch.setattr(FilesBufferOnDevice, "iter_local_tensors", record_local)
    monkeypatch.setattr(FilesBufferOnDevice, "get_tensor", record_source)
    before = framework.get_mem_used()
    before_count, before_bytes = live_allocation_count(), live_allocation_bytes()
    loader = ParallelLoader(
        None,
        [input_files[0]],
        device=device,
        nogds=True,
        use_tqdm_on_load=False,
        queue_size=queue_size,
        overlap_io=overlap,
        max_batch_bytes=largest,
        accumulate_resident=False,
        borrowed_tensors=borrowed,
    )
    copied = {}
    retained = {}
    try:
        assert len(loader.weight_files_batches) > 1
        with closing(loader.iterate_weights()) as weights:
            for name, tensor in weights:
                assert (tensor.data_ptr() == source_ptrs[name]) is borrowed
                # This blocking copy finishes reading the borrowed storage
                # before advancing the iterator, on both CPU and CUDA.
                copied[name] = tensor.to("cpu", copy=True)
                view = tensor.reshape(-1)[:1]
                if not borrowed:
                    retained[name] = tensor
                if early_exit:
                    break
    finally:
        loader.close()

    assert framework.get_mem_used() == before
    # Retaining the last Python tensor and derived view must not pin any chunk
    # in either mode. Borrowed aliases are deliberately not dereferenced here.
    assert live_allocation_count() == before_count
    assert live_allocation_bytes() == before_bytes
    if not borrowed:
        assert torch.equal(view.cpu(), expected[name].reshape(-1)[:1])
    assert set(copied) == (set(expected) if not early_exit else set(source_ptrs))
    for name, tensor in copied.items():
        assert torch.equal(tensor, expected[name])
    # The default mode's tensors remain valid after batch and iterator cleanup.
    for name, tensor in retained.items():
        assert torch.equal(tensor.cpu(), expected[name])


@pytest.mark.parametrize("borrowed,expected_queue_size", [(False, -1), (True, 0)])
def test_borrowed_tensors_fit_without_yield_clone(
    input_files, framework, borrowed, expected_queue_size
):
    if framework.get_name() != "pytorch":
        pytest.skip("PyTorch tensor comparison")
    import torch
    from safetensors.torch import load_file

    expected = load_file(input_files[0])
    largest = max(t.numel() * t.element_size() for t in expected.values())
    copied = {name: torch.empty_like(tensor) for name, tensor in expected.items()}
    loader = ParallelLoader(
        None,
        [input_files[0]],
        device="cpu",
        nogds=True,
        use_tqdm_on_load=False,
        queue_size=0,
        device_memory_budget=2 * largest,
        accumulate_resident=False,
        borrowed_tensors=borrowed,
    )
    seen = set()
    try:
        # Two chunk buffers fit only if there is no extra yield clone.
        assert loader.queue_size == expected_queue_size
        with closing(loader.iterate_weights()) as weights:
            for name, tensor in weights:
                copied[name].copy_(tensor)
                seen.add(name)
    finally:
        loader.close()
    assert seen == set(expected)
    for name, tensor in copied.items():
        assert torch.equal(tensor, expected[name])


def test_low_level_borrow_rejects_distributed_group_before_io():
    loader = BaseSafeTensorsFileLoader.__new__(BaseSafeTensorsFileLoader)
    loader.pg = MagicMock()
    loader.pg.size.return_value = 2
    with pytest.raises(ValueError, match="single-process loader group"):
        loader.copy_files_to_device(borrowed_tensors=True)


def test_shutdown_closes_unconsumed_queued_buffer(
    open_tensor_buffer, input_files, framework
):
    before = framework.get_mem_used()
    with closing(
        ParallelLoader(
            None,
            input_files,
            device="cpu",
            nogds=True,
            framework=framework.get_name(),
            use_tqdm_on_load=False,
        )
    ) as pipeline:
        fb = open_tensor_buffer()
        assert framework.get_mem_used() > before
        pipeline.batch_queue.put(FileBatch(fb, list(fb.key_to_rank_lidx), 0))
        pipeline._drain_queue()
        assert fb.rank_loaders == {}
        assert framework.get_mem_used() == before
