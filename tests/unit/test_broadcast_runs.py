# SPDX-License-Identifier: Apache-2.0

import json
import os
import struct
from contextlib import closing

import pytest

from fastsafetensors import ParallelLoader, cpp
from fastsafetensors.common import is_gpu_found
from fastsafetensors.dlpack import from_cuda_buffer
from fastsafetensors.frameworks._torch import TorchOp, TorchTensor
from fastsafetensors.st_types import Device, DType


def test_bounded_dlpack_source_run():
    import torch

    device = Device.from_str("cpu")
    backing = torch.arange(16, dtype=torch.uint8)
    parts = [
        TorchTensor(
            device,
            DType.U8,
            torch.from_dlpack(
                from_cuda_buffer(
                    backing.data_ptr() + offset, [8], [1], DType.U8, device
                )
            ),
        )
        for offset in (0, 8)
    ]
    assert all(t.get_raw().untyped_storage().nbytes() == 8 for t in parts)
    actual = TorchOp._flat_source_run(parts, [8, 8]).clone()
    backing.zero_()
    assert torch.equal(actual, torch.arange(16, dtype=torch.uint8))
    with pytest.raises(ValueError, match="contiguous"):
        TorchOp._flat_source_run(parts[::-1], [8, 8])


def _checkpoint(path, prefix):
    """Mixed dtypes with an unaligned span, an empty tensor and a scalar."""
    import torch

    values = {
        prefix + "a": torch.tensor([1], dtype=torch.uint8),
        prefix + "b": torch.arange(4, dtype=torch.float32),
        prefix + "c": torch.arange(5, dtype=torch.bfloat16),
        prefix + "d": torch.arange(3, dtype=torch.int32),
        prefix + "e": torch.empty(0, dtype=torch.float32),
        prefix + "f": torch.tensor(42, dtype=torch.int64),
        prefix + "g": torch.tensor([9, 10], dtype=torch.uint8),
    }
    # The reference per-tensor Gloo path does not support FP8. NCCL tests
    # exercise it (and compare exact bits) through both delivery paths.
    if torch.cuda.is_available() and hasattr(torch, "float8_e4m3fn"):
        values[prefix + "h"] = torch.arange(8).to(torch.float8_e4m3fn)
    if hasattr(torch, "float4_e2m1fn_x2"):
        values[prefix + "i"] = torch.tensor([18, 52], dtype=torch.uint8).view(
            torch.float4_e2m1fn_x2
        )
    offset = 0
    header = {}
    payload = bytearray()
    for name, tensor in values.items():
        wrapped = TorchOp().wrap_tensor(tensor)
        nbytes = wrapped.get_nbytes()
        shape = list(tensor.shape)
        if wrapped.dtype == DType.F4:
            shape[-1] *= 2
        header[name] = dict(
            dtype=wrapped.dtype.value,
            shape=shape,
            data_offsets=[offset, offset + nbytes],
        )
        payload.extend(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        offset += nbytes
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    temporary = path + ".tmp"
    with open(temporary, "wb") as output:
        output.write(struct.pack("<Q", len(encoded)) + encoded + payload)
    os.replace(temporary, path)
    return values


@pytest.mark.parametrize("chunked", [True, False])
def test_pipeline_runs_own_storage(pg, framework, tmp_dir, chunked):
    if framework.get_name() != "pytorch":
        pytest.skip("PyTorch broadcast optimization")
    import torch
    import torch.distributed as dist

    group = framework.get_process_group(pg)
    device = f"cuda:{group.rank()}" if is_gpu_found() else "cpu"
    paths = [os.path.join(tmp_dir, f"broadcast-runs-{i}.safetensors") for i in range(2)]
    if group.rank() == 0:
        for i, path in enumerate(paths):
            _checkpoint(path, f"s{i}.")
    if group.size() > 1:
        dist.barrier()
    from safetensors.torch import load_file

    expected = {}
    for path in paths:
        expected.update(load_file(path))
    reference_loader = ParallelLoader(
        pg,
        paths,
        device=device,
        nogds=True,
        use_tqdm_on_load=False,
        broadcast_run_bytes=0,
        max_batch_bytes=32 if chunked else None,
    )
    try:
        with closing(reference_loader.iterate_weights()) as weights:
            expected_order = [name for name, _ in weights]
    finally:
        reference_loader.close()
    observed_runs = []
    op = framework.broadcast_contiguous_run

    def record(*args, **kwargs):
        result = op(*args, **kwargs)
        observed_runs.append(len(args[2]))
        return result

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(framework, "broadcast_contiguous_run", record)
        loader = ParallelLoader(
            pg,
            paths,
            device=device,
            nogds=True,
            use_tqdm_on_load=False,
            broadcast_run_bytes=32,
            broadcast_run_tensors=3,
            max_batch_bytes=32 if chunked else None,
        )
        try:
            with closing(loader.iterate_weights()) as weights:
                actual = dict(weights)
            order = list(actual)
        finally:
            loader.close()
    assert order == expected_order
    # Test after the buffers and iterator have been closed, on every rank.
    for name, tensor in actual.items():
        want = expected[name]
        assert tensor.shape == want.shape and tensor.dtype == want.dtype
        assert torch.equal(
            tensor.cpu().reshape(-1).view(torch.uint8),
            want.reshape(-1).view(torch.uint8),
        ), name
    if group.size() > 1:
        assert observed_runs and max(observed_runs) <= 3
        assert any(n > 1 for n in observed_runs)
    assert framework.get_mem_used() == 0
    assert cpp.get_cpp_metrics().bounce_buffer_bytes == 0


def test_pipeline_runs_early_close(input_files, pg, framework):
    if framework.get_name() != "pytorch":
        pytest.skip("PyTorch stream lifetime")
    import torch

    group = framework.get_process_group(pg)
    device = f"cuda:{group.rank()}" if is_gpu_found() else "cpu"
    loader = ParallelLoader(
        pg, input_files, device=device, nogds=True, use_tqdm_on_load=False
    )
    iterator = loader.iterate_weights()
    name, first = next(iterator)
    iterator.close()
    loader.close()
    from safetensors.torch import load_file

    expected = load_file(input_files[0])[name]
    assert torch.equal(first.cpu(), expected)
    assert framework.get_mem_used() == 0
    assert cpp.get_cpp_metrics().bounce_buffer_bytes == 0
