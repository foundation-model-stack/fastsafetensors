# SPDX-License-Identifier: Apache-2.0
"""Save CUDA-resident tensors through the public ParallelSaver API."""

import pytest
from safetensors import safe_open

from fastsafetensors import ParallelSaver

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is not available"
)


@pytest.mark.parametrize("num_threads", [1, 4])
def test_parallel_saver_round_trips_cuda_tensors(tmp_path, num_threads):
    device = torch.device("cuda", torch.cuda.current_device())
    tensors = {
        "float32": torch.linspace(-3, 3, 512 * 1024, device=device)
        .reshape(512, 1024)
        .requires_grad_(),
        "float16": torch.arange(1024 * 64, dtype=torch.float32, device=device)
        .remainder(997)
        .half()
        .reshape(1024, 64),
        "bfloat16_transposed": torch.arange(
            257 * 128, dtype=torch.float32, device=device
        )
        .remainder(31)
        .bfloat16()
        .reshape(257, 128)
        .t(),
        "float64": torch.linspace(-1, 1, 1024, dtype=torch.float64, device=device),
        "int64": torch.arange(129, dtype=torch.int64, device=device),
        "bool": torch.arange(129, device=device).remainder(2) == 0,
        "sliced": torch.arange(513, dtype=torch.int32, device=device)[1::3],
        "scalar": torch.tensor(3, dtype=torch.float16, device=device),
        "empty": torch.empty(0, 3, device=device),
    }
    assert all(tensor.device == device for tensor in tensors.values())
    assert not tensors["bfloat16_transposed"].is_contiguous()
    assert not tensors["sliced"].is_contiguous()
    assert tensors["sliced"].storage_offset() > 0

    saver = ParallelSaver(num_shards=3, num_threads=num_threads)
    paths = saver.save(
        {name: t for name, t in tensors.items()},
        str(tmp_path / "model"),
        metadata={"source": "cuda"},
    )

    assert len(paths) == 3
    restored = {}
    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            assert checkpoint.metadata()["source"] == "cuda"
            for name in checkpoint.keys():
                assert name not in restored
                restored[name] = checkpoint.get_tensor(name)
    assert restored.keys() == tensors.keys()
    for name, tensor in tensors.items():
        assert tensor.device == device
        torch.testing.assert_close(
            restored[name], tensor.detach().cpu(), rtol=0, atol=0
        )
