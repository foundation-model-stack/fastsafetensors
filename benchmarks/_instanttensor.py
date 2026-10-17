# SPDX-License-Identifier: Apache-2.0
"""InstantTensor adapter matching fst-sweep's vLLM reader (copy=True)."""

import importlib.metadata
import sys
from contextlib import closing


def require_instanttensor(device):
    import torch

    if sys.platform != "linux" or device.type != "cuda" or torch.version.hip:
        raise RuntimeError("instanttensor requires Linux and NVIDIA CUDA")
    try:
        import instanttensor
    except ImportError as error:
        raise RuntimeError(
            "Install instanttensor with the benchmark and instanttensor extras"
        ) from error
    return instanttensor


def instanttensor_iterator(files, pg, device, settings):
    from benchmarks._common import synchronize

    instanttensor = require_instanttensor(device)
    # vLLM passes every checkpoint and does not apply its EP tensor filter to
    # this reader. Consumers ignore unowned experts after receiving them.
    with instanttensor.safe_open(
        [str(path) for path in files],
        framework="pt",
        device=device,
        process_group=pg,
        copy=True,
    ) as loader:
        backend = getattr(loader, "backend", None)
        settings.append(
            dict(
                version=importlib.metadata.version("instanttensor"),
                copy=True,
                process_group=pg is not None,
                backend=getattr(backend, "name", None),
                **{
                    name: getattr(loader, name, None)
                    for name in (
                        "buffer_size",
                        "chunk_size",
                        "concurrency",
                        "io_depth",
                        "total_tensor_size",
                    )
                },
            )
        )
        with closing(loader.tensors()) as iterator:
            try:
                for name, tensor in iterator:
                    yield name, tensor
            finally:
                synchronize(device)
