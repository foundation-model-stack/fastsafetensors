# SPDX-License-Identifier: Apache-2.0

"""Loading on a process group that is a strict subset of the world.

Run with exactly 4 ranks. Collectives take the rank of the source or
destination within the group; on a subgroup that is not the prefix of the
world, a group rank and a global rank differ. The layouts cover both
failure modes: a group rank that names a process outside the group
([0, 3], [2, 3]) and one that names the wrong member ([1, 2]).
"""

import os
from contextlib import closing
from datetime import timedelta

import pytest

from fastsafetensors import SafeTensorsFileLoader
from fastsafetensors.common import is_gpu_found
from fastsafetensors.parallel_loader import PipelineParallel

WORLD_SIZE = 4
LAYOUTS = [[[0, 3], [1, 2]], [[0, 1], [2, 3]]]
TIMEOUT = timedelta(seconds=60)


def _global_rank(framework) -> int:
    if framework.get_name() == "pytorch":
        import torch.distributed as dist

        return dist.get_rank()
    import paddle.distributed as dist

    return dist.get_rank()


def _new_groups(framework, layout):
    """Create every group of the layout on every rank; return this rank's."""
    rank = _global_rank(framework)
    mine = None
    for ranks in layout:
        if framework.get_name() == "pytorch":
            import torch.distributed as dist

            group = dist.new_group(ranks, timeout=TIMEOUT)
        else:
            import paddle.distributed as dist

            group = dist.new_group(ranks, timeout=TIMEOUT)
        if rank in ranks:
            mine = group
    return mine


def _world_barrier(framework) -> None:
    if framework.get_name() == "pytorch":
        import torch.distributed as dist
    else:
        import paddle.distributed as dist
    dist.barrier()


def _load_file(framework, path):
    if framework.get_name() == "pytorch":
        from safetensors.torch import load_file
    else:
        from safetensors.paddle import load_file
    return load_file(path)


def _equal(actual, want) -> bool:
    return bool((actual == want).all())


def _device(framework) -> str:
    if not is_gpu_found():
        return "cpu"
    prefix = "cuda" if framework.get_name() == "pytorch" else "gpu"
    return f"{prefix}:{_global_rank(framework)}"


@pytest.fixture(scope="module", autouse=True)
def require_world(pg, framework):
    if int(os.getenv("WORLD_SIZE", "1")) != WORLD_SIZE:
        pytest.skip(f"needs {WORLD_SIZE} ranks")
    if framework.get_name() == "paddle":
        # Each Paddle gloo group also runs an unprefixed rendezvous in the
        # global store, so creating groups after the default one often fails
        # to connect (PaddlePaddle/Paddle#79814).
        pytest.skip("paddle cannot reliably create additional gloo groups")


@pytest.mark.parametrize("layout", LAYOUTS, ids=["0-3_1-2", "0-1_2-3"])
def test_subgroup_file_loader(input_files, framework, layout):
    """broadcast, scatter and send/recv from group rank 1 on a subgroup."""
    group = _new_groups(framework, layout)
    wrapped = framework.get_process_group(group)
    group_rank = wrapped.rank()
    device = _device(framework)
    expected = _load_file(framework, input_files[0])

    loader = SafeTensorsFileLoader(
        pg=group, device=device, nogds=True, framework=framework.get_name()
    )
    loader.add_filenames({1: input_files})
    bufs = loader.copy_files_to_device()
    try:
        # broadcast
        name = "h.0.attn.c_proj.bias"
        actual = bufs.get_tensor(name)
        assert _equal(actual, expected[name].to(device))

        # scatter
        name = "h.0.mlp.c_fc.weight"
        full = expected[name]
        block = (full.shape[0] + 1) // 2
        actual = bufs.get_sharded(name, 0)
        want = full[group_rank * block : (group_rank + 1) * block]
        assert _equal(actual, want.to(device))

        # send/recv: group rank 1 holds the file and pushes to group rank 0
        name = "h.1.attn.c_proj.bias"
        pushed = bufs.push_tensor(name, 0)
        if group_rank == 0:
            assert pushed is not None
            assert _equal(pushed, expected[name].to(device))
    finally:
        bufs.close()
        loader.close()
        _world_barrier(framework)


@pytest.mark.parametrize("layout", LAYOUTS, ids=["0-3_1-2", "0-1_2-3"])
def test_subgroup_pipeline(input_files, tmp_dir, framework, layout):
    """A two-shard load where group rank 1 owns the second shard."""
    rank = _global_rank(framework)
    expected = _load_file(framework, input_files[0])
    names = sorted(expected)
    shards = [
        os.path.join(tmp_dir, f"subgroup-{framework.get_name()}-{i}.safetensors")
        for i in range(2)
    ]
    if rank == 0:
        if framework.get_name() == "pytorch":
            from safetensors.torch import save_file
        else:
            from safetensors.paddle import save_file
        half = len(names) // 2
        for path, keys in zip(shards, [names[:half], names[half:]]):
            save_file({k: expected[k] for k in keys}, path)
    _world_barrier(framework)

    group = _new_groups(framework, layout)
    wrapped = framework.get_process_group(group)
    device = _device(framework)
    file_loader = SafeTensorsFileLoader(
        group, device=device, nogds=True, framework=framework.get_name()
    )
    loader = PipelineParallel(
        wrapped, file_loader, shards, queue_size=0, use_tqdm_on_load=False
    )
    try:
        with closing(loader.iterate_weights()) as weights:
            actual = dict(weights)
        assert sorted(actual) == names
        for name, tensor in actual.items():
            assert _equal(tensor, expected[name].to(device))
    finally:
        loader.close()
        _world_barrier(framework)


if __name__ == "__main__":
    import sys

    os.environ["PADDLE_DISTRI_BACKEND"] = "nccl" if is_gpu_found() else "gloo"
    sys.exit(pytest.main(sys.argv[1:]))
