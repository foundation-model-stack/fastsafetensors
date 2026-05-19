# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from fastsafetensors import SafeTensorsFileLoader
from fastsafetensors import cpp as fstcpp
from fastsafetensors.common import is_gpu_found
from fastsafetensors.st_types import DType


def test_shuffle(fstcpp_log, input_files, pg, framework):
    print("test_shuffle")
    if framework.get_name() == "pytorch":
        from safetensors.torch import load_file

        rank = pg.rank()
        world_size = pg.size()
        device = f"cuda:{rank}" if is_gpu_found() else "cpu"
    elif framework.get_name() == "paddle":
        from safetensors.paddle import load_file

        rank = pg.process_group.rank()
        world_size = pg.process_group.size()
        device = f"gpu:{rank}" if is_gpu_found() else "cpu"
    else:
        raise Exception(f"Unknown framework: {framework.get_name()}")
    loader = SafeTensorsFileLoader(
        pg=pg, device=device, nogds=True, framework=framework.get_name(), debug_log=True
    )
    loader.add_filenames({0: input_files})
    bufs = loader.copy_files_to_device()
    key_dims = {key: -1 for key in loader.get_keys()}
    for i in range(0, 12):
        key_dims[f"h.{i}.mlp.c_proj.weight"] = 1
        key_dims[f"h.{i}.mlp.c_fc.weight"] = 0
        key_dims[f"h.{i}.attn.c_proj.weight"] = 0
    tensors = bufs.as_dict(key_dims)
    f = load_file(input_files[0])
    origs = {}
    for key in tensors.keys():
        dim = key_dims[key]
        if dim == 0 or dim == 1:
            t = f[key]
            rank_slices = ()
            shape = t.shape
            size = shape[dim]
            block_size = (size + world_size - 1) // world_size
            for i in range(0, len(shape)):
                if i < dim:
                    rank_slices += (slice(None, None, None),)
                elif i == dim:
                    rank_slices += (
                        slice(rank * block_size, (rank + 1) * block_size, 1),
                    )
                    break
            t = t[rank_slices]
            t = t.clone().detach()
        else:
            t = f[key]
        t = t.to(device=device)
        origs[key] = t
        assert framework.is_equal(tensors[key], t)

    bufs.close()
    loader.reset()

    loader.add_filenames({0: input_files})
    bufs = loader.copy_files_to_device()
    if world_size > 1:
        pushed = bufs.push_tensor("h.0.attn.c_proj.bias", 1)
        assert pushed is not None if rank == 1 else pushed is None
        pushed1 = bufs.push_tensor("h.1.attn.c_proj.bias", 0)
        assert pushed1 is not None if rank == 0 else pushed1 is None

        tensors2 = bufs.get_tensor_wrapped("h.0.attn.c_proj.bias")  # cached load
        assert (
            framework.is_equal(tensors2, origs["h.0.attn.c_proj.bias"])
            if rank == 1
            else pushed is None
        )

    actual = bufs.get_multi_cols(
        ["h.0.attn.c_proj.weight", "h.1.attn.c_proj.weight"], dim=0
    )
    if framework.get_name() == "pytorch":
        import torch

        exp = torch.concat(
            [origs["h.0.attn.c_proj.weight"], origs["h.1.attn.c_proj.weight"]], dim=0
        )
        assert framework.is_equal(actual, exp)
    elif framework.get_name() == "paddle":
        import paddle

        exp = paddle.concat(
            [origs["h.0.attn.c_proj.weight"], origs["h.1.attn.c_proj.weight"]], axis=0
        )
        assert framework.is_equal(actual, exp)

    actual = bufs.get_multi_cols(
        ["h.0.mlp.c_proj.weight", "h.1.mlp.c_proj.weight"], dim=1
    )
    if framework.get_name() == "pytorch":
        exp = torch.concat(
            [origs["h.0.mlp.c_proj.weight"], origs["h.1.mlp.c_proj.weight"]], dim=1
        )
        assert framework.is_equal(actual, exp)
    elif framework.get_name() == "paddle":
        exp = paddle.concat(
            [origs["h.0.mlp.c_proj.weight"], origs["h.1.mlp.c_proj.weight"]], axis=1
        )
        assert framework.is_equal(actual, exp)

    bufs.close()
    loader.close()
    assert framework.get_mem_used() == 0
    assert fstcpp.get_cpp_metrics().bounce_buffer_bytes == 0


def test_float4_shuffle_last_dim(fstcpp_log, tmp_dir, pg, framework):
    if framework.get_name() != "pytorch":
        pytest.skip("F4 is only available in PyTorch")
        return

    import torch
    import torch.distributed as dist
    from safetensors.torch import load_file, save_file

    if not hasattr(torch, "float4_e2m1fn_x2"):
        pytest.skip("torch.float4_e2m1fn_x2 requires PyTorch 2.10+")
        return

    rank = pg.rank()
    world_size = pg.size()
    device = f"cuda:{rank}" if is_gpu_found() else "cpu"
    filename = os.path.join(tmp_dir, "multi_f4.safetensors")

    if rank == 0:
        # Native PyTorch shape [8, 16] is safetensors logical shape [8, 32].
        a_u8 = torch.arange(128, dtype=torch.uint8).reshape(8, 16)
        b_u8 = (torch.arange(128, dtype=torch.uint8) + 128).reshape(8, 16)
        save_file(
            {
                "f4_a": a_u8.view(torch.float4_e2m1fn_x2),
                "f4_b": b_u8.view(torch.float4_e2m1fn_x2),
            },
            filename,
            {"fst": "sample"},
        )
    if world_size > 1:
        dist.barrier(group=pg)

    loader = SafeTensorsFileLoader(
        pg=pg, device=device, nogds=True, framework=framework.get_name(), debug_log=True
    )
    loader.add_filenames({0: [filename]})
    bufs = loader.copy_files_to_device()
    logical_shape = loader.get_shape("f4_a")
    assert logical_shape == [8, 32]

    tensors = bufs.as_dict({"f4_a": 1, "f4_b": 1})
    refs = load_file(filename, device=device)
    block_size = (logical_shape[1] + world_size - 1) // world_size
    rank_slice = (
        slice(None, None, None),
        slice(rank * block_size, (rank + 1) * block_size, 1),
    )
    native_slice = framework.get_native_slices(DType.F4, logical_shape, rank_slice)

    assert framework.is_equal(tensors["f4_a"], refs["f4_a"][native_slice])
    assert framework.is_equal(tensors["f4_b"], refs["f4_b"][native_slice])

    bufs.close()
    loader.reset()

    loader.add_filenames({0: [filename]})
    bufs = loader.copy_files_to_device()
    actual = bufs.get_multi_cols(["f4_a", "f4_b"], dim=1)
    expected_u8 = torch.cat(
        [
            refs["f4_a"][native_slice].view(torch.uint8),
            refs["f4_b"][native_slice].view(torch.uint8),
        ],
        dim=1,
    )
    expected = expected_u8.view(torch.float4_e2m1fn_x2)
    assert framework.is_equal(actual, expected)

    bufs.close()
    loader.close()
    assert framework.get_mem_used() == 0
    assert fstcpp.get_cpp_metrics().bounce_buffer_bytes == 0


def test_float4_float8_e8m0_collectives(fstcpp_log, tmp_dir, pg, framework):
    if framework.get_name() != "pytorch":
        pytest.skip("F4 and F8_E8M0 are only available in PyTorch")
        return

    import torch
    import torch.distributed as dist
    from safetensors.torch import load_file, save_file

    if not hasattr(torch, "float4_e2m1fn_x2"):
        pytest.skip("torch.float4_e2m1fn_x2 requires PyTorch 2.10+")
        return
    if not hasattr(torch, "float8_e8m0fnu"):
        pytest.skip("torch.float8_e8m0fnu requires PyTorch 2.5+")
        return

    rank = pg.rank()
    world_size = pg.size()
    device = f"cuda:{rank}" if is_gpu_found() else "cpu"
    filename = os.path.join(tmp_dir, "multi_collectives_f4_f8e8m0.safetensors")

    if rank == 0:
        f4_u8 = torch.arange(128, dtype=torch.uint8).reshape(8, 16)
        f8_u8 = torch.arange(128, 192, dtype=torch.uint8).reshape(8, 8)
        save_file(
            {
                "f4": f4_u8.view(torch.float4_e2m1fn_x2),
                "f8_e8m0": f8_u8.view(torch.float8_e8m0fnu),
            },
            filename,
            {"fst": "sample"},
        )
    if world_size > 1:
        dist.barrier(group=pg)

    refs = load_file(filename, device=device)

    loader = SafeTensorsFileLoader(
        pg=pg, device=device, nogds=True, framework=framework.get_name(), debug_log=True
    )
    loader.add_filenames({0: [filename]})
    bufs = loader.copy_files_to_device()
    assert framework.is_equal(bufs.get_tensor_wrapped("f4"), refs["f4"])
    assert framework.is_equal(bufs.get_tensor_wrapped("f8_e8m0"), refs["f8_e8m0"])
    bufs.close()
    loader.reset()

    loader.add_filenames({0: [filename]})
    bufs = loader.copy_files_to_device()
    tensors = bufs.as_dict({"f4": 1, "f8_e8m0": 1})
    f4_logical_shape = loader.get_shape("f4")
    f4_block_size = (f4_logical_shape[1] + world_size - 1) // world_size
    f4_logical_slice = (
        slice(None, None, None),
        slice(rank * f4_block_size, (rank + 1) * f4_block_size, 1),
    )
    f4_native_slice = framework.get_native_slices(
        DType.F4, f4_logical_shape, f4_logical_slice
    )
    f8_block_size = (refs["f8_e8m0"].shape[1] + world_size - 1) // world_size
    f8_slice = (
        slice(None, None, None),
        slice(rank * f8_block_size, (rank + 1) * f8_block_size, 1),
    )
    assert framework.is_equal(tensors["f4"], refs["f4"][f4_native_slice])
    assert framework.is_equal(tensors["f8_e8m0"], refs["f8_e8m0"][f8_slice])
    bufs.close()
    loader.reset()

    loader.add_filenames({0: [filename]})
    bufs = loader.copy_files_to_device()
    if world_size > 1:
        pushed_f4 = bufs.push_tensor("f4", 1)
        pushed_f8 = bufs.push_tensor("f8_e8m0", 1)
        if rank == 1:
            assert pushed_f4 is not None
            assert pushed_f8 is not None
            assert bool(
                torch.equal(pushed_f4.view(torch.uint8), refs["f4"].view(torch.uint8))
            )
            assert bool(
                torch.equal(
                    pushed_f8.view(torch.uint8), refs["f8_e8m0"].view(torch.uint8)
                )
            )
        else:
            assert pushed_f4 is None
            assert pushed_f8 is None
    bufs.close()
    loader.close()

    assert framework.get_mem_used() == 0
    assert fstcpp.get_cpp_metrics().bounce_buffer_bytes == 0


if __name__ == "__main__":
    import os
    import sys

    os.environ["PADDLE_DISTRI_BACKEND"] = "nccl" if is_gpu_found() else "gloo"
    sys.exit(pytest.main(sys.argv[1:]))
