# Copyright 2024- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0

import pytest

from fastsafetensors import SafeTensorsFileLoader
from fastsafetensors import cpp as fstcpp


def test_shuffle(fstcpp_log, input_files, pg, framework):
    print("test_shuffle")
    if framework.get_name() == "pytorch":
        from safetensors.torch import load_file

        rank = pg.rank()
        world_size = pg.size()
        device = "cuda:0" if fstcpp.is_cuda_found() else "cpu"
    elif framework.get_name() == "paddle":
        from safetensors.paddle import load_file

        rank = pg.process_group.rank()
        world_size = pg.process_group.size()
        device = "gpu:0" if fstcpp.is_cuda_found() else "cpu"
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


if __name__ == "__main__":
    import os
    import sys

    os.environ["PADDLE_DISTRI_BACKEND"] = "gloo"
    sys.exit(pytest.main(sys.argv[1:]))
