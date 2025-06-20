# Copyright 2024- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0

import paddle
import pytest
from safetensors import safe_open

from fastsafetensors import SafeTensorsFileLoader
from fastsafetensors import frameworks


def test_shuffle_paddle(fstcpp_log, input_files, pg):
    if frameworks.OP.get_name() != "paddle":
        return
    device = "gpu" if paddle.device.cuda.device_count() else "cpu"
    loader = SafeTensorsFileLoader(
        pg, device, nogds=True, debug_log=True, framework="paddle"
    )
    loader.add_filenames({0: input_files})
    bufs = loader.copy_files_to_device()
    key_dims = {key: -1 for key in loader.get_keys()}
    for i in range(0, 12):
        key_dims[f"h.{i}.mlp.c_proj.weight"] = 0
        key_dims[f"h.{i}.mlp.c_fc.weight"] = 1
    tensors = bufs.as_dict(key_dims)
    with safe_open(input_files[0], framework="pt") as f:
        for key in tensors.keys():
            dim = key_dims[key]
            if dim == 0 or dim == 1:
                t = f.get_slice(key)
                rank_slices = ()
                shape = t.get_shape()
                size = shape[dim]
                block_size = (
                    size + pg.process_group.size() - 1
                ) // pg.process_group.size()
                for i in range(0, len(shape)):
                    if i < dim:
                        rank_slices += (slice(None, None, None),)
                    elif i == dim:
                        rank_slices += (
                            slice(
                                pg.process_group.rank() * block_size,
                                (pg.process_group.rank() + 1) * block_size,
                                1,
                            ),
                        )
                        break
                t = t[rank_slices]
                t = t.clone().detach()
            else:
                t = f.get_tensor(key)
            assert paddle.all(
                paddle.to_tensor(t.numpy(), place=loader.device).equal(tensors[key])
            )
    bufs.close()
    loader.close()
