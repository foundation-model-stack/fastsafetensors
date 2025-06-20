# Copyright 2024- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0

import pytest
from safetensors import safe_open

from fastsafetensors import SafeTensorsFileLoader
from fastsafetensors import cpp as fstcpp
from fastsafetensors import frameworks


def test_shuffle(fstcpp_log, input_files, pg):
    print("test_shuffle")
    if frameworks.OP.get_name() == "pytorch":
        rank = pg.rank()
        world_size = pg.size()
        device = "cuda:0" if fstcpp.is_cuda_found() else "cpu"
    elif frameworks.OP.get_name() == "paddle":
        rank = pg.process_group.rank()
        world_size = pg.process_group.size()
        device = "gpu:0" if fstcpp.is_cuda_found() else "cpu"
    else:
        raise Exception(f"Unknown framework: {frameworks.OP.get_name()}")
    loader = SafeTensorsFileLoader(device=device, pg=pg, nogds=True, debug_log=True)
    loader.add_filenames({0: input_files})
    bufs = loader.copy_files_to_device()
    key_dims = {key: -1 for key in loader.get_keys()}
    for i in range(0, 12):
        key_dims[f"h.{i}.mlp.c_proj.weight"] = 0
        key_dims[f"h.{i}.mlp.c_fc.weight"] = 1
    tensors = bufs.as_dict(key_dims)
    with safe_open(input_files[0], framework=frameworks.OP.get_name()) as f:
        for key in tensors.keys():
            dim = key_dims[key]
            if dim == 0 or dim == 1:
                t = f.get_slice(key)
                rank_slices = ()
                shape = t.get_shape()
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
                t = f.get_tensor(key)
            assert frameworks.OP.is_equal(tensors[key], t.to(device=device))
    bufs.close()
    loader.close()
