# Copyright 2025 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0

import pytest
import os
import torch
from safetensors.torch import save_file
from fastsafetensors import fastsafe_open
from safetensors import safe_open

def _test_type(tmp_dir, dtype, device):
    filename = os.path.join(tmp_dir, f"a.safetensors")
    t0 = torch.randn((8, 16), dtype=torch.float32).to(dtype=dtype)
    save_file({f"a": t0}, filename, metadata={"fst": "sample"})
    with fastsafe_open(filenames=[filename], nogds=True, device=device, debug_log=True) as f:
        for key in f.get_keys():
            t1 = f.get_tensor(key).clone().detach()
    with safe_open(filename, framework='pt', device=device) as f:
        for key in f.keys():
            t2 = f.get_tensor(key)
    assert torch.all(t2.eq(t1))

def test_float8_e5m2_cpu(fstcpp_log, tmp_dir):
    _test_type(tmp_dir, torch.float8_e5m2, "cpu")

def test_float8_e4m3fn_cpu(fstcpp_log, tmp_dir):
    _test_type(tmp_dir, torch.float8_e4m3fn, "cpu")

def test_float8_e4m3fn_cuda(fstcpp_log, tmp_dir):
    if torch.cuda.is_available:
        _test_type(tmp_dir, torch.float8_e4m3fn, "cuda:0")

def test_float8_e5m2_cuda(fstcpp_log, tmp_dir):
    if torch.cuda.is_available:
        _test_type(tmp_dir, torch.float8_e4m3fn, "cuda:0")