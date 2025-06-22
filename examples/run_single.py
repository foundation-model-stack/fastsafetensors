#!/usr/bin/env python3


def run_torch():
    import torch

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return device


def run_paddle():
    import paddle

    device = "gpu" if paddle.device.cuda.device_count() else "cpu"
    return device


runs = {
    "torch": run_torch,
    "paddle": run_paddle,
}

if __name__ == "__main__":
    import sys

    from fastsafetensors import cpp as fstcpp
    from fastsafetensors import fastsafe_open

    framework = "torch"
    if len(sys.argv) > 1:
        framework = sys.argv[1]

    device = runs[framework]()
    with fastsafe_open(
        ["a.safetensors", "b.safetensors"],
        device=device,
        nogds=not fstcpp.is_cufile_found(),
        framework=framework,
    ) as f:
        print(f"a0: {f.get_tensor(name='a0')}")
        print(f"b0: {f.get_tensor(name='b0')}")
