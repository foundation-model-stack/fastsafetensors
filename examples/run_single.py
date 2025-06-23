#!/usr/bin/env python3

if __name__ == "__main__":
    import sys

    from fastsafetensors import cpp as fstcpp
    from fastsafetensors import fastsafe_open

    framework = "torch"
    filenames = ["a.safetensors", "b.safetensors"]
    if len(sys.argv) > 1:
        framework = sys.argv[1]
    if framework == "torch":
        import torch

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif framework == "paddle":
        import paddle

        device = "gpu" if paddle.device.cuda.device_count() else "cpu"
        filenames = ["a_paddle.safetensors", "b_paddle.safetensors"]
    else:
        raise Exception(f"unknown framework: {framework}")

    with fastsafe_open(
        filenames,
        device=device,
        nogds=not fstcpp.is_cufile_found(),
        framework=framework,
    ) as f:
        print(f"a0: {f.get_tensor(name='a0')}")
        print(f"b0: {f.get_tensor(name='b0')}")
