def gen_torch():
    import torch
    from safetensors.torch import save_file

    t0 = torch.concat(
        [torch.full((1, 8), i, dtype=torch.float16) for i in range(0, 16)], dim=0
    )

    for file_prefix in ["a", "b"]:
        save_file(
            {f"{file_prefix}0": t0},
            f"{file_prefix}.safetensors",
            metadata={"fst": "sample"},
        )


def gen_paddle():
    import paddle
    from safetensors.paddle import save_file

    t0 = paddle.concat(
        [paddle.full((1, 8), i, dtype=paddle.float16) for i in range(0, 16)], axis=0
    )

    for file_prefix in ["a", "b"]:
        save_file(
            {f"{file_prefix}0": t0},
            f"{file_prefix}_paddle.safetensors",
            metadata={"fst": "sample"},
        )


gens = {
    "torch": gen_torch,
    "paddle": gen_paddle,
}

if __name__ == "__main__":
    import sys

    framework = "torch"
    if len(sys.argv) > 1:
        framework = sys.argv[1]
    gens[framework]()
