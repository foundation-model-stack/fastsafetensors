import os
import paddle
t0 = paddle.concat([paddle.full((1,8), i, dtype=paddle.float16) for i in range(0, 16)], dim=0)
from safetensors.paddle import save_file
for file_prefix in ["a", "b"]:
    save_file({f"{file_prefix}0": t0}, f"{file_prefix}_paddle.safetensors", metadata={"fst": "sample"})
