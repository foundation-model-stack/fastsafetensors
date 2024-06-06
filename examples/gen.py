import os
import torch
t0 = torch.concat([torch.full((1,8), i, dtype=torch.float16) for i in range(0, 16)], dim=0)
from safetensors.torch import save_file
for file_prefix in ["a", "b"]:
    save_file({f"{file_prefix}0": t0}, f"{file_prefix}.safetensors", metadata={"fst": "sample"})
