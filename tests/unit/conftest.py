# SPDX-License-Identifier: Apache-2.0

import os
from typing import List

import pytest

from fastsafetensors import SingleGroup
from fastsafetensors import cpp as fstcpp
from fastsafetensors.common import is_gpu_found, resolve_runtime_lib_name
from fastsafetensors.cpp import load_library_functions
from fastsafetensors.frameworks import FrameworkOpBase, get_framework_op
from fastsafetensors.st_types import Device

# Add tests directory to path to import platform_utils
TESTS_DIR = os.path.dirname(__file__)
from platform_utils import get_platform_info

REPO_ROOT = os.path.dirname(os.path.dirname(TESTS_DIR))
DATA_DIR = os.path.join(REPO_ROOT, ".testdata")
TF_DIR = os.path.join(DATA_DIR, "transformers_cache")
TMP_DIR = os.path.join(DATA_DIR, "tmp")
GENERATED_DIR = os.path.join(DATA_DIR, "generated")
os.makedirs(TF_DIR, 0o777, True)
os.makedirs(TMP_DIR, 0o777, True)
os.makedirs(GENERATED_DIR, 0o777, True)

load_library_functions(resolve_runtime_lib_name())
FRAMEWORK = get_framework_op(os.getenv("TEST_FASTSAFETENSORS_FRAMEWORK", "please set"))

# Print platform information at test startup
platform_info = get_platform_info()
print("\n" + "=" * 60)
print("Platform Detection:")
print("=" * 60)
for key, value in platform_info.items():
    print(f"  {key}: {value}")
print("=" * 60 + "\n")


@pytest.fixture(scope="session", autouse=True)
def framework() -> FrameworkOpBase:
    return FRAMEWORK


@pytest.fixture(scope="session", autouse=True)
def input_files() -> List[str]:
    if os.environ.get("FASTSAFETENSORS_TEST_USE_HF_GPT2") != "1":
        return [_ensure_tiny_gpt2_safetensors(FRAMEWORK)]

    gpt_dir = os.path.join(TF_DIR, "models--gpt2")
    if not os.path.exists(gpt_dir):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        AutoModelForCausalLM.from_pretrained(
            "gpt2", trust_remote_code=True, use_safetensors=True, cache_dir=TF_DIR
        )
        AutoTokenizer.from_pretrained("gpt2", cache_dir=TF_DIR)
    src_files = []
    for dir, _, files in os.walk(gpt_dir):
        for filename in files:
            if filename.endswith(".safetensors"):
                src_files.append(f"{dir}/{filename}")
                print(src_files[-1])
    return src_files


def _ensure_tiny_gpt2_safetensors(framework: FrameworkOpBase) -> str:
    filename = os.path.join(
        GENERATED_DIR, f"tiny-gpt2-{framework.get_name()}.safetensors"
    )
    if os.path.exists(filename):
        return filename

    tmp_filename = f"{filename}.{os.getpid()}.tmp"
    if framework.get_name() == "pytorch":
        import torch
        from safetensors.torch import save_file

        dtype = torch.float16

        def make_tensor(rows: int, cols: int, offset: int = 0):
            return torch.arange(rows * cols, dtype=dtype).reshape(rows, cols) + offset

        def make_bias(size: int, offset: int = 0):
            return torch.arange(size, dtype=dtype) + offset

    elif framework.get_name() == "paddle":
        import paddle
        from safetensors.paddle import save_file

        dtype = paddle.float32

        def make_tensor(rows: int, cols: int, offset: int = 0):
            return (
                paddle.arange(rows * cols, dtype=dtype).reshape([rows, cols]) + offset
            )

        def make_bias(size: int, offset: int = 0):
            return paddle.arange(size, dtype=dtype) + offset

    else:
        raise Exception(f"Unknown framework: {framework.get_name()}")

    tensors = {}
    for layer in range(12):
        base = layer * 1000
        tensors[f"h.{layer}.mlp.c_proj.weight"] = make_tensor(8, 16, base)
        tensors[f"h.{layer}.mlp.c_fc.weight"] = make_tensor(16, 8, base + 100)
        tensors[f"h.{layer}.attn.c_proj.weight"] = make_tensor(8, 8, base + 200)
        tensors[f"h.{layer}.attn.c_proj.bias"] = make_bias(8, base + 300)

    save_file(tensors, tmp_filename, metadata={"fst": "tiny-gpt2"})
    os.replace(tmp_filename, filename)
    return filename


@pytest.fixture(scope="session", autouse=True)
def pg():
    rank = int(os.environ.get("RANK", "0"))
    if is_gpu_found():
        dev_str = f"cuda:{rank}" if FRAMEWORK.get_name() == "pytorch" else f"gpu:{rank}"
    else:
        dev_str = "cpu"
    FRAMEWORK.set_device(Device.from_str(dev_str))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    backend = "nccl" if is_gpu_found() else "gloo"
    if world_size > 1:
        if FRAMEWORK.get_name() == "pytorch":
            import torch.distributed as dist

            dist.init_process_group(backend=backend)
            dist.barrier()
            return dist.group.WORLD
        elif FRAMEWORK.get_name() == "paddle":
            # The following code can only be successfully
            # executed by running the code using
            # `python -m paddle.distributed.launch`
            import paddle.distributed as dist

            dist.init_parallel_env()
            return dist.new_group(ranks=list(range(world_size)), backend=backend)
    return SingleGroup()


@pytest.fixture(scope="session", autouse=True)
def dev_init() -> None:
    rank = int(os.environ.get("RANK", "0"))
    if is_gpu_found():
        dev_str = f"cuda:{rank}" if FRAMEWORK.get_name() == "pytorch" else f"gpu:{rank}"
    else:
        dev_str = "cpu"
    FRAMEWORK.set_device(Device.from_str(dev_str))


@pytest.fixture(scope="function")
def fstcpp_log() -> None:
    fstcpp.set_debug_log(True)


@pytest.fixture(scope="function")
def tmp_dir() -> str:
    return TMP_DIR
