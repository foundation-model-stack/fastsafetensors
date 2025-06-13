import os
from typing import List

import pytest

from fastsafetensors import cpp as fstcpp
from fastsafetensors.frameworks import FRAMEWORK, init_framework_op
from fastsafetensors.st_types import Device

init_framework_op(os.getenv("TEST_FASTSAFETENSORS_FRAMEWORK", "please set"))

TESTS_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.dirname(os.path.dirname(TESTS_DIR))
DATA_DIR = os.path.join(REPO_ROOT, ".testdata")
TF_DIR = os.path.join(DATA_DIR, "transformers_cache")
TMP_DIR = os.path.join(DATA_DIR, "tmp")
os.makedirs(TF_DIR, 0o777, True)
os.makedirs(TMP_DIR, 0o777, True)


@pytest.fixture(scope="session", autouse=True)
def input_files() -> List[str]:
    os.environ["HF_HOME"] = TF_DIR
    os.environ["HUGGINGFACE_HUB_CACHE"] = TF_DIR
    from transformers import AutoModelForCausalLM, AutoTokenizer

    AutoModelForCausalLM.from_pretrained("gpt2")
    AutoTokenizer.from_pretrained("gpt2")
    src_files = []
    for dir, _, files in os.walk(os.path.join(TF_DIR, "models--gpt2")):
        for filename in files:
            if filename.endswith(".safetensors"):
                src_files.append(f"{dir}/{filename}")
    return src_files


@pytest.fixture(scope="session", autouse=True)
def pg():
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    if world_size > 1:
        if FRAMEWORK.get_name() == "pytorch":
            import torch.distributed as dist

            dist.init_process_group(backend="gloo")
            dist.barrier()
            return FRAMEWORK.get_process_group(dist.group.WORLD)
        elif FRAMEWORK.get_name() == "paddle":
            # The following code can only be successfully
            # executed by running the code using
            # `python -m paddle.distributed.launch`
            import paddle
            import paddle.distributed as dist

            dist.init_parallel_env()
            backend = "nccl" if paddle.device.cuda.device_count() else "gloo"
            FRAMEWORK.get_process_group(
                dist.new_group(ranks=list(range(world_size)), backend=backend)
            )
    return FRAMEWORK.get_process_group(None)


@pytest.fixture(scope="session", autouse=True)
def dev_init() -> None:
    if fstcpp.is_cuda_found():
        dev_str = "cuda:0" if FRAMEWORK.get_name() == "pytorch" else "gpu:0"
    else:
        dev_str = "cpu"
    FRAMEWORK.set_device(Device.from_str(dev_str))


@pytest.fixture(scope="function")
def fstcpp_log() -> None:
    fstcpp.set_debug_log(True)


@pytest.fixture(scope="function")
def tmp_dir() -> str:
    return TMP_DIR
