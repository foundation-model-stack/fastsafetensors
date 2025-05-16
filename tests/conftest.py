import os
import pytest
import torch
import torch.distributed as dist
from fastsafetensors import cpp as fstcpp
from fastsafetensors import SingleGroup
from typing import List

TESTS_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.dirname(os.path.dirname(TESTS_DIR))
DATA_DIR = os.path.join(REPO_ROOT, ".testdata")
TF_DIR = os.path.join(DATA_DIR, "transformers_cache")
TMP_DIR = os.path.join(DATA_DIR, "tmp")
os.makedirs(TF_DIR, 0o777, True)
os.makedirs(TMP_DIR, 0o777, True)

@pytest.fixture(scope='session', autouse=True)
def input_files() -> List[str]:
    os.environ["HF_HOME"] = TF_DIR
    os.environ["HUGGINGFACE_HUB_CACHE"] = TF_DIR
    from transformers import AutoModelForCausalLM, AutoTokenizer
    AutoModelForCausalLM.from_pretrained("gpt2")
    AutoTokenizer.from_pretrained("gpt2")
    src_files = []
    for dir, _, files in os.walk(TF_DIR):
        for filename in files:
            if filename.endswith(".safetensors"):
                src_files.append(f"{dir}/{filename}")
    return src_files

@pytest.fixture(scope='session', autouse=True)
def pg():
    world_size = os.getenv("WORLD_SIZE")
    if world_size is not None and int(world_size) > 1:
        dist.init_process_group(backend="gloo")
        dist.barrier()
        PG = dist.group.WORLD
    else:
        PG = SingleGroup()
    return PG

@pytest.fixture(scope='session', autouse=True)
def dev_init() -> None:
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

@pytest.fixture(scope='function')
def fstcpp_log() -> None:
    fstcpp.set_debug_log(True)

@pytest.fixture(scope='function')
def tmp_dir() -> str:
    return TMP_DIR