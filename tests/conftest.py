import os
import pytest
import torch
from fastsafetensors import cpp as fstcpp
from typing import List

TESTS_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.dirname(os.path.dirname(TESTS_DIR))
DATA_DIR = os.path.join(REPO_ROOT, ".testdata")
os.makedirs(DATA_DIR, 0o777, True)

@pytest.fixture(scope='session', autouse=True)
def input_files() -> List[str]:
    os.environ["HF_HOME"] = DATA_DIR
    os.environ["HUGGINGFACE_HUB_CACHE"] = DATA_DIR
    from transformers import AutoModelForCausalLM, AutoTokenizer
    AutoModelForCausalLM.from_pretrained("gpt2")
    AutoTokenizer.from_pretrained("gpt2")
    src_files = []
    for dir, _, files in os.walk(DATA_DIR):
        for filename in files:
            if filename.endswith(".safetensors"):
                src_files.append(f"{dir}/{filename}")
    return src_files

@pytest.fixture(scope='session', autouse=True)
def dev_init() -> None:
    if not fstcpp.is_stub():
        torch.cuda.set_device(0)

@pytest.fixture(scope='function')
def fstcpp_log() -> None:
    fstcpp.set_debug_log(True)
