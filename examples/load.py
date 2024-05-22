import sys
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastsafetensors import SafeTensorsFileLoader, SingleGroup
from typer import List

def download(target_dir, model)->List[str]:
    os.environ["HF_HOME"] = target_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = target_dir
    AutoModelForCausalLM.from_pretrained(model)
    AutoTokenizer.from_pretrained(model)
    src_files = []
    for dir, _, files in os.walk(target_dir):
        for filename in files:
            if filename.endswith(".safetensors"):
                src_files.append(f"{dir}/{filename}")
    return src_files

def single(target_dir):
    src_files = download(target_dir, "gpt2")
    loader = SafeTensorsFileLoader(SingleGroup, torch.device("cpu"), nogds=True, debug_log=True)
    loader.add_filenames({0: src_files})
    bufs = loader.copy_files_to_device()
    key_dims = {key: -1 for key in loader.get_keys()}
    tensors = bufs.as_dict(key_dims)
    for key, tensor in tensors:
        print(f"{key}: {tensor}")
    loader.close()

if __name__ == "__main__":
    if len(sys.argv) == 1:
        single("~/.cache/huggingface/hub")
    elif len(sys.argv) == 2:
        single(sys.argv[1])