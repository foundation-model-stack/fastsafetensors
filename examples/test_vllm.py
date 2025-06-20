import os
import sys
from pathlib import Path

import vllm
from vllm.config import LoadFormat


def drop_cache(model_dir: str):
    total = 0
    for f in Path(model_dir).rglob("*"):
        if f.suffix == ".safetensors":
            fd = os.open(f.resolve(), os.O_RDONLY)
            s = os.fstat(fd)
            os.posix_fadvise(fd, 0, s.st_size, os.POSIX_FADV_DONTNEED)
            os.close(fd)
            print(f"DROP_CACHE: {f}, {s.st_size/1024/1024/1024} GiB")
            total += s.st_size
    print(f"total={total/1024/1024/1024}GiB from {model_dir}")


if __name__ == "__main__":
    load_format = LoadFormat.AUTO
    if len(sys.argv) > 1 and sys.argv[1] == "1":
        load_format = LoadFormat.FASTSAFETENSORS
        os.environ["FASTSAFETENSORS_ENABLE_INIT_LOG"] = "1"
        print("export FASTSAFETENSORS_ENABLE_INIT_LOG=1")
    if len(sys.argv) > 2 and sys.argv[2] == "1":
        from transformers.utils import TRANSFORMERS_CACHE

        drop_cache(
            os.path.join(
                TRANSFORMERS_CACHE,
                "models--ibm-granite--granite-3.0-8b-instruct/snapshots",
            )
        )
    _ = vllm.LLM(model="ibm-granite/granite-3.0-8b-instruct", load_format=load_format)
