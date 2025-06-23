import os
import sys
from typing import Dict, List

from safetensors.torch import load_file

from fastsafetensors import SafeTensorsFileLoader, SingleGroup

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("specify a directory containing safetensors files")
        sys.exit(1)
    loader = SafeTensorsFileLoader(SingleGroup(), device="cpu", nogds=True)
    input_file_or_dir = sys.argv[1]
    src_files: Dict[int, List[str]] = {0: []}
    orig_keys = {}
    if os.path.isdir(input_file_or_dir):
        for dir, _, files in os.walk(input_file_or_dir):
            for filename in files:
                if filename.endswith(".safetensors"):
                    src_files[0].append(f"{dir}/{filename}")
    elif os.path.exists(input_file_or_dir) and input_file_or_dir.endswith(
        ".safetensors"
    ):
        src_files[0].append(input_file_or_dir)
        orig_keys = load_file(input_file_or_dir)
    loader.add_filenames(src_files)
    fb = loader.copy_files_to_device()
    if len(orig_keys) > 0:
        for key in loader.get_keys():
            print(
                f'"{key}",{loader.get_shape(key)},{loader.frames[key].data_offsets},{fb.get_tensor(key).dtype},{orig_keys[key].dtype}'
            )
    else:
        for key in loader.get_keys():
            print(
                f'"{key}",{loader.get_shape(key)},{loader.frames[key].data_offsets},{fb.get_tensor(key).dtype}'
            )
