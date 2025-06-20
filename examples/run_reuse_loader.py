import sys

import torch

from fastsafetensors import SafeTensorsFileLoader, SingleGroup

sys.path.insert(0, "/nvme/manish/repos/fastsafetensors/fastsafetensors")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loader = SafeTensorsFileLoader(SingleGroup(), device, nogds=True, debug_log=True)

loader.add_filenames({0: ["a.safetensors"]})  # {rank: files}
fb = loader.copy_files_to_device()
keys = list(fb.key_to_rank_lidx.keys())
for k in keys:
    t = fb.get_tensor(k)
    print(f" k, shape = {k, t.shape}\n")
fb.close()

loader.reset()  # reset the loader for reusing with different set of files
loader.add_filenames({0: ["b.safetensors"]})  # {rank: files}
fb = loader.copy_files_to_device()
keys = list(fb.key_to_rank_lidx.keys())
for k in keys:
    t = fb.get_tensor(k)
    print(f" k, shape = {k, t.shape}\n")
fb.close()
loader.close()
