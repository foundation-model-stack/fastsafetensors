import torch
import torch.distributed as dist
from fastsafetensors import SafeTensorsFileLoader
dist.init_process_group(backend="gloo")
dist.barrier()
pg = dist.group.WORLD
loader = SafeTensorsFileLoader(pg, torch.device("cpu"), nogds=True, debug_log=True)
loader.add_filenames({0: ["a.safetensors"], 1:["b.safetensors"]}) # {rank: files}
fb = loader.copy_files_to_device()
tensor_a0 = fb.get_sharded(tensor_name="a0", dim=0)
tensor_b0 = fb.get_sharded(tensor_name="b0", dim=1)
print(f"RANK {pg.rank()}: tensor_name={tensor_a0}")
print(f"RANK {pg.rank()}: tensor_name={tensor_b0}")
loader.close()