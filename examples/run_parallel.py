#!/usr/bin/env python3

# cd examples
# torchrun --nnodes=2 --master_addr=0.0.0.0 --master_port=1234 --node_rank=0 run_parallel.py &
# PIDS+=$($!)
# torchrun --nnodes=2 --master_addr=0.0.0.0 --master_port=1234 --node_rank=1 run_parallel.py &
# PIDS+=$($!)
# wait ${PIDS[@]}

import torch
import torch.distributed as dist
from fastsafetensors import SafeTensorsFileLoader
dist.init_process_group(backend="gloo")
dist.barrier()
pg = dist.group.WORLD
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loader = SafeTensorsFileLoader(pg, device, nogds=False, debug_log=True)
loader.add_filenames({0: ["a.safetensors"], 1:["b.safetensors"]}) # {rank: files}

# load a.safetensors to rank 0 GPU and b.safetensors to rank 1 GPU
fb = loader.copy_files_to_device()

# every rank must call get_tensor and get_sharded in the same order since they internally call torch.distributed collective ops
tensor_a0 = fb.get_tensor(tensor_name="a0") # broadcast
tensor_b0_sharded = fb.get_sharded(tensor_name="b0", dim=1) # partition and scatter
print(f"RANK {pg.rank()}: tensor_a0={tensor_a0}")
print(f"RANK {pg.rank()}: tensor_b0_sharded={tensor_b0_sharded}")
fb.close()
loader.close()
