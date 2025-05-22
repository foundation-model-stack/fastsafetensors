# !/usr/bin/env python3
# PIDS=()

# runner="python -m paddle.distributed.launch"

# cd paddle_case
# ${runner} --nnodes=2 --master=127.0.0.1:12345 --rank=0 run_parallel.py &
# PIDS+=($!)
# ${runner} --nnodes=2 --master=127.0.0.1:12345 --rank=1 run_parallel.py &
# PIDS+=($!)
# wait "${PIDS[@]}"

import paddle
import paddle.distributed as dist
from fastsafetensors import SafeTensorsFileLoader
dist.init_parallel_env()
backend = "nccl" if paddle.device.cuda.device_count() else "gloo"
pg = dist.new_group(ranks=[0,1], backend=backend)
device = "gpu" if paddle.device.cuda.device_count() else "cpu"
loader = SafeTensorsFileLoader(pg, device, nogds=False, debug_log=True, framework="paddle")
loader.add_filenames({0: ["a_paddle.safetensors"], 1:["b_paddle.safetensors"]}) # {rank: files}

# load a.safetensors to rank 0 GPU and b.safetensors to rank 1 GPU
fb = loader.copy_files_to_device()

# every rank must call get_tensor and get_sharded in the same order since they internally call paddle.distributed collective ops
tensor_a0 = fb.get_tensor(tensor_name="a0") # broadcast
tensor_b0_sharded = fb.get_sharded(tensor_name="b0", dim=1) # partition and scatter
print(f"RANK {pg.process_group.rank()}: tensor_a0={tensor_a0}")
print(f"RANK {pg.process_group.rank()}: tensor_b0_sharded={tensor_b0_sharded}")
fb.close()
loader.close()
