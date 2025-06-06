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
# pg = dist.new_group(ranks=[0,1], backend=backend)
# device = f"gpu:{pg.process_group.rank()}" if paddle.device.cuda.device_count() else "cpu"
pg = dist.get_group()
world_size = dist.get_world_size()
device = f"gpu:{pg.process_group.rank()}" if paddle.is_compiled_with_cuda() else "cpu"
def weight():
    loader = SafeTensorsFileLoader(pg, device, nogds=True, debug_log=True, framework="paddle")
    loader.add_filenames({0: ["large_modela0.safetensors"],1: ["large_modela1.safetensors"]}) # {rank: files}

    # load a.safetensors to rank 0 GPU and b.safetensors to rank 1 GPU
    try:
        fb = loader.copy_files_to_device()
        try:
            keys = list(fb.key_to_rank_lidx.keys())
            for k in keys:
            # every rank must call get_tensor and get_sharded in the same order since they internally call paddle.distributed collective ops
                tensor_a0 = fb.get_tensor(tensor_name=k) # broadcast
                yield k,tensor_a0
        finally:
            fb.close()
    finally:
        loader.close()

test=weight()


def slice_concat_by_axis(weight, fuse_tensor_parts, tensor_parallel_degree, tensor_parallel_rank, axis=0):
    total_splits = fuse_tensor_parts * tensor_parallel_degree
    dim_size = weight.shape[axis]
    split_size = dim_size // total_splits

    slices = []
    for idx in range(tensor_parallel_rank, total_splits, tensor_parallel_degree):
        start = idx * split_size
        # 最后一块切片到末尾，防止除不尽
        end = (start + split_size) if (idx != total_splits - 1) else dim_size

        # 构造切片索引，axis 维度切片，其它维度用 slice(None)
        slice_idx = [slice(None)] * len(weight.shape)
        slice_idx[axis] = slice(start, end)

        block = weight[tuple(slice_idx)]
        slices.append(block)

    result = paddle.concat(slices, axis=axis)
    return result

for name,weight in test:
    print(f"RANK {pg.process_group.rank()}: {name}={weight}")
    # tensor_b0_sharded = fb.get_sharded(tensor_name="b0", dim=1) # partition and scatter
    size=weight.shape[-1]//world_size
    print(f"RANK {pg.process_group.rank()}: {name}_test={slice_concat_by_axis(weight,2,world_size,pg.process_group.rank(),axis=-1)}")