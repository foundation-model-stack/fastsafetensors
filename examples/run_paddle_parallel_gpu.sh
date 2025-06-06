# !/usr/bin/env python3
PIDS=()

runner="python -m paddle.distributed.launch"

cd paddle_case
rm -rf log
# It can only be used on the GPU version of paddlepaddle-gpu
# A machine multy gpu (case : 1 machine 2 gpus)
# Different to torch script because the paddle distributed use nccl to communicate in gpus
CUDA_VISIBLE_DEVICES=0,1 ${runner}  --gpus 0,1 run_parallel.py