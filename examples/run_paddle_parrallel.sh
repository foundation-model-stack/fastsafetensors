# !/usr/bin/env python3
PIDS=()

runner="python -m paddle.distributed.launch"
# runner="torchrun"

cd paddle_case
rm -rf log
# one machine multy gpu (case : 1 machine 2 gpus)
# different to torch script because the paddle distributed use nccl to communicate in gpus
CUDA_VISIBLE_DEVICES=0 ${runner} --nnodes=2 --master=127.0.0.1:8800 --rank=0 run_parallel.py &
PIDS+=($!)
CUDA_VISIBLE_DEVICES=1 ${runner} --nnodes=2 --master=127.0.0.1:8800 --rank=1 run_parallel.py &
PIDS+=($!)
wait "${PIDS[@]}"