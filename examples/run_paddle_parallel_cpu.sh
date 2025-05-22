# !/usr/bin/env python3
PIDS=()

runner="python -m paddle.distributed.launch"

cd paddle_case
rm -rf log
# It can only be used on the CPU version of paddlepaddle
${runner} --nproc_per_node 2  run_parallel.py