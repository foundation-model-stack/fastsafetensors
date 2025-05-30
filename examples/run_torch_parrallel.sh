# !/usr/bin/env python3
PIDS=()

torchrun --nnodes=2 --master_addr=0.0.0.0 --master_port=1234 --node_rank=0 run_parallel.py &
PIDS+=$($!)
torchrun --nnodes=2 --master_addr=0.0.0.0 --master_port=1234 --node_rank=1 run_parallel.py &
PIDS+=$($!)
wait ${PIDS[@]}