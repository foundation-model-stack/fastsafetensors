export devices=0,1,2,3
export CUDA_VISIBLE_DEVICES=${devices}

python -m paddle.distributed.launch \
    --gpus ${devices} \
    run_parallel.py \