# Performance of safetensors on AMD GPUs

## DeepSeek-R1 vLLM Model Weight Loading Speed

This benchmark compares the performance of `safetensors` vs `fastsafetensors` when loading model weights on AMD GPUs.

NOTES: `fastsafetensors` does not support GDS feature on ROCm as there are no GDS alternative on ROCm.

### Benchmark Methodology

1. **Clear system cache** to ensure consistent starting conditions:
   ```bash
   sudo sh -c 'sync && echo 3 > /proc/sys/vm/drop_caches'
   ```

2. **Launch vLLM** with either `--load-format safetensors` or `--load-format fastsafetensors`:

    ```bash
    MODEL=EmbeddedLLM/deepseek-r1-FP8-Dynamic

    VLLM_USE_V1=1 \
    VLLM_ROCM_USE_AITER=1 \
    vllm serve $MODEL \
    --tensor-parallel-size 8 \
    --disable-log-requests \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --trust-remote-code \
    --load-format fastsafetensors \
    --block-size 1
    ```

### Results

The experiments are carried on MI300X.

**Cache Scenarios:**
- **No cache**: Model weights are loaded after clearing the system cache (cold start).
- **Cached**: Model weights are loaded immediately after a previous load. The weights are cached in the filesystem and RAM (warm start).

<img src="./images/fastsafetensors-rocm.png" alt="FastSafeTensors on ROCm" width="70%">



