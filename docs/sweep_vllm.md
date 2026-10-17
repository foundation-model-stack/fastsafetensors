# vLLM model loading sweep

## Measurement conditions

- Each trial starts a fresh vLLM server with `--enforce-eager` and checkpoints already cached locally; checkpoint downloading is outside the measurement.
- Timing uses vLLM's `Model loading took … seconds` log marker. For each run, we take the maximum reported time across ranks. The A100/H100 tables use the completed 2026-10-16 results and report mean ± sample standard deviation in seconds; `n` is the number of trials per loader (6 for A100, 9 for H100). The GB10 tables report mean seconds from 10 repeats for safetensors and 9 repeats for instanttensor/fastsafetensors.
- The interval includes model construction, weight loading, and model-specific processing within vLLM's loading phase. Process startup, subsequent serving warmup, and inference latency are outside this interval. These results differ in scope from the standalone [ParallelLoader benchmarks](benchmark_parallel.md).
- Before every run, the scripts evict checkpoint page cache using `posix_fadvise(POSIX_FADV_DONTNEED)`.
- A100/H100 package versions: fastsafetensors `0.5`, InstantTensor `0.2.2`, and safetensors `0.9.0rc1`.
- vLLM code is modified to enable prefetch in safetensors (https://github.com/vllm-project/vllm/pull/57973) and customization for fastsafetensors 0.4 and 0.5 (https://github.com/vllm-project/vllm/pull/55985).

| Table label | vLLM configuration |
|---|---|
| safetensors | `--load-format safetensors --safetensors-load-strategy lazy` (mmap) |
| safetensors-prefetch | `--load-format safetensors`, strategy left at auto |
| instanttensor | `--load-format instanttensor` |
| fastsafetensors | `--load-format fastsafetensors`, queue size 0, automatic device memory budget (`-1`) |

The auto/prefetch configuration uses local vLLM changes integrating safetensors' CUDA prefetch API and distributed delivery.
The A100/H100 fastsafetensors runs use shard ownership and broadcast delivery (`VLLM_FASTSAFETENSORS_ALL_LOCAL=0`), with EP tensor ownership filtering applied by the consumer.
For the A100/H100 DeepSeek and Nemotron runs, expert parallelism (EP) and EP weight filtering are enabled; this filtering bypasses the CUDA prefetch path.
Thus, `safetensors-prefetch` names the requested configuration and does not mean prefetch is active in every row.
The two-host GB10 runs use TP=2 with EP and EP weight filtering disabled (`mode=noep`); fastsafetensors uses shard ownership and broadcast delivery (`VLLM_FASTSAFETENSORS_ALL_LOCAL=0`).

## Main observations

- On one GB10, fastsafetensors is 1.31–1.40× faster than instanttensor across all five models.
- Across two GB10 hosts, fastsafetensors is 1.22–1.34× faster than instanttensor across all four models.
- On A100, fastsafetensors has the lowest mean in 16 of 19 rows. DeepSeek takes 18.21 s at TP=4 and 16.53 s at TP=8 with fastsafetensors, versus 21.16 s and 20.45 s with instanttensor.
- On H100, fastsafetensors has the lowest mean in 8 of 13 rows. DeepSeek TP=4 takes 12.89 s with fastsafetensors versus 14.49 s with instanttensor. Prefetch has the lowest mean for LLM-jp TP=4 and Granite TP=4.

Bold values mark the lowest observed mean in each row.

## Single host with eight NVIDIA A100-SXM4-80GB GPUs.

TP specifies the number of GPUs used by each run.

| Model | TP | n | safetensors | safetensors-prefetch | instanttensor | fastsafetensors |
|---|---:|---:|---:|---:|---:|---:|
| Granite 4.2 30B | 1 | 6 | 33.16 ± 3.95 | 16.61 ± 0.44 | 6.70 ± 0.58 | **6.45 ± 0.43** |
| LLM-jp 4 33B Thinking | 1 | 6 | 41.22 ± 1.45 | 16.49 ± 0.53 | 6.47 ± 0.38 | **6.29 ± 0.08** |
| Muse Glimmer 30B | 1 | 6 | 37.09 ± 5.62 | 18.12 ± 1.47 | 7.39 ± 0.55 | **6.34 ± 0.30** |
| Qwen 3.8 27B | 1 | 6 | 35.87 ± 0.54 | 13.90 ± 0.95 | **7.41 ± 0.38** | 7.46 ± 0.37 |
| Granite 4.2 30B | 2 | 6 | 21.31 ± 1.10 | 6.85 ± 0.51 | 5.89 ± 0.48 | **5.09 ± 0.22** |
| LLM-jp 4 33B Thinking | 2 | 6 | 23.59 ± 0.80 | 7.07 ± 1.17 | 7.08 ± 1.90 | **5.84 ± 0.38** |
| Muse Glimmer 30B | 2 | 6 | 22.64 ± 1.03 | 11.76 ± 0.41 | 6.17 ± 0.49 | **5.66 ± 0.51** |
| Qwen 3.8 27B | 2 | 6 | 23.54 ± 0.66 | 7.99 ± 0.34 | 7.44 ± 1.16 | **6.46 ± 0.16** |
| DeepSeek V4 Flash 0731 | 4 | 6 | 190.74 ± 3.21 | 191.29 ± 1.57 | 21.16 ± 0.24 | **18.21 ± 0.66** |
| Granite 4.2 30B | 4 | 6 | 18.37 ± 0.40 | 6.54 ± 1.22 | 6.08 ± 0.37 | **5.79 ± 1.13** |
| LLM-jp 4 33B Thinking | 4 | 6 | 18.88 ± 0.56 | 7.11 ± 1.09 | 6.90 ± 1.23 | **6.37 ± 0.57** |
| Muse Glimmer 30B | 4 | 6 | 17.35 ± 0.85 | 11.56 ± 0.61 | 6.83 ± 0.36 | **6.10 ± 0.56** |
| Qwen 3.8 27B | 4 | 6 | 21.69 ± 0.77 | 7.32 ± 0.54 | 8.71 ± 2.00 | **6.70 ± 0.42** |
| DeepSeek V4 Flash 0731 | 8 | 6 | 36.75 ± 2.46 | 36.51 ± 4.11 | 20.45 ± 0.51 | **16.53 ± 1.71** |
| Granite 4.2 30B | 8 | 6 | 41.47 ± 0.98 | **5.59 ± 0.34** | 6.83 ± 0.41 | 5.88 ± 0.41 |
| LLM-jp 4 33B Thinking | 8 | 6 | 44.20 ± 1.21 | 6.17 ± 0.54 | 7.33 ± 0.54 | **6.12 ± 0.28** |
| Muse Glimmer 30B | 8 | 6 | 41.48 ± 1.11 | 11.78 ± 0.96 | 6.88 ± 0.21 | **6.32 ± 0.20** |
| Nemotron 3 Ultra 550B A55B FP8 | 8 | 6 | 52.00 ± 1.75 | 52.26 ± 2.23 | **31.64 ± 1.39** | 34.24 ± 6.19 |
| Qwen 3.8 27B | 8 | 6 | 40.18 ± 0.81 | 7.16 ± 0.64 | 8.11 ± 0.44 | **6.34 ± 0.22** |

DeepSeek load times are recorded, but serving startup fails with `Unsupported architecture` on A100. These rows do not indicate successful inference.

## Single host with four NVIDIA H100 80GB HBM3 GPUs.

| Model | TP | n | safetensors | safetensors-prefetch | instanttensor | fastsafetensors |
|---|---:|---:|---:|---:|---:|---:|
| Granite 4.2 30B | 1 | 9 | 25.13 ± 0.50 | 12.35 ± 0.71 | **5.31 ± 0.11** | 5.44 ± 0.15 |
| LLM-jp 4 33B Thinking | 1 | 9 | 29.66 ± 1.49 | 13.63 ± 0.64 | **5.54 ± 0.17** | 5.78 ± 0.26 |
| Muse Glimmer 30B | 1 | 9 | 26.49 ± 0.58 | 15.34 ± 0.62 | 5.78 ± 0.31 | **5.72 ± 0.28** |
| Qwen 3.8 27B | 1 | 9 | 25.00 ± 0.62 | 10.17 ± 0.47 | **6.53 ± 0.21** | 6.71 ± 0.16 |
| Granite 4.2 30B | 2 | 9 | 17.47 ± 0.55 | 5.37 ± 0.34 | 5.24 ± 0.18 | **4.86 ± 0.12** |
| LLM-jp 4 33B Thinking | 2 | 9 | 18.98 ± 0.45 | 5.59 ± 0.31 | 5.60 ± 0.56 | **5.19 ± 0.17** |
| Muse Glimmer 30B | 2 | 9 | 18.13 ± 0.46 | 9.55 ± 0.18 | 5.30 ± 0.23 | **4.75 ± 0.17** |
| Qwen 3.8 27B | 2 | 9 | 19.22 ± 0.52 | 6.11 ± 0.21 | 6.22 ± 0.14 | **5.76 ± 0.12** |
| DeepSeek V4 Flash 0731 | 4 | 9 | 92.20 ± 5.69 | 96.40 ± 15.23 | 14.49 ± 0.18 | **12.89 ± 0.31** |
| Granite 4.2 30B | 4 | 9 | 22.05 ± 0.89 | **4.99 ± 0.16** | 5.51 ± 0.19 | 5.10 ± 0.35 |
| LLM-jp 4 33B Thinking | 4 | 9 | 22.34 ± 1.13 | **5.18 ± 0.19** | 5.65 ± 0.11 | 5.45 ± 0.40 |
| Muse Glimmer 30B | 4 | 9 | 21.24 ± 1.52 | 9.39 ± 0.24 | 5.53 ± 0.16 | **5.23 ± 0.50** |
| Qwen 3.8 27B | 4 | 9 | 22.24 ± 0.46 | 5.99 ± 0.15 | 6.62 ± 0.28 | **5.76 ± 0.25** |

Qwen TP=2 fastsafetensors includes one trial interrupted after its load time was recorded, before inference validation. All nine load times are included.

## Single GB10 (pgx-0), TP=1

| Model | safetensors | instanttensor | fastsafetensors |
|---|---:|---:|---:|
| Granite 4.2 30B | 267.85 | 20.51 | **14.65** |
| LLM-jp 4 33B Thinking | 352.04 | 22.34 | **17.01** |
| Muse Glimmer 30B | 411.21 | 21.09 | **15.50** |
| Qwen 3.6 35B A3B NVFP4 | 154.92 | 20.92 | **15.09** |
| Qwen 3.8 27B | 375.25 | 21.85 | **16.12** |

## Two GB10 hosts (pgx-0-1), TP=2, EP disabled

| Model | safetensors | instanttensor | fastsafetensors |
|---|---:|---:|---:|
| DeepSeek V4 Flash 0731 | 154.86 | 27.81 | **22.72** |
| GLM 4.7 NVFP4 | 612.60 | 35.49 | **27.77** |
| MiniMax M2.7 NVFP4 | 413.31 | 28.77 | **22.46** |
| Qwen 3.8 Flash Next NVFP4 | 484.91 | 40.13 | **29.85** |

## Checkpoints

| Source model key | Hugging Face checkpoint |
|---|---|
| `granite` | `ibm-granite/granite-4.2-30b` |
| `llm-jp` | `llm-jp/llm-jp-4-33b-thinking` |
| `muse-glimmer` | `meta-models/Muse-Glimmer-30B` |
| `qwen3.8-27b` | `Qwen/Qwen3.8-27B` |
| `deepseek` | `deepseek-ai/DeepSeek-V4-Flash-0731` |
| `nemotron` | `RedHatAI/NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8-dynamic` |
| `qwen3.6-nvfp4` | `nvidia/Qwen3.6-35B-A3B-NVFP4` |
| `glm-4.7` | `Salyut1/GLM-4.7-NVFP4` |
| `minimax-m2.7` | `nvidia/MiniMax-M2.7-NVFP4` |
| `qwen3.8-flash-next` | `RadixArk/Qwen3.8-Flash-Next-NVFP4` |
