# ParallelLoader benchmarks

Results from [benchmark_parallel.py](../benchmarks/benchmark_parallel.py), measured on 2026-10-15.
Each configuration was measured 10 times, with a fresh process for each trial. Values are the mean ± sample standard deviation in seconds.
The results include 320 trials across 32 configurations.

## Measurement conditions

- Models are loaded from the Hugging Face cache on local NVMe storage. Each trial uses a cold page cache, verified after cache eviction. fastsafetensors uses O_DIRECT.
- Timing includes loader construction, iterator consumption, copies into preallocated output tensors, loader cleanup, and GPU synchronization. For TP > 1, each trial reports the maximum elapsed time across ranks.
- Process group initialization, output tensor allocation, cache preparation, and validation after loading are excluded from timing. Sample byte comparisons passed for every included trial.
- This standalone benchmark reproduces vLLM tensor parallelism (TP) slicing and expert parallelism (EP) tensor ownership. It does not measure full vLLM loading, which includes model-specific weight transformations and quantization.
- Library versions: safetensors `0.9.0rc1` and InstantTensor `0.2.1`.

## A100

Eight NVIDIA A100-SXM4-80GB GPUs connected via NVLink.

| Model | TP | FST local | FST broadcast | InstantTensor | prefetch |
|---|---:|---:|---:|---:|---:|
| Granite 4.2 30B | 1 | 2.99 ± 0.05 | — | 3.69 ± 0.17 | 15.24 ± 1.06 |
| Qwen 3.8 27B | 1 | 2.94 ± 0.16 | — | 3.50 ± 0.20 | 7.69 ± 0.29 |
| LLM-jp 4 33B | 1 | 3.49 ± 0.11 | — | 3.89 ± 0.18 | 17.53 ± 1.57 |
| Muse Glimmer 30B | 1 | 2.93 ± 0.12 | — | 3.64 ± 0.19 | 14.97 ± 0.49 |
| DeepSeek V4 Flash | 4 | 9.94 ± 0.11 | 7.65 ± 0.10 | 7.64 ± 0.13 | 36.16 ± 6.18 ※ |
| Nemotron 3 Ultra | 8 | 34.67 ± 0.12 | 24.63 ± 0.13 | 23.94 ± 0.09 | 43.96 ± 0.54 ※ |

`FST local` uses fastsafetensors with `all_local=True` and a per-rank `tensor_filter` when EP filtering is active with TP > 1.
`FST broadcast` corresponds to `fastsafetensors-broadcast`: it disables filtering in the reader, loads and distributes tensors with `all_local=False`, and selects tensors according to EP ownership in the consumer.
This comparison is omitted for TP=1, shown as `—`.

※ For DeepSeek and Nemotron, `prefetch` falls back to regular safetensors with CPU mmap to match the EP filtering path. These values do not measure prefetch.
All other `prefetch` results use the `safetensors-prefetch-sharded` path.

## GB10, all TP=1

NVIDIA GB10 with 128 GB of unified memory.

| Model | fastsafetensors | InstantTensor | prefetch |
|---|---:|---:|---:|
| Granite 4.2 30B | 10.89 ± 0.04 | 17.02 ± 0.22 | 12.28 ± 0.07 |
| Qwen 3.8 27B | 10.63 ± 0.12 | 16.66 ± 0.24 | 12.38 ± 0.14 |
| LLM-jp 4 33B | 13.02 ± 0.13 | 18.83 ± 0.26 | 16.47 ± 0.29 |
| Muse Glimmer 30B | 11.48 ± 0.08 | 17.55 ± 0.26 | 23.49 ± 0.57 |

`prefetch` uses the `safetensors-prefetch-sharded` path for all models.
On this unified memory system, file cache pages from the previous model affected the automatic prefetch memory budget. We therefore evicted caches between models and repeated all 10 trials for each model. The table uses these replacement results.
The initial 40 prefetch trials are excluded from the statistics.
