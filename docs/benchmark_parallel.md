# ParallelLoader benchmarks

Results from [benchmark_parallel.py](../benchmarks/benchmark_parallel.py). The A100 and GB10 tables were remeasured on 2026-10-17 with the default queue and memory budget settings.
Each configuration was measured 10 times, with a fresh process for each trial. Values are the mean ± sample standard deviation in seconds.
The A100 and GB10 tables include 320 trials across 32 configurations. The H100 comparison below is a separate set of 36 trials.

## Measurement conditions

- Models are loaded from the Hugging Face cache on local NVMe storage. Each trial uses a cold page cache, verified after cache eviction. fastsafetensors uses O_DIRECT.
- Timing includes loader construction, iterator consumption, copies into preallocated output tensors, loader cleanup, and GPU synchronization. For TP > 1, each trial reports the maximum elapsed time across ranks.
- Process group initialization, output tensor allocation, cache preparation, and validation after loading are excluded from timing. Sample byte comparisons passed for every included trial.
- This standalone benchmark reproduces vLLM tensor parallelism (TP) slicing and expert parallelism (EP) tensor ownership. It does not measure full vLLM loading, which includes model-specific weight transformations and quantization.
- Library versions: safetensors `0.9.0rc1`; InstantTensor `0.2.2` on both hosts, matching the vLLM measurements.
- fastsafetensors runs on both hosts use the defaults: an automatic device memory budget and queue size 0, matching the [vLLM runs](sweep_vllm.md). The automatic budget is free GPU memory after destination allocation, minus the larger of 1 GiB or 5% of free memory; distributed ranks use the minimum budget across ranks.

## Reproduce the tables

Install the [benchmark dependencies](../benchmarks/README.md#setup), including InstantTensor, and download the checkpoints listed in [`sweep.py`](../benchmarks/sweep.py) into a local Hugging Face cache before running.
Run the following commands from the repository root on the corresponding host, replacing `/data/huggingface/hub` with that host's cache path. Each output directory must be empty or absent.

For the A100 table, the model presets select TP=1 for Granite, Qwen, LLM-jp and Muse Glimmer, TP=4 for DeepSeek, and TP=8 for Nemotron:

```bash
python benchmarks/sweep.py \
  --model all --repeat 10 \
  --loaders fastsafetensors fastsafetensors-broadcast instanttensor safetensors-prefetch \
  --hub-cache /data/huggingface/hub \
  --outdir benchmark-results/a100
```

For the GB10 table, select the four TP=1 models:

```bash
python benchmarks/sweep.py \
  --model granite qwen3.8-27b llm-jp muse-glimmer --tp 1 --repeat 10 \
  --loaders fastsafetensors instanttensor safetensors-prefetch \
  --hub-cache /data/huggingface/hub \
  --outdir benchmark-results/gb10
```

Both commands use the default cold-cache preparation, library I/O policy (`auto`, selecting O_DIRECT on this local NVMe filesystem), automatic memory budget and queue size 0.

The sweep uses the `vllm` loader profile, one measured round and no warmup per fresh process. It enables EP filtering for DeepSeek and Nemotron and skips the broadcast comparison when TP=1. It also evicts the selected checkpoints' page cache before launching workers and when changing models, including on GB10.
Other sweep settings retain their defaults, including 16 I/O threads and a 16 MiB bounce buffer. Each output directory contains a `manifest.json` with resolved checkpoint revisions, per-trial commands, logs and JSON reports, and an aggregated `summary.csv`.

## A100

Eight NVIDIA A100-SXM4-80GB GPUs connected via NVLink.
Pre-load residency checks found at most one resident checkpoint page per rank, within the harness's cold-cache threshold.

| Model | TP | FST local | FST broadcast | InstantTensor | prefetch |
|---|---:|---:|---:|---:|---:|
| Granite 4.2 30B | 1 | 3.05 ± 0.14 | — | 3.61 ± 0.19 | 15.73 ± 0.48 |
| Qwen 3.8 27B | 1 | 2.85 ± 0.04 | — | 3.40 ± 0.15 | 7.94 ± 0.56 |
| LLM-jp 4 33B | 1 | 3.59 ± 0.10 | — | 3.92 ± 0.13 | 17.43 ± 1.40 |
| Muse Glimmer 30B | 1 | 2.96 ± 0.14 | — | 3.67 ± 0.29 | 15.18 ± 0.49 |
| DeepSeek V4 Flash | 4 | 10.01 ± 0.16 | 7.70 ± 0.11 | 7.74 ± 0.16 | 38.90 ± 5.09 ※ |
| Nemotron 3 Ultra | 8 | 34.63 ± 0.19 | 24.28 ± 0.18 | 23.87 ± 0.12 | 42.77 ± 0.60 ※ |

`FST local` uses fastsafetensors with `all_local=True` and a per-rank `tensor_filter` when EP filtering is active with TP > 1.
`FST broadcast` corresponds to `fastsafetensors-broadcast`: it disables filtering in the reader, loads and distributes tensors with `all_local=False`, and selects tensors according to EP ownership in the consumer.
This comparison is omitted for TP=1, shown as `—`.

※ For DeepSeek and Nemotron, `prefetch` falls back to regular safetensors with CPU mmap to match the EP filtering path. These values do not measure prefetch.
All other `prefetch` results use the `safetensors-prefetch-sharded` path.

## GB10, all TP=1

NVIDIA GB10 with 128 GB of unified memory.

| Model | fastsafetensors | InstantTensor | prefetch |
|---|---:|---:|---:|
| Granite 4.2 30B | 10.78 ± 0.02 | 16.92 ± 0.35 | 12.36 ± 0.08 |
| Qwen 3.8 27B | 10.71 ± 0.06 | 16.51 ± 0.47 | 12.33 ± 0.11 |
| LLM-jp 4 33B | 12.94 ± 0.20 | 18.65 ± 0.19 | 16.31 ± 0.16 |
| Muse Glimmer 30B | 11.48 ± 0.09 | 17.25 ± 0.25 | 22.96 ± 0.60 |

`prefetch` uses the `safetensors-prefetch-sharded` path for all models.
On this unified memory system, file cache pages also affect the automatic memory budgets. The sweep evicts the selected checkpoints before workers start and the previous model's cache when switching models. Each trial uses the available memory to select its budget; automatic prefetch windows can therefore vary between trials.

## H100 queue and memory budget comparison

On 2026-10-17, the standalone harness was rerun on `cccxc705` (four NVIDIA H100 80GB HBM3 GPUs) to compare the original settings with the queue and budget settings used by vLLM:

- Original: `--memory-budget-mib 7168 --queue-size 2`.
- vLLM settings: `--memory-budget-mib auto --queue-size 0`. The automatic budget is free GPU memory after destination allocation, minus the larger of 1 GiB or 5% of free memory; distributed ranks use the minimum budget across ranks.

Each configuration was measured three times per condition, alternating the two conditions in fresh processes (36 trials total). Both conditions use O_DIRECT, the `vllm` loader profile, 16 I/O threads, a 16 MiB bounce buffer, one measured round and no warmup. Checkpoint revisions match the original tables, and the loader implementation is unchanged. All trials passed cold-cache residency checks and sample byte comparisons; no OOM messages were found. These are standalone loading times, not full vLLM loading times.

Values are mean ± sample standard deviation in seconds. A positive change means the vLLM settings were slower.

| Model / mode | TP | Original | vLLM settings | Change |
|---|---:|---:|---:|---:|
| Granite 4.2 30B | 1 | 2.850 ± 0.005 | 2.864 ± 0.017 | +0.50% |
| Qwen 3.8 27B | 1 | 2.743 ± 0.002 | 2.758 ± 0.006 | +0.55% |
| LLM-jp 4 33B | 1 | 3.260 ± 0.014 | 3.338 ± 0.001 | +2.40% |
| Muse Glimmer 30B | 1 | 2.774 ± 0.019 | 2.745 ± 0.005 | −1.05% |
| DeepSeek V4 Flash, local | 4 | 9.724 ± 0.044 | 9.732 ± 0.032 | +0.09% |
| DeepSeek V4 Flash, broadcast | 4 | 7.468 ± 0.054 | 7.450 ± 0.029 | −0.24% |

No large latency change was observed in these configurations. This comparison does not cover Nemotron TP=8, which requires more GPUs than this host provides.

The automatic budgets were 15.68–38.39 GiB, but actual additional allocations depend on batch size and queue depth. The table below reports peak additional PyTorch allocation above the preallocated destination tensors, taking the maximum across all ranks and trials, in GiB. It does not include allocations outside the PyTorch allocator.

| Model / mode | Original effective queue | Original peak extra GiB | vLLM settings peak extra GiB |
|---|---:|---:|---:|
| Granite 4.2 30B | 2 | 6.12 | 3.06 |
| Qwen 3.8 27B | 0 | 7.00 | 9.47 |
| LLM-jp 4 33B | 1 | 7.00 | 7.50 |
| Muse Glimmer 30B | 0 | 7.00 | 10.02 |
| DeepSeek V4 Flash, local | 2 | 7.00 | 3.95 |
| DeepSeek V4 Flash, broadcast | 2 | 6.60 | 4.95 |

All vLLM-setting runs used effective queue size 0. Removing the fixed 7 GiB cap increased peak additional allocation for Qwen, LLM-jp and Muse Glimmer, even though queue depth did not increase.
