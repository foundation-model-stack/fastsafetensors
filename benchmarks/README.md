# Loader benchmarks

These scripts compare fastsafetensors with safetensors and optional readers,
without importing vLLM or other model frameworks:

- `benchmark_loader.py`: shard-group loading with `SafeTensorsFileLoader`.
- `benchmark_parallel.py`: pipelined loading and distributed broadcasts with
  `ParallelLoader.iterate_weights()`.

Timing includes loader construction, iteration, copies into preallocated tensors,
cleanup and device synchronization. Model initialization, weight transforms and
inference are outside the benchmark. Results are saved as JSON.

## Setup

Run from the repository root with a CPU or CUDA/ROCm PyTorch installation
appropriate to the machine:

```bash
pip install -e '.[benchmark]'
```

To include InstantTensor (Linux and NVIDIA CUDA), install
`pip install -e '.[benchmark,instanttensor]'` instead.

## Run a benchmark

Use an already downloaded Hugging Face model. The scripts never download models;
`--hub-cache` selects the directory containing `models--...` entries.

```bash
python -m benchmarks.benchmark_loader \
  --model openai-community/gpt2 --hub-cache /data/huggingface/hub \
  --output /tmp/loader.json

python -m benchmarks.benchmark_parallel \
  --model openai-community/gpt2 --hub-cache /data/huggingface/hub \
  --output /tmp/parallel.json
```

Both scripts default to fastsafetensors and CPU-mmap safetensors, with one
warmup and three measured rounds. Use `--readers` to select readers,
`--rounds` and `--warmup` to change repetitions, and `--cache cold|warm|unchanged`
to control the page cache.

For multiple GPUs, launch either script with `torchrun`:

```bash
torchrun --standalone --nproc_per_node=2 -m benchmarks.benchmark_parallel \
  --model openai-community/gpt2 --hub-cache /data/huggingface/hub \
  --output /tmp/parallel-tp2.json
```

Destinations are replicated by default. Use `--shard-map` or
`--partition-profile` for tensor partitioning.

## Run a sweep

`sweep.py` runs fresh processes across model presets, TP sizes and readers:

```bash
python benchmarks/sweep.py \
  --model deepseek --tp 4 --repeat 3 \
  --loaders fastsafetensors fastsafetensors-broadcast instanttensor safetensors-prefetch \
  --hub-cache /data/huggingface/hub --outdir benchmark-results/deepseek-tp4
```

Use `--model all --loaders all` for all presets and comparison readers;
this requires InstantTensor and up to eight visible GPUs with the default TP
settings.
