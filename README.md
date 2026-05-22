fastsafetensors
================

fastsafetensors is an efficient safetensors loader. If you develop your own code that loads large safetensors files, you can try fastsafetensors APIs (see [docs](./docs/overview.md)). For example, vLLM and SGLang have `--load-format fastsafetensors` command-line argument to speed up their initialization.

This library supports Linux/CUDA, ROCm without GDS, Windows, [3FS](https://github.com/deepseek-ai/3fs), and unified-memory systems such as DGX Spark. Our CI tests Python 3.10-3.14 with PyTorch 2.11.0.

# Performance Highlights

Performance highlights from the [CLOUD 2025 paper](https://arxiv.org/abs/2505.23072) and benchmark docs:
- Standalone model loading was **4.8x-7.5x faster** than the default `safetensors` deserializer on Llama, Falcon, and Bloom models, and reached **26.4 GB/s** NVMe read throughput for Llama-70B on four GPUs with GDS.
- In the paper's vLLM integration experiment, startup time dropped from **12.39s to 4.74s** for Llama-2-13B on 4x L40S GPUs, and from **16.04s to 6.88s** on 1x A100.
- On AMD ROCm without GDS, the documented `nogds` path reached **6.02 GB/s** for GPT-2 Medium versus **1.28 GB/s** with `mmap` (**4.7x** throughput), and **2.62 GB/s** for GPT-2 versus **1.01 GB/s** with `mmap` (**2.6x** throughput). See the [report](./docs/amd-perf.md) for more details.

# Quick Start

```bash
pip install fastsafetensors
pip install vllm # for quick demo
vllm serve Qwen/Qwen3-0.6B --load-format fastsafetensors
...
Loading safetensors using Fastsafetensor loader:   0% Completed | 0/1 [00:00<?, ?it/s]
Loading safetensors using Fastsafetensor loader: 100% Completed | 1/1 [00:00<00:00,  1.23it/s]
```

# Design Details

See [Overview](./docs/overview.md) for features, basic API usage, and configuration.

# Code of Conduct

Please refer to [Foundation Model Stack Community Code of Conduct](https://github.com/foundation-model-stack/foundation-model-stack/blob/main/code-of-conduct.md).

# Development

See [Development](./docs/development.md).

# Publication

Takeshi Yoshimura, Tatsuhiro Chiba, Manish Sethi, Daniel Waddington, Swaminathan Sundararaman. (2025) Speeding up Model Loading with fastsafetensors [arXiv:2505.23072](https://arxiv.org/abs/2505.23072) and IEEE CLOUD 2025.
