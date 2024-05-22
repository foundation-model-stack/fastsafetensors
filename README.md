fastsafetensors is an efficient safetensors model loader.
We introduced three major features to optimize model loading performance:
1. Batched, lazy tensor instantiations
2. GPU offloading for sharding, type conversions, and device pointer alignment.
3. GPU Direct Storage enablement for file loading from storage to GPU memory

A major design difference from the original safetensors file loader is *NOT* to use `mmap`.
It loads tensors on-demand with mmap'ed files,
but unfortunately, it cannot fully utilize high-throughput I/O such as NVMe SSDs.
So, we asynchronously transfer files in parallel to saturate storage throughput.
Then, fastsafetensors lazily instantiates tensors at GPU device memory with DLPack.

Another design change is to offload sharding and other manipulations on tensors to GPUs.
The original loader provides slicing for sharding at user programs before copying to device memory. However, it incurrs high CPU usages for host memory accesses.
So, we introduce a special APIs to run sharding with `torch.distributed` collective operations such as `broadcast` and `scatter`.
The offloading is also applied to other tensor manipulations such as type conversions.

The above two design can be naturally extended to utilize device-to-device data transfers with GPU Direct Storage.
The technology helps to minimize copy overheads from NVMe SSDs to GPU memory with host CPU and memory bypassed.

## Dependencies

We currently test fastsafetensors only with python 3.11, pytorch 2.1, and cuda-12.
Note: when using different versions of pytorch, you may require changes on build environments for libpytorch since it seems slightly changing ABIs.

## Install from PyPi (TBD)

```bash
pip install fastsfaetensors
```

## Local installation

Prerequisites: Install torch, cuda, and numa headers

```bash
make install
```

## Package build

Prerequisites: Install Docker (libtorch 2.1, cuda, and numa are automatically pulled)

```bash
make dist
```

## Unit tests

```bash
make install-test # install stub'ed fastsafetensors without torch, cuda, and numa
make unittest
```

## Sample code

see `example/load.py`
