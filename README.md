fastsafetensors is an efficient safetensors model loader.
This library is tested with python 3.9-13 and pytorch 2.1-2.7.

Disclaimer: This repository contains a research prototype. It should be used with caution.

# Features

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

## Basic API usages

`SafeTensorsFileLoader` is a low-level entrypoint. To use it, pass either `SingleGroup()` for simple inference or `ProcessGroup()` (from `torch.distributed`) for tensor-parallel inference. The loader supports both CPU and CUDA devices, with optional GPU Direct Storage (GDS) support. You can specify the device and GDS settings using the `device` and `nogds` arguments, respectively. Note that if GDS is not available, the loader will fail to open files when `nogds=False`. For more information on enabling GDS, please refer to the NVIDIA documentation.

After creating a `SafeTensorsFileLoader` instance, first map target files and a rank using the `.add_filenames()` method. Then, call `.copy_file_to_device()` to trigger the actual file copies on aggregated GPU memory fragments and directly instantiate a group of Tensors. Once the files are loaded, you can retrieve a tensor using the `.get_tensor()` method. Additionally, you can obtain sharded tensors by `.get_sharded()`, which internally run collective operations in `torch.distributed`.

Important: To release the GPU memory allocated for tensors, you must explicitly call the `.close()` method. This is because Fastsafetensors allows multiple tensors to share a limited number of GPU memory fragments. As a result, it is the user's responsibility to ensure that all tensors are properly released before calling `.close()`, which will then safely release the underlying GPU memory.

`fastsafe_open` is an easier entrypoint. You can force turning off GDS and run in the fallback mode if `nogds==True`. However, users must be aware of the above tricky memory management model, which should be fixed in future releases.

```python
with fastsafe_open(filenames=[filename], nogds=True, device="cpu", debug_log=True) as f:
    for key in f.get_keys():
        t = f.get_tensor(key).clone().detach() # clone if t is used outside
```

## Code of Conduct

Please refer to [Foundation Model Stack Community Code of Conduct](https://github.com/foundation-model-stack/foundation-model-stack/blob/main/code-of-conduct.md).

## Publication

Takeshi Yoshimura, Tatsuhiro Chiba, Manish Sethi, Daniel Waddington, Swaminathan Sundararaman. (2025) Speeding up Model Loading with fastsafetensors [arXiv:2505.23072](https://arxiv.org/abs/2505.23072) and IEEE CLOUD 2025.


## Install from PyPI

See https://pypi.org/project/fastsafetensors/

```bash
pip install fastsafetensors
```

## Install from source

```bash
pip install .
```
