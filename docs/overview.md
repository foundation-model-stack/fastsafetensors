Overview
=========

# Features

Fastsafetensors introduces three major features to optimize model loading performance:
1. Batched, lazy tensor instantiation.
2. GPU offloading for sharding, type conversions, and device pointer alignment.
3. GPU Direct Storage enablement for file loading from storage to GPU memory.

A major design difference from the original safetensors file loader is that fastsafetensors does *NOT* use `mmap`.
The original loader loads tensors on demand from memory-mapped files,
but unfortunately, it cannot fully utilize high-throughput I/O such as NVMe SSDs.
Therefore, we asynchronously transfer files in parallel to saturate storage throughput.
The loader then lazily instantiates tensors in GPU device memory with DLPack.

Another design change is to offload sharding and other tensor manipulations to GPUs.
The original loader provides slicing for sharding in user programs before copying to device memory. However, it incurs high CPU usage for host memory accesses.
Therefore, we introduce special APIs to run sharding with `torch.distributed` collective operations such as `broadcast` and `scatter`.
The offloading is also applied to other tensor manipulations such as type conversions.

The above two designs can be naturally extended to utilize device-to-device data transfers with GPU Direct Storage.
The technology helps minimize copy overheads from NVMe SSDs to GPU memory by bypassing host CPU and memory.

# Basic API usage

`SafeTensorsFileLoader` is a low-level entrypoint. To use it, pass either `SingleGroup()` for simple inference or `ProcessGroup()` (from `torch.distributed`) for tensor-parallel inference. The loader supports both CPU and CUDA devices, with optional GPU Direct Storage (GDS) support. You can specify the device and GDS settings using the `device` and `nogds` arguments, respectively. Note that if GDS is not available, the loader will fail to open files when `nogds=False`. For more information on enabling GDS, please refer to the NVIDIA documentation.

After creating a `SafeTensorsFileLoader` instance, first map target files and a rank using the `.add_filenames()` method. Then, call `.copy_file_to_device()` to trigger the actual file copies on aggregated GPU memory fragments and directly instantiate a group of tensors. Once the files are loaded, you can retrieve a tensor using the `.get_tensor()` method. Additionally, you can obtain sharded tensors by `.get_sharded()`, which internally runs collective operations in `torch.distributed`.

Important: To release the GPU memory allocated for tensors, you must explicitly call the `.close()` method. This is because fastsafetensors allows multiple tensors to share a limited number of GPU memory fragments. As a result, it is the user's responsibility to ensure that all tensors are properly released before calling `.close()`, which will then safely release the underlying GPU memory.

`fastsafe_open` is an easier entrypoint. You can force GDS off and run in fallback mode if `nogds=True`. However, users must be aware of the above tricky memory management model, which should be fixed in future releases.

```python
with fastsafe_open(filenames=[filename], nogds=True, device="cpu", debug_log=True) as f:
    for key in f.get_keys():
        t = f.get_tensor(key).clone().detach() # clone if t is used outside
```

# AutoLoader configuration

`AutoLoader` supports file-based configuration for loader type, pipeline mode, copy settings, and more.
See [Configuration Guide](./configuration.md) for defaults, examples, and all available options.

# ROCm

On ROCm, there is no GDS-equivalent support, so fastsafetensors only supports `nogds=True` mode.
The performance gain example can be found at [amd-perf.md](./amd-perf.md).

# Windows

From [PR#72](https://github.com/foundation-model-stack/fastsafetensors/pull/72):

On Linux, GDS uses cuFile to DMA data directly from NVMe into GPU memory. Windows has no cuFile — instead, it offers [DirectStorage](https://devblogs.microsoft.com/directx/directstorage-api-available-on-pc/), a DirectX 12 API designed for the same purpose.

Since DirectStorage writes into D3D12 resources (not CUDA buffers), we bridge the two APIs through CUDA external memory interop:

```
NVMe -> [DirectStorage] -> D3D12 shared buffer -> [cudaImportExternalMemory] -> CUDA device pointer
```

The key steps are:

1. Create a D3D12 committed resource with D3D12_HEAP_FLAG_SHARED so it can be exported
2. DirectStorage reads from NVMe into this D3D12 buffer via IDStorageQueue
3. Export the D3D12 resource as an NT handle via CreateSharedHandle
4. Import into CUDA via cudaImportExternalMemory + cudaExternalMemoryGetMappedBuffer to get a regular CUDA device pointer
5. Synchronize using a D3D12 fence imported as a cudaExternalSemaphore

All DirectStorage, D3D12, and DXGI libraries are loaded at runtime via LoadLibrary/GetProcAddress — no link-time SDK dependency on DirectStorage is required.
