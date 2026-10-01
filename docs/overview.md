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

`SafeTensorsFileLoader` is a low-level entrypoint. To use it, pass either `SingleGroup()` for simple inference or `ProcessGroup()` (from `torch.distributed`) for tensor-parallel inference. The loader supports both CPU and CUDA devices, with optional GPU Direct Storage (GDS) support. You can specify the device and GDS settings using the `device` and `nogds` arguments, respectively. If GDS turns out to be unavailable at runtime (e.g., file handle registration fails), the loader logs a warning and falls back to the bounce-buffer (`nogds`) path instead of failing; you can also set `nogds=True` explicitly to skip GDS initialization. For more information on enabling GDS, please refer to the NVIDIA documentation.

The nogds reader keeps a fixed set of workers for the loader's lifetime. Each
worker alternates two bounce buffers and GPU streams to overlap `pread` with
host-to-device copies. `bbuf_size_kb` is the total host buffer budget, rounded
to whole KiB per slot and divided across `max_threads` workers and their two
buffers. For example, 8 workers and `bbuf_size_kb=256*1024` provide sixteen
16 MiB slots. Closing the loader drains outstanding copies and releases its
workers, streams and host buffers.

On Linux with libnuma and known GPU NUMA topology, `set_numa=True` places
nogds workers on the GPU's CPU node and prefers that node for their memory.
Pinned host buffers are allocated on a separate thread with the same policy,
leaving the caller's CPU affinity and memory policy unchanged. Other memory
nodes remain available if the preferred node is full. Set `set_numa=False`
to retain inherited placement; CPU readers and unavailable NUMA support also
retain inherited placement.

After creating a `SafeTensorsFileLoader` instance, first map target files and a rank using the `.add_filenames()` method. Then, call `.copy_files_to_device()` to trigger the actual file copies on aggregated GPU memory fragments and directly instantiate a group of tensors. Once the files are loaded, you can retrieve a tensor using the `.get_tensor()` method. Additionally, you can obtain sharded tensors by `.get_sharded()`, which internally runs collective operations in `torch.distributed`.

Important: the loader's own `.close()` does not free the device memory that holds loaded tensors (the *load buffers*). Close the object that owns them:

- `SafeTensorsFileLoader`: `.copy_files_to_device()` returns a `FilesBufferOnDevice`, whose `.close()` frees the load buffers. Tensors from it may borrow those buffers, so clone any tensor you need to keep after closing it, and close it before the loader. The loader's `.close()` releases only host-side state (file registrations, the copier and its host bounce buffers), and the loader cannot be used afterwards.
- `ParallelLoader` and `AutoLoader`: the `iterate_weights()` iterator owns the load buffers. It frees each batch's buffers once it moves past that batch's last tensor, and the rest when it is exhausted or closed. If the loop can stop early (`break`, `return`, an exception), close the iterator, e.g. `with contextlib.closing(loader.iterate_weights()) as weights:`, before calling `loader.close()`. Yielded tensors are independent copies and stay valid.

To check for leaks, call `get_framework_op("pytorch").get_mem_used()` (from `fastsafetensors.frameworks`). It returns the bytes of load buffers not yet freed, summed over every loader in the process, so it reads 0 once all are closed. It is not total GPU memory: yielded tensors, pinned host buffers and PyTorch's allocator cache are not counted.

`fastsafe_open` is an easier entrypoint. You can force GDS off and run in fallback mode if `nogds=True`. Leaving the `with` block closes its buffer, so, as with `FilesBufferOnDevice` above, clone any tensor you use outside the block. (This memory model may be simplified in future releases.)

```python
with fastsafe_open(filenames=[filename], nogds=True, device="cpu", debug_log=True) as f:
    for key in f.keys():
        t = f.get_tensor(key).clone().detach() # clone if t is used outside
```

# AutoLoader configuration

`AutoLoader` supports file-based configuration for loader type, pipeline mode, copy settings, and more.
See [Configuration Guide](./configuration.md) for defaults, examples, and all available options.

# ROCm

On ROCm, direct storage-to-GPU loading is supported through hipFile (ROCm >= 7.2): when `libhipfile.so` is available, the GDS code path uses it transparently. On older ROCm without hipFile, the loader falls back to the bounce-buffer (`nogds`) path.
A performance gain example with the `nogds` path can be found at [amd-perf.md](./amd-perf.md).

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

If the DirectStorage DLLs are not installed, the Windows loader falls back to
the bounce-buffer (`nogds`) path. Set `FASTSAFETENSORS_DSTORAGE_DLL_DIR` to an
absolute directory containing `dstoragecore.dll` and `dstorage.dll` to enable
DirectStorage; an invalid explicit directory remains an error.

The fallback resolves a framework-bundled CUDA runtime first (for example,
PyTorch's `torch\lib` directory), then checks system CUDA Toolkit locations.
`FASTSAFETENSORS_CUDART_LIB` can specify an absolute runtime DLL explicitly and
takes precedence over automatic discovery.
