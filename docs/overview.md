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

After creating a `SafeTensorsFileLoader` instance, map files to ranks with `.add_filenames()`, then call `.copy_files_to_device()` to load them and return a `FilesBufferOnDevice`. Retrieve whole tensors with `.get_tensor()` or distributed slices with `.get_sharded()`.

## Lifetime contract

By default, low-level tensor retrieval takes shared ownership of the backing allocation. Tensors and derived views stay valid after the buffer's `.close()`, which drops its own references. The allocation is released when its last owner disappears. A small view can retain an entire file or chunk allocation; `tensor.numel() * tensor.element_size()` does not describe all the memory it retains.

`ParallelLoader` preserves its existing delivery behavior: single-process outputs are cloned into independent storage. Distributed outputs own their receive or broadcast-run storage. Retaining an output does not keep its loader chunk alive, and the existing resident and yield-clone budget accounting is preserved.

For explicit borrowed access, use `copy_files_to_device(borrowed_tensors=True)` in a single-process low-level loader. Tensors and derived views do not hold an allocation owner. Complete every read, including asynchronous device work, before closing the buffer; aliases left in Python do not postpone release. Owning and borrowed storage are selected at materialization, not by `detach()` or slicing an owning tensor.

`ParallelLoader(..., borrowed_tensors=True, accumulate_resident=False)` likewise skips yield clones and returns non-owning views. Complete their use before requesting the next tensor or closing the iterator. Copy into independent storage if data must survive. This mode requires a single-process loader group (`pg=None` or `all_local=True`). Close the iterator on early exit before closing the loader.

The loader's own `close()` releases registrations and copier resources. Close
the low-level buffer, or exhaust/close the `ParallelLoader`/`AutoLoader` iterator,
before closing the loader. `loader.close()` does not close an active iterator.

`live_allocation_count()` / `live_allocation_bytes()` report the load allocations
fastsafetensors still owns, including allocations retained by exported owning
storage. They exclude independent clones, other framework tensors, pinned host
pools and allocator caches. `get_framework_op("pytorch").get_mem_used()` likewise
tracks load-buffer bytes, rather than total GPU memory. In owning mode, closing
every buffer is not enough to return these counters to zero: release all owning
tensors and their derived storage too.

`fastsafe_open` is an easier entrypoint. You can force GDS off and run in fallback mode if `nogds=True`.

```python
with fastsafe_open(filenames=[filename], nogds=True, device="cpu", debug_log=True) as f:
    for key in f.keys():
        t = f.get_tensor(key)  # stays valid after the block; no clone needed
```

# AutoLoader configuration

`AutoLoader` supports file-based configuration for loader type, pipeline mode, copy settings, and more.
See [Configuration Guide](./configuration.md) for defaults, examples, and all available options.

# Saving checkpoints

`ParallelSaver` saves a mapping of native framework tensors as size-balanced
safetensors shards. The destination is a filename prefix; returned paths can be passed
to `ParallelLoader`. Settings can be reused across saves, while metadata is
supplied for each checkpoint. Saves finish and release their file resources
before returning; no explicit close is needed.

```python
from fastsafetensors import ParallelSaver

saver = ParallelSaver(num_shards=8, num_threads=8, framework="pytorch")
paths = saver.save(tensors, "/cache/model", metadata={"version": "1"})
# /cache/model-00001-of-00008.safetensors, ...
```

Use `save_entries` for `WriteEntry` descriptions containing the on-disk dtype,
shape, byte size, and an opaque `source`. Pass a `fill` callback to `ParallelSaver`
to encode entries directly into the supplied `(buffer, entry)` pairs. Determine
encoded byte sizes before shard planning; variable-size encoders must prepare
their payloads first. Application-specific formats, layout metadata, and
compatibility checks belong to the caller.

`num_threads` controls default framework tensor writes; custom callbacks
control their own parallelism. A fill callback must complete all writes before
returning and release references to the supplied buffers.
`shard_metadata` optionally computes string metadata from each shard's entries;
fastsafetensors stores these values without interpreting them.

`metadata` is repeated in every shard; `shard0_metadata` is stored only in the
first shard. The lower-level `plan_shards`, `write_shards`, and `save_sharded`
functions remain available for custom planning and fill callbacks. Completed
shards are published by individual file renames; the set is not published
atomically.

`save` accepts native PyTorch tensors by default; use `framework="paddle"`
for native Paddle tensors. Applications can pass the tensor objects they
already hold. The saver adapts them internally without copying their storage
and without importing a tensor framework until native tensor access is needed.
Existing `TensorBase` wrappers from `get_tensor_wrapped` are also accepted.
Custom fill and shard metadata callbacks receive the original `source` objects,
including native tensors, without wrapping. `save_entries` with a custom fill
callback accepts arbitrary sources and does not load a framework adapter.
Device tensors are copied to host memory before files are persisted;
GPU Direct Storage writing is not implemented.

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
