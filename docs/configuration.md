# Configuration Guide

## Configuration Discovery

`AutoLoader` loads configuration in the following priority (highest first):

1. **Environment variable** — `FASTSAFETENSORS_CONFIG=/path/to/config.json`
2. **Default path** — `./fastsafetensors.json` in the working directory (if it exists)
3. **Built-in defaults** — `LoaderConfig()` dataclass defaults

All fields are optional. Unspecified fields fall back to built-in defaults.

## Default Configuration

When no config file is found, `AutoLoader` uses these defaults:

```json
{
  "loader": "base",
  "framework": "pytorch",
  "parallel": {
    "use_pipeline": false,
    "max_batch_bytes": null,
    "use_chunk_budget_as_allocation_size": false,
    "device_memory_budget": null
  },
  "debug": {
    "debug_log": false,
    "set_numa": true,
    "disable_cache": true
  }
}
```

The base loader extension defaults to `copier_type: "gds"` (GPU Direct Storage).

Available `copier_type` values:

| Value        | Description                                                                                                                                         |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `"gds"`      | NVIDIA GPUDirect Storage via cuFile (default on Linux when available)                                                                               |
| `"fgds"`     | Alternative GPU Direct Storage implementation via [FGDS](https://github.com/Storage-and-OS-for-AI/fgds)                                                                                            |
| `"nogds"`    | Bounce-buffer pread path (no GPU Direct); fallback when GDS is unavailable                                                                          |
| `"unified"`  | Unified-memory copier for shared CPU/GPU memory systems (e.g., DGX Spark); automatically selected on unified-memory hosts when `nogds` is requested |
| `"dstorage"` | DirectStorage backend (Windows only)                                                                                                                |

## queue_size Semantics

| `queue_size` | Mode | Live chunk buffers per rank | Behavior |
|---|---|---|---|
| `-1` | Fully serial | 1 | `copy_files → broadcast → copy_files → ...` |
| `0` | Unbuffered pipeline | Up to 2 | 1 batch copying + 1 batch broadcasting concurrently |
| `>0` | Buffered pipeline | Up to `queue_size+2` | Queued batches + producer + consumer |

`use_pipeline: false` forces `queue_size=-1` (serial, minimal GPU memory).
Without chunking, each buffer covers a whole shard. These counts exclude
resident tensors, broadcast receive tensors, yield clones, and copier staging
memory, which the memory planner accounts for separately where applicable.

## Distributed Tensor Delivery

PyTorch distributed loading coalesces adjacent tensors from one source file
into byte broadcasts, preserving the iterator's order. The default limits are
`broadcast_run_bytes=16777216` (16 MiB) and `broadcast_run_tensors=64` in the
Python `ParallelLoader` API. Every rank must use identical limits. Setting
either limit to zero restores per-tensor broadcasts. Individual tensors larger
than the byte limit are sent alone; file boundaries, gaps and dtype alignment
also split runs. Single-process and Paddle loading use the existing paths.

NCCL communication establishes a dependency on the consumer's current CUDA
stream instead of synchronizing the whole device for every tensor. The consumer
stream is drained before each batch's backing buffers are released, including
when the iterator closes early. Consumers using other CUDA streams must arrange
their usual stream dependencies before accessing the yielded tensors.

Yielded tensors remain valid after the iterator and loader close. Adjacent
outputs share one independently owned run allocation; retaining a view retains
its entire run. With `accumulate_resident=False`, relocate and release each
tensor before requesting the next. The consumer stream is drained between runs
to bound pending staging allocations while checkpoint reads continue on other
streams. A `resident_tensor` predicate keeps the per-tensor delivery path, so
retaining selected tensors does not retain unrelated bytes from shared runs.

## Tensor IO overlap

`ParallelLoader` and `PipelineParallel` default to `overlap_io=True`. The NoGDS,
GDS, and Unified Memory O_DIRECT copiers create tensor views while their workers
read and transfer the chunk. A view does not imply ready data: NoGDS access
waits for the completed DMA prefix that covers the tensor, and Unified Memory
access waits for completed 16 MiB blocks. Different workers may complete out of
order; access waits until every block covering the requested tensor is complete.
GDS publishes a completed prefix after each synchronous cuFileRead block returns.
Files requiring in-place device pointer alignment repair retain whole-file GDS
waits, since relocating bytes while DMA writes remain in flight would race.
Closing a buffer or iterator drains all outstanding writes before freeing
device storage or closing input files.
This uses the existing bounce pool and chunk allocations; memory planner limits
and independent ownership of iterator outputs remain the same.

Use `overlap_io=False` to compare against whole-chunk waits. The low-level
`SafeTensorsFileLoader.copy_files_to_device()` continues to block by default;
pass `allow_inflight=True` to opt into guarded tensor access. Online dtype
conversion and copiers without readiness support retain blocking IO. Unified
Memory keeps one active O_DIRECT DMA job per loader, so its reusable pinned pool
stays within the existing fixed worker allowance even with queued chunks. Each
worker fences its own DMA stream before publishing bytes. O_DIRECT failures
after asynchronous submission are fatal and are drained on close; rereading
with mmap after partial delivery would be unsafe. Network filesystems and
`FASTSAFETENSORS_DMA_THREADS=0` retain blocking mmap/pinning. FGDS and
DirectStorage do not yet publish partial completion here.

Coalesced broadcasts wait for every tensor in the source run before cloning
or broadcasting its bytes. Once that run is ready, communication and consumer
work can overlap the remaining checkpoint reads. Disabling coalescing restores
per-tensor broadcasts, which synchronize the whole device and limit this overlap.

## Bounded Device Memory

`max_batch_bytes` caps each sub-file chunk. It must be at least as large as
the largest selected tensor because tensors are not split across chunks.

With `max_batch_bytes` or `device_memory_budget`, each distinct shard's header
is parsed once and reused for planning and subsequent chunk batches. Keep the
checkpoint files unchanged for the duration of a load. A new pipeline reads
fresh headers, and closing the loader releases the cached metadata. Standalone
`SafeTensorsFileLoader` registration continues to read headers on each call.

`device_memory_budget` bounds resident tensors, transient chunk buffers, and
fixed copier pools across the load. Under distributed broadcast, every rank
must use the same value to produce an identical plan.

If the requested queue depth does not fit, the loader reduces it automatically,
down to `queue_size=-1`. If even serial loading cannot fit, loading raises
`BudgetInfeasibleError`, available as
`from fastsafetensors import BudgetInfeasibleError`. It subclasses `ValueError`
so callers can distinguish an infeasible plan from other invalid arguments.

The loader subtracts the selected copier's fixed device pools from this budget
before fitting the queue and planning chunks. The budget still excludes
allocator rounding and memory used outside the loader. When deriving it from
free device memory, leave a reserve for those costs;
`max(5% of free memory, 1 GiB)` is a starting point, not a guarantee for every
workload. The unified O_DIRECT reader reserves 16 MiB per worker (128 MiB by
default) and keeps its pinned pool for the life of the process. It reserves for
the configured worker limit, even if fewer buffers have been allocated so far.
If the pool was already populated before measuring free memory, this subtraction
can count its bytes again. Reserve any post-load conversion memory separately.

This estimates one loader's configured pool, not the process-wide high-water
mark. Account separately for memory retained by earlier loads with more workers
or used by concurrent loaders. Custom chunk-capable copiers using
`device_memory_budget` must implement both `chunk_transient_multiplier(paths)`
and `fixed_device_overhead(paths)`.

`use_chunk_budget_as_allocation_size: true` allocates each chunk buffer at its
planner budget instead of its exact byte span. The loader still reads only the
selected byte ranges. Stable allocation sizes improve caching-allocator reuse
and require either `max_batch_bytes` or `device_memory_budget`. Custom copiers
must support `set_chunk(byte_ranges, names, allocation_size)` to use this option.
Legacy two-argument `set_chunk` implementations are supported only when
`use_chunk_budget_as_allocation_size` is disabled; enabling it raises `TypeError`.

```json
{
  "parallel": {
    "use_pipeline": true,
    "queue_size": 0,
    "max_batch_bytes": 4294967296,
    "use_chunk_budget_as_allocation_size": true,
    "device_memory_budget": 12884901888
  }
}
```

Direct `ParallelLoader` users may also set `accumulate_resident=False` when
yielded tensors are copied into destinations allocated before loading. Leave
it at its default, `True`, when yielded tensors remain resident.
For consumers that retain only some yielded tensors on the device, keep
`accumulate_resident=True` and pass `resident_tensor(name) -> bool` to identify
those tensors. This affects memory accounting, not which tensors are read or
where they are moved. Each non-resident tensor must be relocated and its device
storage released before requesting the next tensor. Under broadcast, the
predicate must give identical results on every rank and depend only on the
tensor name. Passing it with `accumulate_resident=False` raises `ValueError`.

Direct Python API users can opt into `ParallelLoader(...,
borrowed_tensors=True, accumulate_resident=False)` for a single-process loader
group (`pg=None` or `all_local=True`). This skips yield clones and their planner
reservation. Returned tensors and derived views do not retain loader memory;
complete all use, including asynchronous device reads, before advancing or
closing the iterator. Copy into independent storage to retain data. The default
is `False`, preserving independent outputs and the existing memory accounting.
This option is not exposed by `AutoLoader` configuration.

Low-level `copy_files_to_device()` returns shared-owning tensors by default;
`copy_files_to_device(borrowed_tensors=True)` selects non-owning storage valid
until buffer close. Holding a small owning view retains the complete allocation.
See [Lifetime contract](./overview.md#lifetime-contract) for both modes and the
live-allocation metrics.

## Configuration Examples

### 1. Minimal — All Defaults (no config file needed)

```python
from contextlib import closing

from fastsafetensors import SingleGroup, AutoLoader

pg = SingleGroup()
loader = AutoLoader(pg, files, device="cuda:0")
with closing(loader.iterate_weights()) as weights:
    for key, tensor in weights:
        process(key, tensor)
loader.close()
```

No config file. Uses `loader="base"`, `gds`, serial mode.

`closing()` frees the load buffers even if the loop exits early; `loader.close()` does not. See [Basic API usage](./overview.md#basic-api-usage).

### 2. Base Loader with GDS

```json
{
  "loader": "base",
  "base": {
    "copier_type": "gds"
  }
}
```

Enables GPU Direct Storage for NVMe-to-GPU transfers, bypassing host CPU/memory.

### 3. Base Loader with FGDS

```json
{
  "loader": "base",
  "base": {
    "copier_type": "fgds",
    "max_threads": 16
  }
}
```

Uses the FGDS (alternative GPU Direct Storage) backend via `libfgds.so`. This provides
another direct NVMe-to-GPU path, useful on systems where FGDS is preferred over cuFile GDS.
When `libfgds.so` is not available, the loader gracefully falls back to the `nogds` copier.

### 4. Base Loader with Pipeline Mode

```json
{
  "parallel": {
    "use_pipeline": true,
    "max_concurrent_producers": 1,
    "queue_size": 0,
    "use_tqdm_on_load": true
  }
}
```

Overlaps `copy_files` with `broadcast` for higher throughput.

### 5. 3FS Loader

```json
{
  "loader": "3fs",
  "3fs": {
    "mount_point": "/mnt/3fs",
    "entries": 64,
    "io_depth": 0,
    "buffer_size": 67108864
  }
}
```

Uses ThreeFSLoader with 3FS USRBIO backend.

### 6. Full Reference

```json
{
  "loader": "base",
  "framework": "pytorch",
  "base": {
    "copier_type": "gds",
    "bbuf_size_kb": 16384,
    "max_threads": 16
  },
  "3fs": {
    "mount_point": "/mnt/3fs",
    "entries": 64,
    "io_depth": 0,
    "buffer_size": 67108864
  },
  "parallel": {
    "use_pipeline": false,
    "max_concurrent_producers": 1,
    "queue_size": 0,
    "use_tqdm_on_load": true,
    "max_batch_bytes": null,
    "use_chunk_budget_as_allocation_size": false,
    "device_memory_budget": null
  },
  "debug": {
    "debug_log": false,
    "set_numa": true,
    "disable_cache": true
  }
}
```

Each loader type has its own extension section (e.g., `base`, `3fs`).
Adding a new loader only requires a new section — no changes to `config.py`.
