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

## Bounded Device Memory

`max_batch_bytes` caps each sub-file chunk. It must be at least as large as
the largest selected tensor because tensors are not split across chunks.

`device_memory_budget` bounds resident tensors and transient chunk buffers
across the load. Under distributed broadcast, every rank must use the same
value to produce an identical plan.

If the requested queue depth does not fit, the loader reduces it automatically,
down to `queue_size=-1`. If even serial loading cannot fit, loading raises
`BudgetInfeasibleError`, available as
`from fastsafetensors import BudgetInfeasibleError`. It subclasses `ValueError`
so callers can distinguish an infeasible plan from other invalid arguments.

The budget excludes allocator rounding, copier fixed pools, and memory used
outside the loader. When deriving it from free device memory, leave a reserve
for those costs; `max(5% of free memory, 1 GiB)` is a starting point, not a
guarantee for every workload. Reserve any post-load conversion memory separately.

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

## Configuration Examples

### 1. Minimal — All Defaults (no config file needed)

```python
from fastsafetensors import SingleGroup, AutoLoader

pg = SingleGroup()
loader = AutoLoader(pg, files, device="cuda:0")
for key, tensor in loader.iterate_weights():
    process(key, tensor)
loader.close()
```

No config file. Uses `loader="base"`, `gds`, serial mode.

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
