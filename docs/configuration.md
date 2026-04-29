# Configuration Guide

## Configuration Discovery

`UnifiedLoader` loads configuration in the following priority (highest first):

1. **Environment variable** — `FASTSAFETENSORS_CONFIG=/path/to/config.yaml`
2. **Default path** — `./fastsafetensors.yaml` in the working directory (if it exists)
3. **Built-in defaults** — `LoaderConfig()` dataclass defaults

All fields are optional. Unspecified fields fall back to built-in defaults.

## Default Configuration

When no config file is found, `UnifiedLoader` uses these defaults:

```yaml
loader: "base"
framework: "pytorch"

# parallel
use_pipeline: false

# debug
debug_log: false
set_numa: true
disable_cache: true
```

The base loader extension defaults to `copier_type: "gds"` (GPU Direct Storage).

## Default Call Chain

```
UnifiedLoader(pg, files, device)
  ├─ load_config() → no config file → LoaderConfig()
  ├─ _resolve_loader_class("base") → SafeTensorsFileLoader
  ├─ get_extension_config("base") → {}
  │   └─ process_extension_config({}) → {nogds: False}
  ├─ SafeTensorsFileLoader(pg, device, nogds=False, ...)
  ├─ create_parallel_kwargs() → {queue_size: -1}  (use_pipeline=False)
  └─ PipelineParallel(pg, loader, files, queue_size=-1)
       └─ serial: copy_files → broadcast → copy_files → ...
```

## queue_size Semantics

| `queue_size` | Mode | GPU Memory | Behavior |
|---|---|---|---|
| `-1` | Fully serial | 1 batch | `copy_files → broadcast → copy_files → ...` |
| `0` | Unbuffered pipeline | Up to 2 batches | 1 batch copying + 1 batch broadcasting concurrently |
| `>0` | Buffered pipeline | Up to `queue_size+1` batches | Producer fills queue while consumer broadcasts |

`use_pipeline: false` forces `queue_size=-1` (serial, minimal GPU memory).

## Configuration Examples

### 1. Minimal — All Defaults (no config file needed)

```python
from fastsafetensors import SingleGroup, UnifiedLoader

pg = SingleGroup()
loader = UnifiedLoader(pg, files, device="cuda:0")
for key, tensor in loader.iterate_weights():
    process(key, tensor)
loader.close()
```

No config file. Uses `loader="base"`, `nogds`, serial mode.

### 2. Base Loader with GDS

```yaml
loader: "base"
base:
  copier_type: "gds"
```

Enables GPU Direct Storage for NVMe-to-GPU transfers, bypassing host CPU/memory.

### 3. Base Loader with Pipeline Mode

```yaml
parallel:
  use_pipeline: true
  max_concurrent_producers: 1
  queue_size: 0               # unbuffered: up to 2 batches in GPU memory
  use_tqdm_on_load: true
```

Overlaps `copy_files` with `broadcast` for higher throughput.

### 4. 3FS Loader

```yaml
loader: "3fs"
"3fs":
  mount_point: "/mnt/3fs"
  entries: 64
  io_depth: 0
  buffer_size: 67108864       # 64 MB
```

Uses ThreeFSLoader with 3FS USRBIO backend.

### 5. Full Reference

```yaml
loader: "base"                    # "base" | "3fs"
framework: "pytorch"              # "pytorch" | "paddle"

base:
  copier_type: "gds"              # "gds" | "nogds"
  bbuf_size_kb: 16384             # bounce buffer size in KB
  max_threads: 16                 # max host CPU copy threads

"3fs":
  mount_point: "/mnt/3fs"
  entries: 64
  io_depth: 0
  buffer_size: 67108864           # 64 MB

parallel:
  use_pipeline: false             # false = serial (queue_size forced to -1)
  max_concurrent_producers: 1     # ignored if use_pipeline=false
  queue_size: 0                   # ignored if use_pipeline=false
  use_tqdm_on_load: true          # ignored if use_pipeline=false

debug:
  debug_log: false
  set_numa: true
  disable_cache: true
```

Each loader type has its own extension section (e.g., `base:`, `3fs:`).
Adding a new loader only requires a new section — no changes to `config.py`.
