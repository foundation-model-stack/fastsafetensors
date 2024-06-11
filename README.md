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

Check more details in [doc/overview.md](doc/overview.md)

## Dependencies

We currently test fastsafetensors only with python 3.11, pytorch 2.1, and cuda-12.
Note: when using different versions of pytorch, you may require changes on build environments for libpytorch since it seems slightly changing ABIs.

## Install from PyPi

```bash
pip install fastsafetensors
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

After installing fastsafetensors with `pip` or `make install`, run

```bash
make unittest
```

## Basic API usages

`SafeTensorsFileLoader` is the primary entrypoint of the fastsafetensors library. To use it, pass either `SingleGroup()` for simple inference or `ProcessGroup()` (from `torch.distributed`) for tensor-parallel inference. The loader supports both CPU and CUDA devices, with optional GPU Direct Storage (GDS) support. You can specify the device and GDS settings using the `device` and `nogds` arguments, respectively. Note that if GDS is not available, the loader will fail to open files when `nogds=False`. For more information on enabling GDS, please refer to the NVIDIA documentation.

After creating a `SafeTensorsFileLoader` instance, first map target files and a rank using the `.add_filenames()` method. Then, call `.copy_file_to_device()` to trigger the actual file copies on aggregated GPU memory fragments and directly instantiate a group of Tensors. Once the files are loaded, you can retrieve a tensor using the `.get_tensor()` method. Additionally, you can obtain sharded tensors by `.get_sharded()`, which internally run collective operations in `torch.distributed`.

Important: To release the GPU memory allocated for tensors, you must explicitly call the `.close()` method. This is because Fastsafetensors allows multiple tensors to share a limited number of GPU memory fragments. As a result, it is the user's responsibility to ensure that all tensors are properly released before calling `.close()`, which will then safely release the underlying GPU memory.

## Example: single run

examples/run_single.py:

```python
import torch
from fastsafetensors import SafeTensorsFileLoader, SingleGroup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loader = SafeTensorsFileLoader(SingleGroup(), device, nogds=True, debug_log=True)
loader.add_filenames({0: ["a.safetensors", "b.safetensors"]}) # {rank: files}
fb = loader.copy_files_to_device()
tensor_a0 = fb.get_tensor(tensor_name="a0")
print(f"a0: {tensor_a0}")
loader.close()
```

```
cd examples
python run_single.py
```

Example output:

```
add_filenames 1: path=a.safetensors
[DEBUG] raw_device_pointer: raw_alloc: 0x7acf000, length=256, elapsed=3 us
[DEBUG] nogds_file_reader.submit_read: cudaHostAlloc, size=1048576, elapsed=10 us
[DEBUG] nogds_file_reader.submit_read #3, thread_id=1
[DEBUG] nogds_file_reader._thread: read (mmap=0), fd=4, offset=104, count=256, c=256, copy=13 us, cuda_copy=0 us
wait_io: tensor=a0
a0: tensor([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
        [ 2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.],
        [ 3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.],
        [ 4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.],
        [ 5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.],
        [ 6.,  6.,  6.,  6.,  6.,  6.,  6.,  6.],
        [ 7.,  7.,  7.,  7.,  7.,  7.,  7.,  7.],
        [ 8.,  8.,  8.,  8.,  8.,  8.,  8.,  8.],
        [ 9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.],
        [10., 10., 10., 10., 10., 10., 10., 10.],
        [11., 11., 11., 11., 11., 11., 11., 11.],
        [12., 12., 12., 12., 12., 12., 12., 12.],
        [13., 13., 13., 13., 13., 13., 13., 13.],
        [14., 14., 14., 14., 14., 14., 14., 14.],
        [15., 15., 15., 15., 15., 15., 15., 15.]], dtype=torch.float16)
[DEBUG] ~nogds_file_reader: elapsed=28 us
[DEBUG] ~raw_device_pointer: torch_raw_delete: 0x7acf000, elapsed=0 us
```

## Example: parallel run

examples/run_parallel.py:
```python
import torch
import torch.distributed as dist
from fastsafetensors import SafeTensorsFileLoader
dist.init_process_group(backend="gloo")
dist.barrier()
pg = dist.group.WORLD
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loader = SafeTensorsFileLoader(pg, device, nogds=True, debug_log=True)
loader.add_filenames({0: ["a.safetensors"], 1:["b.safetensors"]}) # {rank: files}
fb = loader.copy_files_to_device()
tensor_a0 = fb.get_tensor(tensor_name="a0") # broadcast
tensor_b0_sharded = fb.get_sharded(tensor_name="b0", dim=1) # partition and scatter
print(f"RANK {pg.rank()}: tensor_a0={tensor_a0}")
print(f"RANK {pg.rank()}: tensor_b0_sharded={tensor_b0_sharded}")
loader.close()
```

You can test the script with torchrun

```bash
cd examples
torchrun --nnodes=2 --master_addr=0.0.0.0 --master_port=1234 --node_rank=0 run_parallel.py &
PIDS+=$($!)
torchrun --nnodes=2 --master_addr=0.0.0.0 --master_port=1234 --node_rank=1 run_parallel.py &
PIDS+=$($!)
wait ${PIDS[@]}
```

Example output:

```
add_filenames 1: path=a.safetensors
[DEBUG] raw_device_pointer: raw_alloc: 0x6ba1000, length=256, elapsed=2 us
[DEBUG] nogds_file_reader.submit_read: cudaHostAlloc, size=1048576, elapsed=10 us
[DEBUG] nogds_file_reader.submit_read #3, thread_id=1
[DEBUG] nogds_file_reader._thread: read (mmap=0), fd=15, offset=104, count=256, c=256, copy=15 us, cuda_copy=0 us
wait_io: tensor=a0
shuffle: broadcast, tensor_name=a0, shape=torch.Size([16, 8]), self.rank=0, pg.rank()=0, has_tensor=True
add_filenames 2: path=b.safetensors
[DEBUG] raw_device_pointer: raw_alloc: 0x7cbb000, length=256, elapsed=2 us
[DEBUG] nogds_file_reader.submit_read: cudaHostAlloc, size=1048576, elapsed=12 us
[DEBUG] nogds_file_reader.submit_read #3, thread_id=1
[DEBUG] nogds_file_reader._thread: read (mmap=0), fd=15, offset=104, count=256, c=256, copy=15 us, cuda_copy=0 us
wait_io: tensor=b0
shuffle: broadcast, tensor_name=a0, shape=torch.Size([16, 8]), self.rank=0, pg.rank()=1, has_tensor=False
_get_tensor: free_dev_ptrs, lidx=0, src=a.safetensorsshuffle: use cache, tensor_name=a0

[DEBUG] ~raw_device_pointer: torch_raw_delete: 0x6ba1000, elapsed=0 us
shuffle: use cache, tensor_name=a0
_get_tensor: free_dev_ptrs, lidx=0, src=a.safetensors
shuffle: scatter, tensor_name=b0, shape=torch.Size([16, 8])->torch.Size([16, 4]), self.rank=1, pg.rank()=0, rank_slices=[(slice(None, None, None), slice(0, 4, 1)), (slice(None, None, None), slice(4, 8, 1))], len(scatter_list)=0
shuffle: scatter, tensor_name=b0, shape=torch.Size([16, 8])->torch.Size([16, 4]), self.rank=1, pg.rank()=1, rank_slices=[(slice(None, None, None), slice(0, 4, 1)), (slice(None, None, None), slice(4, 8, 1))], len(scatter_list)=2
_get_tensor: free_dev_ptrs, lidx=0, src=b.safetensors
[DEBUG] ~raw_device_pointer: torch_raw_delete: 0x7cbb000, elapsed=0 us
RANK 0: tensor_a0=tensor([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
        [ 2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.],
        [ 3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.],
        [ 4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.],
        [ 5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.],
        [ 6.,  6.,  6.,  6.,  6.,  6.,  6.,  6.],
        [ 7.,  7.,  7.,  7.,  7.,  7.,  7.,  7.],
        [ 8.,  8.,  8.,  8.,  8.,  8.,  8.,  8.],
        [ 9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.],
        [10., 10., 10., 10., 10., 10., 10., 10.],
        [11., 11., 11., 11., 11., 11., 11., 11.],
        [12., 12., 12., 12., 12., 12., 12., 12.],
        [13., 13., 13., 13., 13., 13., 13., 13.],
        [14., 14., 14., 14., 14., 14., 14., 14.],
        [15., 15., 15., 15., 15., 15., 15., 15.]], dtype=torch.float16)RANK 1: tensor_a0=tensor([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
        [ 2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.],
        [ 3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.],
        [ 4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.],
        [ 5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.],
        [ 6.,  6.,  6.,  6.,  6.,  6.,  6.,  6.],
        [ 7.,  7.,  7.,  7.,  7.,  7.,  7.,  7.],
        [ 8.,  8.,  8.,  8.,  8.,  8.,  8.,  8.],
        [ 9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.],
        [10., 10., 10., 10., 10., 10., 10., 10.],
        [11., 11., 11., 11., 11., 11., 11., 11.],
        [12., 12., 12., 12., 12., 12., 12., 12.],
        [13., 13., 13., 13., 13., 13., 13., 13.],
        [14., 14., 14., 14., 14., 14., 14., 14.],
        [15., 15., 15., 15., 15., 15., 15., 15.]], dtype=torch.float16)

RANK 1: tensor_b0_sharded=tensor([[ 0.,  0.,  0.,  0.],
        [ 1.,  1.,  1.,  1.],
        [ 2.,  2.,  2.,  2.],
        [ 3.,  3.,  3.,  3.],
        [ 4.,  4.,  4.,  4.],
        [ 5.,  5.,  5.,  5.],
        [ 6.,  6.,  6.,  6.],
        [ 7.,  7.,  7.,  7.],
        [ 8.,  8.,  8.,  8.],
        [ 9.,  9.,  9.,  9.],
        [10., 10., 10., 10.],
        [11., 11., 11., 11.],
        [12., 12., 12., 12.],
        [13., 13., 13., 13.],
        [14., 14., 14., 14.],
        [15., 15., 15., 15.]], dtype=torch.float16)RANK 0: tensor_b0_sharded=tensor([[ 0.,  0.,  0.,  0.],
        [ 1.,  1.,  1.,  1.],
        [ 2.,  2.,  2.,  2.],
        [ 3.,  3.,  3.,  3.],
        [ 4.,  4.,  4.,  4.],
        [ 5.,  5.,  5.,  5.],
        [ 6.,  6.,  6.,  6.],
        [ 7.,  7.,  7.,  7.],
        [ 8.,  8.,  8.,  8.],
        [ 9.,  9.,  9.,  9.],
        [10., 10., 10., 10.],
        [11., 11., 11., 11.],
        [12., 12., 12., 12.],
        [13., 13., 13., 13.],
        [14., 14., 14., 14.],
        [15., 15., 15., 15.]], dtype=torch.float16)
```