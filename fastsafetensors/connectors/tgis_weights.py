# Copyright 2024 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0

import os
import torch
import torch.distributed as dist
from typing import List, Tuple, Optional, Dict
from ..loader import SafeTensorsFileLoader

class Weights:
    def __init__(self, filenames:List[str],
                 device: torch.device,
                 dtype: torch.dtype,
                 pg: dist.ProcessGroup,
                 debug_log: bool=False,
                 aliases: Optional[Dict[str, List[str]]] = None,
                 prefix: Optional[str] = None,
                 memtrace: bool = False,
                 nogds: bool = False,
                 max_copier_threads: int = 16, # should be same as the number of physical CPUs on a node
                 bbuf_size_kb_total = 16 * 1024, # should be same as L2 cache size
            ):
        self.memtrace = memtrace
        if self.memtrace:
            torch.cuda.memory._record_memory_history(max_entries=100000)
        import time
        t0 = time.time_ns()
        self.loader = SafeTensorsFileLoader(pg, device, bbuf_size_kb=bbuf_size_kb_total//pg.size(), max_threads=max_copier_threads, nogds=nogds, debug_log=debug_log)
        # ensure ordering so that we can eagerly free GPU memory at shuffling (layers are often loaded sequentially)
        filenames = sorted(filenames, key=lambda x: os.path.basename(x))
        rank_filenames: Dict[str, List[str]] = {rank: [] for rank in range(0, pg.size())}
        file_sizes = []
        total_size = 0
        for idx in range(0, len(filenames)):
            rank_filenames[idx % pg.size()].append(filenames[idx])
            s = os.stat(filenames[idx])
            total_size += s.st_size
            file_sizes.append(s.st_size)
        self.loader.add_filenames(rank_filenames)
        file_sizes = sorted(file_sizes)
        max_copy_block_size = file_sizes[-1]
        if len(filenames) < max_copier_threads:
            max_copy_block_size = total_size // max_copier_threads
            if max_copy_block_size % bbuf_size_kb_total*1024 > 0:
                max_copy_block_size = max_copy_block_size - max_copy_block_size % (bbuf_size_kb_total*1024) + (bbuf_size_kb_total*1024)
        self.fb = self.loader.copy_files_to_device(dtype, max_copy_block_size=max_copy_block_size)
        t1 = time.time_ns()
        if debug_log:
            print(f"Weights.init: {(t1 - t0) / 1000 / 1000 / 1000} sec")
        self.device = device
        self.dtype = dtype
        self.debug_log = debug_log
        if aliases is None:
            aliases = {}
        self.prefix = prefix
        self.aliases = aliases
        self.process_group = pg

    def close(self):
        if self.process_group.size() > 1:
            self.fb.close()
            self.loader.close()
            torch.cuda.empty_cache()
        if self.memtrace:
            torch.cuda.memory._dump_snapshot(f"memtrace-{self.process_group.rank()}.pickle")
            torch.cuda.memory._record_memory_history(enabled=None)

    def get_filename(self, tensor_name: str) -> Tuple[str, str]:
        if self.prefix is not None:
            tensor_name = f"{self.prefix}.{tensor_name}"
        filename = self.fb.get_filename(tensor_name)
        if filename is not None:
            return (filename, tensor_name)
        if filename not in self.aliases:
            raise ValueError(f"get_filename, tensor_name {tensor_name} was not found in files")
        tensor_name = self.aliases[tensor_name]
        filename = self.fb.get_filename(tensor_name)
        if filename not in self.aliases:
            raise ValueError(f"get_filename, tensor_name {tensor_name} was not found in files")
        return (filename, tensor_name)

    def get_shape(self, tensor_name: str)->torch.Size:
        return self.fb.get_shape(tensor_name)

    def get_tensor(self, tensor_name: str)->torch.Tensor:
        return self.fb.get_tensor(tensor_name, device=self.device, dtype=self.dtype)

    def push_tensor(self, tensor_name: str, dst_rank: int)->torch.Tensor:
        return self.fb.push_tensor(tensor_name, dst_rank, device=self.device, dtype=self.dtype)

    def get_partial_sharded(self, tensor_name: str, dim: int)->torch.Tensor:
        return self.fb.get_sharded(tensor_name, dim, device=self.device, dtype=self.dtype)

    def get_sharded(self, tensor_name: str, dim: int=1)->torch.Tensor:
        return self.fb.get_sharded(tensor_name, dim, device=self.device, dtype=self.dtype)

    def get_weights_col_packed_qkv(self, prefix: str, quantize: str)->torch.Tensor:
        if quantize in ["gptq", "awq"]:
            raise NotImplementedError("Quantization is not supported yet")
        return self.fb.get_sharded_packed_qkv(f"{prefix}.weight", device=self.device, dtype=self.dtype)

    def get_multi_weights_col(self, prefixes: List[str], quantize: str, dim: int)->torch.Tensor:
        if quantize in ["gptq", "awq"]:
            raise NotImplementedError("Quantization is not supported yet")
        tensor_names = [f"{prefix}.weight" for prefix in prefixes]
        return self.fb.get_multi_cols(tensor_names, dim, device=self.device, dtype=self.dtype)

    def get_tensor_shard(self, var, dim):
        world_size = self.process_group.size()
        rank = self.process_group.rank()
        block_size = var.size()[dim] // world_size
        start = rank * block_size
        stop = (rank + 1) * block_size
        if dim == 0:
            tensor = var[start:stop]
        elif dim == 1:
            tensor = var[:, start:stop]
        else:
            raise NotImplementedError("Let's make that generic when needed")
        tensor = tensor.to(dtype=self.dtype)
        tensor = tensor.to(device=self.device)
        return tensor

    def get_multi_weights_row(self, prefix: str, quantize: str)->torch.Tensor:
        if quantize in ["gptq", "awq"]:
            raise NotImplementedError("Quantization is not supported yet")
        return self.fb.get_sharded(f"{prefix}.weight", 1, device=self.device, dtype=self.dtype)