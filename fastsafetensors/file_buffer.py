# Copyright 2024 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0

import os
import torch
import torch.distributed as dist
from typing import Dict, List, Tuple
from collections import OrderedDict

from .tensor_factory import LazyTensorFactory

class FilesBufferOnDevice:
    r""" Device buffer for .safetensors files.
        Users can call get_tensor(), get_sharded(), etc. to instantiate (sharded) tensors from the device buffer.
        Note that for multi-process loading, users must follow the single-program multiple-data (SPMD) paradigm, which is common for torch.distributed programs.
        In other words, users must ensure that every worker process calls the methods here in the same order.
        This is because methods here reuse torch.distributed operations: broadcast, scatter, recv, and send.
        They synchornously wait all the workers to execute copies among processes.

        Users should create this instance with SafeTensorsFileLoader.submit_io().

    Args:
        rank_loaders (Dict<rank, list(LazyTensorFacotry)>): Tensor factories per rank, which hold device pointers for buffers.
        pg (dist.ProcessGroup): process group for pytorch distributed. SingleGroup is available for single GPU use-cases.
        auto_mem_delete (bool): automatically release device buffers when all the tensors are shuffled.

    Examples:
        See examples/run_single.py and examples/run_parallel.py.
    """
    def __init__(self, rank_loaders: Dict[int, List[LazyTensorFactory]], pg: dist.ProcessGroup, auto_mem_delete=True):
        self.rank_loaders: Dict[int, List[LazyTensorFactory]] = rank_loaders
        self.key_to_rank_lidx: Dict[str, Tuple[int, int]] = {}
        self.instantiated: Dict[int, Dict[int, Dict[str, bool]]] = {} # rank, key name
        for rank, loaders in rank_loaders.items():
            self.instantiated[rank] = {}
            for lidx, loader in enumerate(loaders):
                for key in loader.metadata.tensors.keys():
                    if key in self.key_to_rank_lidx:
                        raise Exception(f"FilesBufferOnDevice: key {key} must be unique among files")
                    self.key_to_rank_lidx[key] = (rank, lidx)
                self.instantiated[rank][lidx] = {}
        self.auto_mem_delete = auto_mem_delete and pg.size() > 1
        self.pg = pg

    def close(self):
        for _, loaders in self.rank_loaders.items():
            for loader in loaders:
                loader.free_dev_ptrs()
        self.rank_loaders = {}

    def get_filename(self, tensor_name: str)->str:
        if tensor_name not in self.key_to_rank_lidx:
            return None
        (rank, lidx) = self.key_to_rank_lidx[tensor_name]
        return self.rank_loaders[rank][lidx].metadata.src

    def get_shape(self, tensor_name: str)->torch.Size:
        (rank, lidx) = self._get_rank_lidx(tensor_name)
        return self.rank_loaders[rank][lidx].metadata.tensors[tensor_name].shape

    def _get_rank_lidx(self, tensor_name: str)->Tuple[int, int]:
        if tensor_name not in self.key_to_rank_lidx:
            raise ValueError(f"_get_rank: key {tensor_name} was not found in files")
        return self.key_to_rank_lidx[tensor_name]

    def _get_tensor(self, rank: int, lidx: int, tensor_name: str, ret: torch.Tensor, device: torch.device, dtype: torch.dtype)->torch.Tensor:
        loader = self.rank_loaders[rank][lidx]
        if self.auto_mem_delete:
            self.instantiated[rank][lidx][tensor_name] = True
            if len(self.instantiated[rank][lidx]) == len(loader.metadata.tensors):
                if loader.debug_log and self.pg.rank() == rank:
                    print(f"_get_tensor: free_dev_ptrs, lidx={lidx}, src={loader.metadata.src}")
                loader.free_dev_ptrs()
        if ret is None:
            return ret
        if (device is not None and (device != ret.device)) or (dtype is not None and (dtype != ret.dtype)):
            ret = ret.to(device=device, dtype=dtype)
        return ret

    def get_sharded(self, tensor_name: str, dim: int, device: torch.device=None, dtype: torch.dtype=None)->torch.Tensor:
        """
        partition a tensor instance with the key tensor_name at the dimension dim and return it.
        In multi-process loading, this eventually calls torch.distributed.scatter.
        A special dim is -1, which broadcast a tensor to all the ranks (== get_tensor()).
        """
        (rank, lidix) = self._get_rank_lidx(tensor_name)
        t = self.rank_loaders[rank][lidix].shuffle(self.pg, tensor_name, dim)
        return self._get_tensor(rank, lidix, tensor_name, t, device, dtype)

    def get_tensor(self, tensor_name: str, device: torch.device=None, dtype: torch.dtype=None)->torch.Tensor:
        """
        get a tensor instance with the key tensor_name from a local or remote rank.
        In multi-process loading, this eventually calls torch.distributed.broadcast.
        So, every rank will allocate the same tensor at each device memroy.
        In single-process loading, this directly instantiates a tensor from the device buffer with zero copy.
        """
        return self.get_sharded(tensor_name, -1, device, dtype)

    def push_tensor(self, tensor_name: str, dst_rank: int,  device: torch.device=None, dtype: torch.dtype=None) -> torch.Tensor:
        """
        push a tensor instance with the key tensor_name from a rank to a destination rank dst_rank.
        In multi-process loading, this eventually calls torch.distributed.send if the rank has the tensor instance.
        The destination rank will call torch.distributed.recv.
        Other ranks do nothing.
        """
        (rank, lidix) = self._get_rank_lidx(tensor_name)
        t = self.rank_loaders[rank][lidix].push(self.pg, tensor_name, dst_rank, rank)
        return self._get_tensor(rank, lidix, tensor_name, t, device, dtype)

    def get_sharded_packed_qkv(self, tensor_name: str, device: torch.device=None, dtype: torch.dtype=None)->torch.Tensor:
        (rank, lidix) = self._get_rank_lidx(tensor_name)
        t = self.rank_loaders[rank][lidix].shuffle_packed_qkv(self.pg, tensor_name)
        return self._get_tensor(rank, lidix, tensor_name, t, device, dtype)

    def get_multi_cols(self, tensor_names: List[str], dim: int, device: torch.device=None, dtype: torch.dtype=None)->torch.Tensor:
        rank_lidixs: Dict[Tuple[int, int], List[str]] = {}
        for tensor_name in tensor_names:
            ranklidx = self._get_rank_lidx(tensor_name)
            if ranklidx in rank_lidixs:
                rank_lidixs[ranklidx].append(tensor_name)
            else:
                rank_lidixs[ranklidx] = [tensor_name]
        ts: List[torch.Tensor] = []
        for (rank, lidix), tns in sorted(rank_lidixs.items(), key=lambda x:x[0]):
            ts.append(self.rank_loaders[rank][lidix].shuffle_multi_cols(self.pg, tns, dim))
        if len(ts) == 1:
            # fastpath: tensors at the same layer are often in the same file
            return self._get_tensor(rank, lidix, rank_lidixs[(rank, lidix)][0], ts[0], device, dtype)
        ret = torch.cat(ts, dim=dim)
        if self.auto_mem_delete:
            for tensor_name in tensor_names:
                (rank, lidx) = self._get_rank_lidx(tensor_name)
                loader = self.rank_loaders[rank][lidix]
                self.instantiated[rank][lidx][tensor_name] = True
                if len(self.instantiated[rank][lidx]) == len(loader.metadata.tensors):
                    if loader.debug_log and self.pg.rank() == rank:
                        print(f"get_multi_cols: free_dev_ptrs, rank={rank}, lidx={lidx}, src={loader.metadata.src}")
                    loader.free_dev_ptrs()
        if (device is not None and (device != ret.device)) or (dtype is not None and (dtype != ret.dtype)):
            ret = ret.to(device=device, dtype=dtype)
        return ret

    def as_dict(self, tensor_shard_dim: OrderedDict[str, int])->Dict[str, torch.Tensor]:
        tensors: Dict[str, torch.Tensor] = {}
        for tensor_name, dim in tensor_shard_dim.items():
            (rank, lidx) = self._get_rank_lidx(tensor_name)
            loader = self.rank_loaders[rank][lidx]
            tensors[tensor_name] = loader.shuffle(self.pg, tensor_name, dim)
            if self.auto_mem_delete:
                self.instantiated[rank][lidx][tensor_name] = True
                if len(self.instantiated[rank][lidx]) == len(loader.metadata.tensors):
                    if loader.debug_log and self.pg.rank() == rank:
                        print(f"as_dict: free_dev_ptrs, rank={rank}, src={loader.metadata.src}")
                    loader.free_dev_ptrs()
        if self.auto_mem_delete:
            self.loaders = {}
        return tensors
