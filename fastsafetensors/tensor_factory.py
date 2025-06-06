# Copyright 2024 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.distributed as dist
from typing import Dict, List, Tuple
from collections import OrderedDict

from . import cpp as fstcpp
from .common import SafeTensorsMetadata, free_tensor_memory, paddle_loaded
from .copier.gds import GdsFileCopier
from .copier.nogds import NoGdsFileCopier

if paddle_loaded:
    import paddle
    import paddle.distributed as pdist

class LazyTensorFactory:
    def __init__(self, metadata: SafeTensorsMetadata, device: torch.device, rank: int, local_rank: bool, factory_idx_bits: int, lidx: int, nogds: bool, reader, debug_log: bool=False):
        self.metadata = metadata
        self.device = device
        if not local_rank:
            self.copier = None
        elif nogds:
            self.copier = NoGdsFileCopier(metadata, device, reader, debug_log)
        else:
            self.copier = GdsFileCopier(metadata, device, reader, debug_log)
        self.tensors: Dict[str, torch.Tensor] = {}
        self.shuffled: Dict[str, torch.Tensor] = {}
        self.gbuf: fstcpp.gds_device_buffer = None
        self.debug_log = debug_log
        self.rank = rank
        self.factory_idx_bits = factory_idx_bits
        self.lidx = lidx
        self.next_tag = 1

    def submit_io(self, use_buf_register: bool, max_copy_block_size: int):
        if self.copier is not None:
            self.gbuf = self.copier.submit_io(use_buf_register, max_copy_block_size)

    def wait_io(self, dtype: torch.dtype=None, noalign: bool=False):
        if self.copier is not None:
            self.tensors = self.copier.wait_io(self.gbuf, dtype=dtype, noalign=noalign)
            if self.debug_log:
                for name in self.tensors.keys():
                    print(f"wait_io: tensor={name}")
            self.copier = None

    def push(self, pg: dist.ProcessGroup, tensor_name: str, dst_rank: int, src_rank: int, group = None)->torch.Tensor:
        if pg.size() == 1:
            return self.tensors[tensor_name]
        tag = (self.next_tag << self.factory_idx_bits) + self.lidx
        self.next_tag += 1
        if pg.rank() != dst_rank and pg.rank() != src_rank:
            if self.debug_log:
                print(f"push: skip, tensor_name={tensor_name}, dst_rank={dst_rank}, pg.rank()={pg.rank()}, tag={tag}")
            return None
        elif pg.rank() == dst_rank and src_rank == dst_rank:
            if self.debug_log:
                print(f"push: nocopy, tensor_name={tensor_name}, dst_rank={dst_rank}, pg.rank()={pg.rank()}, tag={tag}")
            return self.tensors[tensor_name].clone().detach()
        frame = self.metadata.tensors[tensor_name]
        if pg.rank() == src_rank:
            if tensor_name not in self.tensors:
                raise Exception(f"push: tensor {tensor_name} was not found. released? lidx={self.lidx}")
            t = self.tensors[tensor_name].clone().detach()
            if self.debug_log:
                print(f"push: send, tensor_name={tensor_name}, shape={frame.shape}, dst_rank={dst_rank}, pg.rank()={pg.rank()}, tag={tag}")
            if self.metadata.framework == "pytorch":
                dist.send(t, dst_rank, group=pg, tag=tag)
            elif paddle_loaded and self.metadata.framework == "paddle":
                pdist.send(t, dst_rank, group=group)
            return None
        
        if self.debug_log:
            print(f"push: recv, tensor_name={tensor_name}, shape={frame.shape}, src_rank={src_rank}, pg.rank()={pg.rank()}, tag={tag}")
        
        if self.metadata.framework == "pytorch":
            t = torch.empty(size=frame.shape, dtype=frame.dtype, device=self.device)
            dist.recv(t, src_rank, group=pg, tag=tag)
        elif paddle_loaded and self.metadata.framework == "paddle":
            t = paddle.to_tensor(paddle.empty(size=frame.shape, dtype=frame.dtype), place=self.device)
            pdist.recv(t,src_rank, group=group)
        return t

    def shuffle(self, pg: dist.ProcessGroup, tensor_name: str, dim: int, group = None)->torch.Tensor:
        if pg.size() == 1:
            return self.tensors[tensor_name]
        if tensor_name in self.shuffled:
            if self.debug_log:
                print(f"shuffle: use cache, tensor_name={tensor_name}")
            t = self.shuffled[tensor_name].clone().detach()
            return t
        frame = self.metadata.tensors[tensor_name]
        if dim == -1:
            if tensor_name in self.tensors:
                dst = self.tensors[tensor_name].clone().detach()
            else:
                if self.metadata.framework == "pytorch":
                    dst = torch.empty(size=frame.shape, dtype=frame.dtype, device=self.device)
                elif paddle_loaded and self.metadata.framework == "paddle":
                    dst = paddle.to_tensor(paddle.empty(shape=frame.shape, dtype=frame.dtype), place=self.device)
                    
            if self.debug_log:
                print(f"shuffle: broadcast, tensor_name={tensor_name}, shape={frame.shape}, self.rank={self.rank}, pg.rank()={pg.rank()}, has_tensor={tensor_name in self.tensors}")
            if self.metadata.framework == "pytorch":
                dist.broadcast(dst, self.rank, group=pg)
            elif paddle_loaded and self.metadata.framework == "paddle":
                pdist.broadcast(dst, self.rank, group=group)
        else:
            rank_slices: List[Tuple] = [() for i in range(0, pg.size())]
            size = frame.shape[dim]
            block_size = (size + pg.size() - 1) // pg.size()
            for rank in range(0, pg.size()):
                for i in range(0, len(frame.shape)):
                    if i < dim:
                        rank_slices[rank] += (slice(None,None,None),)
                    elif i == dim:
                        rank_slices[rank] += (slice(rank * block_size, (rank + 1) * block_size, 1),)
                        break
            scatter_list: List[torch.Tensor] = []
            new_frame = frame[rank_slices[pg.rank()]]
            
            if self.metadata.framework == "pytorch":
                dst = torch.empty(size=new_frame.shape, dtype=new_frame.dtype, device=self.device)
            elif paddle_loaded and self.metadata.framework == "paddle":
                dst = paddle.to_tensor(paddle.empty(shape=new_frame.shape, dtype=frame.dtype), place=self.device)
            if self.rank == pg.rank():
                if tensor_name not in self.tensors:
                    raise Exception(f"shuffle: tensor {tensor_name} was not found, released? lidx={self.lidx}")
                t = self.tensors[tensor_name]
                scatter_list = [t[rank_slices[rank]].contiguous() for rank in range(0, pg.size())] # scatter requires contiguous tensor
            if self.debug_log:
                print(f"shuffle: scatter, tensor_name={tensor_name}, shape={frame.shape}->{new_frame.shape}, self.rank={self.rank}, pg.rank()={pg.rank()}, rank_slices={rank_slices}, len(scatter_list)={len(scatter_list)}")
            if self.metadata.framework == "pytorch":
                dist.scatter(dst, scatter_list=scatter_list, src=self.rank, group=pg)
            elif paddle_loaded and self.metadata.framework == "paddle":
                pdist.scatter(dst, tensor_list=scatter_list, src=self.rank, group=group)
        self.shuffled[tensor_name] = dst
        return dst

    def shuffle_packed_qkv(self, pg: dist.ProcessGroup, tensor_name: str, group = None)->torch.Tensor:
        if tensor_name in self.shuffled:
            if self.debug_log:
                print(f"shuffle: use cache, tensor_name={tensor_name}")
            t = self.shuffled[tensor_name].clone().detach()
            return t
        frame = self.metadata.tensors[tensor_name]
        total_size = frame.shape[0]
        single_size = total_size // 3
        block_size = (single_size + pg.size() - 1) // pg.size()
        scatter_list: List[torch.Tensor] = []
        if tensor_name in self.tensors:
            t = self.tensors[tensor_name]
            for rank in range(0, pg.size()):
                q = t[(slice(rank * block_size, (rank + 1) * block_size, 1))]
                k = t[(slice(single_size + rank * block_size, single_size + (rank + 1) * block_size, 1))]
                v = t[(slice(single_size * 2 + rank * block_size, single_size * 2 + (rank + 1) * block_size, 1))]
                if self.metadata.framework == "pytorch":
                    cat_res = torch.cat([q, k, v], dim=0)
                elif paddle_loaded and self.metadata.framework == "paddle":
                    cat_res = paddle.concat([q, k, v], axis=0)
                scatter_list.append(cat_res)
        if pg.size() == 1:
            self.shuffled[tensor_name] = scatter_list[0]
            return scatter_list[0]
        new_shape = (block_size * 3,) + frame.shape[1:]
        
        if self.debug_log:
            print(f"shuffle_packed_qkv: scatter, tensor_name={tensor_name}, shape={frame.shape}->{new_shape}, self.rank={self.rank}, pg.rank()={pg.rank()}, len(scatter_list)={len(scatter_list)}")
        if self.metadata.framework == "pytorch":
            dst = torch.empty(size=new_shape, dtype=frame.dtype, device=self.device)
            dist.scatter(dst, scatter_list=scatter_list, src=self.rank, group=pg)
        elif paddle_loaded and self.metadata.framework == "paddle":
            dst = paddle.to_tensor(paddle.empty(shape=new_shape, dtype=frame.dtype),place=self.device)
            pdist.scatter(dst, tensor_list=scatter_list, src=self.rank, group=group)
        self.shuffled[tensor_name] = dst
        return dst

    def shuffle_multi_cols(self, pg: dist.ProcessGroup, tensor_names: List[str], dim: int, group = None)->torch.Tensor:
        rank_tensors: List[List[torch.Tensor]] = [[] for i in range(0, pg.size())]
        new_shape: List = []
        for tensor_name in tensor_names:
            frame = self.metadata.tensors[tensor_name]
            total_size = frame.shape[0]
            block_size = (total_size + pg.size() - 1) // pg.size()
            if len(new_shape) == 0:
                new_shape = [block_size] + list(frame.shape[1:])
            elif dim == 0:
                new_shape[0] += block_size
            else:
                new_shape[dim] += frame.shape[dim]
            if self.rank == pg.rank():
                if tensor_name not in self.tensors:
                    raise Exception(f"shuffle_multi_cols: tensor {tensor_name} was not found, released? lidx={self.lidx}")
                t = self.tensors[tensor_name]
                for rank in range(0, pg.size()):
                    rank_tensors[rank].append(t[(slice(rank * block_size, (rank + 1) * block_size, 1))])
        if pg.size() == 1:
            if self.metadata.framework == "pytorch":
                return torch.cat(rank_tensors[self.rank], dim=dim)
            elif paddle_loaded and self.metadata.framework == "paddle":
                return paddle.concat(rank_tensors[self.rank], axis=dim)
            return None
        scatter_list: List[torch.Tensor] = []
        
        if len(rank_tensors[0]) > 0:
            for rank in range(0, pg.size()):
                scatter_list.append(torch.cat(rank_tensors[rank], dim=dim))
        if self.debug_log:
            print(f"shuffle_multi_cols: scatter, tensor_name={tensor_name}, shape={frame.shape}->{new_shape}, self.rank={self.rank}, pg.rank()={pg.rank()}, len(scatter_list)={len(scatter_list)}")
        if self.metadata.framework == "pytorch":
            dst = torch.empty(size=new_shape, dtype=frame.dtype, device=self.device) # dst should be eariler than scatter_list for less fragmentation
            dist.scatter(dst, scatter_list=scatter_list, src=self.rank, group=pg)
        elif paddle_loaded and self.metadata.framework == "paddle":
            dst = paddle.to_tensor(paddle.empty(shape=new_shape, dtype=frame.dtype), place=self.device )# dst should be eariler than scatter_list for less fragmentation
            pdist.scatter(dst, tensor_list=scatter_list, src=self.rank, group=group)
        return dst

    def free_dev_ptrs(self):
        self.tensors = {}
        if self.gbuf is not None:
            free_tensor_memory(self.gbuf, self.device, self.metadata.framework)
            self.gbuf = None

    def shuffle_all(self, pg: dist.ProcessGroup, tensor_shard_dim: OrderedDict[str, int])->Tuple[int, Dict[str, torch.Tensor]]:
        ret: Dict[str, torch.Tensor] = {}
        for tensor_name, dim in tensor_shard_dim.items():
            if tensor_name in self.metadata.tensors:
                ret[tensor_name] = self.shuffle(pg, tensor_name, dim)
        return (0, ret)
