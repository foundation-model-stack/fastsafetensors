# Copyright 2024 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

from . import cpp as fstcpp
from .common import SafeTensorsMetadata
from .copier.gds import GdsFileCopier
from .copier.nogds import NoGdsFileCopier
from .frameworks import FrameworkOpBase, ProcessGroupBase, TensorBase
from .st_types import Device, DType


class LazyTensorFactory:
    def __init__(
        self,
        metadata: SafeTensorsMetadata,
        device: Device,
        rank: int,
        local_rank: bool,
        factory_idx_bits: int,
        lidx: int,
        reader: Union[fstcpp.gds_file_reader, fstcpp.nogds_file_reader],
        framework: FrameworkOpBase,
        debug_log: bool = False,
    ):
        self.framework = framework
        self.metadata = metadata
        self.device = device
        self.copier: Optional[Union[NoGdsFileCopier, GdsFileCopier]] = None
        if local_rank:
            if isinstance(reader, fstcpp.nogds_file_reader):
                self.copier = NoGdsFileCopier(
                    metadata, device, reader, framework, debug_log
                )
            else:
                self.copier = GdsFileCopier(
                    metadata, device, reader, framework, debug_log
                )
        self.tensors: Dict[str, TensorBase] = {}
        self.shuffled: Dict[str, TensorBase] = {}
        self.gbuf: Optional[fstcpp.gds_device_buffer] = None
        self.debug_log = debug_log
        self.rank = rank
        self.factory_idx_bits = factory_idx_bits
        self.lidx = lidx
        self.next_tag = 1

    def submit_io(self, use_buf_register: bool, max_copy_block_size: int):
        if self.copier is not None:
            self.gbuf = self.copier.submit_io(use_buf_register, max_copy_block_size)

    def wait_io(self, dtype: DType = DType.AUTO, noalign: bool = False):
        if self.copier is not None and self.gbuf is not None:
            self.tensors = self.copier.wait_io(self.gbuf, dtype=dtype, noalign=noalign)
            if self.debug_log:
                for name in self.tensors.keys():
                    print(f"wait_io: tensor={name}")
            self.copier = None

    def push(
        self,
        pg: ProcessGroupBase,
        tensor_name: str,
        dst_rank: int,
        src_rank: int,
    ) -> Optional[TensorBase]:
        if pg.size() == 1:
            return self.tensors[tensor_name]
        tag = (self.next_tag << self.factory_idx_bits) + self.lidx
        self.next_tag += 1
        if pg.rank() != dst_rank and pg.rank() != src_rank:
            if self.debug_log:
                print(
                    f"push: skip, tensor_name={tensor_name}, dst_rank={dst_rank}, pg.rank()={pg.rank()}, tag={tag}"
                )
            return None
        elif pg.rank() == dst_rank and src_rank == dst_rank:
            if self.debug_log:
                print(
                    f"push: nocopy, tensor_name={tensor_name}, dst_rank={dst_rank}, pg.rank()={pg.rank()}, tag={tag}"
                )
            return self.tensors[tensor_name].clone().detach()
        frame = self.metadata.tensors[tensor_name]
        if pg.rank() == src_rank:
            if tensor_name not in self.tensors:
                raise Exception(
                    f"push: tensor {tensor_name} was not found. released? lidx={self.lidx}"
                )
            t = self.tensors[tensor_name].clone().detach()
            if self.debug_log:
                print(
                    f"push: send, tensor_name={tensor_name}, shape={frame.shape}, dst_rank={dst_rank}, pg.rank()={pg.rank()}, tag={tag}"
                )
            pg.send(t, dst_rank, tag=tag)
            return None

        if self.debug_log:
            print(
                f"push: recv, tensor_name={tensor_name}, shape={frame.shape}, src_rank={src_rank}, pg.rank()={pg.rank()}, tag={tag}"
            )

        t = self.framework.get_empty_tensor(frame.shape, frame.dtype, self.device)
        pg.recv(t, src_rank, tag=tag)
        return t

    def shuffle(self, pg: ProcessGroupBase, tensor_name: str, dim: int) -> TensorBase:
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
                dst = self.framework.get_empty_tensor(
                    frame.shape, frame.dtype, self.device
                )
            if self.debug_log:
                print(
                    f"shuffle: broadcast, tensor_name={tensor_name}, shape={frame.shape}, self.rank={self.rank}, pg.rank()={pg.rank()}, has_tensor={tensor_name in self.tensors}"
                )
            pg.broadcast(dst, self.rank)
        else:
            rank_slices: List[Tuple] = [() for i in range(0, pg.size())]
            size = frame.shape[dim]
            block_size = (size + pg.size() - 1) // pg.size()
            for rank in range(0, pg.size()):
                for i in range(0, len(frame.shape)):
                    if i < dim:
                        rank_slices[rank] += (slice(None, None, None),)
                    elif i == dim:
                        rank_slices[rank] += (
                            slice(rank * block_size, (rank + 1) * block_size, 1),
                        )
                        break
            scatter_list: List[TensorBase] = []
            new_frame = frame[rank_slices[pg.rank()]]
            dst = self.framework.get_empty_tensor(
                new_frame.shape, new_frame.dtype, self.device
            )
            if self.rank == pg.rank():
                if tensor_name not in self.tensors:
                    raise Exception(
                        f"shuffle: tensor {tensor_name} was not found, released? lidx={self.lidx}"
                    )
                t = self.tensors[tensor_name]
                scatter_list = [
                    t[rank_slices[rank]].contiguous() for rank in range(0, pg.size())
                ]  # scatter requires contiguous tensor
            if self.debug_log:
                print(
                    f"shuffle: scatter, tensor_name={tensor_name}, shape={frame.shape}->{new_frame.shape}, self.rank={self.rank}, pg.rank()={pg.rank()}, rank_slices={rank_slices}, len(scatter_list)={len(scatter_list)}"
                )
            pg.scatter(dst, scatter_list=scatter_list, src=self.rank)
        self.shuffled[tensor_name] = dst
        return dst

    def shuffle_multi_cols(
        self, pg: ProcessGroupBase, tensor_names: List[str], dim: int
    ) -> TensorBase:
        rank_tensors: List[List[TensorBase]] = [[] for i in range(0, pg.size())]
        new_shape: List[int] = []
        for tensor_name in tensor_names:
            frame = self.metadata.tensors[tensor_name]
            total_size = frame.shape[dim]
            block_size = (total_size + pg.size() - 1) // pg.size()
            if len(new_shape) == 0:
                new_shape = frame.shape
                new_shape[dim] = 0
            else:
                for dim2 in range(0, len(frame.shape)):
                    if dim2 != dim and frame.shape[dim2] != new_shape[dim2]:
                        raise Exception(
                            f"dim {dim2} mismatch: tensor {tensor_name} has {frame.shape} vs. {new_shape} (dim={dim})"
                        )
            new_shape[dim] += block_size
            if self.rank == pg.rank():
                if tensor_name not in self.tensors:
                    raise Exception(
                        f"shuffle_multi_cols: tensor {tensor_name} was not found, released? lidx={self.lidx}"
                    )
                t = self.tensors[tensor_name]
                for rank in range(0, pg.size()):
                    rank_slices: Tuple[slice, ...] = ()
                    for i in range(0, len(frame.shape)):
                        if i < dim:
                            rank_slices += (slice(None, None, None),)
                        elif i == dim:
                            rank_slices += (
                                slice(rank * block_size, (rank + 1) * block_size, 1),
                            )
                            break
                    rank_tensors[rank].append(t[rank_slices])
        if pg.size() == 1:
            return self.framework.concat_tensors(rank_tensors[self.rank], dim=dim)
        scatter_list: List[TensorBase] = []

        if self.rank == pg.rank():
            for rank in range(0, pg.size()):
                scatter_list.append(
                    self.framework.concat_tensors(rank_tensors[rank], dim=dim)
                )
        if self.debug_log:
            print(
                f"shuffle_multi_cols: scatter, tensor_name={tensor_name}, shape={frame.shape}->{new_shape}, self.rank={self.rank}, pg.rank()={pg.rank()}, len(scatter_list)={len(scatter_list)}"
            )
        dst = self.framework.get_empty_tensor(new_shape, frame.dtype, self.device)
        pg.scatter(dst, scatter_list=scatter_list, src=self.rank)
        return dst

    def free_dev_ptrs(self):
        self.tensors = {}
        if self.gbuf is not None:
            self.framework.free_tensor_memory(self.gbuf, self.device)
            self.gbuf = None
