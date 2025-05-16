# Copyright 2024 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.distributed as dist
import os
import math
from . import cpp as fstcpp
from typing import List, Dict, Tuple, OrderedDict, Union
import warnings

from .common import SafeTensorsMetadata, ALIGN, CUDA_PTR_ALIGN, TensorFrame, get_device_numa_node, SingleGroup
from .tensor_factory import LazyTensorFactory
from .file_buffer import FilesBufferOnDevice

initialized: bool = False

class SafeTensorsFileLoader:
    r""" Load .safetensors files lazily.

    Args:
        pg (dist.ProcessGroup): process group for pytorch distributed. SingleGroup is available for single GPU use-cases.
        devcie (torch.device): target device.
        bbuf_size_kb (int): bounce buffer size for file copies.
        max_pinned_memory_in_kb (int): maximum KiB of pinned memory for GDS configuration.
        max_threads (int): maximum number of threads for memory copies.
        nogds (bool): if True, trun off GDS and fallback to pread with bounce buffer.
        debug_log (bool): enable debug logs.

    Examples:
        >> from fastsafetensors import SafeTensorsFileLoader, SingleGroup
        >> src_files = download(target_dir, "gpt2")
        >> loader = SafeTensorsFileLoader(SingleGroup, torch.device("cpu"), nogds=True, debug_log=True)
        >> loader.add_filenames({0: src_files})
        >> bufs = loader.copy_files_to_device()
        >> print(bufs.get_tensor(loader.get_keys()[0]))
        >> loader.close()
    """
    def __init__(self, pg: dist.ProcessGroup, device: torch.device, bbuf_size_kb: int = 16 * 1024, max_pinned_memory_in_kb: int = 64 * 1024 * 1024, max_threads: int=16, nogds: bool=False, debug_log: bool=False):
        self.device = device
        self.debug_log = debug_log
        self.meta: Dict[str, Tuple[SafeTensorsMetadata, int]] = {}
        self.need_gds_close = False
        self.frames: OrderedDict[str, TensorFrame] = {}
        self.pg = pg
        global initialized
        if not initialized:
            fstcpp.load_nvidia_functions()
            fstcpp.set_debug_log(debug_log)
            node = get_device_numa_node(device.index)
            if node is not None:
                fstcpp.set_numa_node(node)
            if fstcpp.is_cufile_found() and not nogds: # TODO: init_gds should be called but too slow for parallel initialization
                if fstcpp.init_gds(bbuf_size_kb, max_pinned_memory_in_kb) != 0:
                    raise Exception(f"[FAIL] init_gds max_io_block_in_kb={max_io_block_in_kb}, max_pinned_memory_in_kb={max_pinned_memory_in_kb}")
                self.need_gds_close = True
            initialized = True
        if not device.type == "cpu" and not fstcpp.is_cuda_found():
            raise Exception("[FAIL] libcudart.so does not exist")
        if not fstcpp.is_cufile_found() and not nogds:
            warnings.warn("libcufile.so does not exist but nogds is False. use nogds=True", UserWarning)
            nogds = True
        if nogds:
            self.reader = fstcpp.nogds_file_reader(False, bbuf_size_kb, max_threads, device.type != "cpu")
        else:
            self.reader = fstcpp.gds_file_reader(max_threads, device != "cpu")
        self.nogds = nogds

    def reset(self):
        self.frames = {}
        self.meta = {}

    def close(self):
        if self.need_gds_close:
            fstcpp.close_gds()
            self.need_gds_close = False

    def get_keys(self) -> List[str]:
        return self.frames.keys()

    def get_shape(self, tensor_name: str) -> torch.Size:
        return self.frames[tensor_name].shape

    def add_filenames(self, filenames: Dict[int, List[str]]):
        """
        Register files to ranks to be copied at copy_file_to_device().
        """
        # shuffle files in a round-robin fashion to avoid OoM
        rank_next_idx = {rank: 0 for rank in filenames.keys()}
        completed = 0
        while completed < len(filenames.keys()):
            completed = 0
            for rank in filenames.keys():
                next_idx = rank_next_idx[rank]
                if next_idx < len(filenames[rank]):
                    realpath = filenames[rank][next_idx] #os.path.realpath(filename)
                    metadata = SafeTensorsMetadata.from_file(realpath)
                    self.meta[realpath] = (metadata, rank)
                    self.frames.update(metadata.tensors)
                    if self.debug_log and rank == self.pg.rank():
                        print(f"add_filenames {len(self.meta)}: path={realpath}")
                    rank_next_idx[rank] = next_idx + 1
                else:
                    completed += 1

    def copy_files_to_device(self, dtype: torch.dtype=None, use_buf_register: bool=False, max_copy_block_size: int=16*1024*1024*1024)->FilesBufferOnDevice:
        """
        trigger copying all the files to device buffers.
        At this moment, we do not instantiate tensors but just creating copies at device buffers with or without GDS.
        Users can instantiate and/or partition tensors with FilesBufferOnDevice returned by this function.
        """
        if self.device.type != "cpu":
            torch.cuda.set_device(self.device)

        need_wait: List[LazyTensorFactory] = []
        factories: Dict[int, List[LazyTensorFactory]] = {}
        for i in range(0, self.pg.size()):
            factories[i] = []

        factory_idx_bits = math.ceil(math.log2(len(self.meta) + 1))
        lidx = 1
        
        for _, (meta, rank) in sorted(self.meta.items(), key=lambda x: x[0]):
            self_rank = self.pg.rank() == rank
            factory = LazyTensorFactory(meta, self.device, rank, self_rank, factory_idx_bits, lidx, self.nogds, self.reader, self.debug_log)
            factory.submit_io(use_buf_register, max_copy_block_size)
            factories[rank].append(factory)
            if self_rank:
                need_wait.append(factory)
            lidx += 1
        for factory in need_wait:
            factory.wait_io(dtype=dtype, noalign=self.nogds)
        return FilesBufferOnDevice(factories, pg=self.pg)

class fastsafe_open:
    """
    Opens a safetensors lazily and returns tensors as asked
    This is an enhanced version of safe_open in the original safetensors library to consume file list

    Args:
        filenames (:obj:`str`|`list[str]`|`dict[int, str]`): The filename(s) or rank-file map to open
        framework (:obj:`str`): `pt` is only supported currently
        device (:obj:`str`, defaults to :obj:`"cpu"`): The device on which you want the tensors.
    """

    def __init__(self, filenames: Union[str, List[str], Dict[int, str]],
                       framework: str="pt", pg: dist.ProcessGroup=SingleGroup(),
                       device: Union[str, torch.device]="cpu",
                       nogds: bool=False,
                       debug_log: bool=False):
        if framework != "pt":
            raise NotImplementedError("pytorch is only a framework that current fastsafetensors supports")
        if isinstance(device, str):
            device = torch.device(device)
        self.loader = SafeTensorsFileLoader(pg, device, nogds=nogds, debug_log=debug_log)
        if isinstance(filenames, str):
            filenames = [filenames]
        if isinstance(filenames, list):
            self.loader.add_filenames({0: filenames})
        elif isinstance(filenames, dict):
            self.loader.add_filenames(filenames)
        self.fb = self.loader.copy_files_to_device()

    def metadata(self)->Dict[str, Dict[str, str]]:
        ret = {}
        for filename, (metadata, _) in self.loader.meta.items():
            ret[filename] = metadata.metadata
        return ret

    def get_keys(self)->List[str]:
        return self.fb.key_to_rank_lidx.keys()

    def get_tensor(self, name: str)->torch.Tensor:
        return self.fb.get_tensor(name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if self.fb:
            self.fb.close()
        if self.loader:
            self.loader.close()
