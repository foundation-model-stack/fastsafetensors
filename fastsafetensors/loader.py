# Copyright 2024 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0

import math
import warnings
from typing import Dict, List, Optional, OrderedDict, Tuple, Union

import torch
import torch.distributed as dist

from . import cpp as fstcpp
from .common import (
    ALIGN,
    CUDA_PTR_ALIGN,
    SafeTensorsMetadata,
    TensorFrame,
    get_device_numa_node,
    paddle_loaded,
)
from .file_buffer import FilesBufferOnDevice
from .st_types import SingleGroup, STDevice, STDeviceType, STDType, STEnv
from .tensor_factory import LazyTensorFactory

if paddle_loaded:
    import paddle

initialized: bool = False
loaded_nvidia: bool = False
if not loaded_nvidia:
    fstcpp.load_nvidia_functions()
    loaded_nvidia = True

support_framework = ["pytorch", "pt"]
if paddle_loaded:
    support_framework.append("paddle")


class SafeTensorsFileLoader:
    r"""Load .safetensors files lazily.

    Args:
        pg (dist.ProcessGroup): process group for pytorch distributed. SingleGroup is available for single GPU use-cases.
        devcie (STDevice): target device.
        bbuf_size_kb (int): bounce buffer size for file copies.
        max_pinned_memory_in_kb (int): maximum KiB of pinned memory for GDS configuration.
        max_threads (int): maximum number of threads for memory copies.
        nogds (bool): if True, trun off GDS and fallback to pread with bounce buffer.
        debug_log (bool): enable debug logs.

    Examples:
        >> from fastsafetensors import SafeTensorsFileLoader, SingleGroup
        >> src_files = download(target_dir, "gpt2")
        >> loader = SafeTensorsFileLoader(SingleGroup, STDevice("cpu"), nogds=True, debug_log=True)
        >> loader.add_filenames({0: src_files})
        >> bufs = loader.copy_files_to_device()
        >> print(bufs.get_tensor(loader.get_keys()[0]))
        >> loader.close()
    """

    def __init__(
        self,
        pg: Union[dist.ProcessGroup, SingleGroup],
        device: str,
        bbuf_size_kb: int = 16 * 1024,
        max_pinned_memory_in_kb: int = 64 * 1024 * 1024,
        max_threads: int = 16,
        nogds: bool = False,
        debug_log: bool = False,
        framework="pytorch",
    ):
        self.device = STDevice.from_str(device)
        self.framework = STEnv.from_str(framework)
        self.debug_log = debug_log
        self.meta: Dict[str, Tuple[SafeTensorsMetadata, int]] = {}
        self.need_gds_close = False
        self.frames = OrderedDict[str, TensorFrame]()
        if self.framework == STEnv.Pytorch or isinstance(pg, SingleGroup):
            self.pg = pg
            self.group = pg
        elif paddle_loaded and self.framework == STEnv.Paddle:
            self.pg = pg.process_group
            self.group = pg
        self.nogds = nogds
        global initialized
        if not initialized:
            fstcpp.set_debug_log(debug_log)
            d_id = self.device.index
            if paddle_loaded and self.framework == STEnv.Paddle:
                if self.device == STDeviceType.CPU:
                    d_id = None
                else:
                    if isinstance(self.pg, SingleGroup):
                        # For single (gpu:x, gpu)
                        # gpu:x, like gpu:0, gpu:1, ...
                        d_id = self.device.index if self.device.index else 0
                    else:
                        # For distributed
                        # The gpu determines the current rank
                        # rank0 use gpu:0, rank1 use gpu:1
                        d_id = self.pg.rank() % paddle.device.cuda.device_count()
                        self.device = STDevice(STDeviceType.GPU, d_id)
            node = get_device_numa_node(d_id)
            if node is not None:
                fstcpp.set_numa_node(node)
            if (
                False and fstcpp.is_cufile_found() and not nogds
            ):  # TODO: init_gds should be called but too slow for parallel initialization
                if fstcpp.init_gds(bbuf_size_kb, max_pinned_memory_in_kb) != 0:
                    raise Exception(
                        f"[FAIL] init_gds bbuf_size_kb={bbuf_size_kb}, max_pinned_memory_in_kb={max_pinned_memory_in_kb}"
                    )
                self.need_gds_close = True
            initialized = True
        device_is_not_cpu = not (
            paddle_loaded
            and self.framework == STEnv.Paddle
            and self.device == STDeviceType.CPU
        ) and not (self.framework == STEnv.Pytorch and device == STDeviceType.CPU)
        if device_is_not_cpu and not fstcpp.is_cuda_found():
            raise Exception("[FAIL] libcudart.so does not exist")
        if not fstcpp.is_cufile_found() and not nogds:
            warnings.warn(
                "libcufile.so does not exist but nogds is False. use nogds=True",
                UserWarning,
            )
            nogds = True
        self.reader: Optional[
            Union[fstcpp.nogds_file_reader, fstcpp.gds_file_reader]
        ] = None
        if nogds:
            self.reader = fstcpp.nogds_file_reader(
                False, bbuf_size_kb, max_threads, device_is_not_cpu
            )
        else:
            self.reader = fstcpp.gds_file_reader(max_threads, device_is_not_cpu)
        self.nogds = nogds

    def reset(self):
        self.frames = {}
        self.meta = {}

    def close(self):
        if self.need_gds_close:
            fstcpp.close_gds()
            self.need_gds_close = False

    def get_keys(self) -> List[str]:
        return list(self.frames.keys())

    def get_shape(self, tensor_name: str) -> List[int]:
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
                    realpath = filenames[rank][next_idx]  # os.path.realpath(filename)
                    metadata = SafeTensorsMetadata.from_file(realpath, self.framework)
                    self.meta[realpath] = (metadata, rank)
                    self.frames.update(metadata.tensors)
                    if self.debug_log and rank == self.pg.rank():
                        print(f"add_filenames {len(self.meta)}: path={realpath}")
                    rank_next_idx[rank] = next_idx + 1
                else:
                    completed += 1

    def copy_files_to_device(
        self,
        dtype: STDType = STDType.AUTO,
        use_buf_register: bool = True,
        max_copy_block_size: int = 16 * 1024 * 1024 * 1024,
    ) -> FilesBufferOnDevice:
        """
        trigger copying all the files to device buffers.
        At this moment, we do not instantiate tensors but just creating copies at device buffers with or without GDS.
        Users can instantiate and/or partition tensors with FilesBufferOnDevice returned by this function.
        """
        if self.framework == STEnv.Pytorch:
            if self.device != STDeviceType.CUDA:
                torch.cuda.set_device(self.device.as_str())
        elif paddle_loaded and self.framework == STEnv.Paddle:
            if self.device != STDeviceType.GPU:
                paddle.set_device(self.device.as_str())

        need_wait: List[LazyTensorFactory] = []
        factories: Dict[int, List[LazyTensorFactory]] = {}
        for i in range(0, self.pg.size()):
            factories[i] = []

        factory_idx_bits = math.ceil(math.log2(len(self.meta) + 1))
        lidx = 1

        for _, (meta, rank) in sorted(self.meta.items(), key=lambda x: x[0]):
            self_rank = self.pg.rank() == rank
            factory = LazyTensorFactory(
                meta,
                self.device,
                rank,
                self_rank,
                factory_idx_bits,
                lidx,
                self.nogds,
                self.reader,
                self.debug_log,
            )
            factory.submit_io(use_buf_register, max_copy_block_size)
            factories[rank].append(factory)
            if self_rank:
                need_wait.append(factory)
            lidx += 1
        for factory in need_wait:
            factory.wait_io(dtype=dtype, noalign=self.nogds)
        if self.framework == STEnv.Pytorch:
            return FilesBufferOnDevice(factories, pg=self.pg, framework=self.framework)
        elif paddle_loaded and self.framework == STEnv.Paddle:
            return FilesBufferOnDevice(
                factories, pg=self.group, framework=self.framework
            )
        else:
            raise Exception(f"Unsupported framework {self.framework}")


class fastsafe_open:
    """
    Opens a safetensors lazily and returns tensors as asked
    This is an enhanced version of safe_open in the original safetensors library to consume file list

    Args:
        filenames (:obj:`str`|`list[str]`|`dict[int, str]`): The filename(s) or rank-file map to open
        framework (:obj:`str`): `pt`, `pytorch`, and `paddle` are only supported currently
        device (:obj:`str`, defaults to :obj:`"cpu"`): The device on which you want the tensors.
    """

    def __init__(
        self,
        filenames: Union[str, List[str], Dict[int, List[str]]],
        framework: str = "pt",
        pg: Union[dist.ProcessGroup, SingleGroup] = SingleGroup(),
        device: str = "cpu",
        nogds: bool = False,
        debug_log: bool = False,
        max_copy_block_size: int = 16 * 1024 * 1024 * 1024,
    ):
        self.loader = SafeTensorsFileLoader(
            pg, device, nogds=nogds, debug_log=debug_log, framework=framework
        )
        file_dict: Dict[int, List[str]] = {}
        if isinstance(filenames, str):
            file_dict = {0: [filenames]}
        if isinstance(filenames, list):
            file_dict = {0: filenames}
        elif isinstance(filenames, dict):
            file_dict = filenames
        self.loader.add_filenames(file_dict)
        self.fb = self.loader.copy_files_to_device(
            max_copy_block_size=max_copy_block_size
        )

    def metadata(self) -> Dict[str, Dict[str, str]]:
        ret = {}
        for filename, (metadata, _) in self.loader.meta.items():
            ret[filename] = metadata.metadata
        return ret

    def get_keys(self) -> List[str]:
        return list(self.fb.key_to_rank_lidx.keys())

    def get_tensor(self, name: str) -> torch.Tensor:
        return self.fb.get_tensor(name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if self.fb:
            self.fb.close()
        if self.loader:
            self.loader.close()
