# Copyright 2024 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0

import math
import warnings
from typing import Any, Dict, List, Optional, OrderedDict, Tuple, Union

from . import cpp as fstcpp
from .common import SafeTensorsMetadata, TensorFrame, get_device_numa_node
from .file_buffer import FilesBufferOnDevice
from .frameworks import TensorBase, get_framework_op
from .st_types import DeviceType, DType
from .tensor_factory import LazyTensorFactory

gl_set_numa = False

loaded_nvidia = False


class SafeTensorsFileLoader:
    r"""Load .safetensors files lazily.

    Args:
        devcie (str): target device.
        pg (Optional[Any]): process group-like objects for distributed. None for single GPU use-cases.
        bbuf_size_kb (int): bounce buffer size for file copies.
        max_threads (int): maximum number of threads for memory copies.
        nogds (bool): if True, trun off GDS and fallback to pread with bounce buffer.
        debug_log (bool): enable debug logs.

    Examples:
        >> from fastsafetensors import SafeTensorsFileLoader
        >> src_files = download(target_dir, "gpt2")
        >> loader = SafeTensorsFileLoader(Device("cpu"), nogds=True, debug_log=True)
        >> loader.add_filenames({0: src_files})
        >> bufs = loader.copy_files_to_device()
        >> print(bufs.get_tensor(loader.get_keys()[0]))
        >> loader.close()
    """

    def __init__(
        self,
        pg: Optional[Any],
        device: str = "cpu",
        bbuf_size_kb: int = 16 * 1024,
        max_threads: int = 16,
        nogds: bool = False,
        set_numa: bool = True,
        debug_log: bool = False,
        framework="pytorch",
    ):
        self.framework = get_framework_op(framework)
        self.pg = self.framework.get_process_group(pg)
        self.device = self.framework.get_device(device, self.pg)
        self.debug_log = debug_log
        self.meta: Dict[str, Tuple[SafeTensorsMetadata, int]] = {}
        self.frames = OrderedDict[str, TensorFrame]()
        global loaded_nvidia
        if not loaded_nvidia:
            fstcpp.load_nvidia_functions()
            if fstcpp.init_gds() != 0:
                raise Exception(f"[FAIL] init_gds()")
            loaded_nvidia = True
        global gl_set_numa
        if not gl_set_numa and set_numa:
            node = get_device_numa_node(self.device.index)
            if node is not None:
                fstcpp.set_numa_node(node)
            gl_set_numa = True
        fstcpp.set_debug_log(debug_log)
        device_is_not_cpu = self.device.type != DeviceType.CPU
        if device_is_not_cpu and not fstcpp.is_cuda_found():
            raise Exception("[FAIL] libcudart.so does not exist")
        if not fstcpp.is_cufile_found() and not nogds:
            warnings.warn(
                "libcufile.so does not exist but nogds is False. use nogds=True",
                UserWarning,
            )
            nogds = True
        self.reader: Union[fstcpp.nogds_file_reader, fstcpp.gds_file_reader]
        if nogds:
            self.reader = fstcpp.nogds_file_reader(
                False, bbuf_size_kb, max_threads, device_is_not_cpu
            )
        else:
            self.reader = fstcpp.gds_file_reader(max_threads, device_is_not_cpu)

    def reset(self):
        self.frames = {}
        self.meta = {}

    def close(self):
        self.reset()

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
        dtype: DType = DType.AUTO,
        use_buf_register: bool = True,
        max_copy_block_size: int = 16 * 1024 * 1024 * 1024,
    ) -> FilesBufferOnDevice:
        """
        trigger copying all the files to device buffers.
        At this moment, we do not instantiate tensors but just creating copies at device buffers with or without GDS.
        Users can instantiate and/or partition tensors with FilesBufferOnDevice returned by this function.
        """
        self.framework.set_device(self.device)

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
                self.reader,
                self.framework,
                self.debug_log,
            )
            factory.submit_io(use_buf_register, max_copy_block_size)
            factories[rank].append(factory)
            if self_rank:
                need_wait.append(factory)
            lidx += 1
        for factory in need_wait:
            factory.wait_io(
                dtype=dtype, noalign=isinstance(self.reader, fstcpp.nogds_file_reader)
            )
        return FilesBufferOnDevice(factories, pg=self.pg, framework=self.framework)


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
        pg: Optional[Any] = None,
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

    def keys(self) -> List[str]:
        return list(self.fb.key_to_rank_lidx.keys())

    def get_tensor_wrapped(self, name: str) -> TensorBase:
        return self.fb.get_tensor_wrapped(name)

    def get_tensor(self, name: str) -> Any:
        return self.get_tensor_wrapped(name).get_raw()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if self.fb:
            self.fb.close()
        if self.loader:
            self.loader.close()
