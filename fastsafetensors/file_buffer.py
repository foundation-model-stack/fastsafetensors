# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from .common import init_logger
from .frameworks import FrameworkOpBase, ProcessGroupBase, TensorBase
from .st_types import Device, DType
from .tensor_factory import LazyTensorFactory

logger = init_logger(__name__)


class FilesBufferOnDevice:
    r"""Device buffer for .safetensors files.
        Users can call get_tensor(), get_sharded(), etc. to instantiate (sharded) tensors from the device buffer.
        Note that for multi-process loading, users must follow the single-program multiple-data (SPMD) paradigm, which is common for torch.distributed programs.
        In other words, users must ensure that every worker process calls the methods here in the same order.
        This is because methods here reuse torch.distributed operations: broadcast, scatter, recv, and send.
        They synchornously wait all the workers to execute copies among processes.

        Users should create this instance with SafeTensorsFileLoader.copy_files_to_device().
        Tensors returned from this buffer are valid only while the buffer stays open.
        Clone/copy returned tensors before close() if the tensor data must be used
        after this buffer is closed.

    Args:
        rank_loaders (Dict<rank, list(LazyTensorFacotry)>): Tensor factories per rank, which hold device pointers for buffers.
        pg (ProcessGroupBase): process group for calling distributed ops.
        auto_mem_delete (bool): automatically release device buffers when all the tensors are shuffled.
        keep_tensor (Callable[[str], bool], optional): If set, only tensors for
            which ``keep_tensor(name)`` is True are registered in ``key_to_rank_lidx``;
            others raise ``ValueError`` from ``get_tensor`` / ``get_filename`` /
            ``get_shape``. Subclasses that reimplement the registration loop must
            honor this.

    Examples:
        See examples/run_single.py and examples/run_parallel.py.
    """

    def __init__(
        self,
        rank_loaders: Dict[int, List[LazyTensorFactory]],
        pg: ProcessGroupBase,
        framework: FrameworkOpBase,
        auto_mem_delete: bool = True,
        keep_tensor: Optional[Callable[[str], bool]] = None,
    ):
        self.framework = framework
        self.rank_loaders: Dict[int, List[LazyTensorFactory]] = rank_loaders
        self.key_to_rank_lidx: Dict[str, Tuple[int, int]] = {}
        self.instantiated: Dict[int, Dict[int, Dict[str, bool]]] = {}  # rank, key name
        for rank, loaders in rank_loaders.items():
            self.instantiated[rank] = {}
            for lidx, loader in enumerate(loaders):
                for key in loader.metadata.tensors.keys():
                    if keep_tensor is not None and not keep_tensor(key):
                        continue
                    if key in self.key_to_rank_lidx:
                        raise Exception(
                            f"FilesBufferOnDevice: key {key} must be unique among files"
                        )
                    self.key_to_rank_lidx[key] = (rank, lidx)
                self.instantiated[rank][lidx] = {}
        self.pg = pg
        self.auto_mem_delete = auto_mem_delete and self.pg.size() > 1

    def close(self):
        """Release the backing device buffers.

        Any tensor returned from this FilesBufferOnDevice becomes invalid after
        close() unless the caller cloned/copied it to independent storage.
        """
        for _, loaders in self.rank_loaders.items():
            for loader in loaders:
                loader.free_dev_ptrs()
        self.rank_loaders = {}

    def get_filename(self, tensor_name: str) -> str:
        rank, lidx = self._get_rank_lidx(tensor_name)
        return self.rank_loaders[rank][lidx].metadata.src

    def get_shape(self, tensor_name: str) -> List[int]:
        rank, lidx = self._get_rank_lidx(tensor_name)
        return self.rank_loaders[rank][lidx].metadata.tensors[tensor_name].shape

    def _get_rank_lidx(self, tensor_name: str) -> Tuple[int, int]:
        if tensor_name not in self.key_to_rank_lidx:
            raise ValueError(f"_get_rank: key {tensor_name} was not found in files")
        return self.key_to_rank_lidx[tensor_name]

    def _get_tensor(
        self,
        rank: int,
        lidx: int,
        tensor_name: str,
        ret: TensorBase,
        device: Optional[Device],
        dtype: DType,
    ) -> TensorBase:
        loader = self.rank_loaders[rank][lidx]
        if self.auto_mem_delete:
            self.instantiated[rank][lidx][tensor_name] = True
            if len(self.instantiated[rank][lidx]) == len(loader.metadata.tensors):
                if self.pg.rank() == rank:
                    logger.debug(
                        "_get_tensor: free_dev_ptrs, lidx=%d, src=%s",
                        lidx,
                        loader.metadata.src,
                    )
                loader.free_dev_ptrs()
        return ret.to(device=device, dtype=dtype)

    def get_sharded_wrapped(
        self,
        tensor_name: str,
        dim: int,
        device: Optional[Device] = None,
        dtype: DType = DType.AUTO,
    ) -> TensorBase:
        """Return a wrapped shard of tensor_name.

        The returned tensor must not be used after close() unless the caller
        cloned/copied it to independent storage.
        """
        rank, lidix = self._get_rank_lidx(tensor_name)
        t = self.rank_loaders[rank][lidix].shuffle(self.pg, tensor_name, dim)
        return self._get_tensor(rank, lidix, tensor_name, t, device, dtype)

    def get_sharded(
        self,
        tensor_name: str,
        dim: int,
        device: Optional[Device] = None,
        dtype: DType = DType.AUTO,
    ) -> Any:
        """
        partition a tensor instance with the key tensor_name at the dimension dim and return it.
        In multi-process loading, this eventually calls torch.distributed.scatter.
        A special dim is -1, which broadcast a tensor to all the ranks (== get_tensor()).
        The returned tensor must not be used after close() unless the caller
        cloned/copied it to independent storage.
        """
        return self.get_sharded_wrapped(tensor_name, dim, device, dtype).get_raw()

    def get_tensor_wrapped(
        self,
        tensor_name: str,
        device: Optional[Device] = None,
        dtype: DType = DType.AUTO,
    ) -> TensorBase:
        """Return a wrapped tensor by name.

        The returned tensor must not be used after close() unless the caller
        cloned/copied it to independent storage.
        """
        return self.get_sharded_wrapped(tensor_name, -1, device, dtype)

    def get_tensor(
        self,
        tensor_name: str,
        device: Optional[Device] = None,
        dtype: DType = DType.AUTO,
    ) -> Any:
        """
        get a tensor instance with the key tensor_name from a local or remote rank.
        In multi-process loading, this eventually calls torch.distributed.broadcast.
        So, every rank will allocate the same tensor at each device memroy.
        In single-process loading, this directly instantiates a tensor from the device buffer with zero copy.
        The returned tensor must not be used after close() unless the caller
        cloned/copied it to independent storage.
        """
        return self.get_tensor_wrapped(tensor_name, device, dtype).get_raw()

    def _broadcast_runs(
        self, tensor_names: List[str], max_bytes: int, max_tensors: int
    ) -> Iterator[List[str]]:
        """Keep checkpoint order and split at file, gap, size or alignment."""
        run: List[str] = []
        location = None
        end = None
        size = 0
        for name in tensor_names:
            rank, lidx = self._get_rank_lidx(name)
            frame = self.rank_loaders[rank][lidx].metadata.tensors[name]
            start, stop = frame.data_offsets
            alignment = max(1, int(self.framework.get_dtype_size(frame.dtype)))
            if run and (
                location != (rank, lidx)
                or start != end
                or size + stop - start > max_bytes
                or len(run) >= max_tensors
                or size % alignment != 0
                or start == stop
            ):
                yield run
                run = []
                size = 0
            run.append(name)
            size += stop - start
            location, end = (rank, lidx), stop
            if start == stop:
                yield run
                run = []
                size = 0
        if run:
            yield run

    def _iter_tensors(
        self,
        tensor_names: List[str],
        max_bytes: int,
        max_tensors: int,
    ) -> Iterator[Tuple[str, Any]]:
        """Deliver resident tensors as views of owned contiguous broadcasts."""
        auto_mem_delete = self.auto_mem_delete
        # Keep loader storage until owned source copies finish, including
        # when the iterator closes early.
        self.auto_mem_delete = False
        devices = [
            self.rank_loaders[r][i].device
            for r, i in dict.fromkeys(
                self._get_rank_lidx(name) for name in tensor_names
            )
        ]
        try:
            for run in self._broadcast_runs(tensor_names, max_bytes, max_tensors):
                rank, lidx = self._get_rank_lidx(run[0])
                loader = self.rank_loaders[rank][lidx]
                frames = [loader.metadata.tensors[name] for name in run]
                tensors = None
                if frames[0].data_offsets[1] > frames[0].data_offsets[0]:
                    source = (
                        [loader.tensors[name] for name in run]
                        if self.pg.rank() == rank
                        else []
                    )
                    tensors = self.framework.broadcast_contiguous_run(
                        self.pg, source, frames, rank, loader.device
                    )
                if tensors is None:
                    for name in run:
                        yield name, self.get_tensor(name)
                else:
                    for name, tensor in zip(run, tensors):
                        yield name, tensor.get_raw()
                    # Release the staging run before allocating the next one.
                    del tensor
                    del tensors
        finally:
            try:
                for device in devices:
                    self.framework.synchronize(device)
            finally:
                self.auto_mem_delete = auto_mem_delete

    def push_tensor(
        self,
        tensor_name: str,
        dst_rank: int,
        device: Optional[Device] = None,
        dtype: DType = DType.AUTO,
    ) -> Optional[Any]:
        """
        push a tensor instance with the key tensor_name from a rank to a destination rank dst_rank.
        In multi-process loading, this eventually calls torch.distributed.send if the rank has the tensor instance.
        The destination rank will call torch.distributed.recv.
        Other ranks do nothing.
        The returned tensor must not be used after close() unless the caller
        cloned/copied it to independent storage.
        """
        rank, lidix = self._get_rank_lidx(tensor_name)
        t = self.rank_loaders[rank][lidix].push(self.pg, tensor_name, dst_rank, rank)
        if t:
            return self._get_tensor(
                rank, lidix, tensor_name, t, device, dtype
            ).get_raw()
        return None

    def get_multi_cols(
        self,
        tensor_names: List[str],
        dim: int,
        device: Optional[Device] = None,
        dtype: DType = DType.AUTO,
    ) -> TensorBase:
        """Return concatenated column shards from tensor_names.

        The returned tensor must not be used after close() unless the caller
        cloned/copied it to independent storage.
        """
        rank_lidixs: Dict[Tuple[int, int], List[str]] = {}
        for tensor_name in tensor_names:
            ranklidx = self._get_rank_lidx(tensor_name)
            if ranklidx in rank_lidixs:
                rank_lidixs[ranklidx].append(tensor_name)
            else:
                rank_lidixs[ranklidx] = [tensor_name]
        ts: List[TensorBase] = []
        for (rank, lidix), tns in sorted(rank_lidixs.items(), key=lambda x: x[0]):
            ts.append(
                self.rank_loaders[rank][lidix].shuffle_multi_cols(self.pg, tns, dim)
            )
        if len(ts) == 1:
            # fastpath: tensors at the same layer are often in the same file
            return self._get_tensor(
                rank, lidix, rank_lidixs[(rank, lidix)][0], ts[0], device, dtype
            )
        ret = self.framework.concat_tensors(ts, dim=dim)
        if self.auto_mem_delete:
            for tensor_name in tensor_names:
                rank, lidx = self._get_rank_lidx(tensor_name)
                loader = self.rank_loaders[rank][lidx]
                self.instantiated[rank][lidx][tensor_name] = True
                if len(self.instantiated[rank][lidx]) == len(loader.metadata.tensors):
                    if self.pg.rank() == rank:
                        logger.debug(
                            "get_multi_cols: free_dev_ptrs, rank=%d, lidx=%d, src=%s",
                            rank,
                            lidx,
                            loader.metadata.src,
                        )
                    loader.free_dev_ptrs()
        return ret.to(device=device, dtype=dtype)

    def as_dict(self, tensor_shard_dim: OrderedDict[str, int]) -> Dict[str, TensorBase]:
        """Return tensors keyed by name according to the requested shard dims.

        Returned tensors must not be used after close() unless the caller
        cloned/copied them to independent storage.
        """
        tensors: Dict[str, TensorBase] = {}
        for tensor_name, dim in tensor_shard_dim.items():
            rank, lidx = self._get_rank_lidx(tensor_name)
            loader = self.rank_loaders[rank][lidx]
            tensors[tensor_name] = loader.shuffle(self.pg, tensor_name, dim)
            if self.auto_mem_delete:
                self.instantiated[rank][lidx][tensor_name] = True
                if len(self.instantiated[rank][lidx]) == len(loader.metadata.tensors):
                    if self.pg.rank() == rank:
                        logger.debug(
                            "as_dict: free_dev_ptrs, rank=%d, src=%s",
                            rank,
                            loader.metadata.src,
                        )
                    loader.free_dev_ptrs()
        return tensors
