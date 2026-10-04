# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Dict, List, Optional, Tuple

from . import cpp as fstcpp
from .allocation import SharedDeviceAllocation
from .common import SafeTensorsMetadata, init_logger, is_debug
from .copier.base import CopierInterface, DummyDeviceBuffer
from .frameworks import FrameworkOpBase, ProcessGroupBase, TensorBase
from .st_types import Device, DType

logger = init_logger(__name__)


class LazyTensorFactory:
    def __init__(
        self,
        metadata: SafeTensorsMetadata,
        device: Device,
        rank: int,
        local_rank: bool,
        factory_idx_bits: int,
        lidx: int,
        copier: Optional[CopierInterface],
        framework: FrameworkOpBase,
        disable_cache=True,
    ):
        self.framework = framework
        self.metadata = metadata
        self.device = device
        self.copier: Optional[CopierInterface] = None
        if local_rank:
            self.copier = copier
        self.tensors: Dict[str, TensorBase] = {}
        self.shuffled: Dict[str, TensorBase] = {}
        self.gbuf: Optional[fstcpp.gds_device_buffer] = None
        # Shared owner of gbuf. Created in wait_io and referenced by every
        # owning storage materialized from the buffer. Borrowed storage does
        # not acquire a reference. See SharedDeviceAllocation.
        self.allocation: Optional[SharedDeviceAllocation] = None
        self.borrowed_tensors = False
        self.rank = rank
        self.factory_idx_bits = factory_idx_bits
        self.lidx = lidx
        self.next_tag = 1
        self.disable_cache = disable_cache
        # Bind directly to the copier only when views precede DMA completion.
        # Blocking and unsupported paths have no per-tensor callback overhead.
        self.wait_tensor: Optional[Callable[[str], None]] = None

    def submit_io(
        self,
        use_buf_register: bool,
        max_copy_block_size: int,
        allow_inflight: bool = False,
    ):
        if self.copier is not None:
            if allow_inflight and self.copier.enable_tensor_readiness():
                self.wait_tensor = self.copier.wait_tensor
            self.gbuf = self.copier.submit_io(use_buf_register, max_copy_block_size)
            if self.gbuf:
                logger.debug(
                    "submit_io: new buf, addr=0x%x", self.gbuf.get_base_address()
                )

    def wait_io(
        self,
        dtype: DType = DType.AUTO,
        noalign: bool = False,
        borrowed_tensors: bool = False,
    ):
        self.borrowed_tensors = borrowed_tensors
        if self.copier is not None and self.gbuf is not None:
            # Create the shared owner before materializing tensors so that even
            # if wait_io raises, free_dev_ptrs() can release the buffer. Owning
            # tensors acquire a DLPack reference; borrowed tensors leave the
            # factory as the sole owner, including when derived views survive.
            self.allocation = SharedDeviceAllocation(
                self.gbuf,
                self.framework,
                self.device,
                owns_memory=not isinstance(self.gbuf, DummyDeviceBuffer),
            )
            owner = None if borrowed_tensors else self.allocation
            if self.wait_tensor is not None and dtype == DType.AUTO:
                self.tensors = self.copier.prepare_tensors(self.gbuf, owner=owner)
                return
            self.tensors = self.copier.wait_io(
                self.gbuf,
                dtype=dtype,
                noalign=noalign,
                owner=owner,
            )
            if is_debug(logger):
                for name in self.tensors.keys():
                    logger.debug("wait_io: tensor=%s", name)
            self.copier = None
            self.wait_tensor = None

    def push(
        self,
        pg: ProcessGroupBase,
        tensor_name: str,
        dst_rank: int,
        src_rank: int,
    ) -> Optional[TensorBase]:
        if self.wait_tensor is not None:
            self.wait_tensor(tensor_name)
        if pg.size() == 1:
            return self.tensors[tensor_name]
        tag = (self.next_tag << self.factory_idx_bits) + self.lidx
        self.next_tag += 1
        if pg.rank() != dst_rank and pg.rank() != src_rank:
            logger.debug(
                "push: skip, tensor_name=%s, dst_rank=%d, pg.rank()=%d, tag=%d",
                tensor_name,
                dst_rank,
                pg.rank(),
                tag,
            )
            return None
        elif pg.rank() == dst_rank and src_rank == dst_rank:
            logger.debug(
                "push: nocopy, tensor_name=%s, dst_rank=%d, pg.rank()=%d, tag=%d",
                tensor_name,
                dst_rank,
                pg.rank(),
                tag,
            )
            return self.tensors[tensor_name].clone().detach()
        frame = self.metadata.tensors[tensor_name]
        if pg.rank() == src_rank:
            if tensor_name not in self.tensors:
                raise Exception(
                    f"push: tensor {tensor_name} was not found. released? lidx={self.lidx}"
                )
            t = self.tensors[tensor_name].clone().detach()
            logger.debug(
                "push: send, tensor_name=%s, shape=%s, dst_rank=%d, pg.rank()=%d, tag=%d",
                tensor_name,
                frame.shape,
                dst_rank,
                pg.rank(),
                tag,
            )
            pg.send(t, dst_rank, tag=tag)
            return None

        logger.debug(
            "push: recv, tensor_name=%s, shape=%s, src_rank=%d, pg.rank()=%d, tag=%d",
            tensor_name,
            frame.shape,
            src_rank,
            pg.rank(),
            tag,
        )

        t = self.framework.get_empty_tensor(frame.shape, frame.dtype, self.device)
        pg.recv(t, src_rank, tag=tag)
        return t

    def shuffle(self, pg: ProcessGroupBase, tensor_name: str, dim: int) -> TensorBase:
        if tensor_name in self.shuffled:
            logger.debug("shuffle: use cache, tensor_name=%s", tensor_name)
            t = self.shuffled[tensor_name].clone().detach()
            return t
        if self.wait_tensor is not None:
            self.wait_tensor(tensor_name)
        if pg.size() == 1:
            # Owning tensors retain the allocation; borrowed views expire at close.
            return self.tensors[tensor_name]
        frame = self.metadata.tensors[tensor_name]
        if dim == -1:
            if tensor_name in self.tensors:
                dst = self.tensors[tensor_name].clone().detach()
            else:
                dst = self.framework.get_empty_tensor(
                    frame.shape, frame.dtype, self.device
                )
            logger.debug(
                "shuffle: broadcast, tensor_name=%s, shape=%s, self.rank=%d, pg.rank()=%d, has_tensor=%s",
                tensor_name,
                frame.shape,
                self.rank,
                pg.rank(),
                tensor_name in self.tensors,
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
                    t[
                        self.framework.get_native_slices(
                            frame.dtype, frame.shape, rank_slices[rank]
                        )
                    ].contiguous()
                    for rank in range(0, pg.size())
                ]  # scatter requires contiguous tensor
            logger.debug(
                "shuffle: scatter, tensor_name=%s, shape=%s->%s, self.rank=%d, pg.rank()=%d, rank_slices=%s, len(scatter_list)=%s",
                tensor_name,
                frame.shape,
                new_frame.shape,
                self.rank,
                pg.rank(),
                rank_slices,
                len(scatter_list),
            )
            pg.scatter(dst, scatter_list=scatter_list, src=self.rank)
        if not self.disable_cache:
            # Cache tensor for reuse within the same batch to improve performance.
            # Note: This requires additional (GPU) memory to store the cached tensors.
            # Enable this only if you have sufficient (GPU) memory and required.
            self.shuffled[tensor_name] = dst
        return dst

    def shuffle_multi_cols(
        self, pg: ProcessGroupBase, tensor_names: List[str], dim: int
    ) -> TensorBase:
        rank_tensors: List[List[TensorBase]] = [[] for i in range(0, pg.size())]
        new_shape: List[int] = []
        for tensor_name in tensor_names:
            if self.wait_tensor is not None:
                self.wait_tensor(tensor_name)
            frame = self.metadata.tensors[tensor_name]
            total_size = frame.shape[dim]
            block_size = (total_size + pg.size() - 1) // pg.size()
            shard_start = pg.rank() * block_size
            shard_stop = min((pg.rank() + 1) * block_size, total_size)
            shard_size = max(shard_stop - shard_start, 0)
            if len(new_shape) == 0:
                new_shape = list(frame.shape)
                new_shape[dim] = 0
            else:
                for dim2 in range(0, len(frame.shape)):
                    if dim2 != dim and frame.shape[dim2] != new_shape[dim2]:
                        raise Exception(
                            f"dim {dim2} mismatch: tensor {tensor_name} has {frame.shape} vs. {new_shape} (dim={dim})"
                        )
            new_shape[dim] += shard_size
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
                    rank_tensors[rank].append(
                        t[
                            self.framework.get_native_slices(
                                frame.dtype, frame.shape, rank_slices
                            )
                        ]
                    )
        if pg.size() == 1:
            return self.framework.concat_tensors(rank_tensors[self.rank], dim=dim)
        scatter_list: List[TensorBase] = []

        if self.rank == pg.rank():
            for rank in range(0, pg.size()):
                scatter_list.append(
                    self.framework.concat_tensors(rank_tensors[rank], dim=dim)
                )
        logger.debug(
            "shuffle_multi_cols: scatter, tensor_name=%s, shape=%s->%s, self.rank=%d, pg.rank()=%d, len(scatter_list)=%s",
            tensor_name,
            frame.shape,
            new_shape,
            self.rank,
            pg.rank(),
            len(scatter_list),
        )
        dst = self.framework.get_empty_tensor(new_shape, frame.dtype, self.device)
        pg.scatter(dst, scatter_list=scatter_list, src=self.rank)
        return dst

    def free_dev_ptrs(self):
        """Release this factory's reference to the backing allocation.

        This is a logical close: it drops the buffer-side reference and the
        factory's internal tensor references, but does not forcibly free memory
        that exported tensors still hold. The physical buffer is freed by
        SharedDeviceAllocation once its last reference disappears. Idempotent:
        safe to call repeatedly (e.g. from repeated close()).
        """
        try:
            if self.copier is not None:
                if self.wait_tensor is None and self.gbuf is not None:
                    # A later file can fail before this blocking copier's
                    # wait_io was reached. Drain its writes before cleanup.
                    self.copier.wait_io(self.gbuf, noalign=True)
                else:
                    self.copier.finish_io()
        finally:
            self.copier = None
            self.wait_tensor = None
            self._free_buffer()

    def _free_buffer(self):
        self.tensors = {}
        if self.allocation is not None:
            logger.debug(
                "free_dev_ptrs: release buf, addr=0x%x",
                self.allocation.get_base_address() if self.allocation.live else 0,
            )
            self.allocation.release()
            self.allocation = None
            self.gbuf = None
        elif self.gbuf is not None and not isinstance(self.gbuf, DummyDeviceBuffer):
            # submit_io ran but wait_io never created the allocation (e.g. an
            # error before materialization): free the raw buffer directly.
            self.framework.free_tensor_memory(self.gbuf, self.device)
            logger.debug(
                "free_dev_ptrs: delete buf, addr=0x%x", self.gbuf.get_base_address()
            )
            self.gbuf = None
