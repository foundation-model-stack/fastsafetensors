# SPDX-License-Identifier: Apache-2.0

"""Unified memory copier for systems with shared CPU/GPU memory (DGX Spark, Grace Hopper).

Uses mmap → pin_memory → cudaMemcpyAsync instead of the bounce buffer approach.
On unified memory with ATS, pin_memory on mmap'd pages triggers kernel readahead
and page pinning in a single optimized path, then async DMA transfers at full
memory bandwidth.
"""

import os
from typing import Dict, List, Optional, Tuple

import torch

from .. import cpp as fstcpp
from ..common import SafeTensorsMetadata
from ..frameworks import FrameworkOpBase, TensorBase
from ..st_types import Device, DeviceType, DType
from .base import CopierInterface
from .registry import CopierConstructFunc, register_copier_constructor


class UnifiedMemCopier(CopierInterface):
    """Copier using mmap → pin_memory → cudaMemcpyAsync for unified memory.

    On systems where CPU and GPU share the same physical memory (DGX Spark,
    Grace Hopper), this avoids the unnecessary bounce buffer used by NoGdsFileCopier.
    The mmap + pin_memory path lets the kernel handle readahead and page pinning
    in a single step, then async DMA copies at full memory bandwidth.
    """

    def __init__(
        self,
        metadata: SafeTensorsMetadata,
        device: Device,
        framework: FrameworkOpBase,
    ):
        self.metadata = metadata
        self.device = device
        self.framework = framework
        self._file_tensor: Optional[torch.Tensor] = None
        self._pinned: List[torch.Tensor] = []
        self.byte_ranges: Optional[List[Tuple[int, int]]] = None

    def set_byte_ranges(self, byte_ranges: Optional[List[Tuple[int, int]]]) -> None:
        """Restrict reads to these ``[start, end)`` absolute file-offset runs.

        Only the bytes in the given runs are mmap-faulted, pinned, and copied;
        the rest of the device buffer is left uninitialized (so the corresponding
        tensors must not be requested). Tensor offsets are unchanged. ``None``
        reads the whole data section. Build runs with
        ``SafeTensorsMetadata.select_byte_ranges``.
        """
        self.byte_ranges = byte_ranges

    def submit_io(
        self, use_buf_register: bool, max_copy_block_size: int
    ) -> fstcpp.gds_device_buffer:
        header_length = self.metadata.header_length
        data_length = self.metadata.size_bytes - header_length

        # Allocate CUDA buffer via framework's allocator (proper lifecycle)
        gbuf = self.framework.alloc_tensor_memory(data_length, self.device)

        # mmap the file (lazy, no I/O yet)
        file_tensor = torch.from_file(
            self.metadata.src, size=self.metadata.size_bytes, dtype=torch.uint8
        )
        self._file_tensor = file_tensor

        # Default to the whole data section, reproducing the full-file read.
        # An empty list (vs None) reads nothing — same semantics as nogds.
        runs = self.byte_ranges
        if runs is None:
            runs = [(header_length, self.metadata.size_bytes)]

        base_address = gbuf.get_base_address()
        self._pinned = []
        for start, end in runs:
            # pin_memory faults in + pins only this run's pages, then DMA to the
            # matching offset in gbuf (data section starts at header_length).
            pinned = file_tensor[start:end].pin_memory()
            self._pinned.append(pinned)
            ret = fstcpp.memcpy_h2d_async(  # type: ignore[attr-defined]
                base_address + (start - header_length),
                pinned.data_ptr(),
                end - start,
            )
            if ret != 0:
                self.framework.free_tensor_memory(gbuf, self.device)
                self._pinned = []
                self._file_tensor = None
                raise RuntimeError(
                    f"cudaMemcpyAsync failed with error {ret} for {self.metadata.src}"
                )

        return gbuf

    def wait_io(
        self,
        gbuf: fstcpp.gds_device_buffer,
        dtype: DType = DType.AUTO,
        noalign: bool = False,
    ) -> Dict[str, TensorBase]:
        torch.cuda.synchronize()

        # Alignment note: unlike the GDS copier, we only copy the data section
        # (not the header) into gbuf, so gbuf starts at a CUDA-allocator-aligned
        # address. The copy_start_offset=header_length cancels out in get_tensors'
        # pointer arithmetic, giving correct offsets. No memmove fixup needed.
        tensors = self.metadata.get_tensors(
            gbuf, self.device, self.metadata.header_length, dtype=dtype
        )

        # Release mmap and pinned memory
        self._pinned = []
        self._file_tensor = None

        return tensors


def is_unified_memory_system() -> bool:
    """Detect if this system has unified CPU/GPU memory.

    Currently verified on DGX Spark (GB10). Other unified memory
    platforms (Grace Hopper GH200) may also benefit but are untested.

    Can be overridden via the FASTSAFETENSORS_UNIFIED_MEM environment
    variable: set to "1" to force enable, "0" to force disable.
    """
    override = os.environ.get("FASTSAFETENSORS_UNIFIED_MEM")
    if override is not None:
        return override == "1"

    if not torch.cuda.is_available():
        return False

    try:
        name = torch.cuda.get_device_name(0).lower()
        if "gb10" in name:
            return True
    except Exception:
        pass

    return False


@register_copier_constructor("unified")
def new_unified_copier(device: Device, **kwargs) -> CopierConstructFunc:
    """Factory function for UnifiedMemCopier.

    Returns a constructor that creates UnifiedMemCopier instances.
    """
    from .nogds import load_library_func

    load_library_func()

    def construct_unified_copier(
        metadata: SafeTensorsMetadata,
        device: Device,
        framework: FrameworkOpBase,
    ) -> CopierInterface:
        return UnifiedMemCopier(metadata, device, framework)

    return construct_unified_copier
