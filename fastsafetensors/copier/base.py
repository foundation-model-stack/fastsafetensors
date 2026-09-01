# SPDX-License-Identifier: Apache-2.0

import operator
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Tuple

from .. import cpp as fstcpp
from ..common import SafeTensorsMetadata
from ..frameworks import TensorBase
from ..st_types import DType


def validated_byte_ranges(
    metadata: SafeTensorsMetadata,
    byte_ranges: Optional[List[Tuple[int, int]]],
) -> Optional[List[Tuple[int, int]]]:
    """Validate ``[start, end)`` absolute file-offset runs against *metadata*
    and return a defensive copy.

    Runs must be integer pairs within the data section
    (``header_length <= start < end <= size_bytes``), sorted and
    non-overlapping. ``None`` (full read) passes through; an empty list reads
    nothing. Shared by every copier so range checks live at one API boundary.
    """
    if byte_ranges is None:
        return None
    checked: List[Tuple[int, int]] = []
    prev_end = metadata.header_length
    for i, run in enumerate(byte_ranges):
        try:
            start, end = run
        except (TypeError, ValueError):
            raise ValueError(
                f"byte_ranges[{i}]: expected a (start, end) pair, got {run!r}"
            ) from None
        if isinstance(start, bool) or isinstance(end, bool):
            raise ValueError(f"byte_ranges[{i}]: offsets must be int, got {run!r}")
        try:
            start, end = operator.index(start), operator.index(end)
        except TypeError:
            raise ValueError(
                f"byte_ranges[{i}]: offsets must be int, got {run!r}"
            ) from None
        if end <= start:
            raise ValueError(f"byte_ranges[{i}]: empty or reversed run {run!r}")
        if start < prev_end:
            raise ValueError(
                f"byte_ranges[{i}]: runs must be sorted, non-overlapping, and "
                f"start at or after the data section "
                f"(start={start}, minimum here {prev_end})"
            )
        if end > metadata.size_bytes:
            raise ValueError(
                f"byte_ranges[{i}]: end={end} is beyond the end of "
                f"{metadata.src} ({metadata.size_bytes} bytes)"
            )
        checked.append((start, end))
        prev_end = end
    return checked


def validated_chunk_allocation_size(
    byte_ranges: List[Tuple[int, int]], allocation_size: Optional[int]
) -> Optional[int]:
    """Validate an optional allocation size for a compact chunk."""
    if allocation_size is None:
        return None
    if isinstance(allocation_size, bool):
        raise ValueError(f"allocation_size must be int, got {allocation_size!r}")
    try:
        allocation_size = operator.index(allocation_size)
    except TypeError:
        raise ValueError(
            f"allocation_size must be int, got {allocation_size!r}"
        ) from None
    if allocation_size <= 0:
        raise ValueError(f"allocation_size must be positive, got {allocation_size}")
    if byte_ranges:
        span = max(end for _, end in byte_ranges) - min(
            start for start, _ in byte_ranges
        )
        if allocation_size < span:
            raise ValueError(
                f"allocation_size={allocation_size} is smaller than chunk span={span}"
            )
    return allocation_size


class CopierInterface(ABC):
    metadata: SafeTensorsMetadata

    def set_byte_ranges(self, byte_ranges: Optional[List[Tuple[int, int]]]) -> None:
        """Restrict reads to these ``[start, end)`` absolute file-offset runs.

        The default implementation validates the runs but reads the whole file,
        so the byte-range filter is a correct no-op on copiers that don't
        implement partial reads. Range-capable copiers (``nogds``, ``unified``)
        override this to read only the given runs, leaving the rest of the
        device buffer uninitialized (so skipped tensors must not be requested).
        Build runs with ``SafeTensorsMetadata.select_byte_ranges``; ``None``
        means full read.
        """
        validated_byte_ranges(self.metadata, byte_ranges)

    def set_chunk(
        self,
        byte_ranges: List[Tuple[int, int]],
        names: Set[str],
        allocation_size: Optional[int] = None,
    ) -> None:
        """Load only ``names``, allocating just those runs' span (sub-file
        chunking for ``ParallelLoader(max_batch_bytes=...)``).

        ``allocation_size`` may pad the buffer without expanding the read.
        Unlike ``set_byte_ranges``, a chunk plan cannot be a no-op: silently
        loading the whole file per chunk-batch would break the memory bound
        and multiply full-file reads, so the default refuses. Partial-read
        copiers (``nogds``, ``unified``) override this.
        """
        raise NotImplementedError(
            f"sub-file chunking (max_batch_bytes) requires a copier that "
            f"overrides set_chunk; {type(self).__name__} loads whole files. "
            f"Use the nogds or unified copier, or unset max_batch_bytes."
        )

    @classmethod
    def chunk_transient_multiplier(cls, paths: List[str]) -> int:
        """Transient device bytes this copier holds per in-flight chunk, as a
        multiple of the chunk's span, when loading *paths*.

        The fit planner (``ParallelLoader(device_memory_budget=...)``) charges
        every live buffer this multiple of its budget, so a copier that stages
        a chunk twice must say so or the plan under-counts and OOMs. Fixed
        overheads that do not scale with chunk size (bounce-buffer pools,
        reader thread pools) are not counted here. Like ``set_chunk``, the
        default refuses rather than guessing: chunking copiers override it.
        """
        raise NotImplementedError(
            f"device_memory_budget needs a copier that overrides "
            f"chunk_transient_multiplier; {cls.__name__} does not implement "
            f"sub-file chunking. Use the nogds or unified copier, or unset "
            f"device_memory_budget."
        )

    @abstractmethod
    def submit_io(
        self, use_buf_register: bool, max_copy_block_size: int
    ) -> fstcpp.gds_device_buffer:
        pass

    @abstractmethod
    def wait_io(
        self,
        gbuf: fstcpp.gds_device_buffer,
        dtype: DType = DType.AUTO,
        noalign: bool = False,
    ) -> Dict[str, TensorBase]:
        pass


class DummyDeviceBuffer(fstcpp.gds_device_buffer):
    def __init__(self):
        super().__init__(0, 0, False)
