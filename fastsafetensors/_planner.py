# SPDX-License-Identifier: Apache-2.0

"""Internal planners for bounded-memory loading.

Not part of the public API: the entry points are
``ParallelLoader(max_batch_bytes=..., device_memory_budget=...)``.

plan_chunks partitions one shard into byte-budgeted sub-file chunks; the
fit planner below turns a whole-load device memory budget into per-file
chunk budgets.


Given the byte sizes of every (kept) tensor -- all known upfront from
safetensors headers -- and a total ``device_memory_budget`` the load may
occupy at peak, compute a per-file chunk budget so that

    resident_bytes + live_transient_buffers <= budget

holds at every moment of the load. Budgets are large while cumulative
resident bytes are small (whole-file loads, no chunking overhead) and
decline only as the device fills, so chunking cost is paid only where the
fit actually requires it. The plan is precomputed and deterministic: same
files, same filter, same budget -> same plan. There is no runtime feedback.

Why per-file budgets are safe (the bound): let ``G(i)`` be the last file of
file ``i``'s batch group -- the ``group_size`` files loaded concurrently, one
per rank. While any chunk of file ``i`` is alive, resident bytes are at most
``R[G(i)+1]`` (only tensors of files ``<= G(i)`` have been materialized; the
group's kept bytes are charged up front, because broadcast leaves every rank
holding every file of the group) and every live transient buffer belongs to
file ``i`` or a later file ``j > i``. The per-file budget ``B`` declines
monotonically with ``R``, so every live buffer span is ``<= B[i]``. With at
most ``depth * transient_multiplier`` buffers plus one yield clone alive on
any one rank,

    peak <= R[G(i)+1]
            + (depth * transient_multiplier + yield_clone) * B[i]
         <= budget   (by choice of B[i]).

With ``group_size == 1``, ``G(i) == i`` and this reduces to ``R[i+1]``.
``yield_clone`` is 1 when a non-resident yielded tensor is cloned, else 0.

The budget itself is the caller's to choose -- only the caller knows what
else will live on the device. A caller sizing it from free memory should
keep a reserve for allocator rounding and the copier's fixed pools (5% or
1 GiB, whichever is larger, is a reasonable starting point), e.g.::

    free, _ = torch.cuda.mem_get_info(dev)
    budget = free - max(free // 20, 1 << 30)

and, under broadcast loading, all-reduce(MIN) that value before passing it:
per-rank readings diverge, and differing budgets would give ranks different
plans and deadlock the lockstep broadcast sequence.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Set, Tuple

from .common import SafeTensorsMetadata


def plan_chunks(
    metadata: SafeTensorsMetadata,
    max_batch_bytes: int,
    keep_tensor: Optional[Callable[[str], bool]] = None,
    merge_gap: int = 4096,
) -> List[Tuple[Set[str], List[Tuple[int, int]]]]:
    """Partition a shard's tensors into byte-budgeted sub-file chunks.

    Walks the (optionally ``keep_tensor``-filtered) tensors in file-offset
    order and greedily packs consecutive tensors into a chunk while the
    chunk's byte span stays within ``max_batch_bytes``. Returns a list of
    ``(names, byte_ranges)`` pairs; feed each to a copier's ``set_chunk`` to
    load only that chunk, bounding peak device memory to ~``max_batch_bytes``
    (the device buffer covers the chunk's span). The byte ranges are the
    kept tensors' runs within the chunk (kept tensors separated by at most
    ``merge_gap`` bytes are coalesced), so holes left by a filter are not
    read even though the buffer spans them.

    A tensor is the atomic load unit, so ``max_batch_bytes`` must be at least
    the largest kept tensor -- otherwise that tensor fits in no chunk and a
    ``ValueError`` is raised naming it.
    """
    chunks: List[Tuple[Set[str], List[Tuple[int, int]]]] = []
    cur: List[str] = []
    cur_runs: List[List[int]] = []  # merged kept runs within the chunk
    cur_start = 0

    def _push(name: str, s: int, e: int) -> None:
        if cur_runs and s - cur_runs[-1][1] <= merge_gap:
            cur_runs[-1][1] = max(cur_runs[-1][1], e)
        else:
            cur_runs.append([s, e])

    for name, frame in metadata.tensors.items():  # insertion order == offset order
        if keep_tensor is not None and not keep_tensor(name):
            continue
        s = metadata.header_length + frame.data_offsets[0]
        e = metadata.header_length + frame.data_offsets[1]
        tensor_bytes = frame.data_offsets[1] - frame.data_offsets[0]
        if tensor_bytes > max_batch_bytes:
            raise ValueError(
                f"max_batch_bytes={max_batch_bytes} is smaller than tensor "
                f"'{name}' ({tensor_bytes} bytes); it must be at least the "
                f"largest kept tensor so every tensor fits in one chunk."
            )
        if cur and (e - cur_start) > max_batch_bytes:
            chunks.append((set(cur), [(s0, e0) for s0, e0 in cur_runs]))
            cur, cur_runs = [], []
        if not cur:
            cur_start = s
        cur.append(name)
        _push(name, s, e)
    if cur:
        chunks.append((set(cur), [(s0, e0) for s0, e0 in cur_runs]))
    return chunks


class BudgetInfeasibleError(ValueError):
    """The model cannot be loaded within the given device memory budget."""


@dataclass(frozen=True)
class FileWeightStats:
    """Per-file byte accounting for the fit plan (kept tensors only)."""

    path: str
    kept_bytes: int  # sum of kept tensor bytes: resident growth
    span_bytes: int  # last kept byte - first kept byte: single-chunk buffer size
    largest_tensor: int  # chunk floor: a tensor is the atomic load unit


def pipeline_depth(queue_size: int) -> int:
    """Worst-case number of concurrently live device buffers for a queue size.

    queue_size -1: fully serial, 1 buffer. 0: unbuffered pipeline, the producer
    materializes the next buffer while the consumer holds one -> 2. n > 0: n in
    the queue + 1 being produced + 1 being consumed -> n + 2.
    """
    if queue_size < 0:
        return 1
    return queue_size + 2


def collect_file_stats(
    metas: List[Tuple[str, SafeTensorsMetadata]],
    keep_tensor: Optional[Callable[[str], bool]] = None,
) -> List[FileWeightStats]:
    """Byte accounting per file from already-parsed headers."""
    stats = []
    for path, meta in metas:
        kept = span_start = span_end = largest = 0
        first = True
        for name, frame in meta.tensors.items():
            if keep_tensor is not None and not keep_tensor(name):
                continue
            s, e = frame.data_offsets[0], frame.data_offsets[1]
            kept += e - s
            largest = max(largest, e - s)
            if first:
                span_start, first = s, False
            span_end = max(span_end, e)
        stats.append(
            FileWeightStats(path, kept, span_end - span_start if kept else 0, largest)
        )
    return stats


def plan_file_budgets(
    stats: List[FileWeightStats],
    device_memory_budget: int,
    depth: int,
    max_batch_bytes: Optional[int] = None,
    accumulate_resident: bool = True,
    transient_multiplier: int = 1,
    group_size: int = 1,
    account_for_yield_clone: bool = False,
) -> List[int]:
    """Per-file chunk budgets satisfying the peak-memory bound.

    ``accumulate_resident=True`` models consumers that keep every yielded
    tensor (resident grows by cumulative kept bytes). ``False`` models
    consumers whose destination memory is already allocated before the load
    (e.g. copying into preallocated model parameters): resident growth is 0
    and the plan degenerates to a uniform budget determined by the transient
    depth.

    ``transient_multiplier`` scales the per-buffer transient cost: how many
    times over the copier stages each in-flight chunk. Only the copier knows
    (its reader path decides), so callers pass
    ``CopierInterface.chunk_transient_multiplier(paths)``. Fixed overheads
    that do not scale with chunk size -- bounce-buffer pools, the O_DIRECT
    reader's thread pool (measured on GB10 unified memory: +~150 MB regardless
    of chunk size) -- are not modelled here and must be left outside the
    budget the caller passes.

    ``group_size`` is the number of files loaded concurrently, one per rank
    (``pg.size()`` under broadcast loading, 1 otherwise). Every rank ends up
    with every tensor of the group, so a whole group's kept bytes become
    resident together and each file in it is charged the group total rather
    than its own prefix -- otherwise the bound below under-counts by up to
    ``group_size - 1`` files whenever shard sizes are uneven.

    ``account_for_yield_clone`` reserves one chunk budget for a non-resident
    yielded tensor clone.

    Returns one budget per file; feed each to
    ``SafeTensorsMetadata.plan_chunks``. A budget >= the file's span yields a
    single whole-span chunk (no splitting). Raises ``BudgetInfeasibleError``
    at plan time when some file's largest tensor cannot fit.
    """
    if device_memory_budget <= 0:
        raise BudgetInfeasibleError(
            f"device_memory_budget={device_memory_budget} must be positive"
        )
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")
    if transient_multiplier < 1:
        raise ValueError(
            f"transient_multiplier must be >= 1, got {transient_multiplier}"
        )
    if group_size < 1:
        raise ValueError(f"group_size must be >= 1, got {group_size}")
    eff_depth = depth * transient_multiplier + int(account_for_yield_clone)
    # Cumulative kept bytes through the end of each file's batch group. With
    # group_size == 1 this is just R[i+1]; under broadcast the whole group is
    # in flight at once, so every file in it is charged the group's total.
    kept_through_group = []
    running = 0
    for end in range(len(stats)):
        running += stats[end].kept_bytes
        kept_through_group.append(running)
    group_resident = [
        kept_through_group[min((i // group_size + 1) * group_size, len(stats)) - 1]
        for i in range(len(stats))
    ]
    budgets = []
    for i, st in enumerate(stats):
        resident = group_resident[i] if accumulate_resident else 0
        b = (device_memory_budget - resident) // eff_depth
        if max_batch_bytes is not None:
            b = min(b, max_batch_bytes)
        if b < st.largest_tensor:
            required = resident + eff_depth * st.largest_tensor
            raise BudgetInfeasibleError(
                f"Model does not fit device_memory_budget: loading '{st.path}' "
                f"needs >= {required} bytes ({resident} resident + {eff_depth} x "
                f"{st.largest_tensor} transient), budget is {device_memory_budget}. "
                f"Reduce pipeline depth (queue_size), free device memory, or pass "
                f"a larger explicit budget."
            )
        budgets.append(b)
    return budgets
