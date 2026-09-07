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
group's resident bytes are charged up front, because broadcast leaves every
rank holding every file of the group) and every live transient buffer belongs to
file ``i`` or a later file ``j > i``. The per-file budget ``B`` declines
monotonically with ``R``, so every live buffer span is ``<= B[i]``. With at
most ``depth * transient_multiplier`` buffers plus one yield clone alive on
any one rank,

    peak <= R[G(i)+1]
            + (depth * transient_multiplier + yield_clone) * B[i]
         <= budget   (by choice of B[i]).

With ``group_size == 1``, ``G(i) == i`` and this reduces to ``R[i+1]``.
``yield_clone`` is 1 when a non-resident yielded tensor is cloned, else 0.

``fit_queue_size`` inverts this bound exactly: given a budget it returns the
deepest queue size whose plan still fits. ``ParallelLoader`` clamps its own
pipeline depth to that, so too small a budget costs throughput, not the load.

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
    kept_bytes: int  # sum of kept tensor bytes: what the load reads
    span_bytes: int  # last kept byte - first kept byte: single-chunk buffer size
    largest_tensor: int  # chunk floor: a tensor is the atomic load unit
    # Subset of kept_bytes that stays on the device after the load. Equal to
    # kept_bytes for a consumer that keeps everything, which is why the two
    # were one number; they diverge for a consumer that relocates part of what
    # it is handed (e.g. offloading embedding tables to host memory).
    # None means "not measured": callers treat it as kept_bytes.
    resident_bytes: Optional[int] = None

    @property
    def resident(self) -> int:
        """resident_bytes, defaulting to kept_bytes when not measured."""
        return self.kept_bytes if self.resident_bytes is None else self.resident_bytes


def pipeline_depth(queue_size: int) -> int:
    """Worst-case number of concurrently live device buffers for a queue size.

    queue_size -1: fully serial, 1 buffer. 0: unbuffered pipeline, the producer
    materializes the next buffer while the consumer holds one -> 2. n > 0: n in
    the queue + 1 being produced + 1 being consumed -> n + 2.
    """
    if queue_size < 0:
        return 1
    return queue_size + 2


def load_depth(queue_size: int, group_size: int = 1) -> int:
    """Concurrently live device buffers: ``pipeline_depth`` plus, under
    broadcast, one in-flight receive tensor.

    The ``depth`` ``plan_file_budgets`` expects. ``fit_queue_size`` inverts this
    function, so both live here and cannot drift apart.
    """
    return pipeline_depth(queue_size) + (1 if group_size > 1 else 0)


def _effective_depth(
    depth: int, transient_multiplier: int, account_for_yield_clone: bool
) -> int:
    """Live transient buffers, each charged one chunk budget."""
    return depth * transient_multiplier + int(account_for_yield_clone)


def _group_resident(stats: List[FileWeightStats], group_size: int) -> List[int]:
    """Cumulative resident bytes through the end of each file's batch group: just
    ``R[i+1]`` when ``group_size == 1``, else the group total for every file in
    it, since broadcast puts the whole group in flight at once.
    """
    kept_through_group = []
    running = 0
    for st in stats:
        running += st.resident
        kept_through_group.append(running)
    return [
        kept_through_group[min((i // group_size + 1) * group_size, len(stats)) - 1]
        for i in range(len(stats))
    ]


def collect_file_stats(
    metas: List[Tuple[str, SafeTensorsMetadata]],
    keep_tensor: Optional[Callable[[str], bool]] = None,
    resident_tensor: Optional[Callable[[str], bool]] = None,
) -> List[FileWeightStats]:
    """Byte accounting per file from already-parsed headers.

    ``keep_tensor`` selects what is read; ``resident_tensor`` selects which of
    those stay on the device once the load is done. Default (None) is that
    everything read stays, which is what a consumer keeping every yielded
    tensor does. A consumer that relocates a subset -- offloading embedding
    tables to host memory, say -- passes a predicate so the plan charges only
    the bytes that actually remain.

    ``resident_tensor`` MUST be a pure function of the tensor name, identical
    on every rank, for the same reason ``keep_tensor`` must: the fit plan is
    recomputed independently per rank and the broadcast sequence desyncs if the
    plans differ. Deciding residency from rank-local state -- observed free
    memory, an offload budget consumed as the load proceeds -- breaks that
    silently. Resolve such a policy to a name predicate before planning.

    Residency does not move ``largest_tensor``: a tensor that is relocated
    after the load is still read whole, so it still sets the chunk floor.
    """
    stats = []
    for path, meta in metas:
        kept = span_start = span_end = largest = resident = 0
        first = True
        for name, frame in meta.tensors.items():
            if keep_tensor is not None and not keep_tensor(name):
                continue
            s, e = frame.data_offsets[0], frame.data_offsets[1]
            kept += e - s
            if resident_tensor is None or resident_tensor(name):
                resident += e - s
            largest = max(largest, e - s)
            if first:
                span_start, first = s, False
            span_end = max(span_end, e)
        stats.append(
            FileWeightStats(
                path,
                kept,
                span_end - span_start if kept else 0,
                largest,
                resident,
            )
        )
    return stats


def has_non_resident(stats: List[FileWeightStats]) -> bool:
    """True when any kept bytes do not stay resident.

    Lets a caller decide whether a transient yielded-tensor clone needs its own
    budget without re-deriving the predicate.
    """
    return any(st.resident < st.kept_bytes for st in stats)


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

    ``accumulate_resident=True`` charges the residency recorded in ``stats``
    (``FileWeightStats.resident``, which is every kept byte unless
    ``collect_file_stats`` was given a ``resident_tensor`` predicate).
    ``False`` models consumers whose destination memory is already allocated
    before the load (e.g. copying into preallocated model parameters): resident
    growth is 0 and the plan degenerates to a uniform budget determined by the
    transient depth. The two booleans remain shorthand for the ends of the
    range; a consumer that keeps only some of what it reads expresses that with
    the predicate rather than by choosing the less-wrong bool.

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
    eff_depth = _effective_depth(depth, transient_multiplier, account_for_yield_clone)
    group_resident = _group_resident(stats, group_size)
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
                + (
                    # Already as shallow as a load gets: ParallelLoader clamps
                    # to this, so queue_size is a spent lever by now.
                    "The pipeline is already fully serial (queue_size=-1): free "
                    "device memory or pass a larger explicit budget."
                    if depth <= load_depth(-1, group_size)
                    else "Reduce pipeline depth (queue_size), free device "
                    "memory, or pass a larger explicit budget."
                )
            )
        budgets.append(b)
    return budgets


def fit_queue_size(
    requested: int,
    stats: List[FileWeightStats],
    device_memory_budget: int,
    max_batch_bytes: Optional[int] = None,
    accumulate_resident: bool = True,
    transient_multiplier: int = 1,
    group_size: int = 1,
    account_for_yield_clone: bool = False,
) -> Optional[int]:
    """The deepest queue size ``<= requested`` whose plan fits the budget.

    ``None`` when no queue size fits: some file's largest tensor overflows the
    budget even loaded serially, so only freeing memory helps.

    The exact inverse of the bound ``plan_file_budgets`` enforces, not a search:
    the returned queue size is guaranteed to plan, and one deeper (when the
    result was clamped) is guaranteed to raise. Every input comes from the
    headers, so no trial load is needed.

    Takes ``plan_file_budgets``' arguments minus ``depth`` -- that is the
    unknown. Ranks must already pass an identical budget (see the module
    docstring), which makes the clamp identical too: no collective needed.
    """
    if transient_multiplier < 1:
        raise ValueError(
            f"transient_multiplier must be >= 1, got {transient_multiplier}"
        )
    if group_size < 1:
        raise ValueError(f"group_size must be >= 1, got {group_size}")
    if device_memory_budget <= 0:
        return None
    group_resident = _group_resident(stats, group_size)
    # Largest eff_depth every file can afford. The plan's test,
    # (budget - resident) // eff_depth >= largest, is exactly
    # budget - resident >= largest * eff_depth for eff_depth >= 1, so this
    # inverts with no floor-division slack either way.
    cap = None  # None: no kept tensor constrains the depth
    for i, st in enumerate(stats):
        if st.largest_tensor <= 0:
            continue  # nothing kept in this file: no chunk, no constraint
        if max_batch_bytes is not None and max_batch_bytes < st.largest_tensor:
            # The cap floors the budget below the atomic load unit at every
            # depth, so no depth helps.
            return None
        headroom = device_memory_budget - (
            group_resident[i] if accumulate_resident else 0
        )
        # Floors to <= 0 when even one buffer does not fit (headroom may be
        # negative once resident alone overruns the budget); `room` below turns
        # that into None, so no separate guard is needed here.
        limit = headroom // st.largest_tensor
        cap = limit if cap is None else min(cap, limit)
    if cap is None:
        return requested
    # eff_depth = load_depth(qs, group) * multiplier + clone <= cap, solved for
    # the load_depth term, then inverted.
    room = cap - int(account_for_yield_clone)
    if room < transient_multiplier:
        return None  # not even one live buffer fits alongside the clone
    base = room // transient_multiplier - (1 if group_size > 1 else 0)
    if base < 1:
        return None
    # Not injective at the bottom: queue_size=-1 and no queue at all both give
    # base depth 1, so 1 maps back to -1.
    return min(requested, -1 if base == 1 else base - 2)
