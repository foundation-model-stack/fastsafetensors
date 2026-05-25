# SPDX-License-Identifier: Apache-2.0
"""Expert-parallel (EP) slice helpers.

Under expert parallelism each rank only *uses* the routed experts it owns, yet
file-granular loading makes every rank read the whole shard -- the unowned
experts' bytes are read and then discarded. These helpers build a tensor-name
predicate selecting just this rank's owned experts (plus every non-expert
tensor), so a partial-read-capable loader can skip the unowned bytes:

    from fastsafetensors import SafeTensorsFileLoader
    from fastsafetensors.ep_slice import expert_parallel_filter

    loader = SafeTensorsFileLoader(pg, device, nogds=True)
    loader.set_tensor_filter(expert_parallel_filter(num_experts=256,
                                                     ep_size=2, ep_rank=rank))
    loader.add_filenames(...)
    bufs = loader.copy_files_to_device()

Owned experts use contiguous-block ("linear") assignment: each rank owns
``num_experts // ep_size`` consecutive experts, with any remainder given to the
lowest-numbered ranks. This is a common expert-to-rank convention; the caller is
responsible for ensuring it matches the assignment its runtime expects. No
external dependency is required.
"""
import os
import re
from typing import Callable, Optional, Pattern, Tuple

# Matches the per-expert index in routed-MoE tensor names, e.g.
# "model.layers.3.mlp.experts.42.w1.weight" or DeepSeek's
# "...ffn.experts.42.gate_proj.weight". Override for a different convention.
DEFAULT_EXPERT_PATTERN: Pattern[str] = re.compile(r"\.experts\.(\d+)\.")


def owned_expert_range(num_experts: int, ep_size: int, ep_rank: int) -> Tuple[int, int]:
    """Return the ``[lo, hi)`` routed-expert indices owned by ``ep_rank``.

    Contiguous-block ("linear") assignment: each rank owns a consecutive block
    of experts, with the remainder distributed to the lowest-numbered ranks.
    """
    if ep_size <= 0:
        raise ValueError(f"ep_size must be positive, got {ep_size}")
    if not 0 <= ep_rank < ep_size:
        raise ValueError(f"ep_rank {ep_rank} out of range for ep_size {ep_size}")
    base = num_experts // ep_size
    rem = num_experts % ep_size
    local = base + (1 if ep_rank < rem else 0)
    start = ep_rank * base + min(ep_rank, rem)
    return (start, start + local)


def expert_parallel_filter(
    num_experts: int,
    ep_size: int,
    ep_rank: int,
    pattern: Pattern[str] = DEFAULT_EXPERT_PATTERN,
) -> Callable[[str], bool]:
    """Build a ``keep(name) -> bool`` predicate for this EP rank.

    Non-expert tensors (names not matching ``pattern``) are kept on every rank;
    routed-expert tensors are kept only when their index is in this rank's owned
    range. Pass the predicate to ``SafeTensorsFileLoader.set_tensor_filter`` or
    ``SafeTensorsMetadata.select_byte_ranges``.
    """
    lo, hi = owned_expert_range(num_experts, ep_size, ep_rank)

    def keep(name: str) -> bool:
        m = pattern.search(name)
        if m is None:
            return True
        return lo <= int(m.group(1)) < hi

    return keep


def expert_parallel_filter_from_env() -> Optional[Callable[[str], bool]]:
    """Build an EP filter from environment variables, or ``None`` if disabled.

    Recognized variables (kept compatible with the DGX Spark overlay this
    prototype generalizes):

      ``FASTSAFETENSORS_EP_SLICE=1``        enable EP-slice reading
      ``FASTSAFETENSORS_EP_NUM_EXPERTS=N``  global routed-expert count (required)
      ``FASTSAFETENSORS_EP_SIZE`` / ``_RANK``  override EP size/rank; otherwise
                                            taken from the initialized
                                            torch.distributed group, else from
                                            ``WORLD_SIZE`` / ``RANK``.

    Returns ``None`` (load everything) unless EP-slice is enabled, the expert
    count is known, and ``ep_size > 1``.
    """
    if os.getenv("FASTSAFETENSORS_EP_SLICE", "0") != "1":
        return None
    num_experts = int(os.getenv("FASTSAFETENSORS_EP_NUM_EXPERTS", "0"))
    if num_experts <= 0:
        return None
    ep_size = int(os.getenv("FASTSAFETENSORS_EP_SIZE", "0"))
    ep_rank = int(os.getenv("FASTSAFETENSORS_EP_RANK", "-1"))
    if ep_size <= 0 or ep_rank < 0:
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                ep_size = dist.get_world_size()
                ep_rank = dist.get_rank()
        except Exception:
            pass
    if ep_size <= 0:
        ep_size = int(os.getenv("WORLD_SIZE", "1"))
    if ep_rank < 0:
        ep_rank = int(os.getenv("RANK", "0"))
    if ep_size <= 1:
        return None
    return expert_parallel_filter(num_experts, ep_size, ep_rank)
