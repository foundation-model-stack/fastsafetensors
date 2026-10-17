# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Loading policy adapted from fst-sweep's vLLM EP filter and iterators.

No model code is imported. TP destinations still use the caller's shard map.
"""

import json
import re
import struct

_EXPERT_ID = re.compile(r"\.experts\.(\d+)\.")


def parse_expert_id(name):
    match = _EXPERT_ID.search(name)
    return int(match[1]) if match else None


def compute_local_expert_ids(num_experts, size, rank, placement="linear"):
    if size <= 1:
        return None
    if placement == "round_robin":
        return set(range(rank, num_experts, size))
    if placement != "linear":
        raise ValueError(f"Unknown expert placement: {placement}")
    base, remainder = divmod(num_experts, size)
    start = rank * base + min(rank, remainder)
    return set(range(start, start + base + (rank < remainder)))


def should_skip_weight(name, local_ids):
    expert = parse_expert_id(name)
    return (
        local_ids is not None
        and expert is not None
        and name.endswith((".weight", ".weight_packed"))
        and expert not in local_ids
    )


def num_experts_from_config(path):
    config_path = (path if path.is_dir() else path.parent) / "config.json"
    if not config_path.is_file():
        return None
    config = json.loads(config_path.read_text())
    config = config.get("text_config", config)
    for key in (
        "num_experts",
        "moe_num_experts",
        "n_routed_experts",
        "num_local_experts",
    ):
        if key in config:
            value = config[key]
            if isinstance(value, list):
                if not value or len(set(value)) != 1:
                    raise ValueError(
                        "Layer-specific expert counts require --num-experts"
                    )
                value = value[0]
            if type(value) is not int or value < 0:
                raise ValueError(f"Invalid {key} in {config_path}")
            return value
    blocks = config.get("block_configs", [])
    if blocks:
        return max(
            (
                block.get("n_routed_experts", 0)
                for block in blocks
                if block.get("block_type") == "moe"
            ),
            default=0,
        )
    # Avoid treating unknown remote-code MoE config formats as dense models.
    return None


def configure(args, path, entries, rank, world, device, benchmark):
    from safetensors import safe_open

    profile = args.loader_profile == "vllm"
    if profile and benchmark != "parallel":
        raise ValueError("--loader-profile vllm requires benchmark_parallel")
    if args.reserve_mib is None:
        args.reserve_mib = 1024 if profile else 512
    if hasattr(args, "queue_size") and args.queue_size is None:
        args.queue_size = 0
    count = args.num_experts
    enabled = args.enable_expert_parallel and args.enable_ep_weight_filter and world > 1
    if enabled and count is None:
        count = num_experts_from_config(path)
        if count is None:
            if any(parse_expert_id(name) is not None for name in entries):
                raise ValueError("Cannot determine expert count; specify --num-experts")
            count = 0
    if count is not None and (type(count) is not int or count < 0):
        raise ValueError("--num-experts must be nonnegative")
    if (
        enabled
        and count == 0
        and any(parse_expert_id(name) is not None for name in entries)
    ):
        raise ValueError("Expert tensors require a positive --num-experts/config count")
    local_ids = (
        compute_local_expert_ids(count, world, rank, args.expert_placement)
        if enabled and count
        else None
    )
    if local_ids is not None and any(
        (expert := parse_expert_id(name)) is not None and expert >= count
        for name in entries
    ):
        raise ValueError("Checkpoint expert id exceeds --num-experts/config count")
    args._local_expert_ids = local_ids
    args._tensor_filter = (
        (lambda name: not should_skip_weight(name, local_ids))
        if local_ids is not None
        else None
    )
    broadcast_ep = args.fastsafetensors_ep_mode == "broadcast"
    if broadcast_ep and local_ids is None:
        raise ValueError(
            "--fastsafetensors-ep-mode broadcast requires an active EP filter "
            "(both EP flags, multiple ranks and a positive expert count)"
        )
    if broadcast_ep and args.all_local:
        raise ValueError(
            "--fastsafetensors-ep-mode broadcast conflicts with --all-local"
        )
    # Rank-specific filters require independent reads in the released loader.
    # The broadcast comparison delivers everything, retaining EP only in the
    # consumer so every rank executes the same collective sequence.
    if local_ids is not None:
        args.all_local = not broadcast_ep
    args._fastsafetensors_tensor_filter = None if broadcast_ep else args._tensor_filter
    if args.borrowed_tensors is None:
        args.borrowed_tensors = profile and (world == 1 or args.all_local)
    selected = {
        name: entry
        for name, entry in entries.items()
        if not should_skip_weight(name, local_ids)
    }
    routes = {}
    for reader in args.readers:
        effective, reason = reader, None
        if reader.startswith("safetensors-prefetch"):
            if local_ids is not None:
                effective, reason = (
                    "safetensors",
                    "EP weight filter disables CUDA prefetch",
                )
            elif profile:
                if device.type != "cuda":
                    effective, reason = "safetensors", "CUDA device unavailable"
                elif not hasattr(safe_open, "prefetch"):
                    effective, reason = "safetensors", "CUDA prefetch API unavailable"
                else:
                    effective = "safetensors-prefetch-sharded"
        routes[reader] = dict(effective_reader=effective, fallback_reason=reason)
    ep = dict(
        enabled=args.enable_expert_parallel,
        filter_requested=args.enable_ep_weight_filter,
        filter_active=local_ids is not None,
        num_experts=count,
        size=world,
        rank=rank,
        placement=args.expert_placement,
        local_expert_ids=sorted(local_ids) if local_ids is not None else None,
    )
    return selected, ep, routes


def largest_tensor(files, keep=None):
    largest = 0
    for path in files:
        with path.open("rb") as handle:
            header = json.loads(handle.read(struct.unpack("<Q", handle.read(8))[0]))
        for name, entry in header.items():
            if name == "__metadata__" or (keep is not None and not keep(name)):
                continue
            start, end = entry["data_offsets"]
            largest = max(largest, end - start)
    return largest
