# SPDX-License-Identifier: Apache-2.0
"""Checkpoint slices for model consumers in fst-sweep's vLLM.

Only partitioning is reproduced: destinations remain separate raw tensors.
Unknown names or partitions requiring padding are rejected rather than guessed.
"""

import json
import re


def deepseek_v4_partition(path, entries, world, ep_active):
    config_path = (path if path.is_dir() else path.parent) / "config.json"
    config = json.loads(config_path.read_text())
    if config.get("model_type") != "deepseek_v4":
        raise ValueError("deepseek-v4 partitioning requires a DeepSeek V4 config")
    if world > 1 and not ep_active:
        raise ValueError("deepseek-v4 TP requires --enable-expert-parallel and filter")
    # This profile matches PP=DP=1, MARLIN, no speculative model and no SP.
    # With DP>1 or MegaMoE the shared experts instead become replicated.
    for key in ("num_attention_heads", "o_groups", "vocab_size"):
        if config[key] % world:
            raise ValueError(f"{key} must be divisible by TP for deepseek-v4")
    block = config.get("quantization_config", {}).get("weight_block_size")
    if block != [128, 128] or config.get("expert_dtype") != "fp4":
        raise ValueError("deepseek-v4 profile supports block-128 FP8 / MXFP4 only")
    if config["vocab_size"] % 64:
        raise ValueError("Vocabulary padding is not supported by deepseek-v4 profile")
    if config["moe_intermediate_size"] % (128 * world):
        raise ValueError("Shared-expert TP padding is not supported by this profile")
    mapping, ignored = {}, set()
    for name, entry in entries.items():
        if name.startswith("mtp."):
            # The vLLM weights mapper drops MTP tensors for the target model.
            # They are still yielded by the iterator, just not copied.
            ignored.add(name)
            continue
        dim = -1
        if name in ("embed.weight", "head.weight"):
            dim = 0
        elif name == "norm.weight" or re.fullmatch(r"hc_head_(base|fn|scale)", name):
            pass
        else:
            match = re.fullmatch(r"layers\.(\d+)\.(.+)", name)
            if match is None or int(match[1]) >= config["num_hidden_layers"]:
                raise ValueError(f"Unsupported DeepSeek V4 tensor: {name}")
            suffix = match[2]
            if re.fullmatch(r"ffn\.experts\.\d+\.w[123]\.(weight|scale)", suffix):
                # EP assigns whole experts; nonlocal scales are ignored by the
                # common consumer and nonlocal heavy weights by the iterator.
                pass
            elif suffix == "attn.attn_sink":
                dim = 0
            elif re.fullmatch(r"attn\.(wq_b|wo_a)\.(weight|scale)", suffix):
                dim = 0
            elif re.fullmatch(r"attn\.wo_b\.(weight|scale)", suffix):
                dim = 1
            elif re.fullmatch(r"ffn\.shared_experts\.w[13]\.(weight|scale)", suffix):
                dim = 0
            elif re.fullmatch(r"ffn\.shared_experts\.w2\.(weight|scale)", suffix):
                dim = 1
            elif (
                re.fullmatch(r"attn\.(wq_a|wkv)\.(weight|scale)", suffix)
                or re.fullmatch(r"attn\.(q_norm|kv_norm)\.weight", suffix)
                or re.fullmatch(r"(attn_norm|ffn_norm)\.weight", suffix)
                or re.fullmatch(r"hc_(attn|ffn)_(base|fn|scale)", suffix)
                or re.fullmatch(r"ffn\.gate\.(weight|bias|tid2eid)", suffix)
                or re.fullmatch(
                    r"attn\.(indexer\.)?compressor\.(ape|norm\.weight|w(kv|gate)\.weight)",
                    suffix,
                )
                or re.fullmatch(r"attn\.indexer\.wq_b\.(weight|scale)", suffix)
                or suffix == "attn.indexer.weights_proj.weight"
            ):
                pass
            else:
                raise ValueError(f"Unsupported DeepSeek V4 tensor: {name}")
        if dim >= 0:
            shape = entry["shape"]
            if dim >= len(shape) or shape[dim] % world:
                raise ValueError(f"Tensor cannot be evenly TP-sharded: {name}")
            mapping[name] = dim
    return mapping, ignored


def nemotron_h_partition(path, entries, world, ep_active):
    """Nemotron-H compressed-tensors channel FP8, TP + EP, no SP or MTP.

    Mirrors NemotronHAttention/MLP and MambaMixer2 weight loaders. A fused
    Mamba projection needs one source slice per logical output, not a single
    contiguous fraction of the checkpoint matrix. Quantization transforms and
    destination packing remain outside this raw-copy benchmark.
    """
    config_path = (path if path.is_dir() else path.parent) / "config.json"
    config = json.loads(config_path.read_text())
    if config.get("model_type") != "nemotron_h":
        raise ValueError("nemotron-h partitioning requires a Nemotron-H config")
    if world > 1 and not ep_active:
        raise ValueError("nemotron-h TP requires --enable-expert-parallel and filter")
    quant = config.get("quantization_config", {})
    schemes = list(quant.get("config_groups", {}).values())
    if (
        quant.get("quant_method") != "compressed-tensors"
        or not schemes
        or any(
            scheme.get("weights", {}).get("strategy") != "channel"
            or scheme.get("weights", {}).get("num_bits") != 8
            or scheme.get("weights", {}).get("type") != "float"
            or not scheme.get("input_activations", {}).get("dynamic")
            for scheme in schemes
        )
        or quant.get("transform_config")
    ):
        raise ValueError("nemotron-h profile supports dynamic channel FP8 only")
    hidden = config["hidden_size"]
    heads = config["num_attention_heads"]
    kv_heads = config["num_key_value_heads"]
    head_dim = config.get("head_dim") or hidden // heads
    mamba_heads = config["mamba_num_heads"]
    intermediate = mamba_heads * config["mamba_head_dim"]
    groups = config["n_groups"]
    group_width = groups * config["ssm_state_size"]
    shared = config["moe_shared_expert_intermediate_size"] * config["n_shared_experts"]
    if any(
        value % world
        for value in (heads, mamba_heads, groups, shared, config["vocab_size"])
    ):
        raise ValueError(
            "nemotron-h requires TP to divide attention/Mamba heads, groups, shared MLP and vocabulary"
        )
    if not kv_heads or (kv_heads % world if kv_heads >= world else world % kv_heads):
        raise ValueError("Nemotron-H KV heads and TP must divide one another")
    if config["vocab_size"] % 64:
        raise ValueError("Vocabulary padding is not supported by nemotron-h profile")
    mapping, ignored = {}, set()
    for name, entry in entries.items():
        if name.startswith("mtp."):
            ignored.add(name)
            continue
        dim = -1
        if name in ("backbone.embeddings.weight", "lm_head.weight"):
            dim = 0
        elif name == "backbone.norm_f.weight":
            pass
        else:
            match = re.fullmatch(r"backbone\.layers\.(\d+)\.(.+)", name)
            if match is None or int(match[1]) >= len(config["layers_block_type"]):
                raise ValueError(f"Unsupported Nemotron-H tensor: {name}")
            kind = config["layers_block_type"][int(match[1])]
            suffix = match[2]
            if suffix == "norm.weight":
                pass
            elif kind == "mamba" and suffix in (
                "mixer.in_proj.weight",
                "mixer.in_proj.weight_scale",
                "mixer.in_proj.bias",
            ):
                dim = {
                    "segments": [
                        intermediate,
                        intermediate,
                        group_width,
                        group_width,
                        mamba_heads,
                    ]
                }
            elif kind == "mamba" and suffix in (
                "mixer.conv1d.weight",
                "mixer.conv1d.bias",
            ):
                dim = {"segments": [intermediate, group_width, group_width]}
            elif kind == "mamba" and suffix in (
                "mixer.A_log",
                "mixer.D",
                "mixer.dt_bias",
                "mixer.norm.weight",
            ):
                dim = 0
            elif kind in ("mamba", "attention") and suffix in (
                "mixer.out_proj.weight" if kind == "mamba" else "mixer.o_proj.weight",
            ):
                dim = 1
            elif kind == "attention" and re.fullmatch(
                r"mixer\.q_proj\.(weight|weight_scale|bias)", suffix
            ):
                dim = 0
            elif kind == "attention" and re.fullmatch(
                r"mixer\.[kv]_proj\.(weight|weight_scale|bias)", suffix
            ):
                if entry["shape"][0] != kv_heads * head_dim:
                    raise ValueError(f"Unexpected KV rows for {name}")
                dim = {"dim": 0, "shards": min(world, kv_heads)}
            elif kind in ("mamba", "attention") and suffix in (
                (
                    "mixer.out_proj.weight_scale"
                    if kind == "mamba"
                    else "mixer.o_proj.weight_scale"
                ),
                "mixer.out_proj.bias" if kind == "mamba" else "mixer.o_proj.bias",
            ):
                # Channel scales index output rows, which are replicated for
                # RowParallelLinear; the matrix shards its input columns.
                pass
            elif kind == "moe" and re.fullmatch(
                r"mixer\.experts\.\d+\.(up|down)_proj\.(weight|weight_scale)", suffix
            ):
                pass  # Whole local experts/scales, assigned by the EP consumer.
            elif kind == "moe" and re.fullmatch(
                r"mixer\.shared_experts\.up_proj\.(weight|weight_scale|bias)", suffix
            ):
                dim = 0
            elif kind == "moe" and suffix == "mixer.shared_experts.down_proj.weight":
                dim = 1
            elif kind == "moe" and (
                re.fullmatch(
                    r"mixer\.fc[12]_latent_proj\.(weight|weight_scale|bias)", suffix
                )
                or suffix
                in (
                    "mixer.gate.weight",
                    "mixer.gate.e_score_correction_bias",
                    "mixer.shared_experts.down_proj.weight_scale",
                    "mixer.shared_experts.down_proj.bias",
                )
            ):
                pass  # ReplicatedLinear, router, or row-parallel channel scales.
            else:
                raise ValueError(f"Unsupported Nemotron-H tensor: {name}")
        if isinstance(dim, dict):
            partition_slice(entry["shape"], dim, 0, world)
            mapping[name] = dim
        elif dim >= 0:
            shape = entry["shape"]
            if dim >= len(shape) or shape[dim] % world:
                raise ValueError(f"Tensor cannot be evenly TP-sharded: {name}")
            mapping[name] = dim
    return mapping, ignored


def partition_slice(shape, dim, rank, world):
    if isinstance(dim, dict):
        if set(dim) == {"segments"}:
            segments = dim["segments"]
            if (
                not shape
                or not segments
                or any(type(n) is not int or n <= 0 or n % world for n in segments)
                or sum(segments) != shape[0]
            ):
                raise ValueError(f"Invalid fused TP segments {segments} for {shape}")
            selections, offset = [], 0
            for size in segments:
                start = offset + rank * (size // world)
                selections.append(
                    (slice(start, start + size // world),)
                    + (slice(None),) * (len(shape) - 1)
                )
                offset += size
            return selections, [shape[0] // world, *shape[1:]]
        if set(dim) == {"dim", "shards"}:
            shards = dim["shards"]
            if type(shards) is not int or shards <= 0 or world % shards:
                raise ValueError(f"Invalid replicated TP shards: {shards}")
            return partition_slice(shape, dim["dim"], rank // (world // shards), shards)
        raise ValueError(f"Unknown partition specification: {dim}")
    if type(dim) is not int or dim < -1 or dim >= len(shape):
        raise ValueError(f"Invalid shard dimension: {dim} for {shape}")
    shape = list(shape)
    selection = [slice(None)] * len(shape)
    if dim >= 0:
        block = (shape[dim] + world - 1) // world
        begin, end = min(rank * block, shape[dim]), min((rank + 1) * block, shape[dim])
        selection[dim] = slice(begin, end)
        shape[dim] = end - begin
    return tuple(selection), shape


def partition_views(source, target, selection):
    """Yield views for direct copies; never concatenate fused source pieces."""
    if isinstance(selection, list):
        offset = 0
        for part in selection:
            view = source[part]
            yield view, target[offset : offset + view.shape[0]]
            offset += view.shape[0]
        if offset != target.shape[0]:
            raise ValueError("Fused destination size does not match source slices")
    else:
        yield source[selection], target
