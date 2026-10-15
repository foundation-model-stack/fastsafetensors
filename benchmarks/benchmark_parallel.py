# SPDX-License-Identifier: Apache-2.0
"""Benchmark ParallelLoader.iterate_weights, including distributed broadcast."""

import argparse
import sys
from contextlib import closing
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks._common import add_arguments, run_benchmark, synchronize


def iter_weights(files, pg, device, args, telemetry):
    import torch

    from benchmarks._vllm import largest_tensor
    from fastsafetensors import ParallelLoader

    if args.broadcast_run_mib < 0 or (
        args.batch_mib is not None and args.batch_mib <= 0
    ):
        raise ValueError("Batch size must be positive; broadcast run size may be zero")
    budget = args.memory_budget_mib << 20 if args.memory_budget_mib else None
    if budget is None and device.type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        reserve = args.reserve_mib << 20
        if args.loader_profile == "vllm":
            reserve = max(reserve, free // 20)
        budget = max(0, free - reserve)
        if budget == 0:
            raise ValueError("No device memory remains for staging")
    if (
        budget is not None
        and pg is not None
        and (not args.all_local or args.loader_profile == "vllm")
    ):
        import torch.distributed as dist

        # Every rank must build the same chunk/broadcast sequence, even when
        # destination slices or other allocations leave different free memory.
        budget_pg = args._budget_pg
        budget_device = "cpu" if dist.get_backend(budget_pg) == "gloo" else device
        shared_budget = torch.tensor(budget, dtype=torch.int64, device=budget_device)
        dist.all_reduce(shared_budget, op=dist.ReduceOp.MIN, group=budget_pg)
        budget = shared_budget.item()
    args.effective_memory_budget_bytes = budget
    batch = (
        args.batch_mib << 20
        if args.batch_mib is not None
        else (
            2 * largest_tensor(files, args._fastsafetensors_tensor_filter)
            if args.loader_profile == "vllm"
            else 2048 << 20
        )
    )
    args.effective_loading_passes = []
    args.effective_batch_bytes = batch
    fixed_allocation = args.use_chunk_budget_as_allocation_size
    if fixed_allocation is None:
        fixed_allocation = args.loader_profile == "vllm" and (
            budget is not None or bool(batch)
        )
    args.effective_chunk_budget_as_allocation_size = fixed_allocation
    loader = ParallelLoader(
        pg,
        [str(file) for file in files],
        device=str(device),
        nogds=args.copier == "nogds",
        max_threads=args.threads,
        bbuf_size_kb=args.bounce_mib << 10,
        queue_size=args.queue_size,
        max_batch_bytes=batch or None,
        use_chunk_budget_as_allocation_size=fixed_allocation,
        device_memory_budget=budget,
        accumulate_resident=False,
        use_tqdm_on_load=False,
        all_local=args.all_local,
        tensor_filter=args._fastsafetensors_tensor_filter,
        overlap_io=args.overlap_io,
        borrowed_tensors=args.borrowed_tensors,
        broadcast_run_bytes=args.broadcast_run_mib << 20,
    )
    try:
        args.effective_loading_passes.append(
            dict(
                all_local=args.all_local,
                files=len(files),
                batch_bytes=batch,
                queue_size=loader.queue_size,
                planned_batches=len(loader.weight_files_batches),
            )
        )
        telemetry.append(loader.loader.copier_class.__name__)
        with closing(loader.iterate_weights()) as iterator:
            try:
                yield from iterator
            finally:
                synchronize(device)
    finally:
        loader.close()


def main():
    parser = add_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument(
        "--batch-mib",
        type=int,
        help="generic: 2048 MiB; vllm: twice the largest selected tensor",
    )
    parser.add_argument(
        "--memory-budget-mib",
        type=int,
        help="Staging memory budget; by default free GPU memory minus reserve",
    )
    parser.add_argument(
        "--queue-size", type=int, help="generic: 0; fst-sweep vllm profile: 2"
    )
    parser.add_argument(
        "--broadcast-run-mib",
        type=int,
        default=16,
        help="Zero disables coalesced broadcasts",
    )
    parser.add_argument(
        "--overlap-io", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--borrowed-tensors", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--use-chunk-budget-as-allocation-size",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allocate the chunk budget instead of its actual span; vllm: on, generic: off",
    )
    args = parser.parse_args()
    if (
        (args.queue_size is not None and args.queue_size < -1)
        or (args.batch_mib is not None and args.batch_mib <= 0)
        or args.broadcast_run_mib < 0
        or (args.memory_budget_mib is not None and args.memory_budget_mib <= 0)
    ):
        parser.error("Invalid queue size or memory size")
    run_benchmark(args, iter_weights, "parallel")


if __name__ == "__main__":
    main()
