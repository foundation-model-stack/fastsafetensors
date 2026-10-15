# SPDX-License-Identifier: Apache-2.0
"""Benchmark SafeTensorsFileLoader with shard groups and get_tensor delivery."""

import argparse
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks._common import add_arguments, run_benchmark, synchronize


def iter_weights(files, pg, device, args, telemetry):
    """The traditional shard-group iterator, without an inference runtime."""
    from fastsafetensors import SafeTensorsFileLoader
    from fastsafetensors.common import SingleGroup

    loader_pg = SingleGroup() if args.all_local else pg
    world = loader_pg.size() if loader_pg is not None else 1
    for start in range(0, len(files), world):
        loader = SafeTensorsFileLoader(
            loader_pg,
            str(device),
            nogds=args.copier == "nogds",
            max_threads=args.threads,
            bbuf_size_kb=args.bounce_mib << 10,
        )
        try:
            if args._fastsafetensors_tensor_filter is not None:
                loader.set_tensor_filter(args._fastsafetensors_tensor_filter)
            telemetry.append(loader.copier_class.__name__)
            loader.add_filenames(
                {
                    rank: [str(file)]
                    for rank, file in enumerate(files[start : start + world])
                }
            )
            buffers = loader.copy_files_to_device(
                borrowed_tensors=args.borrowed_tensors
            )
            try:
                for name in buffers.key_to_rank_lidx:
                    tensor = buffers.get_tensor(name)
                    yield name, tensor
                    del tensor
            finally:
                synchronize(device)
                buffers.close()
        finally:
            loader.close()


def main():
    parser = add_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument(
        "--borrowed-tensors",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use borrowed local views; synchronize consumption before advancing",
    )
    run_benchmark(parser.parse_args(), iter_weights, "loader")


if __name__ == "__main__":
    main()
