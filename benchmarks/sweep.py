# SPDX-License-Identifier: Apache-2.0
"""Run the standalone ParallelLoader benchmark in fresh processes per trial."""

import argparse
import csv
import json
import os
import shlex
import signal
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if not __package__:
    sys.path.insert(0, str(ROOT))

from benchmarks._common import (
    checkpoint_files,
    checkpoint_metadata,
    prepare_cache,
    resolve_input,
)
from benchmarks._vllm import num_experts_from_config, parse_expert_id


@dataclass(frozen=True)
class Model:
    label: str
    repo: str
    tp: int = 1
    ep: bool = False
    partition: str = "generic"
    dim0: bool = False


MODELS = (
    Model("granite", "ibm-granite/granite-4.2-30b"),
    Model("qwen3.8-27b", "Qwen/Qwen3.8-27B"),
    Model("llm-jp", "llm-jp/llm-jp-4-33b-thinking"),
    Model("muse-glimmer", "meta-models/Muse-Glimmer-30B", dim0=True),
    Model(
        "deepseek",
        "deepseek-ai/DeepSeek-V4-Flash-0731",
        tp=4,
        ep=True,
        partition="deepseek-v4",
    ),
    Model(
        "nemotron",
        "RedHatAI/NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8-dynamic",
        tp=8,
        ep=True,
        partition="nemotron-h",
    ),
)
LOADERS = (
    "fastsafetensors",
    "fastsafetensors-broadcast",
    "instanttensor",
    "safetensors-prefetch",
    "safetensors",
    "safetensors-prefetch-sharded",
)
# Plain mmap and sharded-prefetch are selectable separately; the latter is
# already selected by the vLLM profile when applicable.
DEFAULT_LOADERS = LOADERS[:4]


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def values(items):
    return list(
        dict.fromkeys(word for item in items for word in item.replace(",", " ").split())
    )


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        nargs="+",
        default=["all"],
        help="Aliases, cached HF IDs, or all; comma/space separated",
    )
    p.add_argument("--tp", nargs="+", help="TP sizes; default: each model's preset")
    p.add_argument(
        "--loaders", nargs="+", default=["all"], help="all or " + ", ".join(LOADERS)
    )
    p.add_argument("--repeat", type=positive, default=3)
    p.add_argument(
        "--revision", default="main", help="Cached revision; pinned before trials"
    )
    p.add_argument("--hf-home", type=Path)
    p.add_argument("--hub-cache", type=Path)
    p.add_argument("--shard-map", type=Path, help="Override partitioning for one model")
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("benchmark-results") / datetime.now().strftime("%Y%m%d-%H%M%S-%f"),
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cache", choices=["cold", "warm", "unchanged"], default="cold")
    p.add_argument("--io", choices=["direct", "buffered", "auto"], default="direct")
    p.add_argument("--threads", type=positive, default=16)
    p.add_argument("--bounce-mib", type=positive, default=16)
    p.add_argument("--memory-budget-mib", type=positive, default=7168)
    p.add_argument("--queue-size", type=int, default=2)
    p.add_argument("--verify", choices=["sample", "full", "none"], default="sample")
    p.add_argument("--timeout", type=positive, default=1800, help="Seconds per trial")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve cached models and print commands without running or writing files",
    )
    return p


def prepare(args):
    """Resolve snapshots once and prepare the existing generic TP maps."""
    by_name = {name: model for model in MODELS for name in (model.label, model.repo)}
    selected = {}
    for name in values(args.model):
        for model in (
            MODELS
            if name == "all"
            else (by_name.get(name, Model(name.replace("/", "--"), name)),)
        ):
            selected[model.repo] = model
    if not selected:
        raise ValueError("--model must select at least one model")
    if args.shard_map and len(selected) != 1:
        raise ValueError("--shard-map requires exactly one model")
    loaders = values(args.loaders)
    loaders = list(
        dict.fromkeys(
            loader
            for name in loaders
            for loader in (DEFAULT_LOADERS if name == "all" else (name,))
        )
    )
    if not loaders or set(loaders) - set(LOADERS):
        raise ValueError("--loaders must select from: " + ", ".join(LOADERS))
    tps = [positive(n) for n in values(args.tp)] if args.tp is not None else None
    if tps == []:
        raise ValueError("--tp must select at least one size")
    models, maps = [], {}
    for model in selected.values():
        snapshot = resolve_input(
            model=model.repo,
            revision=args.revision,
            hf_home=args.hf_home,
            hub_cache=args.hub_cache,
        )
        checkpoint_files(snapshot)  # Check completeness before starting any trial.
        sizes = tps if tps is not None else [model.tp]
        active_ep = model.ep and any(tp > 1 for tp in sizes)
        if active_ep and not num_experts_from_config(snapshot):
            raise ValueError(
                f"{model.repo} requires a positive expert count in config.json"
            )
        partition = "generic" if args.shard_map else model.partition
        shard_map = args.shard_map
        if shard_map and not shard_map.is_file():
            raise FileNotFoundError(shard_map)
        if shard_map is None and model.dim0 and any(tp > 1 for tp in sizes):
            files, weight_map, index_dir = checkpoint_files(snapshot)
            entries = checkpoint_metadata(files, weight_map, index_dir)
            shard_map = args.outdir / f"{model.label}-shard-map.json"
            maps[shard_map] = {
                name: 0
                for name, entry in entries.items()
                if entry["shape"] and (not active_ep or parse_expert_id(name) is None)
            }
        models.append(
            dict(
                label=model.label,
                model=model.repo,
                revision=snapshot.name,
                tps=sizes,
                ep=model.ep,
                partition=partition,
                shard_map=str(shard_map) if shard_map else None,
            )
        )
    return models, maps, loaders


def jobs_for(args, models, loaders):
    jobs, skipped = [], []
    for model in models:
        for tp in model["tps"]:
            applicable = []
            for loader in loaders:
                if loader == "fastsafetensors-broadcast" and not (
                    model["ep"] and tp > 1
                ):
                    skipped.append(
                        dict(
                            model=model["model"],
                            tp=tp,
                            loader=loader,
                            reason="The EP broadcast comparison requires active EP and TP>1",
                        )
                    )
                else:
                    applicable.append(loader)
            for repeat in range(args.repeat):
                # Rotate order so one loader does not always run first.
                start = repeat % max(1, len(applicable))
                for loader in applicable[start:] + applicable[:start]:
                    index = len(jobs) + 1
                    name = f"{index:04d}-{model['label']}-tp{tp}-{loader}-{repeat + 1}"
                    output = args.outdir / (name + ".json")
                    command = [
                        sys.executable,
                        "-m",
                        "torch.distributed.run",
                        "--standalone",
                        f"--nproc_per_node={tp}",
                        "-m",
                        "benchmarks.benchmark_parallel",
                        "--model",
                        model["model"],
                        "--revision",
                        model["revision"],
                        "--loader-profile",
                        "vllm",
                        "--partition-profile",
                        model["partition"],
                        "--readers",
                        (
                            "fastsafetensors"
                            if loader == "fastsafetensors-broadcast"
                            else loader
                        ),
                        "--fastsafetensors-ep-mode",
                        (
                            "broadcast"
                            if loader == "fastsafetensors-broadcast"
                            else "local"
                        ),
                        "--rounds",
                        "1",
                        "--warmup",
                        "0",
                        "--output",
                        str(output),
                    ]
                    for option in (
                        "device",
                        "cache",
                        "io",
                        "threads",
                        "bounce_mib",
                        "memory_budget_mib",
                        "queue_size",
                        "verify",
                        "hf_home",
                        "hub_cache",
                    ):
                        value = getattr(args, option)
                        if value is not None:
                            command += ["--" + option.replace("_", "-"), str(value)]
                    if model["ep"]:
                        command += [
                            "--enable-expert-parallel",
                            "--enable-ep-weight-filter",
                        ]
                    if model["shard_map"] and tp > 1:
                        command += ["--shard-map", model["shard_map"]]
                    jobs.append(
                        dict(
                            model=model["model"],
                            tp=tp,
                            loader=loader,
                            repeat=repeat + 1,
                            command=command,
                            report=str(output),
                            log=str(args.outdir / (name + ".log")),
                            status="pending",
                        )
                    )
    if not jobs:
        raise ValueError("No applicable model/TP/loader combinations")
    return jobs, skipped


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def write_summary(outdir, state):
    groups = {}
    for job in state["jobs"]:
        if job["status"] == "complete":
            key = (
                job["model"],
                job["tp"],
                job["loader"],
                job["effective_reader"],
                job["fallback_reason"],
            )
            groups.setdefault(key, []).append(job["seconds"])
    rows = [
        dict(
            model=model,
            tp=tp,
            loader=loader,
            effective_reader=effective,
            fallback_reason=reason,
            repeats=len(times),
            seconds=times,
            median_seconds=statistics.median(times),
            mean_seconds=statistics.mean(times),
        )
        for (model, tp, loader, effective, reason), times in groups.items()
    ]
    write_json(
        outdir / "summary.json",
        dict(complete=state["status"] == "complete", results=rows),
    )
    csv_path = outdir / "summary.csv.tmp"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "tp",
                "loader",
                "effective_reader",
                "fallback_reason",
                "repeats",
                "median_seconds",
                "mean_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(
            {key: value for key, value in row.items() if key != "seconds"}
            for row in rows
        )
    csv_path.replace(outdir / "summary.csv")


def stop(process):
    if process.poll() is None:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except ProcessLookupError:
                pass
            process.wait()


def execute(args, models, maps, jobs, skipped):
    if args.outdir.exists() and any(args.outdir.iterdir()):
        raise ValueError(f"Output directory is not empty: {args.outdir}")
    args.outdir.mkdir(parents=True, exist_ok=True)
    for path, mapping in maps.items():
        path.write_text(json.dumps(mapping) + "\n")
    state = dict(status="running", models=models, skipped=skipped, jobs=jobs)

    def save():
        write_json(args.outdir / "manifest.json", state)
        write_summary(args.outdir, state)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env.setdefault("OMP_NUM_THREADS", "1")
    save()
    cold_files = {}
    try:
        if args.cache == "cold":
            # Reclaim every selected model's cache before destination allocation:
            # other checkpoint pages also affect unified-memory prefetch budgets.
            for model in models:
                snapshot = resolve_input(
                    model=model["model"],
                    revision=model["revision"],
                    hf_home=args.hf_home,
                    hub_cache=args.hub_cache,
                )
                cold_files[model["model"]] = checkpoint_files(snapshot)[0]
                prepare_cache(cold_files[model["model"]], "cold")
        previous_model = None
        for index, job in enumerate(jobs, 1):
            job["status"] = "running"
            save()
            print(
                f"[{index}/{len(jobs)}] {job['model']} TP={job['tp']} {job['loader']} repeat={job['repeat']}",
                flush=True,
            )
            if args.cache == "cold":
                # On unified-memory GPUs, cached checkpoint pages also reduce
                # the memory available for destination allocation. Evict before
                # launching workers, which still verify coldness before timing.
                if previous_model is not None and previous_model != job["model"]:
                    prepare_cache(cold_files[previous_model], "cold")
                prepare_cache(cold_files[job["model"]], "cold")
            previous_model = job["model"]
            with Path(job["log"]).open("w") as log:
                process = subprocess.Popen(
                    job["command"],
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=os.name == "posix",
                )
                try:
                    job["pid"] = process.pid
                    save()
                    job["returncode"] = process.wait(timeout=args.timeout)
                finally:
                    stop(process)
            if job["returncode"]:
                raise RuntimeError(f"Benchmark failed; see {job['log']}")
            report = json.loads(Path(job["report"]).read_text())
            if not report.get("complete") or len(report.get("samples", [])) != 1:
                raise RuntimeError(f"Incomplete benchmark report: {job['report']}")
            sample = report["samples"][0]
            expected_reader = (
                "fastsafetensors"
                if job["loader"] == "fastsafetensors-broadcast"
                else job["loader"]
            )
            if report["world_size"] != job["tp"] or sample["reader"] != expected_reader:
                raise RuntimeError(
                    f"Unexpected benchmark configuration: {job['report']}"
                )
            job.update(
                status="complete",
                seconds=sample["max_seconds"],
                effective_reader=sample["effective_reader"],
                fallback_reason=sample["fallback_reason"],
            )
            save()
            print(
                f"  {job['seconds']:.3f} s; effective_reader={job['effective_reader']}"
                + (f" ({job['fallback_reason']})" if job["fallback_reason"] else ""),
                flush=True,
            )
        state["status"] = "complete"
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        for job in jobs:
            if job["status"] == "running":
                job.update(status="failed", error=state["error"])
        raise
    finally:
        save()


def main():
    p = parser()
    args = p.parse_args()
    if args.queue_size < -1:
        p.error("--queue-size must be -1 or greater")
    for name in ("outdir", "hf_home", "hub_cache", "shard_map"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    try:
        models, maps, loaders = prepare(args)
        jobs, skipped = jobs_for(args, models, loaders)
    except (ValueError, OSError, argparse.ArgumentTypeError) as error:
        p.error(str(error))
    for item in skipped:
        print(
            f"Skip {item['model']} TP={item['tp']} {item['loader']}: {item['reason']}"
        )
    if args.dry_run:
        for job in jobs:
            print(shlex.join(job["command"]))
        return

    def interrupted(signum, frame):
        raise KeyboardInterrupt()

    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        execute(args, models, maps, jobs, skipped)
    except KeyboardInterrupt:
        p.exit(130, "Sweep interrupted\n")
    except Exception as error:
        p.exit(1, f"{error}\n")
    finally:
        signal.signal(signal.SIGTERM, previous)
    print(f"Results: {args.outdir / 'summary.csv'}")


if __name__ == "__main__":
    main()
