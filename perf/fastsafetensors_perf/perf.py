# Copyright 2024 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0

import json
import os
import re
import subprocess
import sys
import time
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Union

import torch
import torch.distributed as dist
import typer
from fastsafetensors import SafeTensorsFileLoader, SingleGroup
from safetensors import safe_open

app = typer.Typer()

script_path = __file__


class FilesBufferOnMmap:
    def __init__(
        self,
        device: Union[torch.device, None],
        dtype: Union[torch.dtype, None] = None,
        opt: bool = False,
        debug_log: bool = False,
    ):
        self.device = device
        self.dtype = dtype
        self.debug_log = debug_log
        self.shapes: Dict[str, torch.Size] = {}
        self.handles: Dict[str, Any] = {}
        self.opt = opt
        self.key_to_handle: Dict[str, Any] = {}

    def add_filenames(self, filenames: List[str]):
        for filename in filenames:
            f = safe_open(os.path.realpath(filename), framework="pytorch")
            for k in f.keys():
                self.handles[k] = f

    def get_keys(self) -> List[str]:
        return list(self.handles.keys())

    def enable_opt(self):
        self.opt = True

    def get_tensor(self, tensor_name: str) -> torch.Tensor:
        if tensor_name not in self.key_to_handle:
            raise ValueError(f"get_tensor: key {tensor_name} was not found in files")
        f = self.key_to_handle[tensor_name]
        t = f.get_tensor(tensor_name)  # tensor at pageable area (mmap)
        t = t.clone().detach() if self.opt else t
        return t.to(device=self.device, dtype=self.dtype)

    def get_sharded(
        self, pg: dist.ProcessGroup, tensor_name: str, dim: int
    ) -> torch.Tensor:
        if tensor_name not in self.key_to_handle:
            raise ValueError(f"get_sharded: key {tensor_name} was not found in files")
        f = self.key_to_handle[tensor_name]
        t = f.get_slice(tensor_name)  # tensor at pageable area (mmap)
        rank_slices: tuple[slice, ...] = ()
        shape = t.get_shape()
        size = shape[dim]
        block_size = (size + pg.size() - 1) // pg.size()
        for i in range(0, len(shape)):
            if i < dim:
                rank_slices += (slice(None, None, None),)
            elif i == dim:
                rank_slices += (
                    slice(pg.rank() * block_size, (pg.rank() + 1) * block_size, 1),
                )
                break
        t = t[rank_slices]
        t = t.clone().detach() if self.opt else t
        return t.to(device=self.device, dtype=self.dtype)

    def as_dict(self) -> Dict[str, torch.Tensor]:
        tensors: Dict[str, torch.Tensor] = {}
        for key, f in self.handles.items():
            t = f.get_tensor(key)  # tensor at pageable area (mmap)
            t = t.clone().detach() if self.opt else t  # opt==True: copy to pinned area?
            t = t.to(device=self.device, dtype=self.dtype)
            tensors[key] = t
        return tensors

    def as_dict_sharded(
        self, pg: dist.ProcessGroup, tensor_shard_dim: OrderedDict[str, int]
    ) -> Dict[str, torch.Tensor]:
        tensors: Dict[str, torch.Tensor] = {}
        for key, dim in sorted(tensor_shard_dim.items(), key=lambda x: x[0]):
            f = self.handles[key]
            if dim == -1:
                t = f.get_tensor(key)  # tensor at pageable area (mmap)
            else:
                t = f.get_slice(key)
                rank_slices: tuple = ()
                shape = t.get_shape()
                size = shape[dim]
                block_size = (size + pg.size() - 1) // pg.size()
                for i in range(0, len(shape)):
                    if i < dim:
                        rank_slices += (slice(None, None, None),)
                    elif i == dim:
                        rank_slices += (
                            slice(
                                pg.rank() * block_size, (pg.rank() + 1) * block_size, 1
                            ),
                        )
                        break
                t = t[rank_slices]
            t = t.clone().detach() if self.opt else t  # opt==True: copy to pinned area?
            t = t.to(device=self.device, dtype=self.dtype)
            tensors[key] = t
        return tensors


def get_sten_files(
    sten_collection_filepath: str, model_name: str, world_size: int
) -> Dict[int, List[str]]:
    world_size_str = f"world_size_{world_size}"
    with open(sten_collection_filepath, "r") as f:
        m = json.load(f)
        if model_name not in m:
            return {}
        m = m[model_name]
        if world_size_str not in m:
            world_size_str = "world_size_0"
        if world_size_str not in m:
            return {}
        m = m[world_size_str]
        ret = {}
        for rank in range(0, world_size):
            ret[rank] = m[f"rank_{rank}"]
        return ret


def get_key_pats(
    sten_collection_filepath: str, model_name: str
) -> Tuple[Dict[re.Pattern, int], str]:
    with open(sten_collection_filepath, "r") as f:
        m = json.load(f)
        if model_name not in m:
            return {}, ""
        m = m[model_name]
        if "keys" not in m:
            return {}, ""
        m = m["keys"]
        ret = {}
        for key in m["all"]:
            ret[re.compile(key)] = -1
        for key in m["dim_0"]:
            ret[re.compile(key)] = 0
        for key in m["dim_1"]:
            ret[re.compile(key)] = 1
        return ret, m["layer_prefix"]


def get_key_dim(
    keys: List[str], pats: Dict[re.Pattern, int], layer_prefix: str
) -> OrderedDict[str, int]:
    from collections import OrderedDict

    ret: OrderedDict[str, int] = OrderedDict()
    # layer_tmp = {}
    # pat2 = re.compile(f"{layer_prefix}([0-9]+)\..*")
    for key in keys:
        found = False
        for pat, dim in pats.items():
            m = pat.match(key)
            if m is not None:
                # if m[0].startswith(layer_prefix):
                #    layer_tmp[m[0]] = (int(pat2.match(m[0])[1]), dim)
                # else:
                #    ret[m[0]] = dim
                ret[m[0]] = dim
                found = True
                break
        if not found:
            ret[key] = -1
    # for key, (_, dim) in sorted(layer_tmp.items(), key=lambda x:x[1][0]):
    #    ret[key] = dim
    return ret

class MyProc:
    def __init__(self, popen: Union[subprocess.Popen, None]=None):
        self.popen: Union[subprocess.Popen, None] = popen
    def terminate(self):
        if self.popen:
            self.popen.terminate()
    def kill(self):
        if self.popen:
            self.popen.kill()
    def wait(self, timeout=Union[int, None]):
        if self.popen:
            return self.popen.wait(timeout=timeout)

mon_procs: Dict[int, Tuple[MyProc, MyProc, Any, str]] = {}


def start_sysstat(
    model_name: str,
    run_id: Union[str, None] = None,
    gpu_trace: bool = True,
    memtrace_enabled: bool = False,
) -> int:
    memtrace_file = ""
    if gpu_trace and memtrace_enabled:
        torch.cuda.memory._record_memory_history(max_entries=100000)
        if run_id is not None:
            memtrace_file = f"memtrace-{run_id}.pickle"
        else:
            memtrace_file = f"memtrace-{model_name.replace('/', '--')}.pickle"

    if run_id is not None:
        dool_file = f"dool-{run_id}.csv"
        iostat_f = open(f"iostat-{run_id}.log", "w")
    else:
        dool_file = f"dool-{model_name.replace('/', '--')}.csv"
        iostat_f = open(f"iostat-{model_name.replace('/', '--')}.log", "w")
    dool_cmd = [
        "dool",
        "-cmdnpyg",
    ]
    if gpu_trace:
        dool_cmd.append("--nvidia-gpu")
    dool_cmd += ["--output", dool_file]
    try:
        dool = MyProc(subprocess.Popen(
            dool_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        ))
    except:
        dool = MyProc()
    try:
        iostat = MyProc(subprocess.Popen(
            ["iostat", "1"], stdout=iostat_f, stderr=subprocess.DEVNULL
        ))
    except:
        iostat = MyProc()
    id = len(mon_procs)
    mon_procs[id] = (dool, iostat, iostat_f, memtrace_file)
    return id


def stop_sysstat(id: int):
    (dool, iostat, iostat_f, memtrace_file) = mon_procs[id]
    dool.terminate()
    iostat.terminate()
    try:
        dool.wait(timeout=3)
        iostat.wait(timeout=3)
    except subprocess.TimeoutExpired:
        dool.kill()
        dool.wait()
        iostat.kill()
        iostat.wait()
    iostat_f.close()
    if memtrace_file != "":
        torch.cuda.memory._dump_snapshot(memtrace_file)
        torch.cuda.memory._record_memory_history(enabled=None)


def as_safetensors_dtype(dtype_str: str) -> Union[str, None]:
    if dtype_str == "auto":
        return None
    from fastsfaetensors.common import TYPE_MAP
    if dtype_str not in TYPE_MAP:
        raise Exception(f"unsupported type: {dtype_str}. supported types: {TYPE_MAP.keys()}")
    return dtype_str


def get_size(tensor: torch.Tensor) -> int:
    c = 1
    for s in list(tensor.shape):
        c *= s
    c *= tensor.dtype.itemsize
    return c


def as_torch_device(device: str, rank: int) -> torch.device:
    if device.startswith("cuda"):
        return torch.device(f"cuda:{rank}")
    elif device == "cpu":
        return torch.device("cpu")
    return torch.device(device)


@app.command()
def get_config(
    device_index: int,
    model_name: str,
    sten_collection_json: str,
    rank: int,
    world_size: int,
) -> Tuple[int, int]:

    rank_filenames = get_sten_files(sten_collection_json, model_name, world_size)
    filenames = rank_filenames[rank]

    max_copier_threads = int(
        os.getenv("FST_THREADS", "16")
    )  # number of copy threads at host CPU
    bbuf_size_kb_total = int(
        os.getenv("FST_BBUF_SIZE_KB", "163840")
    )  # size of bounce buffer at host memory for FST_NOGDS==1
    from fastsafetensors.common import get_device_numa_node

    node = get_device_numa_node(device_index)
    total_l2_size = 0
    phys_cpus = {}
    failed = False
    import glob

    for cpudir in glob.glob(f"/sys/devices/system/node/node{node}/cpu[0-9]*"):
        try:
            with open(f"{cpudir}/cache/index2/size") as f:  # L2 cache size for a cpu
                size_str = f.read().strip()
                if size_str[-1] != "K":
                    raise Exception(f"cannot parse {cpudir}/cache/index2/size")
                total_l2_size += int(size_str[:-1])
            with open(f"{cpudir}/topology/core_id") as f:  # physical core ID
                phys_cpus[f.read().strip()] = True
        except Exception as e:
            failed = True
            print(f"Failed to auto-configure fastsafetensors. reason: {e}")
            break
    if not failed and total_l2_size > 0:
        bbuf_size_kb_total = total_l2_size
    if not failed and len(phys_cpus) > 0:
        max_copier_threads = len(phys_cpus)

    max_copy_block_size = 1
    total_size = 0
    for _, filename in enumerate(sorted(filenames, key=lambda x: os.path.basename(x))):
        s = os.stat(filename)
        total_size += s.st_size
        if max_copy_block_size < s.st_size:
            max_copy_block_size = s.st_size
    if len(filenames) < max_copier_threads:
        max_copy_block_size = total_size // world_size // max_copier_threads
        if max_copy_block_size % bbuf_size_kb_total * 1024 > 0:
            max_copy_block_size = (
                max_copy_block_size
                - max_copy_block_size % (bbuf_size_kb_total * 1024)
                + (bbuf_size_kb_total * 1024)
            )
    print(
        f"--max-threads={max_copier_threads} --max-direct-io-kb={int(bbuf_size_kb_total)} --max-block-size-mb={int(max_copy_block_size/1024/1024)}"
    )
    return (max_copier_threads, bbuf_size_kb_total)


@app.command()
def drop_cache(
    model_name: str,
    sten_collection_json: str,
    world_size: int = 1,
):
    total = 0
    with open(sten_collection_json, "r") as f:
        m = json.load(f)
    ranks = m[model_name][f"world_size_{world_size}"]
    targets = {}
    for _, filenames in ranks.items():
        for filename in filenames:
            targets[os.path.realpath(filename)] = True
    for filename in targets.keys():
        fd = os.open(filename, os.O_RDONLY)
        s = os.fstat(fd)
        os.posix_fadvise(fd, 0, s.st_size, os.POSIX_FADV_DONTNEED)
        os.close(fd)
        print(f"DROP_CACHE: {filename}, {s.st_size/1024/1024/1024} GiB")
        total += s.st_size
    fd = os.open(sten_collection_json, os.O_RDONLY)
    s = os.fstat(fd)
    os.posix_fadvise(fd, 0, s.st_size, os.POSIX_FADV_DONTNEED)
    os.close(fd)
    print(f"DROP_CACHE: {sten_collection_json}, {s.st_size/1024/1024/1024} GiB")
    total += s.st_size
    print(f"total={total/1024/1024/1024}GiB from {sten_collection_json}")


@app.command()
def run_mmap_sharded_internal(
    model_name: str,
    sten_collection_json: str,
    device: str = "cuda",
    dtype: str = "auto",
    opt: bool = False,
    debug_log: bool = False,
):
    import torch.distributed as dist

    backend = "nccl"
    if device == "cpu":
        backend = "gloo"
    dist.init_process_group(backend=backend)
    dist.barrier()  # ensure nccl is initialized
    pg = dist.group.WORLD
    if pg is None:
        return
    rank_filenames = get_sten_files(sten_collection_json, model_name, pg.size())
    filenames = []
    for _, files in sorted(rank_filenames.items(), key=lambda x: x[0]):
        for f in files:
            filenames.append(f)
    (key_pats, layer_prefix) = get_key_pats(sten_collection_json, model_name)
    torch_dtype = as_safetensors_dtype(dtype)

    t0 = time.time_ns()
    fb = FilesBufferOnMmap(
        device=as_torch_device(device, pg.rank()),
        dtype=torch_dtype,
        opt=opt,
        debug_log=debug_log,
    )
    fb.add_filenames(filenames)
    key_dim = get_key_dim(fb.get_keys(), key_pats, layer_prefix)
    t1 = time.time_ns()
    count = 0
    ts = fb.as_dict_sharded(pg, key_dim)
    for _, t in ts.items():
        count += get_size(t)
    t2 = time.time_ns()
    print(f"{t0},{t1},{t2},{count}")


@app.command()
def run_mmap(
    model_name: str,
    sten_collection_json: str,
    device: str = "cuda",
    dtype: str = "auto",
    run_id: Union[str, None] = None,
    rank: int = 0,
    world_size: int = 1,
    debug_log: bool = False,
    sysstat_enabled: bool = True,
    memtrace_enabled: bool = False,
    opt: bool = False,
    cache_drop: bool = False,
):
    if cache_drop:
        drop_cache(model_name, sten_collection_json, world_size)
    torch_dtype = as_safetensors_dtype(dtype)
    if sysstat_enabled:
        stat_id = start_sysstat(
            model_name,
            run_id,
            device.startswith("cuda"),
            memtrace_enabled and world_size == 1,
        )
    t0 = time.time_ns()
    if world_size == 1:
        filenames = get_sten_files(sten_collection_json, model_name, world_size)[rank]
        fb = FilesBufferOnMmap(
            device=as_torch_device(device, 0),
            dtype=torch_dtype,
            opt=opt,
            debug_log=debug_log,
        )
        fb.add_filenames(filenames)
        t1 = time.time_ns()
        ts = fb.as_dict()
        count = 0.0
        for _, t in ts.items():
            count += get_size(t)
        t2 = time.time_ns()
        init_sec = (t1 - t0) / 1000 / 1000 / 1000
        get_sec = (t2 - t1) / 1000 / 1000 / 1000
        elapsed_sec = (t2 - t0) / 1000 / 1000 / 1000
        count = count / 1024 / 1024 / 1024
    else:
        rank_procs = []
        for rank in range(0, world_size):
            rank_cmd = [
                "torchrun",
                "--nproc-per-node=1",
                f"--nnodes={world_size}",
                "--max-restarts=0",
                "--master_addr=0.0.0.0",
                "--master_port=1234",
                f"--node_rank={rank}",
                script_path,
                "run-mmap-sharded-internal",
                f"--dtype={dtype}",
                f"--device={device}",
                model_name,
                sten_collection_json,
            ]
            if opt:
                rank_cmd += ["--opt"]
            envs = deepcopy(os.environ)
            if world_size == 2:
                envs["CUDA_VISIBLE_DEVICES"] = "4,6"
            else:
                envs["CUDA_VISIBLE_DEVICES"] = ",".join(
                    [str((i + 4) % 8) for i in range(0, world_size)]
                )  # for vela cluster
            rank_procs.append(
                subprocess.Popen(
                    rank_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=envs
                )
            )
        outs = []
        for proc in rank_procs:
            stdout, stderr = proc.communicate()
            outs.append(stdout.decode("utf-8"))
            if len(stderr) > 0:
                print(stderr)

        count = 0.0
        time_matrix = [[-1, -1], [-1, -1], [-1, -1]]  # (min, max) for t_n
        for rank, out in enumerate(outs):
            out_split = out.split(",")
            t_ps = [int(s) for s in out_split]
            count += t_ps[-1]
            p_init_sec = (t_ps[1] - t_ps[0]) / 1000 / 1000 / 1000
            p_get_sec = (t_ps[2] - t_ps[1]) / 1000 / 1000 / 1000
            p_elapsed_sec = (t_ps[2] - t_ps[0]) / 1000 / 1000 / 1000
            print(
                f"rank{rank}: elapsed={p_elapsed_sec}, init={p_init_sec} sec, get={p_get_sec} sec, bytes={t_ps[-1]/1024/1024/1024} GiB, bw={t_ps[-1]/1024/1024/1024/p_elapsed_sec} GiB/s"
            )
            for i, t_p in enumerate(t_ps[:-1]):
                if time_matrix[i][0] == -1 or time_matrix[i][0] > t_p:
                    time_matrix[i][0] = t_p
                if time_matrix[i][1] == -1 or time_matrix[i][1] < t_p:
                    time_matrix[i][1] = t_p
        init_sec = (time_matrix[1][1] - time_matrix[0][0]) / 1000 / 1000 / 1000
        get_sec = (time_matrix[2][1] - time_matrix[1][0]) / 1000 / 1000 / 1000
        elapsed_sec = (time_matrix[2][1] - time_matrix[0][0]) / 1000 / 1000 / 1000
        count = count / 1024 / 1024 / 1024
    if sysstat_enabled:
        stop_sysstat(stat_id)

    print(
        f"elapsed: {elapsed_sec} sec, init: {init_sec} sec, get: {get_sec} sec, bytes={count}GiB, bw={count/elapsed_sec}GiB/s"
    )


@app.command()
def run_gds_sharded_internal(
    model_name: str,
    sten_collection_json: str,
    device: str = "cuda",
    dtype: str = "auto",
    max_block_size_mb: float = 10 * 1024,
    max_direct_io_kb: int = 16 * 1024,
    max_pinned_memory_in_kb: int = 64 * 1024 * 1024,
    max_threads: int = 16,
    use_buf_register: bool = True,
    debug_log: bool = False,
    nogds: bool = False,
    exclude_gds_init: bool = True,
):
    import torch.distributed as dist

    backend = "nccl"
    if device == "cpu":
        backend = "gloo"
    dist.init_process_group(backend=backend)
    dist.barrier()  # ensure nccl is initialized
    pg = dist.group.WORLD
    if pg is None:
        return
    filenames = get_sten_files(sten_collection_json, model_name, pg.size())
    (key_pats, layer_prefix) = get_key_pats(sten_collection_json, model_name)

    t0 = time.time_ns()
    if not nogds and exclude_gds_init:
        import fastsafetensors.cpp as fstcpp

        fstcpp.init_gds(16 * 1024, 80 * 1024 * 1024)
    loader = SafeTensorsFileLoader(
        pg,
        as_torch_device(device, pg.rank()),
        max_direct_io_kb,
        max_pinned_memory_in_kb,
        max_threads,
        nogds=nogds,
        debug_log=debug_log,
    )
    loader.add_filenames(filenames)
    key_dim = get_key_dim(loader.get_keys(), key_pats, layer_prefix)
    t1 = time.time_ns()
    fb = loader.copy_files_to_device(
        dtype=as_safetensors_dtype(dtype),
        use_buf_register=use_buf_register,
        max_copy_block_size=int(max_block_size_mb * 1024 * 1024),
    )
    t2 = time.time_ns()
    ts = fb.as_dict(tensor_shard_dim=key_dim)
    t3 = time.time_ns()
    count = 0
    for _, t in ts.items():
        count += get_size(t)
    t4 = time.time_ns()
    print(f"{t0},{t1},{t2},{t3},{t4},{count}")
    loader.close()
    if not nogds and exclude_gds_init:
        fstcpp.close_gds()


@app.command()
def run_gds(
    model_name: str,
    sten_collection_json: str,
    dtype: str = "auto",
    run_id: Union[str, None] = None,
    device: str = "cuda",
    max_block_size_mb: float = 10 * 1024,
    debug_log: bool = False,
    max_direct_io_kb: int = 16 * 1024,
    max_pinned_memory_in_kb: int = 64 * 1024 * 1024,
    max_threads: int = 16,
    world_size: int = 1,
    use_buf_register: bool = True,
    sysstat_enabled: bool = True,
    memtrace_enabled: bool = False,
    nogds: bool = False,
    cache_drop: bool = False,
    exclude_gds_init: bool = True,
):
    if cache_drop:
        drop_cache(model_name, sten_collection_json, world_size)
    torch_dtype = as_safetensors_dtype(dtype)
    if sysstat_enabled:
        stat_id = start_sysstat(
            model_name,
            run_id,
            device.startswith("cuda"),
            memtrace_enabled and world_size == 1,
        )
    if world_size > 1:
        rank_procs = {}
        for rank in range(0, world_size):
            rank_cmd = [
                "torchrun",
                "--nproc-per-node=1",
                f"--nnodes={world_size}",
                "--max-restarts=0",
                "--master_addr=0.0.0.0",
                "--master_port=1234",
                f"--node_rank={rank}",
                script_path,
                "run-gds-sharded-internal",
                f"--max-threads={max_threads}",
                f"--max-direct-io-kb={max_direct_io_kb}",
                f"--device={device}",
                f"--max-block-size-mb={max_block_size_mb}",
            ]
            if use_buf_register:
                rank_cmd += ["--use-buf-register"]
            if nogds:
                rank_cmd += ["--nogds"]
            if debug_log and rank == 0:
                rank_cmd += ["--debug-log"]
            if exclude_gds_init:
                rank_cmd += ["--exclude-gds-init"]
            rank_cmd += [model_name, sten_collection_json]
            envs = deepcopy(os.environ)
            if device != "cpu":
                if world_size == 2:
                    envs["CUDA_VISIBLE_DEVICES"] = "4,6"
                else:
                    envs["CUDA_VISIBLE_DEVICES"] = ",".join(
                        [str((i + 4) % 8) for i in range(0, world_size)]
                    )  # for vela cluster
            envs["NCCL_CUMEM_ENABLE"] = "0"
            if debug_log:
                print(" ".join(rank_cmd))
            rank_procs[rank] = subprocess.Popen(
                rank_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=envs
            )
        outs = []
        for rank, proc in sorted(rank_procs.items(), key=lambda x: x[0]):
            stdout, stderr = proc.communicate()
            outs.append(stdout.decode("utf-8"))
            if len(stdout) > 0:
                sys.stdout.buffer.write(bytes(f"{rank}: ", "utf-8"))
                sys.stdout.buffer.write(stdout)
            if len(stderr) > 0:
                sys.stderr.buffer.write(bytes(f"{rank}: ", "utf-8"))
                sys.stderr.buffer.write(stderr)
        pat = re.compile("^([0-9]+),([0-9]+),([0-9]+),([0-9]+),([0-9]+),([0-9]+)$")
        count = 0.0
        time_matrix = [
            [-1, -1],
            [-1, -1],
            [-1, -1],
            [-1, -1],
            [-1, -1],
        ]  # (min, max) for t_n
        for rank, out in enumerate(outs):
            for line in out.split("\n"):
                out_split = pat.match(line)
                if out_split is None:
                    continue
                t_ps = []
                for i in range(0, 6):
                    t_ps.append(int(out_split[i + 1]))
                count += t_ps[-1]
                p_init_sec = (t_ps[1] - t_ps[0]) / 1000 / 1000 / 1000
                p_io_sec = (t_ps[2] - t_ps[1]) / 1000 / 1000 / 1000
                p_shuffle_sec = (t_ps[3] - t_ps[2]) / 1000 / 1000 / 1000
                p_get_sec = (t_ps[4] - t_ps[3]) / 1000 / 1000 / 1000
                p_elapsed_sec = (t_ps[4] - t_ps[0]) / 1000 / 1000 / 1000
                print(
                    f"rank{rank}: elapsed={p_elapsed_sec}, init={p_init_sec} sec, io={p_io_sec} sec, shuffle={p_shuffle_sec} sec, get={p_get_sec} sec, bytes={t_ps[-1]/1024/1024/1024} GiB, bw={t_ps[-1]/1024/1024/1024/p_elapsed_sec} GiB/s"
                )
                for i, t_p in enumerate(t_ps[:-1]):
                    if time_matrix[i][0] == -1 or time_matrix[i][0] > t_p:
                        time_matrix[i][0] = t_p
                    if time_matrix[i][1] == -1 or time_matrix[i][1] < t_p:
                        time_matrix[i][1] = t_p
        init_sec = (time_matrix[1][1] - time_matrix[0][0]) / 1000 / 1000 / 1000
        io_sec = (time_matrix[3][1] - time_matrix[1][0]) / 1000 / 1000 / 1000
        get_sec = (time_matrix[4][1] - time_matrix[3][0]) / 1000 / 1000 / 1000
        elapsed_sec = (time_matrix[4][1] - time_matrix[0][0]) / 1000 / 1000 / 1000
        count = count / 1024 / 1024 / 1024
    else:
        filenames = get_sten_files(sten_collection_json, model_name, world_size)
        t0 = time.time_ns()
        if not nogds and exclude_gds_init:
            import fastsafetensors.cpp as fstcpp

            fstcpp.init_gds(16 * 1024, 80 * 1024 * 1024)
        loader = SafeTensorsFileLoader(
            SingleGroup(),
            as_torch_device(device, 0),
            max_direct_io_kb,
            max_pinned_memory_in_kb,
            max_threads,
            nogds=nogds,
            debug_log=debug_log,
        )
        loader.add_filenames(filenames)
        t1 = time.time_ns()
        tensors = loader.copy_files_to_device(
            dtype=torch_dtype,
            use_buf_register=use_buf_register,
            max_copy_block_size=int(max_block_size_mb * 1024 * 1024),
        )
        t2 = time.time_ns()

        count = 0.0
        ts = tensors.as_dict({key: -1 for key in loader.get_keys()})
        for key, t in ts.items():
            c = get_size(t)
            count += c
        t3 = time.time_ns()
        init_sec = (t1 - t0) / 1000 / 1000 / 1000
        io_sec = (t2 - t1) / 1000 / 1000 / 1000
        get_sec = (t3 - t2) / 1000 / 1000 / 1000
        elapsed_sec = (t3 - t0) / 1000 / 1000 / 1000
        count = count / 1024 / 1024 / 1024

    if sysstat_enabled:
        stop_sysstat(stat_id)

    print(
        f"elapsed: {elapsed_sec} sec, init: {init_sec} sec, io: {io_sec} sec, get: {get_sec} sec, bytes={count}GiB, bw={count/elapsed_sec}GiB/s"
    )

    if world_size == 1:
        loader.close()


if __name__ == "__main__":
    app()
