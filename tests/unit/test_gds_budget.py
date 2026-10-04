# SPDX-License-Identifier: Apache-2.0

import gc
import json
import os
import shutil
import struct
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import pytest

from fastsafetensors import BudgetInfeasibleError, ParallelLoader, cpp
from fastsafetensors.common import SafeTensorsMetadata
from fastsafetensors.copier import gds
from fastsafetensors.st_types import Device, DeviceType, DType


@pytest.fixture(params=["cpu", "cuda:0"])
def gds_device(request, framework):
    if framework.get_name() != "pytorch":
        pytest.skip("pytorch-only GDS integration tests")
    import torch

    target = request.param
    if target != "cpu":
        if (
            not torch.cuda.is_available()
            or not os.path.exists("/dev/nvidia-fs0")
            or not cpp.is_cufile_found()
        ):
            pytest.skip("native GDS unavailable")
        torch.cuda.set_device(target)
        gds.init_gds(framework)
    return Device.from_str(target)


def checkpoint(path, header_alignment, gap_size=8192):
    """Sparse selected runs, non-4K lengths, and an empty selected tensor."""
    import torch

    tensors = {
        "skip0": torch.zeros(4096, dtype=torch.uint8),
        "a": torch.arange(1025, dtype=torch.float32),
        "skip1": torch.zeros(gap_size, dtype=torch.uint8),
        "b": torch.arange(513, dtype=torch.float32) + 10000,
        "empty": torch.empty(0, dtype=torch.float32),
        "skip2": torch.zeros(4096, dtype=torch.uint8),
    }
    header, payload = {}, bytearray()
    for name, tensor in tensors.items():
        data = tensor.numpy().tobytes()
        start = len(payload)
        payload.extend(data)
        header[name] = dict(
            dtype="F32" if tensor.dtype == torch.float32 else "U8",
            shape=list(tensor.shape),
            data_offsets=[start, len(payload)],
        )
    encoded = json.dumps(header).encode()
    if header_alignment == 4096:
        encoded += b" " * ((-8 - len(encoded)) % 4096)
    else:
        encoded += b" " * ((-len(encoded)) % 16)  # header_length % 16 == 8
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    return tensors


class RecordingReader:
    def __init__(self, device):
        self.reader = cpp.gds_file_reader(4, device.type != DeviceType.CPU, 0)
        self.calls = []

    def submit_read(self, fh, buf, offset, length, ptr_off, file_length, *args):
        self.calls.append((offset, length, ptr_off))
        return self.reader.submit_read(
            fh, buf, offset, length, ptr_off, file_length, *args
        )

    def wait_read(self, req):
        return self.reader.wait_read(req)

    def wait_read_prefix(self, req, length):
        return self.reader.wait_read_prefix(req, length)


def assert_selected_reads(copier, reader, buf, runs, device, aligned=True):
    """Every selected byte is covered once; CUDA I/O and pointers are aligned."""
    alignment = cpp.get_alignment_size()
    previous_end = 0
    for offset, length, ptr_off in reader.calls:
        assert ptr_off == offset - copier.aligned_offset
        assert offset >= previous_end
        previous_end = offset + length
        if device.type != DeviceType.CPU and aligned:
            assert offset % alignment == length % alignment == 0
            assert (buf.get_base_address() + ptr_off) % alignment == 0
    for start, end in runs:
        covered = start
        for offset, length, _ in reader.calls:
            if offset <= covered < offset + length:
                covered = min(end, offset + length)
        assert covered == end
    requested = sum(end - start for start, end in runs)
    submitted = sum(length for _, length, _ in reader.calls)
    assert requested <= submitted <= requested + 2 * (alignment - 1) * len(runs)
    if device.type == DeviceType.CPU or not aligned:
        assert submitted == requested


@pytest.mark.parametrize("header_alignment", [8, 4096])
@pytest.mark.parametrize("allocation_size", [None, 20000])
@pytest.mark.parametrize("registered", [False, True])
@pytest.mark.parametrize("max_copy_block_size", [1024, 4096, 4097])
def test_gds_compact_sparse_chunk(
    tmp_path,
    framework,
    gds_device,
    header_alignment,
    allocation_size,
    registered,
    max_copy_block_size,
):
    import torch

    path = tmp_path / "chunk.safetensors"
    expected = checkpoint(path, header_alignment)
    meta = SafeTensorsMetadata.from_file(str(path), framework)
    names = {"a", "b", "empty"}
    runs = meta.select_byte_ranges(lambda n: n in names, merge_gap=0)
    reader = RecordingReader(gds_device)
    copier = gds.GdsFileCopier(meta, gds_device, reader, framework)
    copier.set_chunk(runs, names, allocation_size)
    readiness = copier.enable_tensor_readiness()
    buf = copier.submit_io(registered, max_copy_block_size)
    try:
        assert copier._fallback is None, "test must exercise native GDS"
        aligned = (
            gds_device.type != DeviceType.CPU
            and max_copy_block_size >= cpp.get_alignment_size()
        )
        padding = copier.chunk_device_overhead([str(path)]) if aligned else 0
        assert (
            buf.get_length() == (allocation_size or runs[-1][1] - runs[0][0]) + padding
        )
        assert_selected_reads(copier, reader, buf, runs, gds_device, aligned)
        assert all(length <= max_copy_block_size for _, length, _ in reader.calls)
        if readiness:
            got = copier.prepare_tensors(buf)
            for name in names:
                copier.wait_tensor(name)
                assert torch.equal(got[name].get_raw().cpu(), expected[name]), name
            copier.finish_io()
        else:
            got = copier.wait_io(buf)
        assert set(got) == names
        for name in names:
            assert torch.equal(got[name].get_raw().cpu(), expected[name]), name
    finally:
        copier.finish_io()
        framework.free_tensor_memory(buf, gds_device)


@pytest.mark.parametrize("chunk", [False, True])
def test_gds_filtered_ranges_and_eof(tmp_path, framework, gds_device, chunk):
    import torch

    path = tmp_path / "eof.safetensors"
    expected = checkpoint(path, 8)
    meta = SafeTensorsMetadata.from_file(str(path), framework)
    names = {"a", "skip2"}
    runs = meta.select_byte_ranges(lambda n: n in names, merge_gap=0)
    reader = RecordingReader(gds_device)
    copier = gds.GdsFileCopier(meta, gds_device, reader, framework)
    if chunk:
        copier.set_chunk(runs, names)
    else:
        copier.set_byte_ranges(runs)
    assert copier.enable_tensor_readiness()
    buf = copier.submit_io(True, 4096)
    try:
        assert copier._fallback is None
        got = copier.prepare_tensors(buf)
        for name in names:
            copier.wait_tensor(name)
            assert torch.equal(got[name].get_raw().cpu(), expected[name]), name
        assert_selected_reads(copier, reader, buf, runs, gds_device)
        assert (
            max(offset + length for offset, length, _ in reader.calls)
            >= meta.size_bytes
        )
        with pytest.raises(ValueError, match="unread bytes"):
            copier.wait_tensor("skip1")
    finally:
        copier.finish_io()
        framework.free_tensor_memory(buf, gds_device)


def test_gds_chunk_conversion_needs_no_alignment_scratch(
    tmp_path, framework, gds_device, monkeypatch
):
    import torch

    path = tmp_path / "convert.safetensors"
    expected = checkpoint(path, 8)
    meta = SafeTensorsMetadata.from_file(str(path), framework)
    names = {"a", "b"}
    runs = meta.select_byte_ranges(lambda n: n in names, merge_gap=0)
    copier = gds.GdsFileCopier(meta, gds_device, RecordingReader(gds_device), framework)
    copier.set_chunk(runs, names)
    allocations = []
    alloc = framework.alloc_tensor_memory

    def record_alloc(size, device):
        allocations.append(size)
        return alloc(size, device)

    monkeypatch.setattr(framework, "alloc_tensor_memory", record_alloc)
    buf = copier.submit_io(True, 4096)
    try:
        got = copier.wait_io(buf, dtype=DType.F16)
        assert set(got) == names
        for name in names:
            assert torch.equal(got[name].get_raw().cpu(), expected[name].half()), name
        padding = (
            copier.chunk_device_overhead([str(path)])
            if gds_device.type != DeviceType.CPU
            else 0
        )
        assert allocations == [runs[-1][1] - runs[0][0] + padding]
    finally:
        copier.finish_io()
        framework.free_tensor_memory(buf, gds_device)


def test_gds_chunk_fallback_keeps_plan(tmp_path, framework, monkeypatch):
    if framework.get_name() != "pytorch":
        pytest.skip("pytorch-only test")
    import torch

    path = tmp_path / "fallback.safetensors"
    expected = checkpoint(path, 4096)
    meta = SafeTensorsMetadata.from_file(str(path), framework)
    names = {"a", "b", "empty"}
    runs = meta.select_byte_ranges(lambda n: n in names, merge_gap=0)

    def unavailable(*args):
        raise RuntimeError("cuFileHandleRegister failed")

    monkeypatch.setattr(cpp, "gds_file_handle", unavailable)
    device = Device.from_str("cpu")
    copier = gds.GdsFileCopier(meta, device, None, framework)
    copier.set_chunk(runs, names, 20000)
    assert copier.enable_tensor_readiness()
    buf = copier.submit_io(True, 4096)
    try:
        assert copier._fallback is not None
        assert buf.get_length() == 20000
        got = copier.prepare_tensors(buf)
        assert set(got) == names
        for name in names:
            copier.wait_tensor(name)
            assert torch.equal(got[name].get_raw(), expected[name]), name
        got = copier.wait_io(buf)
        assert set(got) == names
    finally:
        copier.finish_io()
        framework.free_tensor_memory(buf, device)


@pytest.mark.parametrize("borrowed", [False, True])
@pytest.mark.parametrize(
    "overlap,fallback", [(False, False), (True, False), (False, True), (True, True)]
)
def test_gds_chunk_storage_lifetime(
    tmp_path, framework, monkeypatch, borrowed, overlap, fallback
):
    if framework.get_name() != "pytorch":
        pytest.skip("PyTorch storage lifetime check")
    import torch

    from fastsafetensors import live_allocation_count
    from fastsafetensors.allocation import SharedDeviceAllocation

    path = tmp_path / "ownership.safetensors"
    expected = checkpoint(path, 8)
    meta = SafeTensorsMetadata.from_file(str(path), framework)
    names = {"a", "b", "empty"}
    runs = meta.select_byte_ranges(lambda n: n in names, merge_gap=0)
    device = Device.from_str("cpu")
    copier = gds.GdsFileCopier(meta, device, RecordingReader(device), framework)
    copier.set_chunk(runs, names)
    if fallback:

        def unavailable(*args):
            raise RuntimeError("cuFileHandleRegister failed")

        monkeypatch.setattr(cpp, "gds_file_handle", unavailable)
    if overlap:
        assert copier.enable_tensor_readiness()
    before = live_allocation_count()
    buf = copier.submit_io(False, 4096)
    allocation = SharedDeviceAllocation(buf, framework, device)
    owner = None if borrowed else allocation
    try:
        if overlap:
            got = copier.prepare_tensors(buf, owner=owner)
            for name in names:
                copier.wait_tensor(name)
            copier.finish_io()
        else:
            got = copier.wait_io(buf, owner=owner)
        assert set(got) == names
        assert torch.equal(got["a"].get_raw(), expected["a"])
        # Keep a derived view after closing the buffer-side owner.
        view = got["a"].get_raw()[:16]
        del got
    finally:
        copier.finish_io()
        allocation.release()
    assert live_allocation_count() == before + int(not borrowed)
    if not borrowed:
        assert torch.equal(view, expected["a"][:16])
    del view
    gc.collect()
    assert live_allocation_count() == before


@pytest.mark.parametrize("fixed_allocation", [False, True])
def test_gds_parallel_budget(
    tmp_path, framework, gds_device, monkeypatch, fixed_allocation
):
    import torch

    path = tmp_path / "budget.safetensors"
    expected = checkpoint(path, 8)
    reader = RecordingReader(gds_device)
    allocations = []
    real_submit = gds.GdsFileCopier.submit_io

    def submit(copier, *args, **kwargs):
        buf = real_submit(copier, *args, **kwargs)
        assert copier._fallback is None, "test must exercise native GDS"
        allocations.append(buf.get_length())
        return buf

    def construct(meta, device, fw):
        return gds.GdsFileCopier(meta, device, reader, fw)

    construct.copier_class = gds.GdsFileCopier
    monkeypatch.setattr(
        "fastsafetensors.loader.create_copier_constructor", lambda **kwargs: construct
    )
    monkeypatch.setattr(gds.GdsFileCopier, "submit_io", submit)
    overhead = gds.GdsFileCopier.fixed_device_overhead([str(path)])
    padding = gds.GdsFileCopier.chunk_device_overhead([str(path)])
    # Filtering creates two chunks. A queue of 4 cannot fit this headroom;
    # fitting must reduce the queue and charge the fixed cuFile cache first.
    kept = 4100 + 2052
    budget = overhead + kept + 2 * (4100 + padding)
    with closing(
        ParallelLoader(
            pg=None,
            hf_weights_files=[str(path)],
            device=gds_device.as_str(),
            queue_size=4,
            device_memory_budget=budget,
            max_batch_bytes=5000,
            use_chunk_budget_as_allocation_size=fixed_allocation,
            tensor_filter=lambda n: n in {"a", "b", "empty"},
            set_numa=False,
            use_tqdm_on_load=False,
        )
    ) as loader:
        assert loader.loader.copier_class is gds.GdsFileCopier
        assert loader.queue_size < 4
        assert len(loader.weight_files_batches) == 2
        got = dict(loader.iterate_weights())
        assert set(got) == {"a", "b", "empty"}
        for name in got:
            assert torch.equal(got[name].cpu(), expected[name]), name
        extra = padding if gds_device.type != DeviceType.CPU else 0
        assert allocations == (
            [4100 + extra, 4100 + extra]
            if fixed_allocation
            else [4100 + extra, 2052 + extra]
        )

    with pytest.raises(BudgetInfeasibleError):
        ParallelLoader(
            pg=None,
            hf_weights_files=[str(path)],
            device=gds_device.as_str(),
            device_memory_budget=overhead + kept + 4100 + padding - 1,
            tensor_filter=lambda n: n in {"a", "b", "empty"},
            set_numa=False,
            use_tqdm_on_load=False,
        )


def test_gds_fixed_overhead_uses_runtime_cache(monkeypatch):
    monkeypatch.setattr(cpp, "gds_device_cache_size", lambda: 384 << 20)
    assert gds.GdsFileCopier.fixed_device_overhead(["file"]) == 384 << 20


def test_unsupported_hip_budget_falls_back_and_keeps_the_plan(
    tmp_path, framework, monkeypatch
):
    if framework.get_name() != "pytorch":
        pytest.skip("pytorch-only integration test")
    import torch

    from fastsafetensors.copier.nogds import NoGdsFileCopier

    path = tmp_path / "hip-fallback.safetensors"
    expected = checkpoint(path, 8)
    monkeypatch.setattr(gds, "load_library_func", lambda framework: None)
    monkeypatch.setattr(gds.platform, "system", lambda: "Linux")
    monkeypatch.setattr(cpp, "is_hip_found", lambda: True)
    monkeypatch.setattr(cpp, "is_hipfile_memory_budget_supported", lambda: False)

    def forbidden(*args, **kwargs):
        pytest.fail("unsupported budget must delegate before initializing GDS")

    monkeypatch.setattr(gds, "init_gds", forbidden)
    monkeypatch.setattr(gds.GdsFileCopier, "fixed_device_overhead", forbidden)
    # Give the fallback a different fixed cost to prove planning uses its
    # accounting, rather than just carrying the GDS plan across unchanged.
    overhead = 8192
    monkeypatch.setattr(
        NoGdsFileCopier,
        "fixed_device_overhead",
        classmethod(lambda cls, paths: overhead),
    )
    allocations = []
    real_submit = NoGdsFileCopier.submit_io

    def submit(self, *args, **kwargs):
        buf = real_submit(self, *args, **kwargs)
        allocations.append(buf.get_length())
        return buf

    monkeypatch.setattr(NoGdsFileCopier, "submit_io", submit)
    kept = 4100 + 2052
    budget = overhead + kept + 2 * 4100
    settings = dict(
        pg=None,
        hf_weights_files=[str(path)],
        device="cpu",  # emulate HIP policy, but perform real CPU I/O in CI
        device_memory_budget=budget,
        queue_size=4,
        max_batch_bytes=5000,
        use_chunk_budget_as_allocation_size=True,
        tensor_filter=lambda n: n in {"a", "b", "empty"},
        set_numa=False,
        use_tqdm_on_load=False,
    )
    with pytest.warns(UserWarning, match="preserving device_memory_budget"):
        loader = ParallelLoader(**settings)
    with closing(loader):
        assert loader.device_memory_budget == budget
        assert loader.loader.copier_class is NoGdsFileCopier
        assert loader.queue_size == 0
        assert len(loader.weight_files_batches) == 2
        got = dict(loader.iterate_weights())
        assert set(got) == {"a", "b", "empty"}
        for name, tensor in got.items():
            assert torch.equal(tensor, expected[name])
        assert allocations == [4100, 4100]

    allocations.clear()
    settings["device_memory_budget"] = overhead + kept + 4100 - 1
    with pytest.warns(UserWarning, match="preserving device_memory_budget"):
        with pytest.raises(BudgetInfeasibleError, match="NoGdsFileCopier"):
            ParallelLoader(**settings)
    assert not allocations  # infeasible fallback fails before any tensor I/O


@pytest.mark.parametrize("budget", [None, 1 << 20])
@pytest.mark.parametrize("supported", [False, True])
def test_hip_budget_policy_preserves_unbudgeted_and_supported_gds(
    framework, monkeypatch, budget, supported
):
    monkeypatch.setattr(gds, "load_library_func", lambda framework: None)
    monkeypatch.setattr(gds.platform, "system", lambda: "Linux")
    monkeypatch.setattr(gds.os.path, "exists", lambda path: False)
    monkeypatch.setattr(cpp, "is_hip_found", lambda: True)
    monkeypatch.setattr(cpp, "is_hipfile_memory_budget_supported", lambda: supported)
    calls = []
    monkeypatch.setattr(gds, "init_gds", lambda framework: calls.append("init"))
    monkeypatch.setattr(cpp, "gds_file_reader", lambda *args: object())
    # The first test covers budgeted unsupported versions using real nogds;
    # the other cases must keep HIP direct I/O despite lacking nvidia-fs.
    if budget is not None and not supported:
        with pytest.warns(UserWarning, match="preserving device_memory_budget"):
            construct = gds.new_gds_file_copier(
                Device.from_str("cpu"),
                framework=framework,
                device_memory_budget=budget,
            )
        assert construct.copier_class is not gds.GdsFileCopier
        assert not calls
    else:
        construct = gds.new_gds_file_copier(
            Device.from_str("cpu"),
            framework=framework,
            device_memory_budget=budget,
        )
        assert construct.copier_class is gds.GdsFileCopier
        assert calls == ["init"]


def test_unsupported_hip_budget_prefers_unified_on_shared_memory(
    framework, monkeypatch
):
    from fastsafetensors.copier import unified
    from fastsafetensors.copier.registry import copier_class_of

    monkeypatch.setattr(gds, "load_library_func", lambda framework: None)
    monkeypatch.setattr(gds, "is_gpu_found", lambda: True)
    monkeypatch.setattr(cpp, "is_hip_found", lambda: True)
    monkeypatch.setattr(cpp, "is_cufile_found", lambda: True)
    monkeypatch.setattr(cpp, "is_gds_supported", lambda device: 1)
    monkeypatch.setattr(cpp, "is_hipfile_memory_budget_supported", lambda: False)
    monkeypatch.setattr(unified, "is_unified_memory_system", lambda framework: True)

    def forbidden(*args, **kwargs):
        pytest.fail("unsupported HIP budget must not initialize GDS")

    monkeypatch.setattr(gds, "init_gds", forbidden)
    with pytest.warns(UserWarning, match="preserving device_memory_budget"):
        construct = gds.new_gds_file_copier(
            Device.from_str("cuda:0"),
            framework=framework,
            device_memory_budget=1 << 20,
        )
    assert copier_class_of(construct) is unified.UnifiedMemCopier


@pytest.mark.parametrize("chunk", [False, True])
def test_gds_padding_coalesces_shared_pages_without_exposing_skipped_tensor(
    tmp_path, framework, gds_device, chunk
):
    import torch

    path = tmp_path / "shared-page.safetensors"
    expected = checkpoint(path, 8, gap_size=4)
    meta = SafeTensorsMetadata.from_file(str(path), framework)
    names = {"a", "b"}
    runs = meta.select_byte_ranges(lambda n: n in names, merge_gap=0)
    reader = RecordingReader(gds_device)
    copier = gds.GdsFileCopier(meta, gds_device, reader, framework)
    if chunk:
        copier.set_chunk(runs, names)
    else:
        copier.set_byte_ranges(runs)
    assert copier.enable_tensor_readiness()
    buf = copier.submit_io(True, 4096)
    try:
        got = copier.prepare_tensors(buf)
        assert_selected_reads(copier, reader, buf, runs, gds_device)
        for name in names:
            copier.wait_tensor(name)
            assert torch.equal(got[name].get_raw().cpu(), expected[name]), name
        with pytest.raises(ValueError, match="unread bytes"):
            copier.wait_tensor("skip1")
    finally:
        copier.finish_io()
        framework.free_tensor_memory(buf, gds_device)


# Native runtime discovery uses isolated processes with mock shared libraries.
_RUNTIME_SOURCE = r"""
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { int err; int driver_err; } Error;
int cudaGetDeviceCount(int *count) { *count = 1; return 0; }
int hipGetDeviceCount(int *count) { *count = 1; return 0; }
#define GPU_STUBS(prefix) \
    int prefix##Memcpy(void *d, const void *s, size_t n, int k) { return 0; } \
    int prefix##DeviceSynchronize(void) { return 0; } \
    int prefix##FreeHost(void *p) { return 0; } \
    int prefix##DeviceGetPCIBusId(char *p, int n, int d) { return 0; } \
    int prefix##Malloc(void **p, size_t n) { return 0; } \
    int prefix##Free(void *p) { return 0; } \
    int prefix##DriverGetVersion(int *v) { *v = 70200000; return 0; } \
    int prefix##DeviceGetAttribute(int *v, int a, int d) { *v = 1; return 0; } \
    int prefix##SetDevice(int d) { return 0; }
GPU_STUBS(cuda)
GPU_STUBS(hip)
int cudaHostAlloc(void **p, size_t n, unsigned f) { return 0; }
int hipHostMalloc(void **p, size_t n, unsigned f) { return 0; }
int hipHostFree(void *p) { return 0; }

#define FILE_STUBS(prefix) \
    Error prefix##DriverOpen(void) { return (Error){0, 0}; } \
    Error prefix##DriverClose(void) { return (Error){0, 0}; } \
    Error prefix##DriverSetMaxDirectIOSize(size_t n) { return (Error){0, 0}; } \
    Error prefix##DriverSetMaxPinnedMemSize(size_t n) { return (Error){0, 0}; } \
    Error prefix##BufRegister(const void *p, size_t n, int f) { return (Error){0, 0}; } \
    Error prefix##BufDeregister(const void *p) { return (Error){0, 0}; } \
    Error prefix##HandleRegister(void **h, void *d) { return (Error){0, 0}; } \
    void prefix##HandleDeregister(void *h) {} \
    long prefix##Read(void *h, void *p, size_t n, long f, long b) { return n; }
FILE_STUBS(cuFile)
FILE_STUBS(hipFile)

#ifndef OMIT_VERSION
Error hipFileGetVersion(unsigned *major, unsigned *minor, unsigned *patch) {
    const char *v = getenv("MOCK_HIPFILE_VERSION");
    if (v && !strcmp(v, "error")) return (Error){5030, 0};
    if (v && !strcmp(v, "driver-error")) return (Error){0, 1};
    sscanf(v ? v : "0.4.0", "%u.%u.%u", major, minor, patch);
    return (Error){0, 0};
}
#endif
Error cuFileGetVersion(int *v) { *v = 1150; return (Error){0, 0}; }
/* Calling the AMD stub is forbidden, even if the symbol exists. */
Error hipFileDriverGetProperties(void *p) { abort(); }
#ifndef OMIT_PROPERTIES
typedef struct {
    struct {
        unsigned major, minor;
        size_t poll, io;
        unsigned status, control;
    } nvfs;
    unsigned flags, cache, per_buffer, pinned, batch, timeout;
} Props;
Error cuFileDriverGetProperties(Props *p) {
    const char *cache = getenv("MOCK_CUFILE_CACHE_KIB");
    p->cache = cache ? strtoul(cache, NULL, 10) : 131072;
    return (Error){getenv("MOCK_CUFILE_QUERY_ERROR") ? 5030 : 0, 0};
}
#endif
"""

_RUNTIME_PROBE = """
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location('cpp', sys.argv[1])
cpp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cpp)
cpp.load_library_functions(sys.argv[2])
assert cpp.is_cufile_found()
assert cpp.is_hip_found() == (sys.argv[3] == 'hip')
assert cpp.init_gds() == 0
try:
    result = {'cache_bytes': cpp.gds_device_cache_size()}
except RuntimeError as e:
    result = {'error': str(e)}
result['hip_budget_supported'] = cpp.is_hipfile_memory_budget_supported()
assert cpp.close_gds() == 0
print(json.dumps(result))
"""


@pytest.fixture(scope="module")
def runtime_libraries(tmp_path_factory):
    if sys.platform != "linux":
        pytest.skip("mock runtime DSOs require Linux")
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("C compiler unavailable")
    root = tmp_path_factory.mktemp("gds-runtime")
    source = root / "runtime.c"
    source.write_text(_RUNTIME_SOURCE)
    for name, defines in [
        ("hip", []),
        ("hip-no-version", ["-DOMIT_VERSION"]),
        ("cuda", []),
        ("cuda-no-properties", ["-DOMIT_PROPERTIES"]),
    ]:
        directory = root / name
        directory.mkdir()
        library = directory / (
            "libhipfile.so" if name.startswith("hip") else "libcufile.so.0"
        )
        subprocess.run(
            [cc, "-shared", "-fPIC", *defines, str(source), "-o", str(library)],
            check=True,
            capture_output=True,
        )
        shutil.copyfile(library, directory / f"lib{name.split('-')[0]}mock.so")
    return root


def _probe_runtime(root, name, **overrides):
    directory = root / name
    platform = name.split("-")[0]
    env = dict(os.environ)
    for key in list(env):
        if key.startswith("MOCK_"):
            del env[key]
    env.update(overrides)
    env["LD_LIBRARY_PATH"] = (
        str(directory) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    )
    env["FASTSAFETENSORS_ENABLE_INIT_LOG"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _RUNTIME_PROBE,
            str(Path(cpp.__file__).resolve()),
            str(directory / f"lib{platform}mock.so"),
            platform,
        ],
        check=True,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return json.loads(result.stdout), result.stderr


@pytest.mark.parametrize("version", ["0.2.0", "0.3.0", "0.4.0", "0.4.7"])
def test_amd_hipfile_has_no_device_cache(runtime_libraries, version):
    result, log = _probe_runtime(runtime_libraries, "hip", MOCK_HIPFILE_VERSION=version)
    assert result == {"cache_bytes": 0, "hip_budget_supported": True}
    assert f"ver: {version}" in log


@pytest.mark.parametrize(
    "version", ["0.1.0", "0.5.0", "1.0.0", "error", "driver-error"]
)
def test_amd_hipfile_rejects_unaudited_cache_policy(runtime_libraries, version):
    result, _ = _probe_runtime(runtime_libraries, "hip", MOCK_HIPFILE_VERSION=version)
    assert "requires a known AMD hipFile version" in result["error"]
    assert not result["hip_budget_supported"]


def test_amd_hipfile_missing_version_query(runtime_libraries):
    result, _ = _probe_runtime(runtime_libraries, "hip-no-version")
    assert "requires a known AMD hipFile version" in result["error"]
    assert not result["hip_budget_supported"]


@pytest.mark.parametrize("cache_kib", [0, 131072, 393216])
def test_nvidia_cache_capacity_including_zero(runtime_libraries, cache_kib):
    result, _ = _probe_runtime(
        runtime_libraries, "cuda", MOCK_CUFILE_CACHE_KIB=str(cache_kib)
    )
    assert result == {
        "cache_bytes": cache_kib * 1024,
        "hip_budget_supported": False,
    }


def test_nvidia_cache_query_failure(runtime_libraries):
    result, _ = _probe_runtime(runtime_libraries, "cuda", MOCK_CUFILE_QUERY_ERROR="1")
    assert "GDS device cache size query failed" in result["error"]


def test_nvidia_cache_query_missing(runtime_libraries):
    result, _ = _probe_runtime(runtime_libraries, "cuda-no-properties")
    assert "requires cuFileDriverGetProperties" in result["error"]
