# SPDX-License-Identifier: Apache-2.0
"""Robustness tests: filesystem-aware I/O paths and graceful degradation."""

import pytest

from fastsafetensors.common import get_fs_type

# ---- get_fs_type: longest-prefix mount matching ----

MOUNTS = """\
/dev/root / ext4 rw 0 0
nas:/vol /mnt/nfs nfs4 rw 0 0
/dev/nvme0n1p2 /data ext4 rw 0 0
nas:/models /data/remote nfs rw 0 0
tmpfs /dev/shm tmpfs rw 0 0
/dev/sdb1 /mnt/with\\040space ext4 rw 0 0
"""


@pytest.fixture()
def mounts_file(tmp_path):
    p = tmp_path / "mounts"
    p.write_text(MOUNTS)
    return str(p)


def test_get_fs_type_basic(mounts_file):
    assert get_fs_type("/mnt/nfs/model.safetensors", mounts_file) == "nfs4"
    assert get_fs_type("/data/m/x.safetensors", mounts_file) == "ext4"
    assert get_fs_type("/somewhere/else", mounts_file) == "ext4"  # root fallback


def test_get_fs_type_longest_prefix_wins(mounts_file):
    # /data is ext4 but /data/remote is an NFS mount inside it
    assert get_fs_type("/data/remote/model.safetensors", mounts_file) == "nfs"


def test_get_fs_type_escaped_mountpoint(mounts_file):
    assert get_fs_type("/mnt/with space/f.safetensors", mounts_file) == "ext4"


def test_get_fs_type_unreadable_mounts(tmp_path):
    assert get_fs_type("/data/x", str(tmp_path / "nope")) == ""


# ---- DirectStorage -> nogds fallback ----


def test_missing_dstorage_dlls_fall_back_to_nogds(monkeypatch, caplog, tmp_path):
    import logging

    from fastsafetensors.copier import dstorage
    from fastsafetensors.st_types import Device

    fallback_constructor = object()
    fallback_calls = []

    def make_fallback(device, **kwargs):
        fallback_calls.append((device, kwargs))
        return fallback_constructor

    monkeypatch.setattr(dstorage.sys, "platform", "win32")
    monkeypatch.delenv(dstorage._DSTORAGE_DLL_DIR_ENV_VAR, raising=False)
    monkeypatch.delenv(dstorage._LEGACY_DSTORAGE_DOWNLOAD_ENV_VAR, raising=False)
    monkeypatch.setattr(
        dstorage, "_get_dstorage_cache_dir", lambda: tmp_path / "missing"
    )
    monkeypatch.setattr(dstorage, "_inited_ds", False)
    monkeypatch.setattr(dstorage, "new_nogds_file_copier", make_fallback)
    monkeypatch.setattr(dstorage, "_warned_dstorage_fallback", False)
    device = Device.from_str("cuda:0")

    with caplog.at_level(logging.WARNING, logger="fastsafetensors.copier.dstorage"):
        result = dstorage.new_dstorage_copier(
            device, framework="pytorch", bbuf_size_kb=4096, max_threads=4
        )
        dstorage.new_dstorage_copier(device, framework="pytorch")

    assert result is fallback_constructor
    assert fallback_calls[0] == (
        device,
        {"framework": "pytorch", "bbuf_size_kb": 4096, "max_threads": 4},
    )
    assert caplog.text.count("falling back to the nogds copier") == 1


def test_explicit_dstorage_configuration_error_does_not_fall_back(
    monkeypatch, tmp_path
):
    from fastsafetensors.copier import dstorage
    from fastsafetensors.st_types import Device

    configured_dir = tmp_path / "missing"
    monkeypatch.setattr(dstorage.sys, "platform", "win32")
    monkeypatch.setenv(dstorage._DSTORAGE_DLL_DIR_ENV_VAR, str(configured_dir))
    monkeypatch.delenv(dstorage._LEGACY_DSTORAGE_DOWNLOAD_ENV_VAR, raising=False)
    monkeypatch.setattr(dstorage, "_inited_ds", False)
    monkeypatch.setattr(
        dstorage,
        "new_nogds_file_copier",
        lambda *args, **kwargs: pytest.fail("unexpected nogds fallback"),
    )

    with pytest.raises(FileNotFoundError, match="FASTSAFETENSORS_DSTORAGE_DLL_DIR"):
        dstorage.new_dstorage_copier(Device.from_str("cuda:0"), framework="pytorch")


# ---- O_DIRECT gating on network filesystems ----


def test_odirect_gating(monkeypatch):
    from fastsafetensors.copier import unified

    monkeypatch.delenv("FASTSAFETENSORS_ODIRECT", raising=False)
    monkeypatch.setattr(unified, "get_fs_type", lambda p: "nfs4")
    assert unified._odirect_ok("/mnt/nfs/f") is False
    monkeypatch.setattr(unified, "get_fs_type", lambda p: "ext4")
    assert unified._odirect_ok("/data/f") is True
    monkeypatch.setattr(unified, "get_fs_type", lambda p: "")  # unknown: allow
    assert unified._odirect_ok("/x") is True


def test_odirect_env_override(monkeypatch):
    from fastsafetensors.copier import unified

    monkeypatch.setattr(unified, "get_fs_type", lambda p: "nfs4")
    monkeypatch.setenv("FASTSAFETENSORS_ODIRECT", "1")
    assert unified._odirect_ok("/mnt/nfs/f") is True  # forced on
    monkeypatch.setattr(unified, "get_fs_type", lambda p: "ext4")
    monkeypatch.setenv("FASTSAFETENSORS_ODIRECT", "0")
    assert unified._odirect_ok("/data/f") is False  # forced off


# ---- chunk plans must fail loudly on copiers without set_chunk ----


def test_chunk_plan_requires_set_chunk(input_files, framework):
    if framework.get_name() != "pytorch":
        pytest.skip("pytorch-only")
    from fastsafetensors import SafeTensorsFileLoader, SafeTensorsMetadata
    from fastsafetensors._planner import plan_chunks
    from fastsafetensors.copier.base import CopierInterface

    loader = SafeTensorsFileLoader(None, "cpu", nogds=True, framework="pytorch")
    loader.add_filenames({0: [input_files[0]]})
    meta = SafeTensorsMetadata.from_file(input_files[0], framework)
    (chunk,) = plan_chunks(meta, meta.size_bytes)  # single whole-span chunk
    names, ranges = chunk
    loader._set_chunk_plan({input_files[0]: (names, ranges, None)})

    class _NoChunkCopier(CopierInterface):  # e.g. gds/dstorage: no set_chunk override
        def __init__(self, metadata):
            self.metadata = metadata

        def submit_io(self, use_buf_register, max_copy_block_size):
            raise AssertionError("chunk plan must be refused before any I/O")

        def wait_io(self, gbuf, dtype=None, noalign=False):
            raise AssertionError("chunk plan must be refused before any I/O")

    loader.copier_constructor = lambda m, d, f: _NoChunkCopier(m)
    with pytest.raises(NotImplementedError, match="set_chunk"):
        loader.copy_files_to_device()


# ---- early consumer stop must not strand the producer thread ----


def test_early_close_terminates_producer(input_files, framework):
    """Stopping iteration mid-shard must unblock and end the producer thread.

    With queue_size<=0 and max_batch_bytes expanding a shard into several
    chunk-batches, a consumer that stops after the first tensor leaves the
    producer waiting on consumer_processed / a full queue; close() must wake
    it so the (non-daemon) thread exits and the process can terminate.
    """
    if framework.get_name() != "pytorch":
        pytest.skip("pytorch-only")
    import threading
    import time

    from fastsafetensors import ParallelLoader

    before = set(threading.enumerate())
    loader = ParallelLoader(
        None,
        [input_files[0]],
        device="cpu",
        nogds=True,
        framework="pytorch",
        queue_size=-1,
        max_batch_bytes=256,  # fixture's largest tensor: several chunk-batches
    )
    it = loader.iterate_weights()
    next(it)
    it.close()

    def _leftover():
        # Only non-daemon threads: the stranded producer is non-daemon (that
        # is the bug), while e.g. tqdm's TMonitor daemon singleton may first
        # spawn inside this test and legitimately outlive it.
        return [t for t in threading.enumerate() if t not in before and not t.daemon]

    deadline = time.time() + 10
    leftover = _leftover()
    while leftover and time.time() < deadline:
        time.sleep(0.05)
        leftover = _leftover()
    loader.close()  # before the assert: a failure must not leak into later tests
    assert not leftover, f"producer thread still alive: {leftover}"


def test_chunk_budget_is_used_as_allocation_size(input_files, framework):
    if framework.get_name() != "pytorch":
        pytest.skip("pytorch-only")
    from fastsafetensors.parallel_loader import PipelineParallel

    budget = 256
    pp = PipelineParallel(
        None,
        _make_loader(framework),
        [input_files[0]],
        queue_size=-1,
        use_tqdm_on_load=False,
        max_batch_bytes=budget,
        use_chunk_budget_as_allocation_size=True,
    )

    entries = [entry for spec in pp.weight_files_batches for entry in spec if entry]
    assert entries
    assert all(entry[3] == budget for entry in entries)


def test_chunk_budget_allocation_loads_all_weights(input_files, framework):
    if framework.get_name() != "pytorch":
        pytest.skip("pytorch-only")
    from fastsafetensors import ParallelLoader, SafeTensorsMetadata

    loader = ParallelLoader(
        None,
        [input_files[0]],
        device="cpu",
        nogds=True,
        queue_size=-1,
        use_tqdm_on_load=False,
        max_batch_bytes=256,
        use_chunk_budget_as_allocation_size=True,
    )
    try:
        loaded = dict(loader.iterate_weights())
    finally:
        loader.close()

    meta = SafeTensorsMetadata.from_file(input_files[0], framework)
    assert set(loaded) == set(meta.tensors)


def test_chunk_budget_allocation_requires_chunking(input_files, framework):
    if framework.get_name() != "pytorch":
        pytest.skip("pytorch-only")
    from fastsafetensors.parallel_loader import PipelineParallel

    with pytest.raises(ValueError, match="requires max_batch_bytes"):
        PipelineParallel(
            None,
            _make_loader(framework),
            [input_files[0]],
            use_chunk_budget_as_allocation_size=True,
        )


# ---- runtime GDS -> nogds fallback ----


def test_gds_copier_falls_back_to_nogds(input_files, framework, monkeypatch):
    if framework.get_name() != "pytorch":
        pytest.skip("pytorch-only")
    from test_fastsafetensors import load_safetensors_file

    from fastsafetensors import SafeTensorsMetadata
    from fastsafetensors import cpp as fstcpp
    from fastsafetensors.copier.gds import GdsFileCopier
    from fastsafetensors.st_types import Device

    def _boom(*a, **k):
        raise RuntimeError(
            "raw_gds_file_handle: cuFileHandleRegister returned an error = 5027"
        )

    monkeypatch.setattr(fstcpp, "gds_file_handle", _boom)

    device = Device.from_str("cpu")
    meta = SafeTensorsMetadata.from_file(input_files[0], framework)
    reader = fstcpp.gds_file_reader(4, False, 0)
    copier = GdsFileCopier(meta, device, reader, framework)
    gbuf = copier.submit_io(False, 10 * 1024 * 1024 * 1024)
    tensors = copier.wait_io(gbuf)
    expected = load_safetensors_file(input_files[0], device, framework)
    assert set(tensors.keys()) == set(expected.keys())
    for k, exp in expected.items():
        assert framework.is_equal(tensors[k], exp), k
    framework.free_tensor_memory(gbuf, device)
    # the fallback's bounce-buffer reader must not outlive the copy cycle
    assert fstcpp.get_cpp_metrics().bounce_buffer_bytes == 0


def test_gds_fallback_warns_once_and_shares_reader(
    input_files, framework, monkeypatch, caplog
):
    if framework.get_name() != "pytorch":
        pytest.skip("pytorch-only")
    import logging

    from fastsafetensors import SafeTensorsMetadata
    from fastsafetensors import cpp as fstcpp
    from fastsafetensors.copier import gds as gds_mod
    from fastsafetensors.st_types import Device

    monkeypatch.setattr(
        fstcpp,
        "gds_file_handle",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("error = 5027")),
    )
    monkeypatch.setattr(gds_mod, "_warned_gds_fallback", False)

    device = Device.from_str("cpu")
    meta = SafeTensorsMetadata.from_file(input_files[0], framework)
    reader = fstcpp.gds_file_reader(4, False, 0)
    cache = []
    with caplog.at_level(logging.WARNING, logger="fastsafetensors.copier.gds"):
        c1 = gds_mod.GdsFileCopier(
            meta, device, reader, framework, fallback_cache=cache
        )
        g1 = c1.submit_io(False, 1 << 30)
        c1.wait_io(g1)
        c2 = gds_mod.GdsFileCopier(
            meta, device, reader, framework, fallback_cache=cache
        )
        g2 = c2.submit_io(False, 1 << 30)
        c2.wait_io(g2)
    assert caplog.text.count("falling back to the nogds copier") == 1  # warn once
    assert len(cache) == 1  # one shared nogds constructor for the whole loader
    framework.free_tensor_memory(g1, device)
    framework.free_tensor_memory(g2, device)


# ---- device_memory_budget in broadcast mode ----


class _FakePG:
    def size(self):
        return 2

    def rank(self):
        return 0


def _make_loader(framework):
    from fastsafetensors import SafeTensorsFileLoader

    return SafeTensorsFileLoader(None, "cpu", nogds=True, framework="pytorch")


def _tight_budget(path, framework):
    """The smallest budget that still admits two copies of *path* under
    broadcast: resident for both shards plus one in-flight chunk per unit of
    pipeline depth. Anything larger stops forcing sub-file chunking."""
    from fastsafetensors import SafeTensorsMetadata
    from fastsafetensors._planner import collect_file_stats, pipeline_depth

    meta = SafeTensorsMetadata.from_file(path, framework)
    (st,) = collect_file_stats([(path, meta)])
    # broadcast adds one depth unit for the in-flight receive tensor
    return 2 * st.kept_bytes + (pipeline_depth(0) + 1) * st.largest_tensor


def test_broadcast_explicit_budget_allowed(input_files, framework, tmp_path):
    if framework.get_name() != "pytorch":
        pytest.skip("pytorch-only")
    import shutil

    from fastsafetensors.parallel_loader import PipelineParallel

    f2 = str(tmp_path / "copy.safetensors")
    shutil.copy(input_files[0], f2)
    pp = PipelineParallel(
        _FakePG(),
        _make_loader(framework),
        [input_files[0], f2],
        queue_size=0,
        use_tqdm_on_load=False,
        device_memory_budget=1 << 30,  # explicit int: deterministic across ranks
    )
    # one file per rank per group -> chunk-batch specs of width 2
    assert pp.weight_files_batches
    assert all(len(spec) == 2 for spec in pp.weight_files_batches)


def test_broadcast_budget_chunks_in_lockstep(input_files, framework, tmp_path):
    """Under broadcast, a budget tight enough to split shards must still give
    every rank the same number of chunk-batches, each full width -- a ragged
    sequence would leave some rank broadcasting while another has finished."""
    if framework.get_name() != "pytorch":
        pytest.skip("pytorch-only")
    import shutil

    from fastsafetensors.parallel_loader import PipelineParallel

    f2 = str(tmp_path / "copy2.safetensors")
    shutil.copy(input_files[0], f2)
    pp = PipelineParallel(
        _FakePG(),
        _make_loader(framework),
        [input_files[0], f2],
        queue_size=0,
        use_tqdm_on_load=False,
        device_memory_budget=_tight_budget(input_files[0], framework),
    )
    specs = pp.weight_files_batches
    # More batches than files => the budget actually forced sub-file chunking.
    assert len(specs) > 1, specs
    assert all(len(spec) == 2 for spec in specs), specs
