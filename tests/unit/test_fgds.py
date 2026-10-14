# SPDX-License-Identifier: Apache-2.0

"""Exercise FGDS orchestration with deferred reads into real CPU allocations.

Only the libfgds boundary is mocked. File contents, metadata, tensor views and
loader ownership use the normal implementation; no FGDS driver/GPU is needed.
"""

import ctypes
import gc
import json
import os
import struct
from contextlib import closing
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from fastsafetensors import SafeTensorsFileLoader, live_allocation_count
from fastsafetensors.common import SafeTensorsMetadata
from fastsafetensors.copier import fgds
from fastsafetensors.copier.nogds import NoGdsFileCopier
from fastsafetensors.copier.registry import copier_class_of
from fastsafetensors.st_types import Device, DType


class MockFgdsReader:
    def __init__(self, runtime, max_threads, device_id):
        self.runtime = runtime
        self.max_threads = max_threads
        self.device_id = device_id
        self.pending = {}
        self.submitted = []
        self.waited = []

    def submit_read(self, handle, dst, offset, length, ptr_off):
        req_id = len(self.submitted) + 1
        if req_id == self.runtime.submit_error:
            raise RuntimeError("injected submit failure")
        request = (handle, dst, offset, length, ptr_off)
        self.pending[req_id] = request
        # Record numeric offsets, without retaining the handle after wait.
        self.submitted.append((dst.addr, offset, length, ptr_off))
        return req_id

    def wait_read(self, req_id):
        handle, dst, offset, length, ptr_off = self.pending.pop(req_id)
        self.waited.append(req_id)
        self.runtime.events.append(("wait", req_id))
        if req_id == self.runtime.wait_error:
            raise RuntimeError("injected wait failure")
        # The native reader passes dst's address and ptr_off separately to
        # fgds_read, which writes to base + ptr_off, not just dst's address.
        assert 0 <= ptr_off <= ptr_off + length <= dst.size
        handle.file.seek(offset)
        data = handle.file.read(length)
        count = self.runtime.read_results.get(req_id, len(data))
        if count > 0:
            ctypes.memmove(dst.addr + ptr_off, data, min(count, len(data)))
        return count


class MockFgdsRuntime:
    def __init__(self):
        self.events = []
        self.readers = []
        self.reg_result = 0
        self.dereg_result = 0
        self.init_result = 0
        self.handle_error = False
        self.reg_error = False
        self.submit_error = None
        self.wait_error = None
        self.read_results = {}

    def file_handle(self, path, direct, device_id):
        self.events.append(("open", path, direct, device_id))
        if self.handle_error:
            raise RuntimeError("injected handle failure")
        runtime = self

        class Handle:
            def __init__(self):
                self.file = open(path, "rb")

            def __del__(self):
                self.file.close()
                runtime.events.append(("close", path))

        return Handle()

    def file_reader(self, max_threads, device_id):
        reader = MockFgdsReader(self, max_threads, device_id)
        self.readers.append(reader)
        return reader

    def regmem(self, device_id, addr, size, target):
        self.events.append(("register", device_id, addr, size, target))
        if self.reg_error:
            raise RuntimeError("injected registration failure")
        return self.reg_result

    def deregmem(self, device_id, addr, size):
        assert all(not reader.pending for reader in self.readers)
        self.events.append(("deregister", device_id, addr, size))
        return self.dereg_result

    def init(self, device_id):
        self.events.append(("init", device_id))
        return self.init_result


class RecordingFramework:
    def __init__(self, framework, runtime):
        self.framework = framework
        self.runtime = runtime
        self.allocated = []
        self.freed = []

    def __getattr__(self, name):
        return getattr(self.framework, name)

    def alloc_tensor_memory(self, size, device):
        assert device.type == Device().type  # memmove must only see CPU pointers.
        buffer = self.framework.alloc_tensor_memory(size, device)
        self.allocated.append(buffer)
        return buffer

    def free_tensor_memory(self, buffer, device):
        assert buffer not in self.freed, "buffer freed twice"
        assert all(not reader.pending for reader in self.runtime.readers)
        self.freed.append(buffer)
        self.runtime.events.append(("free", buffer.get_base_address()))
        self.framework.free_tensor_memory(buffer, device)


@pytest.fixture
def mock_fgds(monkeypatch, framework):
    runtime = MockFgdsRuntime()
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(fgds.platform, "system", lambda: "Linux")
    monkeypatch.setattr(fgds, "_fgds_opened_devices", set())
    monkeypatch.setattr(fgds, "load_library_func", lambda fw: None)
    monkeypatch.setattr(fgds._cpp, "load_fgds_library", lambda: None)
    monkeypatch.setattr(fgds._cpp, "is_fgds_found", lambda: True)
    monkeypatch.setattr(fgds._cpp, "fgds_file_handle", runtime.file_handle)
    monkeypatch.setattr(fgds._cpp, "fgds_file_reader", runtime.file_reader)
    monkeypatch.setattr(
        fgds._cpp,
        "fgds_device_buffer",
        lambda addr, size: SimpleNamespace(addr=addr, size=size),
    )
    monkeypatch.setattr(fgds._cpp, "fgds_regmem", runtime.regmem)
    monkeypatch.setattr(fgds._cpp, "fgds_deregmem", runtime.deregmem)
    monkeypatch.setattr(fgds._cpp, "init_fgds", runtime.init)
    monkeypatch.setattr(
        fgds._cpp,
        "close_fgds",
        lambda device_id: runtime.events.append(("shutdown", device_id)),
    )
    recording = RecordingFramework(framework, runtime)
    yield runtime, recording
    # Do not let mocked device IDs reach the real process-exit handler.
    fgds._fgds_atexit_cleanup()
    for reader in runtime.readers:
        reader.pending.clear()
    for buffer in recording.allocated:
        if buffer not in recording.freed:
            recording.free_tensor_memory(buffer, Device())


@pytest.fixture
def checkpoint(tmp_path, framework):
    def create(header_length=128, payload_size=512):
        header = json.dumps(
            {
                "weight": {
                    "dtype": "F32",
                    "shape": [payload_size // 4],
                    "data_offsets": [0, payload_size],
                }
            }
        ).encode()
        assert len(header) <= header_length - 8
        expected = [float(i % 32) for i in range(payload_size // 4)]
        path = tmp_path / "model.safetensors"
        path.write_bytes(
            struct.pack("<Q", header_length - 8)
            + header.ljust(header_length - 8, b" ")
            + struct.pack(f"<{len(expected)}f", *expected)
        )
        return SafeTensorsMetadata.from_file(str(path), framework), expected

    return create


def make_copier(mock_fgds, metadata):
    runtime, framework = mock_fgds
    reader = runtime.file_reader(3, 0)
    return fgds.FgdsFileCopier(metadata, Device(), reader, framework)


@pytest.mark.parametrize(
    "visible,index,expected",
    [
        (None, 2, 2),
        ("", 0, 0),
        ("  ", 1, 1),
        ("3, 1", 0, 3),
        ("3, 1", 1, 1),
        (" , 4, , 2 ", 1, 2),
        ("GPU-uuid", 0, 0),
        ("0000:01:00.0", 0, 0),
    ],
)
def test_device_mapping(monkeypatch, visible, index, expected):
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    assert fgds.get_fgds_device_id(Device.from_str(f"cuda:{index}")) == expected


def test_device_mapping_out_of_range(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,2")
    with pytest.raises(ValueError, match="exceeds the range"):
        fgds.get_fgds_device_id(Device.from_str("cuda:2"))


@pytest.mark.parametrize(
    "header,payload,block,register",
    [
        (128, 512, 256, False),
        (fgds.ALIGN, fgds.ALIGN, fgds.ALIGN, True),
        (fgds.ALIGN + 128, 512, fgds.ALIGN, False),
        (128, fgds.ALLOC_MINIMAL + 512, fgds.ALIGN, True),
    ],
)
def test_aligned_split_reads_and_tensor_values(
    mock_fgds, checkpoint, header, payload, block, register
):
    runtime, framework = mock_fgds
    metadata, expected = checkpoint(header, payload)
    copier = make_copier(mock_fgds, metadata)
    buffer = copier.submit_io(register, block)
    start = header // fgds.ALIGN * fgds.ALIGN
    length = ((header + payload + fgds.ALIGN - 1) // fgds.ALIGN) * fgds.ALIGN - start
    assert copier.aligned_offset == start
    assert copier.aligned_length == length
    assert buffer.get_length() == max(fgds.ALLOC_MINIMAL, length)
    assert runtime.events[-1] == (
        "register",
        0,
        buffer.get_base_address(),
        buffer.get_length(),
        None,
    )
    assert copier.fgds_reader.submitted == [
        (buffer.get_base_address(), start + pos, min(block, length - pos), pos)
        for pos in range(0, length, block)
    ]
    # EOF-short reads, including requests entirely in tail padding, are valid.
    tensors = copier.wait_io(buffer)
    assert tensors["weight"].get_raw().tolist() == expected
    assert copier.copy_reqs == {}
    assert copier.registered_memory is None and copier.fgds_handle is None
    assert copier.fgds_reader.pending == {}
    assert framework.freed == []  # the caller owns the returned buffer
    del tensors
    framework.free_tensor_memory(buffer, Device())


@pytest.mark.parametrize("block", [0, -1])
def test_invalid_block_size_fails_before_acquiring_resources(
    mock_fgds, checkpoint, block
):
    runtime, framework = mock_fgds
    runtime.submit_error = 2  # bound the regression if a zero block loops again
    copier = make_copier(mock_fgds, checkpoint()[0])
    with pytest.raises(ValueError, match="must be positive"):
        copier.submit_io(False, block)
    assert runtime.events == [] and framework.allocated == []


@pytest.mark.parametrize(
    "failure", ["handle", "allocation", "register_code", "register_exception", "submit"]
)
def test_submission_failure_releases_resources(
    mock_fgds, checkpoint, monkeypatch, failure
):
    runtime, framework = mock_fgds
    copier = make_copier(mock_fgds, checkpoint()[0])
    if failure == "handle":
        runtime.handle_error = True
    elif failure == "allocation":
        monkeypatch.setattr(
            framework,
            "alloc_tensor_memory",
            Mock(side_effect=RuntimeError("injected allocation failure")),
        )
    elif failure == "register_code":
        runtime.reg_result = 7
    elif failure == "register_exception":
        runtime.reg_error = True
    else:
        runtime.submit_error = 2
        runtime.wait_error = 1  # a drain error must not hide the submit error
    if failure == "register_code":
        with pytest.warns(UserWarning, match="fgds_regmem failed"):
            with pytest.raises(RuntimeError, match="fgds_regmem failed"):
                copier.submit_io(False, 256)
    else:
        with pytest.raises(
            RuntimeError,
            match=failure if failure != "register_exception" else "registration",
        ):
            copier.submit_io(False, 256)
    assert copier.gbuf is None
    assert copier.fgds_handle is None and copier.registered_memory is None
    assert copier.copy_reqs == {} and copier.fgds_reader.pending == {}
    assert framework.freed == framework.allocated
    if failure == "submit":
        assert copier.fgds_reader.waited == [1]
        names = [event[0] for event in runtime.events]
        assert names.index("wait") < names.index("deregister") < names.index("free")


@pytest.mark.parametrize("failure", ["negative", "short", "exception"])
def test_read_failure_drains_requests_and_loader_frees_once(
    mock_fgds, checkpoint, monkeypatch, failure
):
    runtime, framework = mock_fgds
    metadata, _ = checkpoint()
    if failure == "exception":
        runtime.wait_error = 1
    else:
        runtime.read_results[1] = -1 if failure == "negative" else 255
    copier = make_copier(mock_fgds, metadata)
    monkeypatch.setattr(
        "fastsafetensors.loader.create_copier_constructor",
        lambda **kw: lambda *args: copier,
    )
    before = live_allocation_count()
    monkeypatch.setattr(
        "fastsafetensors.loader.get_framework_op", lambda name: framework
    )
    with closing(
        SafeTensorsFileLoader(None, "cpu", framework=framework.get_name())
    ) as loader:
        loader.add_filenames({0: [metadata.src]})
        with pytest.raises(
            RuntimeError,
            match=(
                "injected wait failure"
                if failure == "exception"
                else r"failed_reqs=\[1\].*reqs=\[1, 2"
            ),
        ):
            loader.copy_files_to_device(max_copy_block_size=256)
    assert copier.fgds_reader.waited == list(range(1, fgds.ALIGN // 256 + 1))
    assert copier.fgds_reader.pending == {}
    assert copier.fgds_handle is None and copier.registered_memory is None
    assert framework.freed == framework.allocated and len(framework.freed) == 1
    assert live_allocation_count() == before


def test_deregistration_warning_and_cleanup_are_idempotent(mock_fgds, checkpoint):
    runtime, framework = mock_fgds
    copier = make_copier(mock_fgds, checkpoint()[0])
    copier.submit_io(False, fgds.ALIGN)
    runtime.dereg_result = 9
    with pytest.warns(UserWarning, match="fgds_deregmem failed.*9"):
        copier.finish_io()
    copier.finish_io()
    copier._cleanup(free_gbuf=True)
    copier._cleanup(free_gbuf=True)
    assert len(framework.freed) == 1
    assert len([event for event in runtime.events if event[0] == "deregister"]) == 1


def test_wait_forwards_conversion_and_owner(mock_fgds, checkpoint, monkeypatch):
    copier = make_copier(mock_fgds, checkpoint()[0])
    buffer = copier.submit_io(False, fgds.ALIGN)
    tensors = {"converted": object()}
    get_tensors = Mock(return_value=tensors)
    monkeypatch.setattr(copier.metadata, "get_tensors", get_tensors)
    owner = object()
    assert copier.wait_io(buffer, dtype=DType.F16, noalign=True, owner=owner) is tensors
    get_tensors.assert_called_once_with(
        buffer, Device(), copier.aligned_offset, dtype=DType.F16, owner=owner
    )


def test_factory_initializes_each_physical_device_once(
    mock_fgds, checkpoint, monkeypatch
):
    runtime, framework = mock_fgds
    metadata, _ = checkpoint()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5,3")
    monkeypatch.setattr(fgds, "is_gpu_found", lambda: True)
    for index in (0, 0, 1):
        device = Device.from_str(f"cuda:{index}")
        construct = fgds.new_fgds_file_copier(device, framework, max_threads=3)
        assert copier_class_of(construct) is fgds.FgdsFileCopier
        copier = construct(metadata, device, framework)
        assert copier.device_id == (5 if index == 0 else 3)
        assert copier.fgds_reader.device_id == copier.device_id
        assert copier.fgds_reader.max_threads == copier.max_threads == 3
    assert [event for event in runtime.events if event[0] == "init"] == [
        ("init", 5),
        ("init", 3),
    ]
    fgds._fgds_atexit_cleanup()
    fgds._fgds_atexit_cleanup()
    assert sorted(event for event in runtime.events if event[0] == "shutdown") == [
        ("shutdown", 3),
        ("shutdown", 5),
    ]
    assert fgds._fgds_opened_devices == set()
    assert all(
        not os.path.exists(event[1]) for event in runtime.events if event[0] == "open"
    )


@pytest.mark.parametrize("failure", ["platform", "library", "gpu", "probe", "init"])
def test_factory_fallback_preserves_options(mock_fgds, monkeypatch, failure):
    runtime, framework = mock_fgds
    device = Device.from_str("cuda:0") if failure == "gpu" else Device()
    if failure == "platform":
        monkeypatch.setattr(fgds.platform, "system", lambda: "Darwin")
    elif failure == "library":
        monkeypatch.setattr(fgds._cpp, "is_fgds_found", lambda: False)
    elif failure == "gpu":
        monkeypatch.setattr(fgds, "is_gpu_found", lambda: False)
    elif failure == "probe":
        runtime.handle_error = True
    else:
        runtime.init_result = 11
    construct = Mock()
    construct.copier_class = NoGdsFileCopier
    fallback = Mock(return_value=construct)
    monkeypatch.setattr(fgds, "new_nogds_file_copier", fallback)
    with pytest.warns(UserWarning, match="[Ff]alling back to NoGDS"):
        assert (
            fgds.new_fgds_file_copier(
                device, framework, bbuf_size_kb=321, max_threads=7, custom=True
            )
            is construct
        )
    fallback.assert_called_once_with(device, 321, 7, framework=framework, custom=True)
    assert copier_class_of(construct) is NoGdsFileCopier
    assert fgds._fgds_opened_devices == set()
    assert runtime.readers == []
    assert all(
        not os.path.exists(event[1]) for event in runtime.events if event[0] == "open"
    )


def test_loader_tensor_survives_close_with_mock_fgds(
    mock_fgds, checkpoint, monkeypatch
):
    runtime, framework = mock_fgds
    metadata, expected = checkpoint()
    # Use the real FGDS factory and copier, substituting only CPU allocation.
    construct = fgds.new_fgds_file_copier(Device(), framework)
    monkeypatch.setattr(
        "fastsafetensors.loader.create_copier_constructor", lambda **kw: construct
    )
    before = live_allocation_count()
    monkeypatch.setattr(
        "fastsafetensors.loader.get_framework_op", lambda name: framework
    )
    with closing(
        SafeTensorsFileLoader(None, "cpu", framework=framework.get_name())
    ) as loader:
        loader.add_filenames({0: [metadata.src]})
        with closing(loader.copy_files_to_device()) as buffer:
            tensor = buffer.get_tensor("weight")
    assert tensor.tolist() == expected
    assert framework.freed == [] and live_allocation_count() == before + 1
    del tensor
    gc.collect()
    assert len(framework.freed) == 1 and live_allocation_count() == before
    assert runtime.readers[0].pending == {}
