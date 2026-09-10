# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the framework-hinted GPU runtime selection."""

import sys
import tempfile
from pathlib import Path

import pytest

from fastsafetensors import common


class _FakeFramework:
    def __init__(self, cuda_ver, runtime_lib_dirs=None):
        self._ver = cuda_ver
        self._runtime_lib_dirs = runtime_lib_dirs or []

    def get_cuda_ver(self):
        if isinstance(self._ver, Exception):
            raise self._ver
        return self._ver

    def get_runtime_lib_dirs(self):
        return self._runtime_lib_dirs


@pytest.fixture(autouse=True)
def _force_non_windows(monkeypatch):
    # The hint is intentionally a no-op on Windows (cudart resolver owns it).
    monkeypatch.setattr(sys, "platform", "linux")


def test_none_framework_uses_autodetect():
    assert common.resolve_runtime_lib_name(None) == ""


def test_hip_framework_selects_amdhip():
    assert (
        common.resolve_runtime_lib_name(_FakeFramework("hip-7.2.0")) == "libamdhip64.so"
    )


def test_cuda_framework_selects_cudart():
    assert (
        common.resolve_runtime_lib_name(_FakeFramework("cuda-12.1")) == "libcudart.so"
    )


@pytest.mark.parametrize("ver", ["", "weird", "rocm-7.0"])
def test_unknown_vendor_uses_autodetect(ver):
    assert common.resolve_runtime_lib_name(_FakeFramework(ver)) == ""


def test_get_cuda_ver_raises_uses_autodetect():
    assert common.resolve_runtime_lib_name(_FakeFramework(RuntimeError("boom"))) == ""


def test_windows_is_noop(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    assert common.resolve_runtime_lib_name(_FakeFramework("hip-7.2.0")) == ""


def test_windows_uses_framework_bundled_cudart(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("FASTSAFETENSORS_CUDART_LIB", raising=False)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    with tempfile.TemporaryDirectory() as tmp:
        runtime_dir = Path(tmp) / "torch" / "lib"
        runtime_dir.mkdir(parents=True)
        cudart = runtime_dir / "cudart64_12.dll"
        cudart.touch()

        result = common.resolve_runtime_lib_name(
            _FakeFramework("cuda-12.8", [str(runtime_dir)])
        )

        assert result == str(cudart.absolute())


def test_windows_uses_highest_bundled_cudart_major(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("FASTSAFETENSORS_CUDART_LIB", raising=False)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    with tempfile.TemporaryDirectory() as tmp:
        runtime_dir = Path(tmp) / "torch" / "lib"
        runtime_dir.mkdir(parents=True)
        (runtime_dir / "cudart64_9.dll").touch()
        cudart = runtime_dir / "cudart64_12.dll"
        cudart.touch()

        result = common.resolve_runtime_lib_name(
            _FakeFramework("cuda-12.8", [str(runtime_dir)])
        )

        assert result == str(cudart.absolute())


def test_windows_uses_highest_installed_cuda_version(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("FASTSAFETENSORS_CUDART_LIB", raising=False)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    with tempfile.TemporaryDirectory() as tmp:
        cuda_base = Path(tmp) / "NVIDIA GPU Computing Toolkit" / "CUDA"
        old_bin = cuda_base / "v9.0" / "bin"
        old_bin.mkdir(parents=True)
        (old_bin / "cudart64_9.dll").touch()
        new_bin = cuda_base / "v12.6" / "bin"
        new_bin.mkdir(parents=True)
        cudart = new_bin / "cudart64_12.dll"
        cudart.touch()
        monkeypatch.setenv("ProgramFiles", tmp)

        result = common.resolve_runtime_lib_name()

        assert result == str(cudart.absolute())


def test_windows_explicit_cudart_takes_precedence(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    with tempfile.TemporaryDirectory() as tmp:
        override = Path(tmp) / "configured" / "cudart64_12.dll"
        override.parent.mkdir()
        override.touch()
        bundled_dir = Path(tmp) / "torch" / "lib"
        bundled_dir.mkdir(parents=True)
        (bundled_dir / "cudart64_13.dll").touch()
        monkeypatch.setenv("FASTSAFETENSORS_CUDART_LIB", str(override))

        result = common.resolve_runtime_lib_name(
            _FakeFramework("cuda-12.8", [str(bundled_dir)])
        )

        assert result == str(override.absolute())


def test_dstorage_initialization_uses_framework_bundled_cudart(monkeypatch):
    import ctypes

    from fastsafetensors.copier import dstorage, nogds
    from fastsafetensors.st_types import Device

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        runtime_dir = tmp_path / "torch" / "lib"
        runtime_dir.mkdir(parents=True)
        cudart = runtime_dir / "cudart64_12.dll"
        cudart.touch()

        dstorage_dir = tmp_path / "dstorage"
        dstorage_dir.mkdir()
        for dll_name in dstorage._DSTORAGE_DLLS:
            (dstorage_dir / dll_name).touch()

        framework = _FakeFramework("cuda-12.8", [str(runtime_dir)])
        loaded_runtimes = []
        init_calls = []

        class _ReadyStreamReader:
            def is_ready(self):
                return True

        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.delenv("FASTSAFETENSORS_CUDART_LIB", raising=False)
        monkeypatch.delenv("CUDA_HOME", raising=False)
        monkeypatch.delenv("CUDA_PATH", raising=False)
        monkeypatch.setenv("ProgramFiles", str(tmp_path / "no-system-cuda"))
        monkeypatch.setenv(dstorage._DSTORAGE_DLL_DIR_ENV_VAR, str(dstorage_dir))
        monkeypatch.setattr(ctypes, "WinDLL", lambda path: None, raising=False)
        monkeypatch.setattr(
            dstorage.os, "add_dll_directory", lambda path: object(), raising=False
        )
        monkeypatch.setattr(
            nogds.fstcpp, "load_library_functions", loaded_runtimes.append
        )
        monkeypatch.setattr(nogds, "is_gpu_found", lambda: True)
        monkeypatch.setattr(nogds, "_loaded_library", False)
        monkeypatch.setattr(dstorage, "is_gpu_found", lambda: True)
        monkeypatch.setattr(dstorage, "_inited_ds", False)
        monkeypatch.setattr(dstorage, "_dstorage_dll_dir", None)
        monkeypatch.setattr(dstorage, "_dstorage_dll_dir_handle", None)
        monkeypatch.setattr(
            dstorage.fstcpp,
            "init_dstorage",
            lambda *args: init_calls.append(args) or "ok",
            raising=False,
        )
        monkeypatch.setattr(
            dstorage.fstcpp,
            "dstorage_stream_reader",
            _ReadyStreamReader,
            raising=False,
        )

        constructor = dstorage.new_dstorage_copier(
            Device.from_str("cuda:0"), framework=framework
        )

        expected_cudart = str(cudart.absolute())
        assert callable(constructor)
        assert loaded_runtimes == [expected_cudart]
        assert init_calls == [(0, 0, expected_cudart, str(dstorage_dir))]


def test_load_library_func_hint_with_no_gpu_raises(monkeypatch):
    """A hint that finds no GPU is a hard failure"""
    from fastsafetensors.copier import nogds

    calls = []

    def fake_load(lib):
        calls.append(lib)

    monkeypatch.setattr(nogds.fstcpp, "load_library_functions", fake_load)
    monkeypatch.setattr(
        nogds, "resolve_runtime_lib_name", lambda fw=None: "libamdhip64.so"
    )

    monkeypatch.setattr(nogds, "is_gpu_found", lambda: False)
    monkeypatch.setattr(nogds, "_loaded_library", False)

    with pytest.raises(Exception, match="libamdhip64.so"):
        nogds.load_library_func(_FakeFramework("hip-7.2.0"))

    assert calls == ["libamdhip64.so"]
    assert nogds._loaded_library is False


def test_load_library_func_hint_succeeds_no_fallback(monkeypatch):
    from fastsafetensors.copier import nogds

    calls = []
    monkeypatch.setattr(
        nogds.fstcpp, "load_library_functions", lambda lib: calls.append(lib)
    )
    monkeypatch.setattr(
        nogds,
        "resolve_runtime_lib_name",
        lambda fw=None: "libamdhip64.so" if fw is not None else "",
    )
    monkeypatch.setattr(nogds, "is_gpu_found", lambda: True)
    monkeypatch.setattr(nogds, "_loaded_library", False)

    nogds.load_library_func(_FakeFramework("hip-7.2.0"))

    assert calls == ["libamdhip64.so"]
