# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the framework-hinted GPU runtime selection."""

import sys

import pytest

from fastsafetensors import common


class _FakeFramework:
    def __init__(self, cuda_ver):
        self._ver = cuda_ver

    def get_cuda_ver(self):
        if isinstance(self._ver, Exception):
            raise self._ver
        return self._ver


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
