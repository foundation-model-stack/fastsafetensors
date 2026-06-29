# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict

from .. import cpp as fstcpp
from ..common import (
    SafeTensorsMetadata,
    init_logger,
    is_gpu_found,
    resolve_runtime_lib_name,
)
from ..frameworks import FrameworkOpBase, TensorBase
from ..st_types import Device, DeviceType, DType
from .base import CopierInterface
from .registry import CopierConstructFunc, register_copier_constructor

logger = init_logger(__name__)

_inited_ds = False
_dstorage_dll_dir_handle = None
_dstorage_dll_dir: Path | None = None
_DSTORAGE_DLLS = ("dstoragecore.dll", "dstorage.dll")
_DSTORAGE_DLL_DIR_ENV_VAR = "FASTSAFETENSORS_DSTORAGE_DLL_DIR"
_LEGACY_DSTORAGE_DOWNLOAD_ENV_VAR = "FASTSAFETENSORS_DSTORAGE_NUPKG_URL"


def _get_dstorage_cache_dir() -> Path:
    return Path.home() / ".cache" / "fastsafetensors"


def _validate_dstorage_dll_dir(path: Path, source: str) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(path))))
    if not path.is_absolute():
        raise ValueError(f"{source} must be an absolute directory path: {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"{source} does not exist: {path}")

    missing = [name for name in _DSTORAGE_DLLS if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{source} is missing required DirectStorage DLLs: {missing} in {path}"
        )
    return path


def resolve_dstorage_dll_dir() -> Path:
    """Resolve the directory that contains pre-installed DirectStorage DLLs."""
    if sys.platform != "win32":
        raise RuntimeError("DirectStorage is only supported on Windows")

    legacy = os.environ.get(_LEGACY_DSTORAGE_DOWNLOAD_ENV_VAR, "").strip()
    if legacy:
        raise RuntimeError(
            f"{_LEGACY_DSTORAGE_DOWNLOAD_ENV_VAR} is no longer supported for "
            "security reasons. Install DirectStorage DLLs manually and point "
            f"{_DSTORAGE_DLL_DIR_ENV_VAR} to an absolute directory instead."
        )

    override = os.environ.get(_DSTORAGE_DLL_DIR_ENV_VAR, "").strip()
    if override:
        return _validate_dstorage_dll_dir(Path(override), _DSTORAGE_DLL_DIR_ENV_VAR)

    cache_dir = _get_dstorage_cache_dir()
    if cache_dir.is_dir():
        try:
            return _validate_dstorage_dll_dir(cache_dir, str(cache_dir))
        except FileNotFoundError:
            pass

    raise RuntimeError(
        "DirectStorage DLLs were not found. Install dstoragecore.dll and "
        "dstorage.dll into an absolute directory and set "
        f"{_DSTORAGE_DLL_DIR_ENV_VAR}, or place them in {cache_dir}."
    )


def load_dstorage_dlls() -> None:
    """Load DirectStorage DLLs from a pre-installed absolute directory."""
    if sys.platform != "win32":
        return  # DirectStorage is Windows-only

    import ctypes

    global _dstorage_dll_dir_handle
    global _dstorage_dll_dir

    dll_dir = resolve_dstorage_dll_dir()
    if _dstorage_dll_dir == dll_dir:
        return

    if hasattr(os, "add_dll_directory"):
        _dstorage_dll_dir_handle = os.add_dll_directory(str(dll_dir))

    for dll_name in _DSTORAGE_DLLS:
        dll_path = dll_dir / dll_name
        try:
            ctypes.WinDLL(str(dll_path))
        except OSError as e:
            raise RuntimeError(f"Failed to preload DirectStorage DLL {dll_path}: {e}")

    _dstorage_dll_dir = dll_dir


def init_dstorage(device_id: int = 0) -> None:
    global _inited_ds
    if not _inited_ds:
        from .nogds import load_library_func

        load_dstorage_dlls()
        load_library_func()
        if not is_gpu_found():
            raise RuntimeError("CUDA runtime not found")
        # Windows-only; resolve_runtime_lib_name() returns the cudart DLL path here
        cudart_dll = resolve_runtime_lib_name()
        if not cudart_dll:
            raise RuntimeError("Could not find CUDA runtime DLL")
        if _dstorage_dll_dir is None:
            raise RuntimeError("DirectStorage DLL directory was not initialized")
        status = fstcpp.init_dstorage(device_id, 0, cudart_dll, str(_dstorage_dll_dir))
        if status != "ok":
            raise RuntimeError(f"init_dstorage failed: {status}")
        _inited_ds = True


class DStorageFileCopier(CopierInterface):
    """Copier that reads files via DirectStorage with double-buffered staging
    into a standard CUDA (gds_device_buffer) destination."""

    def __init__(
        self,
        metadata: SafeTensorsMetadata,
        device: Device,
        stream_reader: fstcpp.dstorage_stream_reader,
        framework: FrameworkOpBase,
    ):
        self.framework = framework
        self.metadata = metadata
        self.device = device
        self.stream_reader = stream_reader
        self.fh: Any = None  # fstcpp.dstorage_file_handle

    def submit_io(self, use_buf_register: bool, max_copy_block_size: int):
        total_bytes = self.metadata.size_bytes - self.metadata.header_length

        gbuf = self.framework.alloc_tensor_memory(total_bytes, self.device)

        # Open the file via DirectStorage
        self.fh = fstcpp.dstorage_file_handle()
        if not self.fh.open(self.metadata.src):
            raise IOError(f"Failed to open {self.metadata.src} via DirectStorage")

        # DS reads NVMe to staging buf, then cudaMemcpy staging to final CUDA buffer.
        result = self.stream_reader.read_to_cuda(
            self.fh,
            gbuf.get_base_address(),
            self.metadata.header_length,
            total_bytes,
        )
        if result < 0:
            hr = self.stream_reader.last_hresult()
            raise RuntimeError(
                f"dstorage_stream_reader.read_to_cuda failed: result={result}, "
                f"HRESULT=0x{hr & 0xFFFFFFFF:08X}"
            )

        return gbuf

    def wait_io(self, gbuf, dtype=DType.AUTO, noalign=False):
        if self.fh:
            self.fh.close()
        return self.metadata.get_tensors(
            gbuf, self.device, self.metadata.header_length, dtype=dtype
        )


@register_copier_constructor("dstorage")
def new_dstorage_copier(device: Device, **kwargs) -> CopierConstructFunc:
    """Factory for DirectStorage file copier."""
    init_dstorage(device.index if device.index is not None else 0)
    stream_reader = fstcpp.dstorage_stream_reader()
    if not stream_reader.is_ready():
        raise RuntimeError("dstorage_stream_reader failed to initialize")

    def construct(
        metadata: SafeTensorsMetadata, device: Device, framework: FrameworkOpBase
    ) -> CopierInterface:
        return DStorageFileCopier(metadata, device, stream_reader, framework)

    return construct
