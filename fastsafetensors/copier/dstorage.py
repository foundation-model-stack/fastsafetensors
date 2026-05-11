# SPDX-License-Identifier: Apache-2.0
import os
import sys
from typing import Any, Dict

from .. import cpp as fstcpp
from ..common import (
    SafeTensorsMetadata,
    init_logger,
    is_gpu_found,
    resolve_cudart_lib_name,
)
from ..frameworks import FrameworkOpBase, TensorBase
from ..st_types import Device, DeviceType, DType
from .base import CopierInterface
from .registry import CopierConstructFunc, register_copier_constructor

logger = init_logger(__name__)

_inited_ds = False


def load_dstorage_dlls() -> None:
    """Download and install DirectStorage DLLs if not already present."""
    import ctypes
    import io
    import shutil
    import zipfile
    from pathlib import Path
    from urllib.error import URLError
    from urllib.request import Request, urlopen

    cache_dir = Path.home() / ".cache" / "fastsafetensors"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dstorage_dll = cache_dir / "dstorage.dll"
    dlls = ["dstoragecore.dll", "dstorage.dll"]
    arch = "x64" if sys.maxsize > 2**32 else "x86"

    if not dstorage_dll.exists():
        logger.info("Downloading fastsafetensors DirectStorage DLL's")

        nupkg_url = os.environ.get("FASTSAFETENSORS_DSTORAGE_NUPKG_URL") or (
            "https://globalcdn.nuget.org/packages/"
            "microsoft.direct3d.directstorage.1.3.0.nupkg"
            "?packageVersion=1.3.0"
        )
        extract_dir = cache_dir / "directstorage"
        dll_src_dir = extract_dir / "native" / "bin" / arch

        try:
            req = Request(nupkg_url, headers={"User-Agent": "fastsafetensors"})
            with urlopen(req, timeout=60) as resp:
                nupkg_data = resp.read()

            with zipfile.ZipFile(io.BytesIO(nupkg_data)) as zf:
                zf.extractall(extract_dir)

            for dll_name in dlls:
                src = dll_src_dir / dll_name
                dst = cache_dir / dll_name
                if src.is_file():
                    shutil.copy2(src, dst)
                else:
                    raise FileNotFoundError(
                        f"Expected {dll_name} at {src} but not found in NuGet package"
                    )
        except (URLError, OSError, zipfile.BadZipFile, FileNotFoundError) as e:
            logger.warning(f"Failed to download/install DirectStorage DLLs: {e}")
        finally:
            if extract_dir.is_dir():
                shutil.rmtree(extract_dir, ignore_errors=True)

    for dll_name in dlls:
        dll_path = cache_dir / dll_name
        if dll_path.is_file():
            try:
                ctypes.WinDLL(str(dll_path.absolute()))
            except OSError as e:
                logger.warning(f"Failed to preload {dll_path}: {e}")


def init_dstorage(device_id: int = 0) -> None:
    global _inited_ds
    if not _inited_ds:
        from .nogds import load_library_func

        load_dstorage_dlls()
        load_library_func()
        if not is_gpu_found():
            raise RuntimeError("CUDA runtime not found")
        cudart_dll = resolve_cudart_lib_name()
        if not cudart_dll:
            raise RuntimeError("Could not find CUDA runtime DLL")
        status = fstcpp.init_dstorage(device_id, 0, cudart_dll)
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
