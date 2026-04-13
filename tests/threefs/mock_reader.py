# SPDX-License-Identifier: Apache-2.0
"""Mock file reader for CI tests. No 3FS/CUDA/external dependencies required."""

import ctypes
import os


class MockFileReader:
    """Local filesystem-backed mock reader for CI tests.

    Uses os.pread for file I/O and ctypes.memmove for host memory copy.
    dev_ptr is expected to point to host memory (cpu_malloc) in CI environments.
    """

    def __init__(self, mount_point: str = "", **kwargs) -> None:
        self._fd_map: dict[str, int] = {}
        self._mount_point = mount_point

    def read_chunked(
        self, path, dev_ptr, file_offset, total_length, chunk_size=0, **kwargs
    ) -> int:
        if path not in self._fd_map:
            self._fd_map[path] = os.open(path, os.O_RDONLY)
        fd = self._fd_map[path]
        data = os.pread(fd, total_length, file_offset)
        if dev_ptr != 0:
            staging_buf = bytearray(data)
            staging_ptr = ctypes.addressof(
                (ctypes.c_char * len(staging_buf)).from_buffer(staging_buf)
            )
            ctypes.memmove(dev_ptr, staging_ptr, len(data))
        return len(data)

    def read_headers_batch(self, paths, num_threads=8):
        return {}

    def close(self) -> None:
        for fd in self._fd_map.values():
            try:
                os.close(fd)
            except OSError:
                pass
        self._fd_map.clear()


def extract_mount_point(path: str) -> str:
    """Fallback: return the directory containing the file."""
    return os.path.dirname(os.path.abspath(path))
