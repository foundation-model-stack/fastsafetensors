# SPDX-License-Identifier: Apache-2.0

"""Reject malformed checkpoint headers before creating tensor views."""

import json
import os
import struct

import pytest

from fastsafetensors import SafeTensorsMetadata, cpp


def _checkpoint(tmp_path, tensors, payload_size):
    header = json.dumps(tensors).encode()
    header += b" " * (-len(header) % 8)
    content = struct.pack("<Q", len(header)) + header + bytes(payload_size)
    path = tmp_path / "checkpoint.safetensors"
    path.write_bytes(content)
    return path, content


@pytest.mark.parametrize("source", ["file", "buffer"])
@pytest.mark.parametrize("shape", [[-2, -2], [-1, 0], [True], [1.0], "1"])
def test_invalid_shapes_are_rejected(tmp_path, framework, monkeypatch, source, shape):
    path, content = _checkpoint(
        tmp_path, {"w": {"dtype": "F32", "shape": shape, "data_offsets": [0, 16]}}, 16
    )
    with pytest.raises(ValueError, match="nonnegative integers"):
        if source == "file":
            SafeTensorsMetadata.from_file(str(path), framework)
        else:
            monkeypatch.setattr(
                cpp, "read_buffer", lambda ptr, n: content[ptr : ptr + n]
            )
            SafeTensorsMetadata.from_buffer(0, len(content), str(path), framework)


@pytest.mark.parametrize(
    "offsets,shape,payload_size,error",
    [
        ([4, 8], [1], 8, "InvalidOffset"),
        ([0, -4], [1], 4, "InvalidOffset"),
        ([0, 8], [1], 8, "TensorInvalidInfo"),
        ([0, 8], [2], 4, "MetadataIncompleteBuffer"),
    ],
)
def test_invalid_offsets_and_tensor_sizes(
    tmp_path, framework, offsets, shape, payload_size, error
):
    path, _ = _checkpoint(
        tmp_path,
        {"w": {"dtype": "F32", "shape": shape, "data_offsets": offsets}},
        payload_size,
    )
    with pytest.raises(Exception, match=error):
        SafeTensorsMetadata.from_file(str(path), framework)


def test_overlapping_tensor_offsets(tmp_path, framework):
    path, _ = _checkpoint(
        tmp_path,
        {
            name: {"dtype": "F32", "shape": [2], "data_offsets": offsets}
            for name, offsets in [("a", [0, 8]), ("b", [4, 12])]
        },
        12,
    )
    with pytest.raises(Exception, match="InvalidOffset"):
        SafeTensorsMetadata.from_file(str(path), framework)


@pytest.mark.parametrize(
    "content,error",
    [
        (b"short", "HeaderTooSmall"),
        (struct.pack("<Q", 100_000_001), "HeaderTooLarge"),
        (struct.pack("<Q", 16) + b"{}", "InvalidHeaderLength"),
    ],
)
def test_from_fd_rejects_invalid_header_lengths(tmp_path, framework, content, error):
    path = tmp_path / "bad.safetensors"
    path.write_bytes(content)
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    try:
        with pytest.raises(Exception, match=error):
            SafeTensorsMetadata.from_fd(fd, str(path), framework)
        os.fstat(fd)  # The caller retains ownership of the supplied descriptor.
    finally:
        os.close(fd)
