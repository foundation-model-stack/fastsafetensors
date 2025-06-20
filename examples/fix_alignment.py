#!/usr/bin/env python3
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json
import shutil
from copy import deepcopy

from fastsafetensors import SafeTensorsMetadata


def fix_sten_file(src_file: str, dst_file: str):
    pad_key = "p"
    pad_value = "P"
    src_fd = os.open(src_file, os.O_RDONLY, 0o644)
    if src_fd < 0:
        raise Exception(f"FAIL: open, src_file={src_file}")
    meta = SafeTensorsMetadata.from_fd(src_fd, src_file, keep_orig_dict=True)
    print(
        f"src: filename={src_file}, header_len={meta.header_length}, size={meta.size_bytes}"
    )

    min_head_pad_len = len(bytes(f',"{pad_key}":""', encoding="utf-8"))
    dst_header = {"__metadata__": deepcopy(meta.metadata)}
    dst_header.update(meta.ser)
    dst_header_str = str.encode(json.dumps(dst_header, separators=(",", ":")), "utf-8")
    dst_header_len = len(dst_header_str) + 8
    head_pad = 0
    need_copy = True
    CUDA_PTR_ALIGN = meta.framework.get_device_ptr_align()
    if dst_header_len % CUDA_PTR_ALIGN > 0:
        head_pad = CUDA_PTR_ALIGN - dst_header_len % CUDA_PTR_ALIGN
        if head_pad < min_head_pad_len:
            head_pad += CUDA_PTR_ALIGN
        dst_header["__metadata__"][pad_key] = pad_value * (head_pad - min_head_pad_len)
        dst_header_str = str.encode(
            json.dumps(dst_header, separators=(",", ":")), "utf-8"
        )
        dst_header_len = len(dst_header_str) + 8
        print(
            f"dst: filename={dst_file}, header_len={dst_header_len} (pad={head_pad}), size={dst_header_len + meta.size_bytes - meta.header_length}"
        )

        dst_fd = os.open(dst_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        if dst_fd < 0:
            raise Exception(f"FAIL: open, dst_fd={dst_fd}")
        os_write_full(
            dst_fd,
            (dst_header_len - 8).to_bytes(length=8, byteorder="little", signed=False),
        )
        os_write_full(dst_fd, dst_header_str)

        os.lseek(src_fd, meta.header_length, os.SEEK_SET)
        os.lseek(dst_fd, dst_header_len, os.SEEK_SET)
        os_sendfile_full(
            dst_fd, src_fd, dst_header_len, meta.size_bytes - meta.header_length
        )
        os.close(dst_fd)
        need_copy = False

        meta2 = SafeTensorsMetadata.from_file(dst_file)
        print(f"new metadata: {meta2.metadata}")
    else:
        print(f"no fixes are required. skip")
    os.close(src_fd)
    return need_copy


def os_write_full(fd: int, buf: bytes):
    count = 0
    while count < len(buf):
        c = os.write(fd, buf[count:])
        if c == 0:
            break
        elif c < 0:
            raise IOError(
                f"os_write_full: os.write returned error, fd={fd}, len(buf)={len(buf)}, count={count}"
            )
        count += c


def os_sendfile_full(src_fd: int, dst_fd: int, offset: int, length: int):
    count = 0
    while count < length:
        c = os.sendfile(src_fd, dst_fd, None, length - count)
        if c == 0:
            break
        elif c < 0:
            raise IOError(
                f"os_sendfile_full: os.sendfile returned error, src_fd={src_fd}, dst_fd={dst_fd}, offset={offset}, length={length}, count={count}"
            )
        count += c


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "specify a transformers_cache directory, src model name, and dst model name"
        )
        sys.exit(1)
    cache_dir = sys.argv[1]
    src_dir = os.path.join(cache_dir, "models--" + sys.argv[2].replace("/", "--"))
    dst_dir = os.path.join(cache_dir, "models--" + sys.argv[3].replace("/", "--"))
    if not os.path.exists(src_dir):
        print("src_dir must exist")
        sys.exit(1)
    if not os.path.isdir(src_dir):
        print("src_dir must be a directory")
        sys.exit(1)
    print("dir, outdir, src_dir, dir.lstrip")
    for dir, _, files in os.walk(src_dir):
        outdir = f"{dst_dir}/{dir[len(src_dir)+1:]}"
        os.makedirs(outdir, exist_ok=True)
        for filename in files:
            need_copy = True
            if filename.endswith(".safetensors"):
                need_copy = fix_sten_file(f"{dir}/{filename}", f"{outdir}/{filename}")
            if need_copy:
                print(f"copy: {dir}/{filename} --> {outdir}/{filename}")
                shutil.copyfile(f"{dir}/{filename}", f"{outdir}/{filename}")
