# SPDX-License-Identifier: Apache-2.0

"""Smoke-test the compiled C++ extension without importing package __init__."""

import importlib.machinery
import importlib.metadata
import importlib.util
import sys
from pathlib import Path


def _load_cpp_extension():
    dist = importlib.metadata.distribution("fastsafetensors")
    dist_root = Path(dist.locate_file(""))

    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        candidate = dist_root / "fastsafetensors" / f"cpp{suffix}"
        if candidate.exists():
            spec = importlib.util.spec_from_file_location("cpp", candidate)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"failed to create import spec for {candidate}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module

    suffixes = ", ".join(importlib.machinery.EXTENSION_SUFFIXES)
    raise FileNotFoundError(
        f"fastsafetensors.cpp extension was not found under {dist_root}; "
        f"checked suffixes: {suffixes}"
    )


def main() -> None:
    cpp = _load_cpp_extension()
    assert cpp.get_alignment_size() == 4096
    cpp.load_library_functions("")
    assert isinstance(cpp.is_cuda_found(), bool)
    assert isinstance(cpp.is_hip_found(), bool)
    assert isinstance(cpp.is_cufile_found(), bool)


if __name__ == "__main__":
    main()
