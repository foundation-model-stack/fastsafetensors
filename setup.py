# SPDX-License-Identifier: Apache-2.0

import os
import platform

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


def MyExtension(name, sources, mod_name, *args, **kwargs):
    import pybind11

    pybind11_path = os.path.dirname(pybind11.__file__)

    kwargs["define_macros"] = [("__MOD_NAME__", mod_name)]
    kwargs["libraries"] = ["stdc++"]
    kwargs["include_dirs"] = kwargs.get("include_dirs", []) + [
        f"{pybind11_path}/include"
    ]
    kwargs["language"] = "c++"
    kwargs["extra_compile_args"] = ["-fvisibility=hidden", "-std=c++17"]

    # Windows-specific configuration for DirectStorage + D3D12/CUDA interop
    if platform.system() == "Windows":
        sources.append("fastsafetensors/cpp/dstorage_reader.cpp")
        kwargs["libraries"] = []
        # c++20 required for designated initializers at ext.hpp
        kwargs["extra_compile_args"] = ["/std:c++20"]
        # Note: dstorage.dll is loaded at runtime via LoadLibrary, not linked.
        kwargs["libraries"].extend(["ole32", "d3d12", "dxgi", "dxguid", "uuid"])

        # CUDA interop headers: if CUDA_HOME/CUDA_PATH is set, add include path
        # for cudaExternalMemory types used by the interop bridge.
        cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        if cuda_home:
            cuda_include = os.path.join(cuda_home, "include")
            if os.path.isdir(cuda_include):
                kwargs["include_dirs"].append(cuda_include)

    return Extension(name, sources, *args, **kwargs)


package_data_patterns = ["*.hpp", "*.h", "cpp.pyi"]

setup(
    packages=[
        "fastsafetensors",
        "fastsafetensors.copier",
        "fastsafetensors.cpp",
        "fastsafetensors.frameworks",
    ],
    include_package_data=True,
    package_data={"fastsafetensors.cpp": package_data_patterns},
    ext_modules=[
        MyExtension(
            name="fastsafetensors.cpp",
            sources=["fastsafetensors/cpp/ext.cpp"],
            include_dirs=["fastsafetensors/cpp"],
            mod_name="cpp",
        )
    ],
)
