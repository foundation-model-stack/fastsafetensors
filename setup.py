# Copyright 2024 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0

import os
from setuptools import setup, Extension

def MyExtension(name, sources, is_test, mod_name, *args, **kwargs):
    cuda_home = os.getenv('CUDA_HOME', '/usr/local/cuda')
    torch_path = os.getenv('TORCH_PATH', '/libtorch')
    if torch_path == 'auto':
        import torch
        torch_path = os.path.dirname(torch.__file__)

    kwargs['define_macros'] = [("__MOD_NAME__", mod_name)]
    if is_test:
        kwargs['define_macros'].append(("__TEST__", None))
        kwargs['libraries'] = ['stdc++']
        kwargs['include_dirs'] = kwargs.get('include_dirs', []) + [f"{torch_path}/include"] # for pybind11/pybind11.h
    else:
        kwargs['library_dirs'] = kwargs.get('library_dirs', []) + [f"{cuda_home}/lib64", f"{cuda_home}/lib64/stubs", f"{torch_path}/lib"]
        libraries = kwargs.get('libraries', [])
        for lib in ['stdc++', 'c10', 'torch', 'cuda', 'cufile', 'c10_cuda', 'torch_cuda', 'numa']:
            libraries.append(lib)
        kwargs['libraries'] = libraries
        kwargs['include_dirs'] = kwargs.get('include_dirs', [])
        for torch_include_dir in ["include", "include/torch/csrc/api/include", "include/TH", "include/THC"]:
            kwargs['include_dirs'].append(f"{torch_path}/{torch_include_dir}")
        kwargs['include_dirs'].append(f"{cuda_home}/include")
        kwargs['define_macros'].append(("TORCH_API_INCLUDE_EXTENSION_H", None))
        kwargs['define_macros'].append(("PYBIND11_COMPILER_TYPE", "\"_gcc\""))
        kwargs['define_macros'].append(("PYBIND11_STDLIB", "\"_libstdcpp\""))
        kwargs['define_macros'].append(("PYBIND11_BUILD_ABI", "\"_cxxabi1011\""))
        kwargs['define_macros'].append(("TORCH_EXTENSION_NAME", mod_name))
        kwargs['define_macros'].append(("_GLIBCXX_USE_CXX11_ABI", "0"))
    kwargs['language'] = 'c++'

    # https://pybind11.readthedocs.io/en/stable/faq.html#someclass-declared-with-greater-visibility-than-the-type-of-its-field-someclass-member-wattributes
    kwargs['extra_compile_args'] = ['-fvisibility=hidden', '-std=c++17']

    return Extension(name, sources, *args, **kwargs)

setup(
    packages=["fastsafetensors", "fastsafetensors.connectors", "fastsafetensors.copier", "fastsafetensors.cpp"],
    include_package_data=True,
    package_data={"fastsafetensors.cpp": ["*.hpp"]},
    ext_modules=[
        MyExtension(
            name=f"fastsafetensors.cpp",
            sources=["fastsafetensors/cpp/ext.cpp"],
            include_dirs=["fastsafetensors/cpp"],
            is_test='FST_TEST' in os.environ,
            mod_name="cpp",
        )
    ],
)
