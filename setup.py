# Copyright 2024 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0

import os
from setuptools import setup, Extension

def MyExtension(name, sources, mod_name, *args, **kwargs):
    no_cuda = os.getenv('NOCUDA', '0')
    cuda_home = os.getenv('CUDA_HOME', '/usr/local/cuda')
    import pybind11
    pybind11_path = os.path.dirname(pybind11.__file__)

    kwargs['define_macros'] = [("__MOD_NAME__", mod_name)]
    if no_cuda == '1':
        kwargs['define_macros'].append(("NOCUDA", None))
        kwargs['libraries'] = ['stdc++', 'numa']
        kwargs['include_dirs'] = kwargs.get('include_dirs', []) + [f"{pybind11_path}/include"] # for pybind11/pybind11.h
    else:
        kwargs['library_dirs'] = kwargs.get('library_dirs', []) + [f"{cuda_home}/lib64", f"{cuda_home}/lib64/stubs"]
        libraries = kwargs.get('libraries', [])
        for lib in ['stdc++', 'cuda', 'cufile', 'numa']:
            libraries.append(lib)
        kwargs['libraries'] = libraries
        kwargs['include_dirs'] = kwargs.get('include_dirs', []) + [f"{pybind11_path}/include"] # for pybind11/pybind11.h
        kwargs['include_dirs'].append(f"{cuda_home}/include")
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
            mod_name="cpp",
        )
    ],
)
