# Copyright 2024 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0

import os
import re
import shutil
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


def detect_platform():
    """
    Detect if we're on NVIDIA CUDA or AMD ROCm platform.

    Returns:
        tuple: (platform_type, rocm_version, rocm_path)
            platform_type: 'cuda' or 'rocm'
            rocm_version: ROCm version string (e.g., '7.0.1') or None
            rocm_path: Path to ROCm installation or None
    """
    # Check for ROCm installation
    rocm_path = os.environ.get("ROCM_PATH")
    if not rocm_path:
        # Try common ROCm installation paths
        for path in ["/opt/rocm", "/opt/rocm-*"]:
            if "*" in path:
                import glob
                matches = sorted(glob.glob(path), reverse=True)
                if matches:
                    rocm_path = matches[0]
                    break
            elif os.path.exists(path):
                rocm_path = path
                break

    # Check if ROCm is available
    if rocm_path and os.path.exists(rocm_path):
        # Detect ROCm version
        rocm_version = None
        version_file = os.path.join(rocm_path, ".info", "version")
        if os.path.exists(version_file):
            with open(version_file, "r") as f:
                rocm_version = f.read().strip()
        else:
            # Try to extract version from path
            match = re.search(r'rocm[-/](\d+\.\d+(?:\.\d+)?)', rocm_path)
            if match:
                rocm_version = match.group(1)

        print(f"Detected ROCm platform at {rocm_path}")
        if rocm_version:
            print(f"ROCm version: {rocm_version}")
        return ('rocm', rocm_version, rocm_path)

    # Check for CUDA
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if not cuda_home:
        # Try to find nvcc
        nvcc_path = shutil.which("nvcc")
        if nvcc_path:
            cuda_home = os.path.dirname(os.path.dirname(nvcc_path))

    if cuda_home and os.path.exists(cuda_home):
        print(f"Detected CUDA platform at {cuda_home}")
        return ('cuda', None, None)

    # Default to CUDA if nothing detected
    print("No GPU platform detected, defaulting to CUDA")
    return ('cuda', None, None)


def hipify_source_files(rocm_path):
    """
    Automatically hipify CUDA source files to HIP using torch.utils.hipify.
    The cuda_compat.h header handles what hipify doesn't convert.

    Args:
        rocm_path: Path to ROCm installation

    Returns:
        list: Paths to hipified source files
    """
    from torch.utils.hipify.hipify_python import hipify

    cpp_dir = Path("fastsafetensors/cpp").resolve()

    # Prepare source files for hipification
    extra_files = [
        str(cpp_dir / "ext.cpp"),
        str(cpp_dir / "ext.hpp"),
    ]

    print(f"Hipifying files using torch.utils.hipify:")
    for f in extra_files:
        print(f"  - {f}")

    # Use torch's hipify - similar to vLLM's approach
    hipify_result = hipify(
        project_directory=str(cpp_dir.parent),
        output_directory=str(cpp_dir),
        header_include_dirs=[],
        includes=[f"{cpp_dir}/*"],
        extra_files=extra_files,
        show_detailed=False,
        is_pytorch_extension=False,
        hipify_extra_files_only=True,
    )

    hipified_files = []
    for source_path, result in hipify_result.items():
        if hasattr(result, 'hipified_path') and result.hipified_path:
            print(f"Successfully hipified: {source_path} -> {result.hipified_path}")
            hipified_files.append(result.hipified_path)

    # Copy cuda_compat.h to hip directory as hip_compat.h
    # (hipify converts the include statement from cuda_compat.h to hip_compat.h)
    hip_dir = cpp_dir / "hip"
    if hip_dir.exists():
        cuda_compat = cpp_dir / "cuda_compat.h"
        hip_compat = hip_dir / "hip_compat.h"
        shutil.copy2(cuda_compat, hip_compat)
        print(f"Copied {cuda_compat} -> {hip_compat}")

    return hipified_files



def MyExtension(name, sources, mod_name, platform_type, rocm_path=None, *args, **kwargs):
    import pybind11

    pybind11_path = os.path.dirname(pybind11.__file__)

    kwargs["define_macros"] = [("__MOD_NAME__", mod_name)]
    kwargs["libraries"] = ["stdc++"]
    kwargs["include_dirs"] = kwargs.get("include_dirs", []) + [
        f"{pybind11_path}/include"
    ]  # for pybind11/pybind11.h
    kwargs["language"] = "c++"

    # https://pybind11.readthedocs.io/en/stable/faq.html#someclass-declared-with-greater-visibility-than-the-type-of-its-field-someclass-member-wattributes
    kwargs["extra_compile_args"] = ["-fvisibility=hidden", "-std=c++17"]

    # Platform-specific configuration
    if platform_type == 'rocm' and rocm_path:
        # ROCm/HIP configuration
        kwargs["define_macros"].append(("__HIP_PLATFORM_AMD__", "1"))
        kwargs["libraries"].append("amdhip64")
        kwargs["library_dirs"] = [f"{rocm_path}/lib"]
        kwargs["include_dirs"].append(f"{rocm_path}/include")
        kwargs["extra_compile_args"].append("-D__HIP_PLATFORM_AMD__")
        kwargs["extra_link_args"] = [f"-L{rocm_path}/lib", "-lamdhip64"]

    return Extension(name, sources, *args, **kwargs)


class CustomBuildExt(build_ext):
    """Custom build_ext to handle automatic hipification for ROCm platforms"""

    def run(self):
        # Detect platform
        platform_type, rocm_version, rocm_path = detect_platform()

        # Store platform info
        self.platform_type = platform_type
        self.rocm_version = rocm_version
        self.rocm_path = rocm_path

        #  Configure build based on platform
        if platform_type == 'rocm' and rocm_path:
            print("=" * 60)
            print("Building for AMD ROCm platform")
            if rocm_version:
                print(f"ROCm version: {rocm_version}")
            print("=" * 60)

            # Hipify sources
            hipify_source_files(rocm_path)

            # Update extension sources to use hipified files
            for ext in self.extensions:
                new_sources = []
                for src in ext.sources:
                    if 'fastsafetensors/cpp/ext.cpp' in src:
                        # torch.utils.hipify creates files in hip/ subdirectory
                        new_sources.append(src.replace('fastsafetensors/cpp/ext.cpp', 'fastsafetensors/cpp/hip/ext.cpp'))
                    else:
                        new_sources.append(src)
                ext.sources = new_sources

                # Update include dirs to include hip/ subdirectory
                ext.include_dirs.append("fastsafetensors/cpp/hip")

                # Update extension with ROCm-specific settings
                ext.define_macros.append(("__HIP_PLATFORM_AMD__", "1"))
                ext.define_macros.append(("USE_ROCM", "1"))
                ext.libraries.append("amdhip64")
                ext.library_dirs = [f"{rocm_path}/lib"]
                ext.include_dirs.append(f"{rocm_path}/include")
                ext.extra_compile_args.append("-D__HIP_PLATFORM_AMD__")
                ext.extra_compile_args.append("-DUSE_ROCM")
                ext.extra_link_args = [f"-L{rocm_path}/lib", "-lamdhip64"]
        else:
            print("=" * 60)
            print("Building for NVIDIA CUDA platform")
            print("=" * 60)

        # Continue with normal build
        build_ext.run(self)


# Detect platform for package_data
platform_type, _, rocm_path_detected = detect_platform()
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
            name=f"fastsafetensors.cpp",
            sources=["fastsafetensors/cpp/ext.cpp"],
            include_dirs=["fastsafetensors/cpp"],
            mod_name="cpp",
            platform_type=platform_type,
            rocm_path=rocm_path_detected,
        )
    ],
    cmdclass={
        'build_ext': CustomBuildExt,
    },
)
