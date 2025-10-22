/*
 * Copyright 2024 IBM Inc. All rights reserved
 * SPDX-License-Identifier: Apache-2.0
 *
 * CUDA/HIP compatibility layer for fastsafetensors
 * Minimal compatibility header - only defines what hipify-perl doesn't handle
 */

#pragma once

// Platform detection - this gets hipified to check __HIP_PLATFORM_AMD__
#ifdef __HIP_PLATFORM_AMD__
  #ifndef USE_ROCM
    #define USE_ROCM
  #endif
  #include <hip/hip_runtime.h>
#else
  // For CUDA platform, or when CUDA headers aren't available, we define minimal types in ext.hpp
#endif

// Runtime library name - hipify-perl doesn't change string literals
#ifdef USE_ROCM
  #define GPU_RUNTIME_LIB "libamdhip64.so"
#else
  #define GPU_RUNTIME_LIB "libcudart.so"
#endif

// Custom function pointer names that hipify-perl doesn't recognize
// These are our own naming in ext_funcs struct, not standard CUDA API
#ifdef USE_ROCM
  #define cudaDeviceMalloc hipDeviceMalloc
  #define cudaDeviceFree hipDeviceFree
#endif
