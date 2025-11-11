// SPDX-License-Identifier: Apache-2.0
/*
 * CUDA/HIP compatibility layer for fastsafetensors
 * Minimal compatibility header - only defines what hipify-perl doesn't handle
 */

#ifndef __CUDA_COMPAT_H__
#define __CUDA_COMPAT_H__

// Platform detection - this gets hipified to check __HIP_PLATFORM_AMD__
#ifdef __HIP_PLATFORM_AMD__
  #ifndef USE_ROCM
    #define USE_ROCM
  #endif
  // Note: We do NOT include <hip/hip_runtime.h> here to avoid compile-time dependencies.
  // Instead, we dynamically load the ROCm runtime library (libamdhip64.so) at runtime
  // using dlopen(), just like we do for CUDA (libcudart.so).
  // Minimal types are defined in ext.hpp.
#else
  // For CUDA platform, we also avoid including headers and define minimal types in ext.hpp
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

#endif // __CUDA_COMPAT_H__
