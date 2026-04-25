// SPDX-License-Identifier: Apache-2.0
/*
 * CUDA/HIP compatibility layer for fastsafetensors
 *
 * All GPU functions are loaded at runtime via dlopen()/dlsym() — no CUDA or
 * HIP headers are included and no GPU runtime library is linked at build time.
 * This header provides the two things the preprocessor can handle that dlsym
 * string literals cannot: the library filename and the per-platform symbol names.
 */

#ifndef __CUDA_COMPAT_H__
#define __CUDA_COMPAT_H__

#ifdef __HIP_PLATFORM_AMD__
  #ifndef USE_ROCM
    #define USE_ROCM
  #endif
#endif

// Runtime library loaded via dlopen()
#ifdef USE_ROCM
  #define GPU_RUNTIME_LIB "libamdhip64.so"
#else
  #define GPU_RUNTIME_LIB "libcudart.so"
#endif

// Symbol names passed to dlsym().
// Replaces the hipify-perl string-literal transformation so the source
// compiles for either backend without any external build-time tooling.
#ifdef USE_ROCM
  #define GPU_SYM_GET_DEVICE_COUNT      "hipGetDeviceCount"
  #define GPU_SYM_MEMCPY                "hipMemcpy"
  #define GPU_SYM_MEMCPY_ASYNC          "hipMemcpyAsync"
  #define GPU_SYM_DEVICE_SYNCHRONIZE    "hipDeviceSynchronize"
  #define GPU_SYM_HOST_ALLOC            "hipHostMalloc"
  #define GPU_SYM_FREE_HOST             "hipHostFree"
  #define GPU_SYM_DEVICE_GET_PCI_BUS_ID "hipDeviceGetPCIBusId"
  #define GPU_SYM_DEVICE_MALLOC         "hipMalloc"
  #define GPU_SYM_DEVICE_FREE           "hipFree"
  #define GPU_SYM_DRIVER_GET_VERSION    "hipDriverGetVersion"
  #define GPU_SYM_DEVICE_GET_ATTRIBUTE  "hipDeviceGetAttribute"
  #define GPU_SYM_SET_DEVICE            "hipSetDevice"
#else
  #define GPU_SYM_GET_DEVICE_COUNT      "cudaGetDeviceCount"
  #define GPU_SYM_MEMCPY                "cudaMemcpy"
  #define GPU_SYM_MEMCPY_ASYNC          "cudaMemcpyAsync"
  #define GPU_SYM_DEVICE_SYNCHRONIZE    "cudaDeviceSynchronize"
  #define GPU_SYM_HOST_ALLOC            "cudaHostAlloc"
  #define GPU_SYM_FREE_HOST             "cudaFreeHost"
  #define GPU_SYM_DEVICE_GET_PCI_BUS_ID "cudaDeviceGetPCIBusId"
  #define GPU_SYM_DEVICE_MALLOC         "cudaMalloc"
  #define GPU_SYM_DEVICE_FREE           "cudaFree"
  #define GPU_SYM_DRIVER_GET_VERSION    "cudaDriverGetVersion"
  #define GPU_SYM_DEVICE_GET_ATTRIBUTE  "cudaDeviceGetAttribute"
  #define GPU_SYM_SET_DEVICE            "cudaSetDevice"
#endif

// Internal struct field name aliases for ROCm builds
#ifdef USE_ROCM
  #define cudaDeviceMalloc hipDeviceMalloc
  #define cudaDeviceFree hipDeviceFree
#endif

#endif // __CUDA_COMPAT_H__
