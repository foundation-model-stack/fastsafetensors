// SPDX-License-Identifier: Apache-2.0
/*
 * CUDA/HIP compatibility layer for fastsafetensors
 *
 * All GPU functions are loaded at runtime via dlopen()/dlsym() — no CUDA or
 * HIP headers are included and no GPU runtime library is linked at build time.
 * This header provides library filenames and symbol names for both CUDA and
 * ROCm so that a single universal binary can detect the platform at runtime.
 */

#ifndef __GPU_COMPAT_H__
#define __GPU_COMPAT_H__

// CUDA runtime library and symbol names
#define CUDA_RUNTIME_LIB               "libcudart.so"
#define CUDA_SYM_GET_DEVICE_COUNT      "cudaGetDeviceCount"
#define CUDA_SYM_MEMCPY                "cudaMemcpy"
#define CUDA_SYM_MEMCPY_ASYNC          "cudaMemcpyAsync"
#define CUDA_SYM_DEVICE_SYNCHRONIZE    "cudaDeviceSynchronize"
#define CUDA_SYM_HOST_ALLOC            "cudaHostAlloc"
#define CUDA_SYM_FREE_HOST             "cudaFreeHost"
#define CUDA_SYM_DEVICE_GET_PCI_BUS_ID "cudaDeviceGetPCIBusId"
#define CUDA_SYM_DEVICE_MALLOC         "cudaMalloc"
#define CUDA_SYM_DEVICE_FREE           "cudaFree"
#define CUDA_SYM_DRIVER_GET_VERSION    "cudaDriverGetVersion"
#define CUDA_SYM_DEVICE_GET_ATTRIBUTE  "cudaDeviceGetAttribute"
#define CUDA_SYM_SET_DEVICE            "cudaSetDevice"

// ROCm/HIP runtime library and symbol names
#define HIP_RUNTIME_LIB                "libamdhip64.so"
#define HIP_SYM_GET_DEVICE_COUNT       "hipGetDeviceCount"
#define HIP_SYM_MEMCPY                 "hipMemcpy"
#define HIP_SYM_MEMCPY_ASYNC           "hipMemcpyAsync"
#define HIP_SYM_DEVICE_SYNCHRONIZE     "hipDeviceSynchronize"
#define HIP_SYM_HOST_ALLOC             "hipHostMalloc"
#define HIP_SYM_FREE_HOST              "hipHostFree"
#define HIP_SYM_DEVICE_GET_PCI_BUS_ID  "hipDeviceGetPCIBusId"
#define HIP_SYM_DEVICE_MALLOC          "hipMalloc"
#define HIP_SYM_DEVICE_FREE            "hipFree"
#define HIP_SYM_DRIVER_GET_VERSION     "hipDriverGetVersion"
#define HIP_SYM_DEVICE_GET_ATTRIBUTE   "hipDeviceGetAttribute"
#define HIP_SYM_SET_DEVICE             "hipSetDevice"

#endif // __GPU_COMPAT_H__
