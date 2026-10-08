// SPDX-License-Identifier: Apache-2.0

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <fcntl.h>
#include <cstring>
#include <cerrno>
#ifdef _MSC_VER
#include <io.h>
#include <malloc.h>
#include <share.h>
#include <stdio.h>
#include <cstdint>
#include <mutex>
#include <unordered_set>
#ifndef NOMINMAX
#define NOMINMAX
#endif
// Windows-compatible posix_memalign
static inline int posix_memalign(void **memptr, size_t alignment, size_t size) {
    *memptr = _aligned_malloc(size, alignment);
    return (*memptr) ? 0 : errno;
}
// Windows-compatible pread
static inline int64_t pread(int fd, void *buf, size_t count, int64_t offset) {
    int64_t cur = _lseeki64(fd, 0, 1 /*SEEK_CUR*/);
    if (cur < 0) return -1;
    if (_lseeki64(fd, offset, 0 /*SEEK_SET*/) < 0) return -1;
    int rd = _read(fd, buf, (unsigned int)count);
    _lseeki64(fd, cur, 0 /*SEEK_SET*/);
    return rd;
}
// --- Windows equivalents for dlfcn.h ---
#include <windows.h>
#define RTLD_LAZY    0
#define RTLD_GLOBAL  0
#ifndef RTLD_NODELETE
#define RTLD_NODELETE 0x1000
#endif

static std::mutex g_nodelete_handles_mutex;
static std::unordered_set<void*> g_nodelete_handles;

static inline bool is_windows_path_like(const char* filename) {
    if (!filename || !filename[0]) return false;
    return std::strchr(filename, '\\') != nullptr ||
           std::strchr(filename, '/') != nullptr ||
           (std::strlen(filename) > 1 && filename[1] == ':');
}

static inline void* dlopen(const char* filename, int mode) {
    if (!filename) return nullptr;
    DWORD flags = LOAD_LIBRARY_SEARCH_DEFAULT_DIRS;
    if (is_windows_path_like(filename)) {
        flags |= LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR;
    }
    void* handle = reinterpret_cast<void*>(LoadLibraryExA(filename, nullptr, flags));
    if (handle && (mode & RTLD_NODELETE)) {
        std::lock_guard<std::mutex> lock(g_nodelete_handles_mutex);
        g_nodelete_handles.insert(handle);
    }
    return handle;
}
static inline void* dlsym(void* handle, const char* symbol) {
    return reinterpret_cast<void*>(GetProcAddress(reinterpret_cast<HMODULE>(handle), symbol));
}
static inline int dlclose(void* handle) {
    {
        std::lock_guard<std::mutex> lock(g_nodelete_handles_mutex);
        if (g_nodelete_handles.find(handle) != g_nodelete_handles.end()) {
            return 0;
        }
    }
    return FreeLibrary(reinterpret_cast<HMODULE>(handle)) ? 0 : -1;
}

// --- Windows equivalents for mmap/munmap ---
#define PROT_READ   1
#define MAP_PRIVATE 2
#define MAP_FAILED  ((void*)-1)

static inline void* mmap(void* /*addr*/, size_t length, int /*prot*/, int /*flags*/, int fd, int64_t offset) {
    HANDLE hFile = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
    if (hFile == INVALID_HANDLE_VALUE) return MAP_FAILED;
    DWORD offsetHigh = static_cast<DWORD>(offset >> 32);
    DWORD offsetLow  = static_cast<DWORD>(offset & 0xFFFFFFFF);
    HANDLE hMapping = CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!hMapping) return MAP_FAILED;
    void* ptr = MapViewOfFile(hMapping, FILE_MAP_READ, offsetHigh, offsetLow, length);
    CloseHandle(hMapping);  // view keeps the mapping alive
    return ptr ? ptr : MAP_FAILED;
}
static inline int munmap(void* addr, size_t /*length*/) {
    return UnmapViewOfFile(addr) ? 0 : -1;
}

// Map POSIX names to MSVC equivalents
#define open  _open
#define close _close
#undef O_RDONLY
// Tensor and checkpoint files are byte streams. The CRT defaults descriptors
// to text mode, which translates CRLF and treats Ctrl-Z as EOF unless binary
// mode is requested explicitly.
#define O_RDONLY (_O_RDONLY | _O_BINARY)
#ifndef O_DIRECT
#define O_DIRECT 0
#endif
#else
#include <unistd.h>
#include <sys/mman.h>
#include <chrono>
#include <dlfcn.h>
#endif
#include <chrono>
#include <cstdlib>
#include <algorithm>
#include <atomic>
#include <deque>
#include <limits>
#include <sys/stat.h>
#include <thread>
#include <vector>
#include <mutex>

#include "gpu_compat.h"
#include "ext.hpp"

#define ALIGN 4096

#ifdef _MSC_VER
void init_dstorage_bindings(pybind11::module_&);
#endif

bool debug_log = false;  // non-static: fix Windows build
static bool enable_gil_release = false;

static cpp_metrics_t mc = {.bounce_buffer_bytes = 0};

/* cpu_mode functions: for tests and debugs */

static CUfileError_t cpu_cuFileDriverOpen() { return CUfileError_t{.err = CU_FILE_SUCCESS}; }
static CUfileError_t cpu_cuFileDriverClose() { return CUfileError_t{.err = CU_FILE_SUCCESS}; }
static CUfileError_t cpu_cuFileDriverSetMaxDirectIOSize(size_t) { return CUfileError_t{.err = CU_FILE_SUCCESS}; }
static CUfileError_t cpu_cuFileDriverSetMaxPinnedMemSize(size_t) { return CUfileError_t{.err = CU_FILE_SUCCESS}; }
static CUfileError_t cpu_cuFileBufRegister(const void *, size_t, int) { return CUfileError_t{.err = CU_FILE_SUCCESS}; }
static CUfileError_t cpu_cuFileBufDeregister(const void *) { return CUfileError_t{.err = CU_FILE_SUCCESS}; }
static CUfileError_t cpu_cuFileHandleRegister(CUfileHandle_t * in, CUfileDescr_t *) {
    *in = reinterpret_cast<CUfileHandle_t *>(malloc(sizeof(CUfileHandle_t)));
    if (*in != nullptr) {
        return CUfileError_t{.err = CU_FILE_SUCCESS};
    }
    return CUfileError_t{.err = CU_FILE_INTERNAL_ERROR};
}
static void cpu_cuFileHandleDeregister(CUfileHandle_t h) {
    free(reinterpret_cast<void *>(h));
}
static cudaError_t cpu_cudaMemcpy(void * dst, const void * src, size_t size, enum cudaMemcpyKind) {
    std::memcpy(dst, src, size);
    return cudaSuccess;
}
static cudaError_t cpu_cudaMemcpyAsync(void * dst, const void * src, size_t size, enum cudaMemcpyKind kind, cudaStream_t) {
    std::memcpy(dst, src, size);
    return cudaSuccess;
}
static cudaError_t cpu_cudaStreamCreateWithFlags(cudaStream_t *stream, unsigned int) {
    *stream = nullptr; return cudaSuccess;
}
static cudaError_t cpu_cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
static cudaError_t cpu_cudaStreamDestroy(cudaStream_t) { return cudaSuccess; }
static cudaError_t cpu_cudaDeviceSynchronize() { return cudaSuccess; }
static cudaError_t cpu_cudaHostAlloc(void ** p, size_t length, unsigned int) {
    if (posix_memalign(p, ALIGN, length) != 0) {
        return cudaErrorMemoryAllocation;
    }
    return cudaSuccess;
}
static cudaError_t cpu_cudaFreeHost(void * p) {
#ifdef _MSC_VER
    _aligned_free(p);
#else
    free(p);
#endif
    return cudaSuccess;
}
static cudaError_t cpu_cudaDeviceGetPCIBusId(char * in, int s, int) {
    if (s > 0)
        in[0] = 0;
    return cudaSuccess;
}
static cudaError_t cpu_cudaSetDevice(int) { return cudaSuccess; }
static int cpu_numa_run_on_node(int) {return 0; }
static void (*numa_set_preferred)(int) = nullptr;

ext_funcs_t cpu_fns = ext_funcs_t {
    .cuFileDriverOpen = cpu_cuFileDriverOpen,
    .cuFileDriverClose = cpu_cuFileDriverClose,
    .cuFileDriverSetMaxDirectIOSize = cpu_cuFileDriverSetMaxDirectIOSize,
    .cuFileDriverSetMaxPinnedMemSize = cpu_cuFileDriverSetMaxPinnedMemSize,
    .cuFileBufRegister = cpu_cuFileBufRegister,
    .cuFileBufDeregister = cpu_cuFileBufDeregister,
    .cuFileHandleRegister = cpu_cuFileHandleRegister,
    .cuFileHandleDeregister = cpu_cuFileHandleDeregister,
    .cuFileRead = nullptr,
    .cudaMemcpy = cpu_cudaMemcpy,
    .cudaMemcpyAsync = cpu_cudaMemcpyAsync,
    .cudaStreamCreateWithFlags = cpu_cudaStreamCreateWithFlags,
    .cudaStreamSynchronize = cpu_cudaStreamSynchronize,
    .cudaStreamDestroy = cpu_cudaStreamDestroy,
    .cudaDeviceSynchronize = cpu_cudaDeviceSynchronize,
    .cudaHostAlloc = cpu_cudaHostAlloc,
    .cudaFreeHost = cpu_cudaFreeHost,
    .cudaDeviceGetPCIBusId = cpu_cudaDeviceGetPCIBusId,
    .numa_run_on_node = cpu_numa_run_on_node,
    .cudaSetDevice = cpu_cudaSetDevice,
    .cudaImportExternalMemory = nullptr,
    .cudaExternalMemoryGetMappedBuffer = nullptr,
    .cudaDestroyExternalMemory = nullptr,
};
ext_funcs_t cuda_fns;

static bool gpu_found = false;
static bool is_hip_runtime = false;
static bool cufile_found = false;

static int cufile_ver = 0;

// hipFile has a separate version ABI and an error struct with two fields.
// Its AMD driver-property API is a stub, and its properties layout differs
// from CUfileDrvProps_t, so never resolve it into the cuFile function pointer.
struct hipFileError_t { int err; int hip_drv_err; };
static unsigned hipfile_major = 0, hipfile_minor = 0, hipfile_patch = 0;
static bool hipfile_version_known = false;

// FGDS (FGDS_LIB) function pointers and availability flag. Resolved at
// runtime in load_fgds_library() (called on demand by the FGDS copier path);
// never linked at build time. fgds_found mirrors cufile_found for capability
// probing.
static bool fgds_found = false;
static int (*fgds_open)(int) = nullptr;
static int (*fgds_close)(int) = nullptr;
static int (*fgds_regmem)(int, uintptr_t, size_t, void**) = nullptr;
static int (*fgds_deregmem)(int, uintptr_t, size_t) = nullptr;
static ssize_t (*fgds_read)(fgds_fileid, void*, off_t, size_t, off_t) = nullptr;
// Tracks devices opened via init_fgds() so close_fgds() can pair them.
static std::mutex fgds_open_mutex;
static std::map<int, bool> fgds_opened_devices;

template <typename T> void mydlsym(T** h, void* lib, std::string const& name) {
    *h = reinterpret_cast<T*>(dlsym(lib, name.c_str()));
}

// Try to load one GPU runtime library (CUDA or HIP). Returns true and sets
// gpu_found/is_hip_runtime on success; leaves them unchanged on failure.
static bool load_gpu_lib(const std::string& lib_name, bool is_hip, bool init_log, int mode) {
    cudaError_t (*get_device_count)(int*) = nullptr;
    const char* sym_count = is_hip ? HIP_SYM_GET_DEVICE_COUNT : CUDA_SYM_GET_DEVICE_COUNT;

    void* handle = dlopen(lib_name.c_str(), mode);
    if (!handle) {
        if (init_log) fprintf(stderr, "[DEBUG] %s is not installed. fallback\n", lib_name.c_str());
        return false;
    }

    mydlsym(&get_device_count, handle, sym_count);
    if (!get_device_count) {
        if (init_log) fprintf(stderr, "[DEBUG] No %s in %s, fallback!\n", sym_count, lib_name.c_str());
        dlclose(handle);
        return false;
    }

    int count = 0;
    if (get_device_count(&count) != cudaSuccess) count = 0;
    if (init_log) fprintf(stderr, "[DEBUG] %s: device count=%d\n", lib_name.c_str(), count);
    if (count == 0) {
        dlclose(handle);
        return false;
    }

    mydlsym(&cuda_fns.cudaMemcpy,             handle, is_hip ? HIP_SYM_MEMCPY                : CUDA_SYM_MEMCPY);
    mydlsym(&cuda_fns.cudaMemcpyAsync,        handle, is_hip ? HIP_SYM_MEMCPY_ASYNC          : CUDA_SYM_MEMCPY_ASYNC);
    mydlsym(&cuda_fns.cudaStreamCreateWithFlags, handle, is_hip ? "hipStreamCreateWithFlags" : "cudaStreamCreateWithFlags");
    mydlsym(&cuda_fns.cudaStreamSynchronize, handle, is_hip ? "hipStreamSynchronize" : "cudaStreamSynchronize");
    mydlsym(&cuda_fns.cudaStreamDestroy, handle, is_hip ? "hipStreamDestroy" : "cudaStreamDestroy");
    mydlsym(&cuda_fns.cudaDeviceSynchronize,  handle, is_hip ? HIP_SYM_DEVICE_SYNCHRONIZE    : CUDA_SYM_DEVICE_SYNCHRONIZE);
    mydlsym(&cuda_fns.cudaHostAlloc,          handle, is_hip ? HIP_SYM_HOST_ALLOC            : CUDA_SYM_HOST_ALLOC);
    mydlsym(&cuda_fns.cudaFreeHost,           handle, is_hip ? HIP_SYM_FREE_HOST             : CUDA_SYM_FREE_HOST);
    mydlsym(&cuda_fns.cudaDeviceGetPCIBusId,  handle, is_hip ? HIP_SYM_DEVICE_GET_PCI_BUS_ID : CUDA_SYM_DEVICE_GET_PCI_BUS_ID);
    mydlsym(&cuda_fns.cudaDeviceMalloc,       handle, is_hip ? HIP_SYM_DEVICE_MALLOC         : CUDA_SYM_DEVICE_MALLOC);
    mydlsym(&cuda_fns.cudaDeviceFree,         handle, is_hip ? HIP_SYM_DEVICE_FREE           : CUDA_SYM_DEVICE_FREE);
    mydlsym(&cuda_fns.cudaDriverGetVersion,   handle, is_hip ? HIP_SYM_DRIVER_GET_VERSION    : CUDA_SYM_DRIVER_GET_VERSION);
    mydlsym(&cuda_fns.cudaDeviceGetAttribute, handle, is_hip ? HIP_SYM_DEVICE_GET_ATTRIBUTE  : CUDA_SYM_DEVICE_GET_ATTRIBUTE);
    mydlsym(&cuda_fns.cudaSetDevice,          handle, is_hip ? HIP_SYM_SET_DEVICE            : CUDA_SYM_SET_DEVICE);

    // External memory interop is CUDA-only (used by Windows DirectStorage path)
    if (!is_hip) {
        mydlsym(&cuda_fns.cudaImportExternalMemory, handle, "cudaImportExternalMemory");
        mydlsym(&cuda_fns.cudaExternalMemoryGetMappedBuffer, handle, "cudaExternalMemoryGetMappedBuffer");
        mydlsym(&cuda_fns.cudaDestroyExternalMemory, handle, "cudaDestroyExternalMemory");
    } else {
        cuda_fns.cudaImportExternalMemory = nullptr;
        cuda_fns.cudaExternalMemoryGetMappedBuffer = nullptr;
        cuda_fns.cudaDestroyExternalMemory = nullptr;
    }

    bool success = cuda_fns.cudaMemcpy && cuda_fns.cudaDeviceSynchronize;
    success = success && cuda_fns.cudaHostAlloc && cuda_fns.cudaFreeHost;
    success = success && cuda_fns.cudaDeviceGetPCIBusId && cuda_fns.cudaDeviceMalloc;
    success = success && cuda_fns.cudaDeviceFree && cuda_fns.cudaDriverGetVersion;
    success = success && cuda_fns.cudaDeviceGetAttribute && cuda_fns.cudaSetDevice;

    dlclose(handle);

    if (!success) {
        if (init_log) fprintf(stderr, "[DEBUG] %s missing required GPU functions. fallback\n", lib_name.c_str());
        return false;
    }

    if (init_log) fprintf(stderr, "[DEBUG] loaded: %s (hip=%d)\n", lib_name.c_str(), (int)is_hip);
    gpu_found = true;
    is_hip_runtime = is_hip;
    return true;
}

static void load_library_functions(const std::string& cudart_override = "") {
#ifdef _MSC_VER
    const char* numaLib = nullptr;  // NUMA not available on Windows
#else
    const char* numaLib = "libnuma.so.1";
#endif
    bool init_log = getenv(ENV_ENABLE_INIT_LOG);
    int mode = RTLD_LAZY | RTLD_GLOBAL | RTLD_NODELETE;

    if (numaLib) {
        void* handle_numa = dlopen(numaLib, mode);
        if (handle_numa) {
            mydlsym(&cpu_fns.numa_run_on_node, handle_numa, "numa_run_on_node");
            mydlsym(&numa_set_preferred, handle_numa, "numa_set_preferred");
            if (cpu_fns.numa_run_on_node) {
                cuda_fns.numa_run_on_node = cpu_fns.numa_run_on_node;
                if (init_log) {
                    fprintf(stderr, "[DEBUG] loaded: %s\n", numaLib);
                }
            }
            dlclose(handle_numa);
        }
    }
    if (!cpu_fns.numa_run_on_node) {
        if (init_log && numaLib) {
            fprintf(stderr, "[DEBUG] %s is not installed. fallback\n", numaLib);
        }
        cpu_fns.numa_run_on_node = cpu_numa_run_on_node;
        cuda_fns.numa_run_on_node = cpu_numa_run_on_node;
    }

    if (!cudart_override.empty()) {
        // Caller specified exact library — detect platform from name
        bool is_hip = cudart_override.find("hip") != std::string::npos;
        load_gpu_lib(cudart_override, is_hip, init_log, mode);
    } else {
        // Universal detection: try CUDA first, then ROCm
        if (!load_gpu_lib(CUDA_RUNTIME_LIB, false, init_log, mode)) {
            load_gpu_lib(HIP_RUNTIME_LIB, true, init_log, mode);
        }
    }

    if (!gpu_found) {
        cuda_fns.cudaMemcpy = cpu_cudaMemcpy;
        cuda_fns.cudaDeviceSynchronize = cpu_cudaDeviceSynchronize;
        cuda_fns.cudaHostAlloc = cpu_cudaHostAlloc;
        cuda_fns.cudaFreeHost = cpu_cudaFreeHost;
        cuda_fns.cudaDeviceGetPCIBusId = cpu_cudaDeviceGetPCIBusId;
        cuda_fns.cudaSetDevice = cpu_cudaSetDevice;
        cuda_fns.cudaImportExternalMemory = nullptr;
        cuda_fns.cudaExternalMemoryGetMappedBuffer = nullptr;
        cuda_fns.cudaDestroyExternalMemory = nullptr;
    }

#ifdef _MSC_VER
    const char* gdsLib = nullptr; // neither cuFile nor hipFile on Windows
#else
    const char* gdsLib = is_hip_runtime ? HIPFILE_LIB : CUFILE_LIB;
#endif
    cufile_found = false;
    cufile_ver = 0;
    hipfile_major = hipfile_minor = hipfile_patch = 0;
    hipfile_version_known = false;
    cuda_fns.cuFileDriverGetProperties = nullptr;
    if (gpu_found && gdsLib) {
        const bool is_hip = is_hip_runtime;
        void* handle_gds = dlopen(gdsLib, mode);
        if (handle_gds) {
            if (!is_hip) {
                CUfileError_t (*cuFileGetVersion)(int *);
                mydlsym(&cuFileGetVersion, handle_gds, CUFILE_SYM_GET_VERSION);
                if (cuFileGetVersion) {
                    int version;
                    CUfileError_t err = cuFileGetVersion(&version);
                    if (err.err == CU_FILE_SUCCESS) {
                        cufile_ver = version;
                    }
                }
                if (cufile_ver == 0) {
                    fprintf(stderr, "[WARN] %s is loaded but its version is unknown", gdsLib);
                }
            } else {
                hipFileError_t (*hipFileGetVersion)(unsigned *, unsigned *, unsigned *) = nullptr;
                mydlsym(&hipFileGetVersion, handle_gds, HIPFILE_SYM_GET_VERSION);
                if (hipFileGetVersion) {
                    hipFileError_t err = hipFileGetVersion(&hipfile_major, &hipfile_minor, &hipfile_patch);
                    hipfile_version_known = err.err == 0 && err.hip_drv_err == 0;
                }
            }
            mydlsym(&cuda_fns.cuFileDriverOpen, handle_gds, is_hip ? HIPFILE_SYM_DRIVER_OPEN : CUFILE_SYM_DRIVER_OPEN);
            mydlsym(&cuda_fns.cuFileDriverClose, handle_gds, is_hip ? HIPFILE_SYM_DRIVER_CLOSE : CUFILE_SYM_DRIVER_CLOSE);
            // NVIDIA budgeted loads reserve the configured device cache.
            // AMD hipFile uses direct GPU I/O or a host-side fallback buffer.
            if (!is_hip)
                mydlsym(&cuda_fns.cuFileDriverGetProperties, handle_gds, CUFILE_SYM_DRIVER_GET_PROPERTIES);
            mydlsym(&cuda_fns.cuFileDriverSetMaxDirectIOSize, handle_gds, is_hip ? HIPFILE_SYM_DRIVER_SET_MAX_DIO_SIZE : CUFILE_SYM_DRIVER_SET_MAX_DIO_SIZE);
            mydlsym(&cuda_fns.cuFileDriverSetMaxPinnedMemSize, handle_gds, is_hip ? HIPFILE_SYM_DRIVER_SET_MAX_PIN_SIZE : CUFILE_SYM_DRIVER_SET_MAX_PIN_SIZE);
            mydlsym(&cuda_fns.cuFileBufRegister, handle_gds, is_hip ? HIPFILE_SYM_BUF_REGISTER : CUFILE_SYM_BUF_REGISTER);
            mydlsym(&cuda_fns.cuFileBufDeregister, handle_gds, is_hip ? HIPFILE_SYM_BUF_DEREGISTER : CUFILE_SYM_BUF_DEREGISTER);
            mydlsym(&cuda_fns.cuFileHandleRegister, handle_gds, is_hip ? HIPFILE_SYM_HANDLE_REGISTER : CUFILE_SYM_HANDLE_REGISTER);
            mydlsym(&cuda_fns.cuFileHandleDeregister, handle_gds, is_hip ? HIPFILE_SYM_HANDLE_DEREGISTER : CUFILE_SYM_HANDLE_DEREGISTER);
            mydlsym(&cuda_fns.cuFileRead, handle_gds, is_hip ? HIPFILE_SYM_READ : CUFILE_SYM_READ);
            bool success = cuda_fns.cuFileDriverOpen && cuda_fns.cuFileDriverClose && cuda_fns.cuFileDriverSetMaxDirectIOSize;
            success &= cuda_fns.cuFileDriverSetMaxPinnedMemSize && cuda_fns.cuFileBufRegister && cuda_fns.cuFileBufDeregister;
            success &= cuda_fns.cuFileHandleRegister && cuda_fns.cuFileHandleDeregister && cuda_fns.cuFileRead;
            if (!success) {
                if (init_log) {
                    fprintf(stderr, "[DEBUG] %s does not contain required GDS functions. fallback\n", gdsLib);
                }
            } else {
                if (init_log) {
                    if (is_hip) {
                        if (hipfile_version_known)
                            fprintf(stderr, "[DEBUG] loaded: %s (ver: %u.%u.%u)\n", gdsLib, hipfile_major, hipfile_minor, hipfile_patch);
                        else
                            fprintf(stderr, "[DEBUG] loaded: %s (version unknown; budgeted loads use nogds)\n", gdsLib);
                    } else {
                        fprintf(stderr, "[DEBUG] loaded: %s (ver: %d.%d.%d)\n", gdsLib, cufile_ver / 1000, (cufile_ver % 1000) / 10, cufile_ver % 10);
                    }
                }
                cufile_found = true;
            }
            dlclose(handle_gds);
        } else if (init_log) {
            fprintf(stderr, "[DEBUG] %s is not installed. fallback\n", gdsLib);
        }
    }

    if (!cufile_found) {
        cuda_fns.cuFileDriverOpen = cpu_cuFileDriverOpen;
        cuda_fns.cuFileDriverClose = cpu_cuFileDriverClose;
        cuda_fns.cuFileDriverSetMaxDirectIOSize = cpu_cuFileDriverSetMaxDirectIOSize;
        cuda_fns.cuFileDriverSetMaxPinnedMemSize = cpu_cuFileDriverSetMaxPinnedMemSize;
        cuda_fns.cuFileBufRegister = cpu_cuFileBufRegister;
        cuda_fns.cuFileBufDeregister = cpu_cuFileBufDeregister;
        cuda_fns.cuFileHandleRegister = cpu_cuFileHandleRegister;
        cuda_fns.cuFileHandleDeregister = cpu_cuFileHandleDeregister;

        cuda_fns.cuFileRead = nullptr;
    }
}

// Resolve the FGDS (FGDS_LIB) function pointers. Linux-only NVMe-oF GPUDirect
// path. Loaded strictly on demand from the FGDS copier factory
// (init_fgds() -> new_fgds_file_copier), so selecting another copier
// (gds/nogds/unified/dstorage) never dlopens libfgds.so. FGDS is only
// meaningful with a GPU present, so it also requires gpu_found, which
// load_library_functions() sets first. Idempotent: returns immediately once
// the symbols are resolved.
void load_fgds_library()
{
    if (fgds_found) {
        return;
    }
#ifndef _MSC_VER
    if (!gpu_found) {
        return;
    }
    bool init_log = getenv(ENV_ENABLE_INIT_LOG);
    int mode = RTLD_LAZY | RTLD_GLOBAL | RTLD_NODELETE;
    void* handle_fgds = dlopen(FGDS_LIB, mode);
    if (!handle_fgds) {
        if (init_log) {
            fprintf(stderr, "[DEBUG] %s is not installed. fallback\n", FGDS_LIB);
        }
        return;
    }
    mydlsym(&fgds_open, handle_fgds, "fgds_open");
    mydlsym(&fgds_close, handle_fgds, "fgds_close");
    mydlsym(&fgds_regmem, handle_fgds, "fgds_regmem");
    mydlsym(&fgds_deregmem, handle_fgds, "fgds_deregmem");
    mydlsym(&fgds_read, handle_fgds, "fgds_read");
    bool success = fgds_open && fgds_close && fgds_regmem &&
                   fgds_deregmem && fgds_read;
    if (!success) {
        if (init_log) {
            fprintf(stderr, "[DEBUG] %s does not contain required FGDS functions. fallback\n", FGDS_LIB);
        }
        fgds_open = nullptr;
        fgds_close = nullptr;
        fgds_regmem = nullptr;
        fgds_deregmem = nullptr;
        fgds_read = nullptr;
    } else {
        if (init_log) {
            fprintf(stderr, "[DEBUG] loaded: %s done\n", FGDS_LIB);
        }
        fgds_found = true;
    }
    dlclose(handle_fgds);
#endif
}

bool is_cuda_found()
{
    return gpu_found && !is_hip_runtime;
}

bool is_hip_found()
{
    return gpu_found && is_hip_runtime;
}

bool is_cufile_found()
{
    return cufile_found;
}

/* The version is returned as (1000 * major + 10 * minor). */
int cufile_version()
{
    return cufile_ver;
}

bool is_fgds_found()
{
    return fgds_found;
}

// Open the FGDS device exactly once per device_id. Mirrors init_gds(), which
// calls cuFileDriverOpen() once; here fgds_open() is per-device so we track
// opened devices and make the call idempotent. The matching close_fgds() must
// be invoked for each opened device (e.g. via Python atexit).
int init_fgds(int device_id)
{
    if (!fgds_found || !fgds_open) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(fgds_open_mutex);
    if (fgds_opened_devices.count(device_id)) {
        return 0;
    }
    int ret = fgds_open(device_id);
    if (ret == 0) {
        fgds_opened_devices[device_id] = true;
    }
    return ret;
}

int close_fgds(int device_id)
{
    if (!fgds_found || !fgds_close) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(fgds_open_mutex);
    if (!fgds_opened_devices.count(device_id)) {
        return 0;
    }
    int ret = fgds_close(device_id);
    fgds_opened_devices.erase(device_id);
    return ret;
}

int get_alignment_size()
{
    return ALIGN;
}

void set_debug_log(bool _debug_log)
{
    debug_log = _debug_log;
}

void set_gil_release(bool enable) {
    enable_gil_release = enable;
}

bool get_gil_release() {
    return enable_gil_release;
}

void init_gil_release_from_env() {
    const char* env_val = std::getenv("FASTSAFETENSORS_ENABLE_GIL_RELEASE");
    if (env_val != nullptr) {
        std::string env_str(env_val);
        // Convert to lowercase for case-insensitive comparison
        std::transform(env_str.begin(), env_str.end(), env_str.begin(), ::tolower);
        enable_gil_release = (env_str == "1" || env_str == "true" || env_str == "yes" || env_str == "on");
        if (debug_log) {
            std::printf("[DEBUG] GIL release %s via environment variable FASTSAFETENSORS_ENABLE_GIL_RELEASE=%s\n",
                       enable_gil_release ? "enabled" : "disabled", env_val);
        }
    }
}

int is_gds_supported(int deviceId)
{
    int gdr_support = 1;
    int driverVersion = 0;

    cudaError_t err = cuda_fns.cudaDriverGetVersion(&driverVersion);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "is_gds_supported: %s failed, deviceId=%d, err=%d\n",
            is_hip_runtime ? HIP_SYM_DRIVER_GET_VERSION : CUDA_SYM_DRIVER_GET_VERSION, deviceId, err);
        return -1;
    }

    if (is_hip_runtime) {
        // hipFile requires ROCm >= 7.2.
        constexpr int HIPFILE_MIN_HIP_VER = 70200000;
        if (!cufile_found || driverVersion < HIPFILE_MIN_HIP_VER) return 0;
        return gdr_support;
    }

    if (driverVersion > 11030) {
        err = cuda_fns.cudaDeviceGetAttribute(&gdr_support, cudaDevAttrGPUDirectRDMASupported, deviceId);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "is_gds_supported: cudaDeviceGetAttribute failed, deviceId=%d, err=%d\n", deviceId, err);
            return -1;
        }
    }
    return gdr_support;
}

int init_gds()
{
    CUfileError_t err;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    if (cuda_fns.cuFileDriverOpen) {
        err = cuda_fns.cuFileDriverOpen();
        if (err.err != CU_FILE_SUCCESS) {
            std::fprintf(stderr, "init_gds: cuFileDriverOpen returned an error = %d\n", err.err);
            return -1;
        }
    }
    if (debug_log) {
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::printf("[DEBUG] init_gds: cuFileDriverOpen=%" PRId64 " us\n",
            std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
    }
    return 0;
}

int close_gds()
{
    CUfileError_t err;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    if (cuda_fns.cuFileDriverClose) {
        err = cuda_fns.cuFileDriverClose();
        if (err.err != CU_FILE_SUCCESS) {
            std::fprintf(stderr, "close_gds: cuFileDriverClose returned an error = %d\n", err.err);
            return -1;
        }
    }
    if (debug_log) {
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::printf("[DEBUG] close_gds: cuFileDriverClose, elapsed=%" PRId64 " us\n",
            std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
    }
    return 0;
}

bool is_hipfile_memory_budget_supported()
{
    return is_hip_runtime && cufile_found && hipfile_version_known
        && hipfile_major == 0 && hipfile_minor >= 2 && hipfile_minor <= 4;
}

uint64_t gds_device_cache_size()
{
    if (!cufile_found) return 0; // CPU reader has no device-side cache.
    if (is_hip_runtime) {
        // Audited AMD hipFile 0.2--0.4 synchronous reads allocate no extra
        // VRAM cache: fastpath writes into the destination; fallback uses
        // mmap'd host memory. Registration tracks only pointer/range metadata.
        // Keep the version guard: a failed query or a future implementation
        // must not silently become a zero-byte device-memory reservation.
        if (!is_hipfile_memory_budget_supported())
            throw std::runtime_error("GDS memory budgeting requires a known AMD hipFile version in 0.2.x--0.4.x; use nogds or unset device_memory_budget");
        return 0;
    }
    if (!cuda_fns.cuFileDriverGetProperties)
        throw std::runtime_error("GDS memory budgeting requires cuFileDriverGetProperties");
    CUfileDrvProps_t props{};
    CUfileError_t err = cuda_fns.cuFileDriverGetProperties(&props);
    if (err.err != CU_FILE_SUCCESS)
        throw std::runtime_error("GDS device cache size query failed, err=" + std::to_string(err.err));
    // The driver reports its configured device cache capacity in KiB.
    return static_cast<uint64_t>(props.max_device_cache_size) * 1024;
}

std::string get_device_pci_bus(int deviceId) {
    cudaError_t err;
    char pciBusId[32];

    std::memset(pciBusId, 0, 32);
    if (cuda_fns.cudaDeviceGetPCIBusId) {
        err = cuda_fns.cudaDeviceGetPCIBusId(pciBusId, 32, deviceId);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "get_device_pci_bus: cudaDeviceGetPCIBusId failed, deviceId=%d, err=%d\n", deviceId, err);
            return "";
        }
    } else {
        return "";
    }
    return std::string(pciBusId);
}

int set_numa_node(int numa_node) {
    if (numa_node >= 0) {
        if (cpu_fns.numa_run_on_node(numa_node) != 0) {
            std::fprintf(stderr, "set_numa_node: numa_run_on_node(numa_node=%d) failed\n", numa_node);
            return -1;
        }
    }
    return 0;
}

pybind11::bytes read_buffer(uintptr_t _dst, uint64_t length) {
    std::string buf;
    char *c = reinterpret_cast<char *>(_dst);
    buf.insert(buf.end(), c, c+length);
    return pybind11::bytes(buf);
}

uintptr_t cpu_malloc(uint64_t length) {
    void *p;
    if (posix_memalign(&p, ALIGN, length) < 0) {
        return 0;
    }
    return reinterpret_cast<uintptr_t>(p);
}

void cpu_free(uintptr_t addr) {
    void *p = reinterpret_cast<void *>(addr);
#ifdef _MSC_VER
    _aligned_free(p);
#else
    free(p);
#endif
}

uintptr_t gpu_malloc(uint64_t length) {
    void *p;
    if (cuda_fns.cudaDeviceMalloc(&p, length) != cudaSuccess) {
        return 0;
    }
    return reinterpret_cast<uintptr_t>(p);
}

void gpu_free(uintptr_t addr) {
    cuda_fns.cudaDeviceFree(reinterpret_cast<void*>(addr));
}

const int gds_device_buffer::cufile_register(uint64_t offset, uint64_t length) {
    CUfileError_t err;
    void * dst = reinterpret_cast<void*>(this->_devPtr_base->get_uintptr() + offset);

    std::chrono::steady_clock::time_point begin_register = std::chrono::steady_clock::now();
    err = _fns->cuFileBufRegister(dst, length, 0);
    if (err.err != CU_FILE_SUCCESS) {
        std::fprintf(stderr, "gds_device_buffer.cufile_register: cuFileBufRegister returned an error = %d\n", err.err);
        return -1;
    }
    if (debug_log) {
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::printf("[DEBUG] gds_device_buffer.cufile_register: addr=%p, offset=%" PRIu64 ", length=%" PRIu64 ", register=%" PRId64 " us\n", dst, offset, length,
            std::chrono::duration_cast<std::chrono::microseconds>(end - begin_register).count());
    }
    return 0;
}

const int gds_device_buffer::cufile_deregister(uint64_t offset) {
    void * dst = reinterpret_cast<void*>(this->_devPtr_base->get_uintptr() + offset);
    CUfileError_t err;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    err = _fns->cuFileBufDeregister(dst);
    if (err.err != CU_FILE_SUCCESS) {
        std::fprintf(stderr, "gds_device_buffer.cufile_deregister: cuFileBufDeregister (%p) returned an error=%d\n", dst, err.err);
        return -1;
    }
    if (debug_log) {
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::printf("[DEBUG] gds_device_buffer.cufile_deregister: addr=%p, offset=%" PRIu64 ", elapsed=%" PRId64 " us\n", dst, offset,
            std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
    }
    return 0;
}

const int gds_device_buffer::memmove(uint64_t _dst_off, uint64_t _src_off, const gds_device_buffer& _tmp, uint64_t length) {
    cudaError_t err;
    void *dst = reinterpret_cast<void *>(this->_devPtr_base->get_uintptr() + _dst_off);
    void *src = reinterpret_cast<void *>(this->_devPtr_base->get_uintptr() + _src_off);
    void *tmp = const_cast<void *>(_tmp._devPtr_base->get_raw());

    if (this->_length < _dst_off) {
        std::fprintf(stderr, "gds_device_buffer.memmove: length is smaller than request dst_off, tmp.length=%" PRIu64 ", _dst_off=%" PRIu64 "\n", _tmp._length, _dst_off);
        return -1;
    }
    if (this->_length < _src_off) {
        std::fprintf(stderr, "gds_device_buffer.memmove: length is smaller than request dst_off, tmp.length=%" PRIu64 ", _src_off=%" PRIu64 "\n", _tmp._length, _src_off);
        return -1;
    }
    if (_tmp._length < length) {
        std::fprintf(stderr, "gds_device_buffer.memmove: tmp is smaller than request length, tmp.length=%" PRIu64 ", length=%" PRIu64 "\n", _tmp._length, length);
        return -1;
    }
    if (length == 0) {
        return 0;
    }

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    err = _fns->cudaMemcpy(tmp, src, length, cudaMemcpyDefault);
    if (err != cudaSuccess) {
        std::printf("gds_device_buffer.memmove: cudaMemcpy[0](tmp=%p, src=%p, length=%" PRIu64 ") failed, err=%d\n", tmp, src, length, err);
        return -1;
    }
    err = _fns->cudaMemcpy(dst, tmp, length, cudaMemcpyDefault);
    if (err != cudaSuccess) {
        std::printf("gds_device_buffer.memmove: cudaMemcpy[1](dst=%p, tmp=%p, length=%" PRIu64 ") failed, err=%d\n", dst, tmp, length, err);
        return -1;
    }
    if (debug_log) {
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::printf("[DEBUG] gds_device_buffer.memmove: dst=%p, src=%p, tmp=%p, length=%" PRIu64 ", elapsed=%" PRId64 " us\n", dst, src, tmp, length,
            std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
    }
    return 0;
}


// Persistent workers alternate bounce buffers, draining DMA before slot reuse.
struct nogds_file_reader::state {
    struct request {
        int fd;
        uintptr_t destination;
        int64_t offset, length, next = 0;
        uint64_t remaining, prefix = 0;
        std::vector<uint8_t> completed;
        bool failed = false;
    };
    ext_funcs_t *fns;
    int device, numa_node;
    bool use_mmap, streams;
    static constexpr size_t buffers_per_thread = 2;
    uint64_t block_size, buffer_stride, allocation_size = 0;
    void *allocation = nullptr;
    std::vector<cudaStream_t> copy_streams;
    std::vector<std::thread> workers;
    std::mutex mutex;
#ifdef _MSC_VER
    std::mutex file_mutex;
#endif
    std::condition_variable ready, done;
    std::deque<std::shared_ptr<request>> queue;
    std::map<int, std::shared_ptr<request>> requests;
    bool stopping = false;
    int next_id = 1;

    static void check(cudaError_t error) {
        if (error != cudaSuccess)
            throw std::runtime_error("nogds GPU operation failed: " + std::to_string(error));
    }
    void bind_numa() {
        if (numa_node >= 0 && fns->numa_run_on_node(numa_node) == 0)
            numa_set_preferred(numa_node);
    }
    void read(const request &req, int64_t position, int64_t length,
              char *bounce, cudaStream_t stream) {
        const int64_t offset = req.offset + position;
        int64_t source_skip = 0;
        if (use_mmap) {
#ifdef _MSC_VER
            constexpr int granularity = 65536;
#else
            constexpr int granularity = ALIGN;
#endif
            const int64_t map_offset = offset - offset % granularity;
            const size_t map_length = length + offset - map_offset;
            void *source = mmap(nullptr, map_length, PROT_READ, MAP_PRIVATE, req.fd, map_offset);
            if (source == MAP_FAILED) throw std::runtime_error("nogds mmap failed");
            std::memcpy(bounce, static_cast<char *>(source) + offset - map_offset, length);
            munmap(source, map_length);
        } else {
            int64_t read_offset = offset, read_length = length;
#ifndef _MSC_VER
            if ((fcntl(req.fd, F_GETFL) & O_DIRECT) != 0) {
                source_skip = offset % ALIGN;
                read_offset -= source_skip;
                read_length = ((length + source_skip + ALIGN - 1) / ALIGN) * ALIGN;
            }
#endif
            int64_t got;
            do {
#ifdef _MSC_VER
                std::lock_guard<std::mutex> file_lock(file_mutex);
#endif
                got = pread(req.fd, bounce, read_length, read_offset);
            } while (got < 0 && errno == EINTR);
            // The aligned final read may extend past EOF; all requested tensor
            // bytes must nevertheless be present.
            if (got < length + source_skip)
                throw std::runtime_error("nogds read failed or truncated input");
        }
        void *destination = reinterpret_cast<void *>(req.destination + position);
        if (streams) {
            check(fns->cudaMemcpyAsync(destination, bounce + source_skip, length, cudaMemcpyHostToDevice, stream));
        } else {
            check(fns->cudaMemcpy(destination, bounce + source_skip, length, cudaMemcpyHostToDevice));
            check(fns->cudaDeviceSynchronize());
        }
    }
    void complete(const std::shared_ptr<request> &req, uint64_t position, bool failed) {
        std::lock_guard<std::mutex> lock(mutex);
        req->failed |= failed;
        const uint64_t old_prefix = req->prefix;
        if (!req->completed.empty() && !failed) {
            req->completed[position / block_size] = 1;
            // Workers can finish out of order. Publish only a contiguous prefix
            // whose DMA has completed, never merely scheduled bytes.
            while (req->prefix < req->completed.size() && req->completed[req->prefix])
                ++req->prefix;
        }
        --req->remaining;
        if (req->remaining == 0 || req->failed || old_prefix != req->prefix)
            done.notify_all();
    }
    void worker(size_t index) {
        bind_numa();
        const cudaError_t device_error = fns->cudaSetDevice(device);
        std::vector<std::shared_ptr<request>> pending(buffers_per_thread);
        std::vector<uint64_t> pending_positions(buffers_per_thread);
        const size_t first_slot = index * buffers_per_thread;
        size_t slot = 0;
        auto drain = [&](size_t i) {
            if (!pending[i]) return;
            bool failed = fns->cudaStreamSynchronize(copy_streams[first_slot + i]) != cudaSuccess;
            complete(pending[i], pending_positions[i], failed);
            pending[i].reset();
        };
        for (;;) {
            std::shared_ptr<request> req;
            int64_t position, length;
            {
                std::unique_lock<std::mutex> lock(mutex);
                if (queue.empty()) {
                    // Finish pending DMA before sleeping, so wait_read can finish.
                    lock.unlock();
                    for (size_t i = 0; i < buffers_per_thread; ++i) drain(i);
                    lock.lock();
                }
                ready.wait(lock, [&] { return stopping || !queue.empty(); });
                if (queue.empty()) break;
                req = queue.front();
                position = req->next;
                length = std::min<int64_t>(block_size, req->length - position);
                req->next += length;
                if (req->next == req->length) queue.pop_front();
            }
            drain(slot);
            bool failed = false;
            try {
                check(device_error);
                const size_t i = first_slot + slot;
                read(*req, position, length, static_cast<char *>(allocation) + buffer_stride * i, copy_streams[i]);
            } catch (const std::exception &error) {
                // Drain any queued DMA before either source or destination reuse.
                if (streams) fns->cudaStreamSynchronize(copy_streams[first_slot + slot]);
                else fns->cudaDeviceSynchronize();
                std::fprintf(stderr, "%s\n", error.what());
                failed = true;
            }
            if (streams && !failed) {
                pending[slot] = req;
                pending_positions[slot] = position;
            } else complete(req, position, failed);
            slot = (slot + 1) % buffers_per_thread;
        }
    }
    ~state() {
        { std::lock_guard<std::mutex> lock(mutex); stopping = true; }
        ready.notify_all();
        for (auto &worker : workers) if (worker.joinable()) worker.join();
        fns->cudaSetDevice(device);
        if (streams) for (auto stream : copy_streams) fns->cudaStreamDestroy(stream);
        if (allocation) {
            fns->cudaFreeHost(allocation);
            mc.bounce_buffer_bytes -= allocation_size;
        }
    }
};

nogds_file_reader::nogds_file_reader(bool use_mmap, uint64_t bbuf_size_kb,
        uint64_t max_threads, bool use_cuda, int device_id, int numa_node)
        : _state(new state) {
    state &s = *_state;
    s.fns = use_cuda ? &cuda_fns : &cpu_fns;
    s.device = device_id;
    s.numa_node = use_cuda && numa_set_preferred && s.fns->numa_run_on_node ? numa_node : -1;
    s.streams = s.fns->cudaMemcpyAsync && s.fns->cudaStreamCreateWithFlags
        && s.fns->cudaStreamSynchronize && s.fns->cudaStreamDestroy;
    if (bbuf_size_kb == 0 || max_threads == 0)
        throw std::invalid_argument("bounce buffer size and thread count must be positive");
    if (bbuf_size_kb > (1ULL << 30) || max_threads > 1024)
        throw std::invalid_argument("nogds buffer size or thread count is too large");
    const uint64_t slots = max_threads * state::buffers_per_thread;
    s.block_size = ((bbuf_size_kb + slots - 1) / slots) * 1024;
    s.use_mmap = use_mmap;
    state::check(s.fns->cudaSetDevice(device_id));
    s.buffer_stride = ((s.block_size + ALIGN - 1) / ALIGN) * ALIGN + 2 * ALIGN;
    s.allocation_size = s.buffer_stride * slots;
    // Allocate on a temporary GPU-local thread without changing the caller's
    // CPU affinity or memory policy. Prefer the node, allowing OOM fallback.
    cudaError_t allocation_error = cudaSuccess;
    auto allocate = [&] {
        allocation_error = s.fns->cudaSetDevice(device_id);
        if (allocation_error == cudaSuccess)
            allocation_error = s.fns->cudaHostAlloc(&s.allocation, s.allocation_size, 0);
    };
    if (s.numa_node >= 0) {
        std::thread allocator([&] { s.bind_numa(); allocate(); });
        allocator.join();
    } else {
        allocate();
    }
    state::check(allocation_error);
    mc.bounce_buffer_bytes += s.allocation_size;
    for (uint64_t i = 0; i < slots; ++i) {
        cudaStream_t stream = nullptr;
        if (s.streams) state::check(s.fns->cudaStreamCreateWithFlags(&stream, 1));
        s.copy_streams.push_back(stream);
    }
    for (uint64_t i = 0; i < max_threads; ++i)
        s.workers.emplace_back([&s, i] { s.worker(i); });
}

const int nogds_file_reader::submit_read(int fd, const gds_device_buffer &dst,
        int64_t offset, int64_t length, uint64_t ptr_off, bool track_progress) {
    if (offset < 0 || length < 0 || offset > std::numeric_limits<int64_t>::max() - length)
        throw std::invalid_argument("invalid read offset or length");
    if (ptr_off > dst.get_length() || static_cast<uint64_t>(length) > dst.get_length() - ptr_off)
        throw std::out_of_range("destination out of bounds");
    state &s = *_state;
    if (s.use_mmap) {
#ifdef _MSC_VER
        struct _stat64 info{};
        if (_fstat64(fd, &info) || offset + length > info.st_size)
#else
        struct stat info{};
        if (fstat(fd, &info) || offset + length > info.st_size)
#endif
            throw std::runtime_error("nogds mmap input is truncated");
    }
    state::check(s.fns->cudaSetDevice(s.device));
    // Fence after framework allocation: cached storage can have a pending clone.
    state::check(s.fns->cudaDeviceSynchronize());
    auto req = std::make_shared<state::request>();
    req->fd = fd; req->destination = dst.get_base_address() + ptr_off;
    req->offset = offset; req->length = length;
    req->remaining = length ? (length - 1) / s.block_size + 1 : 0;
    if (track_progress) req->completed.resize(req->remaining, 0);
    int id;
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        if (s.next_id == std::numeric_limits<int>::max()) throw std::overflow_error("request id overflow");
        id = s.next_id++;
        s.requests[id] = req;
        if (length) s.queue.push_back(req);
    }
    s.ready.notify_all();
    return id;
}

const uint64_t nogds_file_reader::wait_read_prefix(int id, uint64_t length) {
    state &s = *_state;
    std::unique_lock<std::mutex> lock(s.mutex);
    auto req = s.requests.at(id);
    if (length > static_cast<uint64_t>(req->length))
        throw std::out_of_range("read prefix out of bounds");
    if (req->length && req->completed.empty())
        throw std::invalid_argument("read progress was not enabled");
    const uint64_t blocks = length ? (length - 1) / s.block_size + 1 : 0;
    s.done.wait(lock, [&] { return req->failed || req->prefix >= blocks; });
    if (req->failed) throw std::runtime_error("nogds read failed or truncated input");
    return std::min<uint64_t>(req->length, req->prefix * s.block_size);
}

const uintptr_t nogds_file_reader::wait_read(int id) {
    state &s = *_state;
    std::unique_lock<std::mutex> lock(s.mutex);
    auto req = s.requests.at(id);
    s.done.wait(lock, [&] { return req->remaining == 0; });
    s.requests.erase(id);
    return req->failed ? 0 : req->destination;
}

nogds_file_reader::~nogds_file_reader() = default;

raw_gds_file_handle::raw_gds_file_handle(std::string filename, bool o_direct, bool use_cuda) {
    CUfileHandle_t cf_handle;
    CUfileDescr_t cf_descr;
    CUfileError_t err;
    int fd;
    int flags = O_RDONLY;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
#if defined(O_DIRECT)
    if (o_direct) {
        flags |= O_DIRECT;
    }
#endif
    fd = open(filename.c_str(), flags, 0644);
    if (fd < 0) {
        char msg[256];
        std::snprintf(msg, 256, "raw_gds_file_handle: open returned an error = %d", errno);
        throw std::runtime_error(msg);
    }
    std::memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
    cf_descr.handle.fd = fd;
    cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

    _fns = use_cuda ? &cuda_fns: &cpu_fns;

    err = _fns->cuFileHandleRegister(&cf_handle, &cf_descr);
    if (err.err != CU_FILE_SUCCESS) {
        close(fd);
        char msg[256];
        std::snprintf(msg, 256, "raw_gds_file_handle: cuFileHandleRegister returned an error = %d", err.err);
        throw std::runtime_error(msg);
    }
    if (debug_log) {
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::printf("[DEBUG] raw_gds_file_handle: fd=%d, cf_handle=%p, elapsed=%" PRId64 " us\n", fd, cf_handle,
            std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
    }
    this->_cf_handle = cf_handle;
    this->_fd = fd;
}

raw_gds_file_handle::~raw_gds_file_handle() {
    if (this->_cf_handle != 0) {
        _fns->cuFileHandleDeregister(this->_cf_handle);
        if (debug_log) {
            std::printf("[DEBUG] ~raw_gds_file_handle: cuFileHandleDeregister: cf_handle=%p\n", this->_cf_handle);
        }
    }
    if (this->_fd > 0) {
        close(this->_fd);
        if (debug_log) {
            std::printf("[DEBUG] ~raw_gds_file_handle: close: fd=%d\n", this->_fd);
        }
    }
}

// Split cuFile requests across persistent workers, retaining registered bases.
struct gds_file_reader::state {
    struct request {
        gds_file_handle fh;
        gds_device_buffer dst;
        uint64_t offset, length, file_bytes, ptr_off, next = 0, remaining;
        uint64_t prefix = 0;
        std::vector<uint8_t> completed;
        ssize_t bytes = 0;
        bool failed = false;
        request(const gds_file_handle &f, const gds_device_buffer &d,
                uint64_t o, uint64_t l, uint64_t valid, uint64_t p, uint64_t n)
            : fh(f), dst(d), offset(o), length(l), file_bytes(valid),
              ptr_off(p), remaining(n) {}
    };
    ext_funcs_t *fns;
    int device, node, next_id = 1;
    uint64_t block;
    bool stopping = false;
    std::vector<std::thread> workers;
    std::mutex mutex;
#ifdef _MSC_VER
    std::mutex file_mutex;
#endif
    std::condition_variable ready, done;
    std::deque<std::shared_ptr<request>> queue;
    std::map<int, std::shared_ptr<request>> requests;
    void worker() {
        if (node >= 0 && fns->numa_run_on_node(node) == 0 && numa_set_preferred)
            numa_set_preferred(node);
        const bool device_failed = fns->cudaSetDevice(device) != cudaSuccess;
        for (;;) {
            std::shared_ptr<request> r;
            uint64_t position, length, expected;
            {
                std::unique_lock<std::mutex> lock(mutex);
                ready.wait(lock, [&] { return stopping || !queue.empty(); });
                if (queue.empty()) break;
                r = queue.front(); position = r->next;
                length = std::min(block, r->length - position);
                expected = std::min(length, r->file_bytes - position);
                r->next += expected;
                if (r->next == r->file_bytes) queue.pop_front();
            }
            ssize_t bytes = 0;
            bool failed = device_failed;
            void *base = r->dst._get_raw_pointer(r->ptr_off, r->length);
            // Keep the padded I/O length for O_DIRECT, but finish when the
            // expected file bytes arrive rather than reading again past EOF.
            while (!failed && static_cast<uint64_t>(bytes) < expected) {
                const uint64_t buffer_offset = position + bytes;
                ssize_t count;
                if (fns->cuFileRead) {
                    count = fns->cuFileRead(r->fh._get_cf_handle(), base,
                          length - bytes, r->offset + buffer_offset, buffer_offset);
                } else {
#ifdef _MSC_VER
                    std::lock_guard<std::mutex> file_lock(file_mutex);
#endif
                    count = pread(r->fh._get_fd(), static_cast<char *>(base) + buffer_offset,
                          length - bytes, r->offset + buffer_offset);
                }
                if (count <= 0) { failed = true; break; }
                bytes += std::min<uint64_t>(count, expected - bytes);
            }
            {
                std::lock_guard<std::mutex> lock(mutex);
                r->failed |= failed; r->bytes += bytes;
                const uint64_t old_prefix = r->prefix;
                if (!r->completed.empty() && !failed) {
                    // cuFileRead is synchronous: its successful return fences
                    // this block's DMA. Workers may finish out of order.
                    r->completed[position / block] = 1;
                    while (r->prefix < r->completed.size() && r->completed[r->prefix])
                        ++r->prefix;
                }
                if (--r->remaining == 0 || r->failed || old_prefix != r->prefix)
                    done.notify_all();
            }
        }
    }
    ~state() {
        { std::lock_guard<std::mutex> lock(mutex); stopping = true; }
        ready.notify_all();
        for (auto &worker : workers) if (worker.joinable()) worker.join();
    }
};

gds_file_reader::gds_file_reader(int max_threads, bool use_cuda, int device_id,
                               uint64_t block_size, int numa_node)
        : _state(new state) {
    if (max_threads <= 0 || max_threads > 1024 || !block_size)
        throw std::invalid_argument("invalid cuFile pool settings");
    state &s = *_state;
    s.fns = use_cuda ? &cuda_fns : &cpu_fns;
    s.device = device_id; s.node = numa_node; s.block = block_size;
    for (int i = 0; i < max_threads; ++i)
        s.workers.emplace_back([&s] { s.worker(); });
}

gds_file_reader::~gds_file_reader() = default;

const int gds_file_reader::submit_read(const gds_file_handle &fh,
        const gds_device_buffer &dst, uint64_t offset, uint64_t length,
        uint64_t ptr_off, uint64_t file_length, bool track_progress) {
    if (offset > file_length) throw std::out_of_range("file offset out of bounds");
    if (ptr_off > dst.get_length() || length > dst.get_length() - ptr_off)
        throw std::out_of_range("destination out of bounds");
    dst._get_raw_pointer(ptr_off, length);
    const uint64_t file_bytes = std::min(length, file_length - offset);
    state &s = *_state;
    // Protect allocator reuse against framework operations on other streams.
    if (s.fns->cudaSetDevice(s.device) != cudaSuccess ||
        s.fns->cudaDeviceSynchronize() != cudaSuccess)
        throw std::runtime_error("cuFile allocator fence failed");
    auto r = std::make_shared<state::request>(fh, dst, offset, length, file_bytes,
                        ptr_off, file_bytes ? (file_bytes - 1) / s.block + 1 : 0);
    if (track_progress) r->completed.resize(r->remaining, 0);
    int id;
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        if (s.next_id == std::numeric_limits<int>::max())
            throw std::overflow_error("request id overflow");
        id = s.next_id++; s.requests[id] = r;
        if (file_bytes) s.queue.push_back(r);
    }
    s.ready.notify_all();
    return id;
}

const uint64_t gds_file_reader::wait_read_prefix(int id, uint64_t length) {
    state &s = *_state;
    std::unique_lock<std::mutex> lock(s.mutex);
    auto r = s.requests.at(id);
    // EOF padding is allocated for O_DIRECT, but is never readable tensor data.
    if (length > r->file_bytes) throw std::out_of_range("read prefix out of bounds");
    if (r->file_bytes && r->completed.empty())
        throw std::invalid_argument("read progress was not enabled");
    const uint64_t blocks = length ? (length - 1) / s.block + 1 : 0;
    s.done.wait(lock, [&] { return r->failed || r->prefix >= blocks; });
    if (r->failed) throw std::runtime_error("cuFile read failed or truncated input");
    return std::min(r->file_bytes, r->prefix * s.block);
}

const ssize_t gds_file_reader::wait_read(int id) {
    state &s = *_state;
    std::unique_lock<std::mutex> lock(s.mutex);
    auto r = s.requests.at(id);
    s.done.wait(lock, [&] { return r->remaining == 0; });
    s.requests.erase(id);
    return r->failed ? -1 : r->bytes;
}

// --- FGDS (FGDS_LIB) classes ---

fgds_device_buffer::fgds_device_buffer(const uintptr_t dev_ptr, const uint64_t length)
    : _devPtr(dev_ptr), _length(length) {}

uintptr_t fgds_device_buffer::get_base_address() const {
    return _devPtr;
}

uint64_t fgds_device_buffer::get_length() const {
    return _length;
}

fgds_file_handle::fgds_file_handle(std::string filename, bool o_direct, int device_id)
    : _fd(-1), _device_id(device_id) {
    int flags = O_RDONLY;
#if defined(O_DIRECT)
    if (o_direct) {
        flags |= O_DIRECT;
    }
#endif
    _fd = open(filename.c_str(), flags, 0644);
    if (_fd < 0) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
}

fgds_file_handle::~fgds_file_handle() {
    if (_fd >= 0) {
        close(_fd);
        _fd = -1;
    }
}

int fgds_file_handle::get_device_id() const {
    return _device_id;
}

int fgds_file_handle::get_fd() const {
    return _fd;
}

// FGDS reader thread: invokes fgds_read() directly from the global function
// pointers resolved in load_fgds_library(). No wrapper object is passed
// to the thread, so there is no object-lifetime dependency beyond the
// extension module itself.
static void fgds_reader_thread(int thread_id, int fd, int device_id,
                                uintptr_t dev_ptr, uint64_t length,
                                uint64_t offset, uint64_t ptr_off,
                                std::map<int, ssize_t>* results,
                                std::mutex* result_lock) {
    ssize_t count = 0;
    void* devPtr_base = reinterpret_cast<void*>(dev_ptr);

    try {
        fgds_fileid fid;
        fid.fd = fd;
        fid.device_id = device_id;
        count = fgds_read(fid, devPtr_base, ptr_off, length, offset);
        if (count < 0) {
            std::fprintf(stderr, "fgds_file_reader._thread: fgds_read returned an error: count=%zd\n", count);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "fgds_file_reader._thread: exception: %s\n", e.what());
        count = -1;
    }

    std::lock_guard<std::mutex> guard(*result_lock);
    (*results)[thread_id] = count;
}

fgds_file_reader::fgds_file_reader(const int max_threads, int device_id)
    : _max_threads(max_threads), _device_id(device_id), _threads(nullptr), _next_id(0) {
    _threads = new std::thread*[max_threads];
    for (int i = 0; i < max_threads; ++i) {
        _threads[i] = nullptr;
    }
}

fgds_file_reader::~fgds_file_reader() {
    if (_threads) {
        for (int i = 0; i < _max_threads; ++i) {
            if (_threads[i] != nullptr) {
                _threads[i]->join();
                delete _threads[i];
            }
        }
        delete[] _threads;
    }
}

const int fgds_file_reader::submit_read(const fgds_file_handle& fh, const fgds_device_buffer& dst,
                                         const uint64_t offset, const uint64_t length,
                                         const uint64_t ptr_off) {
    int id = _next_id++;
    size_t thread_index = (size_t)(id % _max_threads);
    if (_threads[thread_index] != nullptr) {
        _threads[thread_index]->join();
        delete _threads[thread_index];
    }

    _threads[thread_index] = new std::thread(
        fgds_reader_thread, id, fh.get_fd(), fh.get_device_id(),
        dst.get_base_address(), length, offset, ptr_off, &_results, &_result_lock);

    return id;
}

const ssize_t fgds_file_reader::wait_read(const int id) {
    size_t thread_index = (size_t)(id % _max_threads);
    if (_threads[thread_index] != nullptr) {
        _threads[thread_index]->join();
        delete _threads[thread_index];
        _threads[thread_index] = nullptr;
    }

    std::lock_guard<std::mutex> guard(_result_lock);
    ssize_t ret = _results[id];
    _results.erase(id);
    return ret;
}

cpp_metrics_t get_cpp_metrics() {
    return mc;
}

// Bindings

// Multithreaded O_DIRECT range reader for the unified copier: reads only the
// [starts[i], ends[i]) file runs into a device buffer, placing file byte F at
// gbuf[F - header_len]. Threads split the concatenated owned-byte space so one
// large run + many small runs still spread evenly; thread boundaries mid-run
// align the O_DIRECT offset down and overlapping reads write identical bytes
// (idempotent). Bypasses the page cache (O_DIRECT) and drives NVMe queue depth,
// which buffered mmap+pin cannot. Uses the dlopen'd cuda_fns table (no cudart
// link) -- pinned bounce + sync cudaMemcpy. header_len is the buffer-base offset,
// so a compacted chunk buffer passes its span start instead of the real header.
// Reusable pinned 16MB bounce buffers, shared across dma_load_runs calls (and
// concurrent producers). Chunked loading makes many small calls; recycling the
// pinned buffers avoids a cudaHostAlloc/cudaFreeHost per chunk per thread.
// Allocated portable (cudaHostAllocPortable / hipHostMallocPortable, both 0x1)
// so a buffer first pinned under one device's context stays valid pinned
// memory when a later call targets a different device.
static std::mutex g_pin_mtx;
static std::vector<void *> g_pin_pool;
static const size_t PIN_CHUNK = 16UL << 20;
static const unsigned int PIN_FLAG_PORTABLE = 0x1;

// Range completion for the Unified Memory O_DIRECT reader. Track selected
// bytes rather than whole file blocks so compact, disjoint runs are supported.
class dma_completion {
public:
    dma_completion(size_t base, const std::vector<size_t> &starts,
                   const std::vector<size_t> &ends) : base(base), limit(base) {
        if (starts.size() != ends.size()) throw std::invalid_argument("invalid DMA ranges");
        size_t previous = base;
        for (size_t i = 0; i < starts.size(); ++i) {
            if (starts[i] < previous || ends[i] < starts[i])
                throw std::invalid_argument("invalid DMA ranges");
            previous = ends[i];
            limit = std::max(limit, ends[i]);
        }
        pending.resize(limit > base ? (limit - base - 1) / PIN_CHUNK + 1 : 0, 0);
        for (size_t r = 0; r < starts.size(); ++r)
            visit(starts[r], ends[r], [&](size_t block, size_t bytes) { pending[block] += bytes; });
    }

    void complete(size_t start, size_t end) {
        std::lock_guard<std::mutex> lock(mutex);
        bool ready = false;
        visit(start, end, [&](size_t block, size_t bytes) {
            pending[block] -= bytes;
            ready |= pending[block] == 0;
        });
        // Wait predicates change only when a block becomes fully readable.
        if (ready) cv.notify_all();
    }

    int wait_range(size_t start, size_t end) {
        if (start < base || end < start || end > limit)
            throw std::out_of_range("DMA range out of bounds");
        if (start == end) return 0;
        const size_t first = (start - base) / PIN_CHUNK;
        const size_t last = (end - 1 - base) / PIN_CHUNK;
        std::unique_lock<std::mutex> lock(mutex);
        bool ready = false;
        cv.wait(lock, [&] {
            ready = std::all_of(pending.begin() + first, pending.begin() + last + 1,
                               [](size_t bytes) { return bytes == 0; });
            return done || ready;
        });
        return ready ? 0 : (rc ? rc : -5);
    }

    int finish(int result) {
        std::lock_guard<std::mutex> lock(mutex);
        rc = result;
        if (!rc) for (size_t bytes : pending) if (bytes) { rc = -5; break; }
        done = true;
        cv.notify_all();
        return rc;
    }

private:
    template <typename F> void visit(size_t start, size_t end, F fn) {
        for (size_t p = start; p < end;) {
            size_t block = (p - base) / PIN_CHUNK;
            // Avoid overflowing base + (block + 1) * PIN_CHUNK.
            size_t stop = p + std::min(end - p, PIN_CHUNK - (p - base) % PIN_CHUNK);
            fn(block, stop - p);
            p = stop;
        }
    }
    size_t base, limit;
    std::vector<size_t> pending;
    bool done = false;
    int rc = 0;
    std::mutex mutex;
    std::condition_variable cv;
};

static void *pin_acquire() {
    {
        std::lock_guard<std::mutex> lk(g_pin_mtx);
        if (!g_pin_pool.empty()) {
            void *p = g_pin_pool.back();
            g_pin_pool.pop_back();
            return p;
        }
    }
    void *p = nullptr;
    if (cuda_fns.cudaHostAlloc(&p, PIN_CHUNK, PIN_FLAG_PORTABLE) != cudaSuccess)
        return nullptr;
    return p;
}

static void pin_release(void *p) {
    if (!p) return;
    std::lock_guard<std::mutex> lk(g_pin_mtx);
    g_pin_pool.push_back(p);
}

static int dma_load_runs(uintptr_t gbuf_dev, const std::string &path,
                         size_t header_len,
                         const std::vector<size_t> &starts,
                         const std::vector<size_t> &ends, int nthreads,
                         int device_id,
                         const std::shared_ptr<dma_completion> &completion = nullptr) {
    if (!cuda_fns.cudaHostAlloc || !cuda_fns.cudaMemcpy || !cuda_fns.cudaFreeHost
        || !cuda_fns.cudaDeviceSynchronize) {
        return -10;
    }
    const size_t n_runs = starts.size();
    if (n_runs == 0 || ends.size() != n_runs) return 0;
    size_t total = 0;
    for (size_t i = 0; i < n_runs; i++) total += ends[i] - starts[i];
    if (total == 0) return 0;
    if (nthreads < 1) nthreads = 4;
    if (nthreads > 32) nthreads = 32;

    char *gbuf = reinterpret_cast<char *>(gbuf_dev);
    const size_t CHUNK = 16UL << 20;
    const size_t ALN = 4096UL;
    std::atomic<int> rc{0};
    std::vector<std::thread> threads;

    for (int ti = 0; ti < nthreads; ti++) {
        size_t gbs = (size_t)((double)ti * total / nthreads);
        size_t gbe = (ti == nthreads - 1) ? total
                                          : (size_t)((double)(ti + 1) * total / nthreads);
        if (gbe <= gbs) continue;
        threads.emplace_back([&, gbs, gbe]() {
            // The current CUDA device is thread-local and defaults to 0 in a
            // fresh thread; select the loader's target before any CUDA call so
            // contexts and copies land on the right device (device_id < 0 =
            // caller doesn't know, e.g. cpu device: leave the default).
            if (device_id >= 0) cuda_fns.cudaSetDevice(device_id);
            void *pinned = pin_acquire();
            if (!pinned) {
                rc = -1;
                return;
            }
            int fd = open(path.c_str(), O_RDONLY | O_DIRECT);
            if (fd < 0) {
                rc = -2;
                pin_release(pinned);
                return;
            }
            cudaStream_t stream = nullptr;
            const bool streams = completion && cuda_fns.cudaMemcpyAsync
                && cuda_fns.cudaStreamCreateWithFlags && cuda_fns.cudaStreamSynchronize
                && cuda_fns.cudaStreamDestroy;
            if (streams && cuda_fns.cudaStreamCreateWithFlags(&stream, 1) != cudaSuccess) {
                rc = -4;
                close(fd);
                pin_release(pinned);
                return;
            }
            size_t cum = 0;
            for (size_t r = 0; r < n_runs && rc.load() == 0; r++) {
                size_t rs = starts[r], re = ends[r], rlen = re - rs;
                size_t b0 = cum, b1 = cum + rlen;  // this run in owned-byte space
                cum = b1;
                size_t ov0 = b0 > gbs ? b0 : gbs;  // overlap with my span
                size_t ov1 = b1 < gbe ? b1 : gbe;
                if (ov0 >= ov1) continue;
                size_t fstart = rs + (ov0 - b0);   // file coords of my portion
                size_t fend = rs + (ov1 - b0);
                size_t astart = fstart & ~(ALN - 1);  // align O_DIRECT offset down
                for (size_t fo = astart; fo < fend; fo += CHUNK) {
                    size_t want = fend - fo;
                    size_t reqlen = (want >= CHUNK) ? CHUNK
                                                    : ((want + ALN - 1) & ~(ALN - 1));
                    ssize_t got = pread(fd, pinned, reqlen, fo);
                    if (got <= 0) { rc = -3; break; }
                    size_t fo_end = fo + (size_t)got;
                    size_t cs = fo > fstart ? fo : fstart;  // copy only [fstart,fend)
                    size_t ce = fo_end < fend ? fo_end : fend;
                    if (cs < ce) {
                        cudaError_t e;
                        if (streams) {
                            e = cuda_fns.cudaMemcpyAsync(
                                gbuf + (cs - header_len), (char *)pinned + (cs - fo),
                                ce - cs, cudaMemcpyHostToDevice, stream);
                            // Fence only this worker's DMA before publishing bytes
                            // or reusing its pinned bounce buffer.
                            cudaError_t sync = cuda_fns.cudaStreamSynchronize(stream);
                            if (e == cudaSuccess) e = sync;
                        } else {
                            e = cuda_fns.cudaMemcpy(
                                gbuf + (cs - header_len), (char *)pinned + (cs - fo),
                                ce - cs, cudaMemcpyHostToDevice);
                            if (completion) {
                                cudaError_t sync = cuda_fns.cudaDeviceSynchronize();
                                if (e == cudaSuccess) e = sync;
                            }
                        }
                        if (e != cudaSuccess) { rc = -4; break; }
                        if (completion) completion->complete(cs, ce);
                    }
                    if (fo_end >= fend) break;
                }
            }
            // Synchronize on this thread (its current device is the target);
            // the calling thread may have a different device current.
            if (streams) {
                // Every successful copy was fenced before publishing bytes.
                // Drain again only on error before releasing the pinned buffer.
                if (rc.load() != 0 && cuda_fns.cudaStreamSynchronize(stream) != cudaSuccess) rc = -4;
                cuda_fns.cudaStreamDestroy(stream);
            } else if (cuda_fns.cudaDeviceSynchronize() != cudaSuccess) rc = -4;
            close(fd);
            pin_release(pinned);
        });
    }
    for (auto &t : threads) t.join();
    return rc.load();
}

// Async host-to-device memcpy for unified memory copier
static int memcpy_h2d_async(uintptr_t dst, uintptr_t src, size_t size) {
    if (!cuda_fns.cudaMemcpyAsync) {
        return -1;
    }
    cudaError_t err = cuda_fns.cudaMemcpyAsync(
        reinterpret_cast<void *>(dst),
        reinterpret_cast<const void *>(src),
        size,
        cudaMemcpyHostToDevice,
        nullptr  // default stream
    );
    return static_cast<int>(err);
}

PYBIND11_MODULE(__MOD_NAME__, m)
{
#ifdef _MSC_VER
    init_dstorage_bindings(m);
#endif
    // Initialize GIL release setting from environment variable on module load
    init_gil_release_from_env();
    m.def("is_cuda_found", &is_cuda_found);
    m.def("is_hip_found", &is_hip_found);
    m.def("is_cufile_found", &is_cufile_found);
    m.def("cufile_version", &cufile_version);
    m.def("set_debug_log", &set_debug_log);
    m.def("get_alignment_size", &get_alignment_size);
    m.def("is_gds_supported", &is_gds_supported);
    m.def("init_gds", &init_gds);
    m.def("gds_device_cache_size", &gds_device_cache_size);
    m.def("is_hipfile_memory_budget_supported", &is_hipfile_memory_budget_supported);
    m.def("close_gds", &close_gds);
    m.def("is_fgds_found", &is_fgds_found);
    m.def("load_fgds_library", &load_fgds_library);
    m.def("init_fgds", &init_fgds, pybind11::arg("device_id"));
    m.def("close_fgds", &close_fgds, pybind11::arg("device_id"));
    m.def("get_device_pci_bus", &get_device_pci_bus);
    m.def("set_numa_node", &set_numa_node);
    m.def("read_buffer", &read_buffer);
    m.def("cpu_malloc", &cpu_malloc);
    m.def("cpu_free", &cpu_free);
    m.def("gpu_malloc", &gpu_malloc);
    m.def("gpu_free", &gpu_free);
    m.def("load_library_functions", &load_library_functions,
          pybind11::arg("cudart_lib_name") = "");
    m.def("memcpy_h2d_async", &memcpy_h2d_async);
    m.def(
        "dma_load_runs",
        [](uintptr_t gbuf_dev, const std::string &path, size_t header_len,
           const std::vector<size_t> &starts, const std::vector<size_t> &ends,
           int nthreads, int device_id) {
            pybind11::gil_scoped_release release;  // blocking O_DIRECT + DMA
            return dma_load_runs(gbuf_dev, path, header_len, starts, ends,
                                 nthreads, device_id);
        },
        pybind11::arg("gbuf_dev"), pybind11::arg("path"),
        pybind11::arg("header_len"), pybind11::arg("starts"),
        pybind11::arg("ends"), pybind11::arg("nthreads") = 8,
        pybind11::arg("device_id") = -1);
    pybind11::class_<dma_completion, std::shared_ptr<dma_completion>>(m, "dma_completion")
        .def(pybind11::init<size_t, const std::vector<size_t> &, const std::vector<size_t> &>())
        .def("wait_range", &dma_completion::wait_range,
             pybind11::call_guard<pybind11::gil_scoped_release>())
        .def("finish", &dma_completion::finish);
    m.def("dma_load_runs_progress",
        [](uintptr_t ptr, const std::string &path, size_t base,
           const std::vector<size_t> &starts, const std::vector<size_t> &ends,
           int threads, int device, const std::shared_ptr<dma_completion> &completion) {
            if (!completion) throw std::invalid_argument("DMA completion is required");
            pybind11::gil_scoped_release release;
            int rc = dma_load_runs(ptr, path, base, starts, ends, threads, device, completion);
            return completion->finish(rc);
        }, pybind11::arg("gbuf_dev"), pybind11::arg("path"), pybind11::arg("header_len"),
        pybind11::arg("starts"), pybind11::arg("ends"), pybind11::arg("nthreads"),
        pybind11::arg("device_id"), pybind11::arg("completion"));
    m.def("get_cpp_metrics", &get_cpp_metrics);
    m.def("set_gil_release", &set_gil_release);
    m.def("get_gil_release", &get_gil_release);

    pybind11::class_<gds_device_buffer>(m, "gds_device_buffer")
        .def(pybind11::init<const uintptr_t, const uint64_t, bool>())
        // These GPU waits can run on a pipeline producer while the consumer
        // still needs Python to enqueue collective operations. Do not hold
        // the GIL across registration or in-place alignment copies.
        .def("cufile_register", &gds_device_buffer::cufile_register,
             pybind11::call_guard<pybind11::gil_scoped_release>())
        .def("cufile_deregister", &gds_device_buffer::cufile_deregister,
             pybind11::call_guard<pybind11::gil_scoped_release>())
        .def("memmove", &gds_device_buffer::memmove,
             pybind11::call_guard<pybind11::gil_scoped_release>())
        .def("get_base_address", &gds_device_buffer::get_base_address)
        .def("get_length", &gds_device_buffer::get_length);

    // Helper lambdas to conditionally apply GIL release
    auto nogds_submit_read = [](nogds_file_reader& self, const int fd, const gds_device_buffer& dst, const int64_t offset, const int64_t length, const uint64_t ptr_off, bool track_progress) {
        if (enable_gil_release) {
            pybind11::gil_scoped_release release;
            return self.submit_read(fd, dst, offset, length, ptr_off, track_progress);
        } else {
            return self.submit_read(fd, dst, offset, length, ptr_off, track_progress);
        }
    };

    auto nogds_wait_read = [](nogds_file_reader& self, const int thread_id) {
        if (enable_gil_release) {
            pybind11::gil_scoped_release release;
            return self.wait_read(thread_id);
        } else {
            return self.wait_read(thread_id);
        }
    };

    pybind11::class_<nogds_file_reader>(m, "nogds_file_reader")
        .def(pybind11::init<bool, uint64_t, uint64_t, bool, int, int>(),
             pybind11::arg("use_mmap"), pybind11::arg("bbuf_size_kb"),
             pybind11::arg("max_threads"), pybind11::arg("use_cuda"),
             pybind11::arg("device_id"), pybind11::arg("numa_node") = -1)
        .def("submit_read", nogds_submit_read, pybind11::arg("fd"), pybind11::arg("dst"),
             pybind11::arg("offset"), pybind11::arg("length"), pybind11::arg("ptr_off"),
             pybind11::arg("track_progress") = false)
        .def("wait_read_prefix", &nogds_file_reader::wait_read_prefix,
             pybind11::call_guard<pybind11::gil_scoped_release>())
        .def("wait_read", nogds_wait_read);

    pybind11::class_<gds_file_handle>(m, "gds_file_handle")
        .def(pybind11::init<std::string, bool, bool>());

    // Helper lambdas for gds_file_reader to conditionally apply GIL release
    auto gds_submit_read = [](gds_file_reader& self, const gds_file_handle &fh, const gds_device_buffer &dst, const uint64_t offset, const uint64_t length, const uint64_t ptr_off, const uint64_t file_length, bool track_progress) {
        if (enable_gil_release) {
            pybind11::gil_scoped_release release;
            return self.submit_read(fh, dst, offset, length, ptr_off, file_length, track_progress);
        } else {
            return self.submit_read(fh, dst, offset, length, ptr_off, file_length, track_progress);
        }
    };

    auto gds_wait_read = [](gds_file_reader& self, const int id) {
        if (enable_gil_release) {
            pybind11::gil_scoped_release release;
            return self.wait_read(id);
        } else {
            return self.wait_read(id);
        }
    };

    pybind11::class_<gds_file_reader>(m, "gds_file_reader")
        .def(pybind11::init<int, bool, int, uint64_t, int>(),
             pybind11::arg("max_threads"), pybind11::arg("use_cuda"),
             pybind11::arg("device_id"), pybind11::arg("block_size") = 16 * 1024 * 1024,
             pybind11::arg("numa_node") = -1)
        .def("submit_read", gds_submit_read, pybind11::arg("fh"), pybind11::arg("dst"),
             pybind11::arg("offset"), pybind11::arg("length"), pybind11::arg("ptr_off"),
             pybind11::arg("file_length"), pybind11::arg("track_progress") = false)
        .def("wait_read_prefix", &gds_file_reader::wait_read_prefix,
             pybind11::call_guard<pybind11::gil_scoped_release>())
        .def("wait_read", gds_wait_read);

    // FGDS classes. Symbols are resolved at runtime; on platforms without
    // FGDS_LIB these classes still exist but fgds operations return errors.
    pybind11::class_<fgds_file_handle>(m, "fgds_file_handle")
        .def(pybind11::init<std::string, bool, int>())
        .def("get_device_id", &fgds_file_handle::get_device_id)
        .def("get_fd", &fgds_file_handle::get_fd);

    pybind11::class_<fgds_device_buffer>(m, "fgds_device_buffer")
        .def(pybind11::init<const uintptr_t, const uint64_t>())
        .def("get_base_address", &fgds_device_buffer::get_base_address)
        .def("get_length", &fgds_device_buffer::get_length);

    // Helper lambdas for fgds_file_reader to conditionally apply GIL release
    auto fgds_submit_read = [](fgds_file_reader& self, const fgds_file_handle &fh, const fgds_device_buffer &dst, const uint64_t offset, const uint64_t length, const uint64_t ptr_off) {
        if (enable_gil_release) {
            pybind11::gil_scoped_release release;
            return self.submit_read(fh, dst, offset, length, ptr_off);
        } else {
            return self.submit_read(fh, dst, offset, length, ptr_off);
        }
    };

    auto fgds_wait_read = [](fgds_file_reader& self, const int id) {
        if (enable_gil_release) {
            pybind11::gil_scoped_release release;
            return self.wait_read(id);
        } else {
            return self.wait_read(id);
        }
    };

    pybind11::class_<fgds_file_reader>(m, "fgds_file_reader")
        .def(pybind11::init<const int, int>())
        .def("submit_read", fgds_submit_read)
        .def("wait_read", fgds_wait_read);
    // FGDS memory registration helpers. fgds_open/fgds_close are exposed as
    // init_fgds/close_fgds above so device lifetime is explicit.
    m.def("fgds_regmem", [](int device_id, uintptr_t addr, size_t size, pybind11::object /*target_addr*/) {
        if (!fgds_regmem) return -1;
        void* target_addr = nullptr;
        return fgds_regmem(device_id, addr, size, &target_addr);
    }, pybind11::arg("device_id"), pybind11::arg("addr"),
       pybind11::arg("size"), pybind11::arg("target_addr"));

    m.def("fgds_deregmem", [](int device_id, uintptr_t addr, size_t size) {
        if (!fgds_deregmem) return -1;
        return fgds_deregmem(device_id, addr, size);
    }, pybind11::arg("device_id"), pybind11::arg("addr"),
       pybind11::arg("size"));

    pybind11::class_<cpp_metrics_t>(m, "cpp_metrics")
        .def(pybind11::init<>())
        .def_readwrite("bounce_buffer_bytes", &cpp_metrics_t::bounce_buffer_bytes);
}
