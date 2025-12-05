// SPDX-License-Identifier: Apache-2.0

#include <fcntl.h>
#include <cstring>
#include <unistd.h>
#include <sys/mman.h>
#include <chrono>
#include <dlfcn.h>
#include <cstdlib>
#include <algorithm>

#include "cuda_compat.h"
#include "ext.hpp"

#define ALIGN 4096

static bool debug_log = false;
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
static cudaError_t cpu_cudaDeviceSynchronize() { return cudaSuccess; }
static cudaError_t cpu_cudaHostAlloc(void ** p, size_t length, unsigned int) {
    if (posix_memalign(p, ALIGN, length) < 0) {
        return cudaErrorMemoryAllocation;
    }
    return cudaSuccess;
}
static cudaError_t cpu_cudaFreeHost(void * p) {
    free(p);
    return cudaSuccess;
}
static cudaError_t cpu_cudaDeviceGetPCIBusId(char * in, int s, int) {
    if (s > 0)
        in[0] = 0;
    return cudaSuccess;
}
static int cpu_numa_run_on_node(int) {return 0; }

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
    .cudaDeviceSynchronize = cpu_cudaDeviceSynchronize,
    .cudaHostAlloc = cpu_cudaHostAlloc,
    .cudaFreeHost = cpu_cudaFreeHost,
    .cudaDeviceGetPCIBusId = cpu_cudaDeviceGetPCIBusId,
    .numa_run_on_node = cpu_numa_run_on_node,
};
ext_funcs_t cuda_fns;

static bool cuda_found = false;
static bool is_hip_runtime = false;  // Track if we loaded HIP (not auto-hipified)
static bool cufile_found = false;

static int cufile_ver = 0;

template <typename T> void mydlsym(T** h, void* lib, std::string const& name) {
    *h = reinterpret_cast<T*>(dlsym(lib, name.c_str()));
}

static void load_library_functions() {
    cudaError_t (*cudaGetDeviceCount)(int*);
    const char* cufileLib = "libcufile.so.0";
    const char* cudartLib = GPU_RUNTIME_LIB;
    const char* numaLib = "libnuma.so.1";
    bool init_log = getenv(ENV_ENABLE_INIT_LOG);
    int mode = RTLD_LAZY | RTLD_GLOBAL | RTLD_NODELETE;

    void* handle_numa = dlopen(numaLib, mode);
    if (handle_numa) {
        mydlsym(&cpu_fns.numa_run_on_node, handle_numa, "numa_run_on_node");
        if (cpu_fns.numa_run_on_node) {
            cuda_fns.numa_run_on_node = cpu_fns.numa_run_on_node;
            if (init_log) {
                fprintf(stderr, "[DEBUG] loaded: %s\n", numaLib);
            }
        }
        dlclose(handle_numa);
    }
    if (!cpu_fns.numa_run_on_node) {
        if (init_log) {
            fprintf(stderr, "[DEBUG] %s is not installed. fallback\n", numaLib);
        }
        cpu_fns.numa_run_on_node = cpu_numa_run_on_node;
        cuda_fns.numa_run_on_node = cpu_numa_run_on_node;
    }

    void* handle_cudart = dlopen(cudartLib, mode);
    if (handle_cudart) {
        mydlsym(&cudaGetDeviceCount, handle_cudart, "cudaGetDeviceCount");
        if (cudaGetDeviceCount) {
            int count;
            if (cudaGetDeviceCount(&count) != cudaSuccess) {
                count = 0; // why cudaGetDeviceCount returns non-zero for errors?
            }
            cuda_found = count > 0;
            // Detect if we loaded HIP runtime (ROCm) vs CUDA runtime
            if (cuda_found && std::string(cudartLib).find("hip") != std::string::npos) {
                is_hip_runtime = true;
            }
            if (init_log) {
                fprintf(stderr, "[DEBUG] device count=%d, cuda_found=%d, is_hip_runtime=%d\n", count, cuda_found, is_hip_runtime);
            }
        } else {
            cuda_found = false;
            if (init_log) {
                fprintf(stderr, "[DEBUG] No cudaGetDeviceCount, fallback!\n");
            }
        }
        if (cuda_found) {
            mydlsym(&cuda_fns.cudaMemcpy, handle_cudart, "cudaMemcpy");
            mydlsym(&cuda_fns.cudaDeviceSynchronize, handle_cudart, "cudaDeviceSynchronize");
            mydlsym(&cuda_fns.cudaHostAlloc, handle_cudart, "cudaHostAlloc");
            mydlsym(&cuda_fns.cudaFreeHost, handle_cudart, "cudaFreeHost");
            mydlsym(&cuda_fns.cudaDeviceGetPCIBusId, handle_cudart, "cudaDeviceGetPCIBusId");
            mydlsym(&cuda_fns.cudaDeviceMalloc, handle_cudart, "cudaMalloc");
            mydlsym(&cuda_fns.cudaDeviceFree, handle_cudart, "cudaFree");
            mydlsym(&cuda_fns.cudaDriverGetVersion, handle_cudart, "cudaDriverGetVersion");
            mydlsym(&cuda_fns.cudaDeviceGetAttribute, handle_cudart, "cudaDeviceGetAttribute");
            bool success = cuda_fns.cudaMemcpy && cuda_fns.cudaDeviceSynchronize;
            success = success && cuda_fns.cudaHostAlloc && cuda_fns.cudaFreeHost;
            success = success && cuda_fns.cudaDeviceGetPCIBusId && cuda_fns.cudaDeviceMalloc;
            success = success && cuda_fns.cudaDeviceFree && cuda_fns.cudaDriverGetVersion;
            success = success && cuda_fns.cudaDeviceGetAttribute;
            if (!success) {
                cuda_found = false;
                if (init_log) {
                    fprintf(stderr, "[DEBUG] %s does not contain required CUDA functions. fallback\n", cudartLib);
                }
            } else if (init_log) {
                fprintf(stderr, "[DEBUG] loaded: %s\n", cudartLib);
            }
        }
        dlclose(handle_cudart);
    } else if (init_log) {
        fprintf(stderr, "[DEBUG] %s is not installed. fallback\n", cudartLib);
    }
    if (!cuda_found) {
        cuda_fns.cudaMemcpy = cpu_cudaMemcpy;
        cuda_fns.cudaDeviceSynchronize = cpu_cudaDeviceSynchronize;
        cuda_fns.cudaHostAlloc = cpu_cudaHostAlloc;
        cuda_fns.cudaFreeHost = cpu_cudaFreeHost;
        cuda_fns.cudaDeviceGetPCIBusId = cpu_cudaDeviceGetPCIBusId;
    }

    cufile_found = false;
    if (cuda_found) {
        void* handle_cufile = dlopen(cufileLib, mode);
        if (handle_cufile) {
            CUfileError_t (*cuFileGetVersion)(int *);
            mydlsym(&cuFileGetVersion, handle_cufile, "cuFileGetVersion");
            if (cuFileGetVersion) {
                int version;
                CUfileError_t err = cuFileGetVersion(&version);
                if (err.err == CU_FILE_SUCCESS) {
                    cufile_ver = version;
                }
            }
            if (cufile_ver == 0) {
                fprintf(stderr, "[WARN] libcufile.so is loaded but its version is unknown");
            }
            mydlsym(&cuda_fns.cuFileDriverOpen, handle_cufile, "cuFileDriverOpen");
            mydlsym(&cuda_fns.cuFileDriverClose, handle_cufile, "cuFileDriverClose");
            mydlsym(&cuda_fns.cuFileDriverSetMaxDirectIOSize, handle_cufile, "cuFileDriverSetMaxDirectIOSize");
            mydlsym(&cuda_fns.cuFileDriverSetMaxPinnedMemSize, handle_cufile, "cuFileDriverSetMaxPinnedMemSize");
            mydlsym(&cuda_fns.cuFileBufRegister, handle_cufile, "cuFileBufRegister");
            mydlsym(&cuda_fns.cuFileBufDeregister, handle_cufile, "cuFileBufDeregister");
            mydlsym(&cuda_fns.cuFileHandleRegister, handle_cufile, "cuFileHandleRegister");
            mydlsym(&cuda_fns.cuFileHandleDeregister, handle_cufile, "cuFileHandleDeregister");
            mydlsym(&cuda_fns.cuFileRead, handle_cufile, "cuFileRead");
            bool success = cuda_fns.cuFileDriverOpen && cuda_fns.cuFileDriverClose && cuda_fns.cuFileDriverSetMaxDirectIOSize;
            success &= cuda_fns.cuFileDriverSetMaxPinnedMemSize && cuda_fns.cuFileBufRegister && cuda_fns.cuFileBufDeregister;
            success &= cuda_fns.cuFileHandleRegister && cuda_fns.cuFileHandleDeregister && cuda_fns.cuFileRead;
            if (!success) {
                if (init_log) {
                    fprintf(stderr, "[DEBUG] %s does not contain required cuFile functions. fallback\n", cufileLib);
                }
            } else {
                if (init_log) {
                    fprintf(stderr, "[DEBUG] loaded: %s (ver: %d.%d.%d)\n", cufileLib, cufile_ver / 1000, (cufile_ver % 1000) / 10, cufile_ver % 10);
                }
                cufile_found = true;
            }
            dlclose(handle_cufile);
        } else if (init_log) {
            fprintf(stderr, "[DEBUG] %s is not installed. fallback\n", cufileLib);
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

// Note: is_cuda_found gets auto-hipified to is_hip_found on ROCm builds
// So this function will be is_hip_found() after hipification on ROCm
bool is_cuda_found()
{
    return cuda_found;
}

// Separate function that always returns false on ROCm (CUDA not available on ROCm)
// This will be used for the "is_cuda_found" Python export on ROCm builds
bool cuda_not_available()
{
    return false;  // On ROCm, CUDA is never available
}

// Separate function for checking HIP runtime detection (not hipified)
// On CUDA: checks if HIP runtime was detected
// On ROCm: not used (is_cuda_found gets hipified to is_hip_found)
bool check_hip_runtime()
{
    return is_hip_runtime;
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
    cudaError_t err;
    int driverVersion = 0;
    int gdr_support = 1;

    err = cuda_fns.cudaDriverGetVersion(&driverVersion);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "is_gds_supported: cudaDriverGetVersion failed, deviceId=%d, err=%d\n", deviceId, err);
        return -1;
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
    free(p);
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


void nogds_file_reader::_thread(const int thread_id, ext_funcs_t *fns, const int fd, const gds_device_buffer& dst, const int64_t offset, const int64_t length, const uint64_t ptr_off, thread_states_t *s) {
    void * src = nullptr;
    cudaError_t err;
    int64_t count;
    bool failed = false;
    void * buffer = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(s->_read_buffer) + s->_bbuf_size_kb * 1024 * (thread_id % s->_max_threads));

    if (s->_use_mmap) {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        src = mmap(NULL, length, PROT_READ, MAP_PRIVATE, fd, offset);
        if (src == MAP_FAILED) {
            std::printf("nogds_file_reader._thread: mmap(fd=%d, offset=%" PRIu64 ", length=%" PRIu64 ") failed\n", fd, offset, length);
            failed = true;
            goto out;
        }
        if (debug_log) {
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            std::printf("[DEBUG] nogds_file_reader._thread: mmap, fd=%d, offset=%" PRIu64 ", length=%" PRIu64 ", elapsed=%" PRId64 " us\n",
                fd, offset, length, std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
        }
    }
    count = 0;
    while (count < length) {
        int64_t l = length - count;
        int64_t c;
        if (l > (int64_t)(s->_bbuf_size_kb * 1024)) {
            l = (int64_t)(s->_bbuf_size_kb * 1024);
        }
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        if (s->_use_mmap) {
            std::memcpy(buffer, (void *)((uintptr_t)src + count), l);
            c = l;
        } else {
            c = pread(fd, buffer, l, offset + count);
            if (c != l) {
                std::printf("nogds_file_reader._thread failed: pread(fd=%d, buffer=%p, offset=%" PRIu64 ", count=%" PRIi64 ", l=%" PRIi64 "), c=%" PRIi64 "\n", fd, buffer, offset, count, l, c);
                failed = true;
                goto out;
            }
        }
        std::chrono::steady_clock::time_point memcpy_begin = std::chrono::steady_clock::now();
        err = fns->cudaMemcpy(dst._get_raw_pointer(ptr_off + count, c), buffer, c, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::printf("nogds_file_reader._thread: cudaMemcpy(%p, %p, %" PRIi64 ") failed, err=%d\n", dst._get_raw_pointer(ptr_off + count, c), buffer, count, err);
            failed = true;
            goto out;
        } else if (c <= 64 * 1024) {
            fns->cudaDeviceSynchronize();
        }
        count += c;
        if (debug_log) {
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            std::printf("[DEBUG] nogds_file_reader._thread: read (mmap=%d), fd=%d, offset=%" PRIu64 ", count=%" PRIi64 ", c=%" PRIi64 ", copy=%" PRId64 " us, cuda_copy=%" PRId64 " us\n",
                s->_use_mmap, fd, offset, count, c, std::chrono::duration_cast<std::chrono::microseconds>(memcpy_begin - begin).count(), std::chrono::duration_cast<std::chrono::microseconds>(end - memcpy_begin).count());
        }
    }
out:
    {
        std::unique_lock lk(s->_result_mutex);
        if (failed) {
            s->_results[thread_id] = nullptr;
        } else {
            s->_results[thread_id] = dst._get_raw_pointer(ptr_off, length);
        }
        s->_result_cond.notify_one();
    }
    if (s->_use_mmap && src != nullptr) {
        munmap(src, length);
    }
}

const int nogds_file_reader::submit_read(const int fd, const gds_device_buffer& dst, const int64_t offset, const int64_t length, const uint64_t ptr_off)
{
    const int thread_id = this->_next_thread_id++;
    if (this->_threads == nullptr) {
        this->_threads = new std::thread*[this->_s._max_threads];
        for (uint64_t i = 0; i < this->_s._max_threads; ++i) {
            this->_threads[i] = nullptr;
        }
    }
    if (this->_s._read_buffer == nullptr) {
        cudaError_t err;
        std::chrono::steady_clock::time_point alloc_begin = std::chrono::steady_clock::now();
        auto buf_len = this->_s._bbuf_size_kb * 1024 * this->_s._max_threads;
        err = _fns->cudaHostAlloc(&this->_s._read_buffer, buf_len, 0);
        if (err != cudaSuccess) {
            std::printf("nogds_file_reader.submit_read: cudaHostAlloc(%" PRIi64 ") failed\n", buf_len);
            return -1;
        }
        mc.bounce_buffer_bytes += buf_len;
        if (debug_log) {
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            std::printf("[DEBUG] nogds_file_reader.submit_read: cudaHostAlloc, addr=%p, size=%" PRIi64 ", elapsed=%" PRId64 " us\n",
                reinterpret_cast<void*>(this->_s._read_buffer),
                buf_len, std::chrono::duration_cast<std::chrono::microseconds>(end - alloc_begin).count());
        }
    }
    std::thread *t = this->_threads[thread_id % this->_s._max_threads];
    if (t != nullptr) {
        t->join();
        delete(t);
    }
    t = new std::thread(nogds_file_reader::_thread, thread_id, _fns, fd, dst, offset, length, ptr_off, &this->_s);
    this->_threads[thread_id % this->_s._max_threads] = t;
    if (debug_log) {
        std::printf("[DEBUG] nogds_file_reader.submit_read #3, thread_id=%d\n", thread_id);
    }
    return thread_id;
}

const uintptr_t nogds_file_reader::wait_read(const int thread_id) {
    void * ret;
    {
        std::unique_lock lk(this->_s._result_mutex);
        while(this->_s._results.count(thread_id) == 0) {
            this->_s._result_cond.wait(lk);
        }
        ret = this->_s._results.at(thread_id);
        this->_s._results.erase(thread_id);
    }
    return reinterpret_cast<const uintptr_t>(ret);
}

nogds_file_reader::~nogds_file_reader() {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    if (this->_s._read_buffer != nullptr) {
        auto buf_len = this->_s._bbuf_size_kb * 1024 * this->_s._max_threads;
        _fns->cudaFreeHost(this->_s._read_buffer);
        if (debug_log) {
            std::printf("[DEBUG] cudaFreeHost, addr=%p, size=%" PRIi64 "\n",
                reinterpret_cast<void *>(this->_s._read_buffer), buf_len);
        }
        this->_s._read_buffer = nullptr;
        mc.bounce_buffer_bytes -= buf_len;
    }
    if (this->_threads != nullptr) {
        for (uint64_t i = 0; i < this->_s._max_threads; ++i) {
            std::thread * t = this->_threads[i];
            if (t != nullptr) {
                t->join();
                delete(t);
            }
        }
        delete(this->_threads);
        this->_threads = nullptr;
    }
    if (debug_log) {
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::printf("[DEBUG] ~nogds_file_reader: elapsed=%" PRId64 " us\n",
            std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
    }
}

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

void gds_file_reader::_thread(const int thread_id, ext_funcs_t *fns, const gds_file_handle &fh, const gds_device_buffer &dst, const uint64_t offset, const uint64_t length, const uint64_t ptr_off, const uint64_t file_length, thread_states_t *s) {
    ssize_t count = 0;
    void * devPtr_base = dst._get_raw_pointer(ptr_off, length);
    std::chrono::steady_clock::time_point begin, begin_notify;

    // NOTE: we cannot call register_buffer here since it apparently fails when cuFileRead runs in background.
    begin = std::chrono::steady_clock::now();
    while (uint64_t(count) < length && offset + uint64_t(count) < file_length) {
        ssize_t c;
        if (!fns->cuFileRead) {
            c = pread(fh._get_fd(), reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(devPtr_base) + count), length - count, offset + count);
        } else {
            c = fns->cuFileRead(fh._get_cf_handle(), devPtr_base, length - count, offset + count, count);
        }
        if (debug_log) {
            std::printf("[DEBUG] gds_file_reader._thread: cuFileRead(fh, %p, length=%" PRIu64 ", off=%" PRIu64 ", ptr_off=%" PRIu64 ", count=%zd)=%zd\n", devPtr_base, length, offset, ptr_off, count, c);
        }
        if (c < 0) {
            std::fprintf(stderr, "gds_file_reader._thread: cuFileRead returned an error: errno=%d\n", errno);
            count = -1;
            break;
        } else if (c == 0) {
            break;
        }
        count += size_t(c);
    }
    begin_notify = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> guard(s->_result_lock);
        s->_results.insert(std::make_pair(thread_id, count));
    }
    if (debug_log) {
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::printf("[DEBUG] gds_file_reader._thread: fh=%p, offset=%" PRIu64 ", length=%" PRIu64 ", count=%zd, read=%" PRId64" us, notify=%" PRId64 " us\n",
            fh._get_cf_handle(), offset, length, count,
            std::chrono::duration_cast<std::chrono::microseconds>(begin_notify - begin).count(),
            std::chrono::duration_cast<std::chrono::microseconds>(end - begin_notify).count());
    }
}

const int gds_file_reader::submit_read(const gds_file_handle &fh, const gds_device_buffer &dst, const uint64_t offset, const uint64_t length, const uint64_t ptr_off, const uint64_t file_length) {
    int id;
    std::thread * t;

    id = this->_next_id++;
    size_t thread_index = (size_t)(id % this->_s._max_threads);

    if (this->_threads == nullptr) {
        this->_threads = new std::thread*[this->_s._max_threads];
        for (int i = 0; i < this->_s._max_threads; i++) {
            this->_threads[i] = nullptr;
        }
    }

    t = this->_threads[thread_index];
    if (t != nullptr) {
        // block if we have too many readers
        // NOTE: caller (i.e., python code) runs on a single thread.  so, we do not care about more than two waiters
        t->join();
        delete(t);
    }
    t = new std::thread(_thread, id, _fns, fh, dst, offset, length, ptr_off, file_length, &this->_s);
    this->_threads[thread_index] = t;
    return id;
}

const ssize_t gds_file_reader::wait_read(const int id) {
    size_t thread_index = (size_t)(id % this->_s._max_threads);
    if (this->_threads != nullptr) {
        std::thread * t = this->_threads[thread_index];
        if (t != nullptr) {
            t->join();
            delete(t);
            this->_threads[thread_index] = nullptr;
        }
    }
    std::lock_guard<std::mutex> guard(this->_s._result_lock);
    ssize_t ret = this->_s._results.at(id);
    this->_s._results.erase(id);
    return ret;
}

cpp_metrics_t get_cpp_metrics() {
    return mc;
}

// Bindings

PYBIND11_MODULE(__MOD_NAME__, m)
{
    // Initialize GIL release setting from environment variable on module load
    init_gil_release_from_env();
    // Export both is_cuda_found and is_hip_found on all platforms
    // Use string concatenation to prevent hipify from converting the export names
#ifdef USE_ROCM
    // On ROCm after hipify:
    // - is_cuda_found() becomes is_hip_found(), so export it as "is_hip_found"
    // - Export cuda_not_available() as "is_cuda_found" (CUDA not available on ROCm)
    m.def(("is_" "cuda" "_found"), &cuda_not_available);  // Returns false on ROCm
    m.def(("is_" "hip" "_found"), &is_cuda_found);  // hipified to is_hip_found, returns hip status
#else
    // On CUDA:
    // - is_cuda_found() checks for CUDA
    // - check_hip_runtime() checks if HIP runtime was loaded
    m.def(("is_" "cuda" "_found"), &is_cuda_found);
    m.def(("is_" "hip" "_found"), &check_hip_runtime);
#endif
    m.def("is_cufile_found", &is_cufile_found);
    m.def("cufile_version", &cufile_version);
    m.def("set_debug_log", &set_debug_log);
    m.def("get_alignment_size", &get_alignment_size);
    m.def("is_gds_supported", &is_gds_supported);
    m.def("init_gds", &init_gds);
    m.def("close_gds", &close_gds);
    m.def("get_device_pci_bus", &get_device_pci_bus);
    m.def("set_numa_node", &set_numa_node);
    m.def("read_buffer", &read_buffer);
    m.def("cpu_malloc", &cpu_malloc);
    m.def("cpu_free", &cpu_free);
    m.def("gpu_malloc", &gpu_malloc);
    m.def("gpu_free", &gpu_free);
    m.def("load_library_functions", &load_library_functions);
    m.def("get_cpp_metrics", &get_cpp_metrics);
    m.def("set_gil_release", &set_gil_release);
    m.def("get_gil_release", &get_gil_release);

    pybind11::class_<gds_device_buffer>(m, "gds_device_buffer")
        .def(pybind11::init<const uintptr_t, const uint64_t, bool>())
        .def("cufile_register", &gds_device_buffer::cufile_register)
        .def("cufile_deregister", &gds_device_buffer::cufile_deregister)
        .def("memmove", &gds_device_buffer::memmove)
        .def("get_base_address", &gds_device_buffer::get_base_address)
        .def("get_length", &gds_device_buffer::get_length);

    // Helper lambdas to conditionally apply GIL release
    auto nogds_submit_read = [](nogds_file_reader& self, const int fd, const gds_device_buffer& dst, const int64_t offset, const int64_t length, const uint64_t ptr_off) {
        if (enable_gil_release) {
            pybind11::gil_scoped_release release;
            return self.submit_read(fd, dst, offset, length, ptr_off);
        } else {
            return self.submit_read(fd, dst, offset, length, ptr_off);
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
        .def(pybind11::init<const bool, const uint64_t, const int, bool>())
        .def("submit_read", nogds_submit_read)
        .def("wait_read", nogds_wait_read);

    pybind11::class_<gds_file_handle>(m, "gds_file_handle")
        .def(pybind11::init<std::string, bool, bool>());

    // Helper lambdas for gds_file_reader to conditionally apply GIL release
    auto gds_submit_read = [](gds_file_reader& self, const gds_file_handle &fh, const gds_device_buffer &dst, const uint64_t offset, const uint64_t length, const uint64_t ptr_off, const uint64_t file_length) {
        if (enable_gil_release) {
            pybind11::gil_scoped_release release;
            return self.submit_read(fh, dst, offset, length, ptr_off, file_length);
        } else {
            return self.submit_read(fh, dst, offset, length, ptr_off, file_length);
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
        .def(pybind11::init<const int, bool>())
        .def("submit_read", gds_submit_read)
        .def("wait_read", gds_wait_read);

    pybind11::class_<cpp_metrics_t>(m, "cpp_metrics")
        .def(pybind11::init<>())
        .def_readwrite("bounce_buffer_bytes", &cpp_metrics_t::bounce_buffer_bytes);
}
