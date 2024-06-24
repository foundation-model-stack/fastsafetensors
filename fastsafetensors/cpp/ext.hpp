/*
 * Copyright 2024 IBM Inc. All rights reserved
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __EXT_HPP__
#define __EXT_HPP__

#include <cstdint>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <map>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifndef __MOD_NAME__
#define __MOD_NAME__ fastsafetensors_cpp
#endif

#include <numa.h>

#ifndef NOCUDA

#include <cuda_runtime.h>
#include <cufile.h>

#else

typedef enum CUfileOpError { CU_FILE_SUCCESS=0, CU_FILE_INTERNAL_ERROR=5030 } CUfileOpError;
enum CUfileFileHandleType { CU_FILE_HANDLE_TYPE_OPAQUE_FD = 1 };
typedef void * CUfileHandle_t;
typedef struct CUfileDescr_t { enum CUfileFileHandleType type; union { int fd; }handle; } CUfileDescr_t;
typedef struct CUfileError { CUfileOpError err; }CUfileError_t;
typedef enum cudaError { cudaSuccess = 0, cudaErrorMemoryAllocation = 2 } cudaError_t;
enum cudaMemcpyKind { cudaMemcpyHostToDevice=2, cudaMemcpyDefault = 4 };

#endif

int get_alignment_size();
void set_debug_log(bool _debug_log);
int init_gds(uint64_t _max_direct_io_size_in_kb, uint64_t max_pinned_memory_size_in_kb);
int close_gds();
std::string get_device_pci_bus(int deviceId);
int set_numa_node(int numa_node);
pybind11::bytes read_buffer(uintptr_t _dst, uint64_t length);
uintptr_t cpu_malloc(uint64_t length);
void cpu_free(uintptr_t addr);

class raw_device_pointer {
private:
    const void * _devPtr_base;
public:
    raw_device_pointer(const uintptr_t devPtr_base): _devPtr_base(reinterpret_cast<const void *>(devPtr_base)) {}
    const uintptr_t get_uintptr() const { return reinterpret_cast<const uintptr_t>(this->_devPtr_base); }
    const void * get_raw() const { return this->_devPtr_base; }
};

class gds_device_buffer {
private:
    const std::shared_ptr<const raw_device_pointer> _devPtr_base;
    const uint64_t _length;
public:
    gds_device_buffer(const uintptr_t devPtr_base, const uint64_t length):
        _devPtr_base(std::make_shared<const raw_device_pointer>((devPtr_base))), _length(length) {}
    const int cufile_register(uint64_t offset, uint64_t length);
    const int cufile_deregister(uint64_t offset);
    const int memmove(uint64_t _dst_off, uint64_t _src_off, const gds_device_buffer& _tmp, uint64_t length);
    void * _get_raw_pointer(uint64_t offset, uint64_t length) const { // not exposed to python
        if (this->_length < offset + length) {
            char msg[256];
            snprintf(msg, 256, "out of bound access: 0x%p, length=%ld, request offset=%ld, request length=%ld", this->_devPtr_base->get_raw(), this->_length, offset, length);
            throw std::out_of_range(msg);
        }
        return reinterpret_cast<void *>(this->_devPtr_base->get_uintptr() + offset);
    }
    const uintptr_t get_base_address() const {
        return this->_devPtr_base->get_uintptr();
    }
};

class nogds_file_reader {
private:
    int _next_thread_id;
    std::mutex _mutex;
    std::condition_variable _cond;
    std::thread ** _threads; // TOFIX

    typedef struct thread_states {
        std::mutex _result_mutex;
        std::condition_variable _result_cond;
        std::map<int, void *> _results;
        void * _read_buffer;
        const bool _use_mmap;
        const uint64_t _bbuf_size_kb;
        const uint64_t _max_threads;
    } thread_states_t;
    thread_states_t _s;
public:
    nogds_file_reader(const bool use_mmap, const uint64_t bbuf_size_kb, const uint64_t max_threads):
        _next_thread_id(1), _threads(nullptr), _s(thread_states_t{_read_buffer: nullptr, _use_mmap: use_mmap, _bbuf_size_kb: (bbuf_size_kb + max_threads - 1)/max_threads, _max_threads: max_threads}) {}

    static void _thread(const int thread_id, const int fd, const gds_device_buffer& dst, const int64_t offset, const int64_t length, const uint64_t ptr_off, thread_states_t *s); // not exposed to python
    const int submit_read(const int fd, const gds_device_buffer& dst, const int64_t offset, const int64_t length, const uint64_t ptr_off);
    const uintptr_t wait_read(const int thread_id);
    ~nogds_file_reader();
};

class raw_gds_file_handle {
private:
    int _fd;
    CUfileHandle_t _cf_handle;
public:
    raw_gds_file_handle(std::string filename);
    ~raw_gds_file_handle();
    const CUfileHandle_t get_cf_handle() const { return this->_cf_handle; }
    const int get_fd() const { return this->_fd; }
};

class gds_file_handle {
private:
    std::shared_ptr<const raw_gds_file_handle> _h;
public:
    gds_file_handle(std::string filename): _h(std::make_shared<const raw_gds_file_handle>(filename)) {}
    const CUfileHandle_t _get_cf_handle() const { return this->_h->get_cf_handle(); }
    const int _get_fd() const { return this->_h->get_fd(); }
};

class gds_file_reader {
private:
    int _next_id;
    std::thread ** _threads; // TOFIX
    typedef struct thread_states {
        std::mutex _result_lock;
        std::map<int, ssize_t> _results;
        const int _max_threads;
    } thread_states_t;
    thread_states_t _s;
public:
    gds_file_reader(const int max_threads): _next_id(1), _threads(nullptr), _s(thread_states_t{_max_threads: max_threads}) {}
    static void _thread(const int thread_id, const gds_file_handle &fh, const gds_device_buffer &dst, const uint64_t offset, const uint64_t length, const uint64_t ptr_off, const uint64_t file_length, thread_states_t *s);
    const int submit_read(const gds_file_handle &fh, const gds_device_buffer &dst, const uint64_t offset, const uint64_t length, const uint64_t ptr_off, const uint64_t file_length);
    const ssize_t wait_read(const int id);
};

typedef struct ext_funcs {
    CUfileError_t (*cuFileDriverOpen)();
    CUfileError_t (*cuFileDriverClose)();
    CUfileError_t (*cuFileDriverSetMaxDirectIOSize)(size_t);
    CUfileError_t (*cuFileDriverSetMaxPinnedMemSize)(size_t);
    CUfileError_t (*cuFileBufRegister)(const void *, size_t, int);
    CUfileError_t (*cuFileBufDeregister)(const void *);
    CUfileError_t (*cuFileHandleRegister)(CUfileHandle_t *, CUfileDescr_t *);
    void (*cuFileHandleDeregister)(CUfileHandle_t);
    ssize_t (*cuFileRead)(const gds_file_handle&, void *, size_t, off_t, off_t);
    cudaError_t (*cudaMemcpy)(void *, const void *, size_t, enum cudaMemcpyKind);
    cudaError_t (*cudaDeviceSynchronize)(void);
    cudaError_t (*cudaHostAlloc)(void **, size_t, unsigned int);
    cudaError_t (*cudaFreeHost)(void *);
    cudaError_t (*cudaDeviceGetPCIBusId)(char *, int, int);
    int (*numa_run_on_node)(int);
} ext_funcs_t;

#endif //__EXT_HPP__