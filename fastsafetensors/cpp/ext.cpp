/*
 * Copyright 2024 IBM Inc. All rights reserved
 * SPDX-License-Identifier: Apache-2.0
 */

#include <fcntl.h>
#include <cstring>
#include <unistd.h>
#include <sys/mman.h>
#include <chrono>

#include "ext.hpp"

#define ALIGN 4096

/* cpu_mode functions: for tests and debugs */

static CUfileError_t cpu_cuFileDriverOpen() { return CUfileError_t{err: CU_FILE_SUCCESS}; }
static CUfileError_t cpu_cuFileDriverClose() { return CUfileError_t{err: CU_FILE_SUCCESS}; }
static CUfileError_t cpu_cuFileDriverSetMaxDirectIOSize(size_t) { return CUfileError_t{err: CU_FILE_SUCCESS}; }
static CUfileError_t cpu_cuFileDriverSetMaxPinnedMemSize(size_t) { return CUfileError_t{err: CU_FILE_SUCCESS}; }
static CUfileError_t cpu_cuFileBufRegister(const void *, size_t, int) { return CUfileError_t{err: CU_FILE_SUCCESS}; }
static CUfileError_t cpu_cuFileBufDeregister(const void *) { return CUfileError_t{err: CU_FILE_SUCCESS}; }
static CUfileError_t cpu_cuFileHandleBufRegister(CUfileHandle_t * in, CUfileDescr_t *) {
    *in = reinterpret_cast<CUfileHandle_t *>(malloc(sizeof(CUfileHandle_t)));
    if (*in != nullptr) {
        return CUfileError_t{err: CU_FILE_SUCCESS};
    }
    return CUfileError_t{err: CU_FILE_INTERNAL_ERROR};
}
static void cpu_cuFileHandleDeregister(CUfileHandle_t h) {
    free(reinterpret_cast<void *>(h));
}
static ssize_t cpu_cuFileRead(const gds_file_handle& h, void * p, size_t l, off_t o, off_t c) {
    return pread(h._get_fd(), reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(p) + c), l, o);
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
    std::snprintf(in, s, "0000:00:00:00.00");
    return cudaSuccess;
}
static int cpu_numa_run_on_node(int) {return 0; }

#ifndef NOCUDA

static ssize_t ax_cuFileRead(const gds_file_handle& h, void * p, size_t l, off_t o, off_t c) {
    return cuFileRead(h._get_cf_handle(), p, l, o, c);
}

ext_funcs_t fns = ext_funcs_t {
    cuFileDriverOpen: cuFileDriverOpen,
    cuFileDriverClose: cuFileDriverClose,
    cuFileDriverSetMaxDirectIOSize: cuFileDriverSetMaxDirectIOSize,
    cuFileDriverSetMaxPinnedMemSize: cuFileDriverSetMaxPinnedMemSize,
    cuFileBufRegister: cuFileBufRegister,
    cuFileBufDeregister: cuFileBufDeregister,
    cuFileHandleRegister: cuFileHandleRegister,
    cuFileHandleDeregister: cuFileHandleDeregister,
    cuFileRead: ax_cuFileRead,
    cudaMemcpy: cudaMemcpy,
    cudaDeviceSynchronize: cudaDeviceSynchronize,
    cudaHostAlloc: cudaHostAlloc,
    cudaFreeHost: cudaFreeHost,
    cudaDeviceGetPCIBusId: cudaDeviceGetPCIBusId,
    numa_run_on_node: numa_run_on_node,
};

static bool use_cuda = true;

#else
ext_funcs_t fns = ext_funcs_t {
    cuFileDriverOpen: cpu_cuFileDriverOpen,
    cuFileDriverClose: cpu_cuFileDriverClose,
    cuFileDriverSetMaxDirectIOSize: cpu_cuFileDriverSetMaxDirectIOSize,
    cuFileDriverSetMaxPinnedMemSize: cpu_cuFileDriverSetMaxPinnedMemSize,
    cuFileBufRegister: cpu_cuFileBufRegister,
    cuFileBufDeregister: cpu_cuFileBufDeregister,
    cuFileHandleRegister: cpu_cuFileHandleBufRegister,
    cuFileHandleDeregister: cpu_cuFileHandleDeregister,
    cuFileRead: cpu_cuFileRead,
    cudaMemcpy: cpu_cudaMemcpy,
    cudaDeviceSynchronize: cpu_cudaDeviceSynchronize,
    cudaHostAlloc: cpu_cudaHostAlloc,
    cudaFreeHost: cpu_cudaFreeHost,
    cudaDeviceGetPCIBusId: cpu_cudaDeviceGetPCIBusId,
    numa_run_on_node: numa_run_on_node,
};
static bool use_cuda = false;

#endif

bool is_cpu_mode() {
    return !use_cuda;
}

void set_cpu_mode() {
    use_cuda = false;
    fns = ext_funcs_t {
        cuFileDriverOpen: cpu_cuFileDriverOpen,
        cuFileDriverClose: cpu_cuFileDriverClose,
        cuFileDriverSetMaxDirectIOSize: cpu_cuFileDriverSetMaxDirectIOSize,
        cuFileDriverSetMaxPinnedMemSize: cpu_cuFileDriverSetMaxPinnedMemSize,
        cuFileBufRegister: cpu_cuFileBufRegister,
        cuFileBufDeregister: cpu_cuFileBufDeregister,
        cuFileHandleRegister: cpu_cuFileHandleBufRegister,
        cuFileHandleDeregister: cpu_cuFileHandleDeregister,
        cuFileRead: cpu_cuFileRead,
        cudaMemcpy: cpu_cudaMemcpy,
        cudaDeviceSynchronize: cpu_cudaDeviceSynchronize,
        cudaHostAlloc: cpu_cudaHostAlloc,
        cudaFreeHost: cpu_cudaFreeHost,
        cudaDeviceGetPCIBusId: cpu_cudaDeviceGetPCIBusId,
        numa_run_on_node: cpu_numa_run_on_node,
    };
}

static bool debug_log = false;

int get_alignment_size()
{
    return ALIGN;
}

void set_debug_log(bool _debug_log)
{
    debug_log = _debug_log;
}

int init_gds(uint64_t _max_direct_io_size_in_kb, uint64_t max_pinned_memory_size_in_kb)
{
    CUfileError_t err;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    err = fns.cuFileDriverOpen();
    if (err.err != CU_FILE_SUCCESS) {
        std::fprintf(stderr, "init_gds: cuFileDriverOpen returned an error = %d\n", err.err);
        return -1;
    }

    std::chrono::steady_clock::time_point begin_set_dio = std::chrono::steady_clock::now();
    err = fns.cuFileDriverSetMaxDirectIOSize(_max_direct_io_size_in_kb);
    if (err.err != CU_FILE_SUCCESS) {
        std::fprintf(stderr, "init_gds: cuFileDriverGetProperties(%ld) returned an error = %d\n", _max_direct_io_size_in_kb, err.err);
        close_gds();
        return -1;
    }

    std::chrono::steady_clock::time_point begin_set_pin = std::chrono::steady_clock::now();
    err = fns.cuFileDriverSetMaxPinnedMemSize(max_pinned_memory_size_in_kb);
    if (err.err != CU_FILE_SUCCESS) {
        std::fprintf(stderr, "init_gds: cuFileDriverSetMaxPinnedMemSize(%ld) returned an error = %d\n", max_pinned_memory_size_in_kb, err.err);
        close_gds();
        return -1;
    }
    if (debug_log) {
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::printf("[DEBUG] init_gds: cuFileDriverOpen=%ld us, cuFileDriverSetMaxDirectIOSize=%ld us, cuFileDriverSetMaxPinnedMemSize=%ld us, elapsed=%ld us\n",
            std::chrono::duration_cast<std::chrono::microseconds>(begin_set_dio - begin).count(),
            std::chrono::duration_cast<std::chrono::microseconds>(begin_set_pin - begin_set_dio).count(),
            std::chrono::duration_cast<std::chrono::microseconds>(end - begin_set_pin).count(),
            std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
    }
    return 0;
}

int close_gds()
{
    CUfileError_t err;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    err = fns.cuFileDriverClose();
    if (err.err != CU_FILE_SUCCESS) {
        std::fprintf(stderr, "close_gds: cuFileDriverClose returned an error = %d\n", err.err);
        return -1;
    }
    if (debug_log) {
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::printf("[DEBUG] close_gds: cuFileDriverClose, elapsed=%ld us\n",
            std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
    }
    return 0;
}

std::string get_device_pci_bus(int deviceId) {
    cudaError_t err;
    char pciBusId[32];

    std::memset(pciBusId, 0, 32);
    err = fns.cudaDeviceGetPCIBusId(pciBusId, 32, deviceId);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "get_device_pci_bus: cudaDeviceGetPCIBusId failed, deviceId=%d, err=%d\n", deviceId, err);
        return "";
    }
    return std::string(pciBusId);
}

int set_numa_node(int numa_node) {
    if (numa_node >= 0) {
        if (fns.numa_run_on_node(numa_node) != 0) {
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

const int gds_device_buffer::cufile_register(uint64_t offset, uint64_t length) {
    CUfileError_t err;
    void * dst = reinterpret_cast<void*>(this->_devPtr_base->get_uintptr() + offset);

    std::chrono::steady_clock::time_point begin_register = std::chrono::steady_clock::now();
    err = fns.cuFileBufRegister(dst, length, 0);
    if (err.err != CU_FILE_SUCCESS) {
        std::fprintf(stderr, "gds_device_buffer.cufile_register: cuFileBufRegister returned an error = %d\n", err.err);
        return -1;
    }
    if (debug_log) {
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::printf("[DEBUG] gds_device_buffer.cufile_register: addr=%p, offset=%lu, length=%lu, register=%ld us\n", dst, offset, length,
            std::chrono::duration_cast<std::chrono::microseconds>(end - begin_register).count());
    }
    return 0;
}

const int gds_device_buffer::cufile_deregister(uint64_t offset) {
    void * dst = reinterpret_cast<void*>(this->_devPtr_base->get_uintptr() + offset);
    CUfileError_t err;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    err = fns.cuFileBufDeregister(dst);
    if (err.err != CU_FILE_SUCCESS) {
        std::fprintf(stderr, "gds_device_buffer.cufile_deregister: cuFileBufDeregister (%p) returned an error=%d\n", dst, err.err);
        return -1;
    }
    if (debug_log) {
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::printf("[DEBUG] gds_device_buffer.cufile_deregister: addr=%p, offset=%ld, elapsed=%ld us\n", dst, offset,
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
        std::fprintf(stderr, "gds_device_buffer.memmove: length is smaller than request dst_off, tmp.length=%ld, _dst_off=%ld\n", _tmp._length, _dst_off);
        return -1;
    }
    if (this->_length < _src_off) {
        std::fprintf(stderr, "gds_device_buffer.memmove: length is smaller than request dst_off, tmp.length=%ld, _src_off=%ld\n", _tmp._length, _src_off);
        return -1;
    }
    if (_tmp._length < length) {
        std::fprintf(stderr, "gds_device_buffer.memmove: tmp is smaller than request length, tmp.length=%ld, length=%ld\n", _tmp._length, length);
        return -1;
    }
    if (length == 0) {
        return 0;
    }

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    err = fns.cudaMemcpy(tmp, src, length, cudaMemcpyDefault);
    if (err != cudaSuccess) {
        std::printf("gds_device_buffer.memmove: cudaMemcpy[0](tmp=%p, src=%p, length=%ld) failed, err=%d\n", tmp, src, length, err);
        return -1;
    }
    err = fns.cudaMemcpy(dst, tmp, length, cudaMemcpyDefault);
    if (err != cudaSuccess) {
        std::printf("gds_device_buffer.memmove: cudaMemcpy[1](dst=%p, tmp=%p, length=%ld) failed, err=%d\n", dst, tmp, length, err);
        return -1;
    }
    if (debug_log) {
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::printf("[DEBUG] gds_device_buffer.memmove: dst=%p, src=%p, tmp=%p, length=%ld, elapsed=%ld us\n", dst, src, tmp, length,
            std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
    }
    return 0;
}


void nogds_file_reader::_thread(const int thread_id, const int fd, const gds_device_buffer& dst, const int64_t offset, const int64_t length, const uint64_t ptr_off, thread_states_t *s) {
    void * src = nullptr;
    cudaError_t err;
    int64_t count;
    bool failed = false;
    void * buffer = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(s->_read_buffer) + s->_bbuf_size_kb * 1024 * (thread_id % s->_max_threads));

    if (s->_use_mmap) {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        src = mmap(NULL, length, PROT_READ, MAP_PRIVATE, fd, offset);
        if (src == MAP_FAILED) {
            std::printf("nogds_file_reader._thread: mmap(fd=%d, offset=%ld, length=%ld) failed\n", fd, offset, length);
            failed = true;
            goto out;
        }
        if (debug_log) {
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            std::printf("[DEBUG] nogds_file_reader._thread: mmap, fd=%d, offset=%ld, length=%ld, elapsed=%ld us\n",
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
                std::printf("nogds_file_reader._thread failed: pread(fd=%d, buffer=%p, offset=%ld, count=%ld, l=%ld), c=%ld\n", fd, buffer, offset, count, l, c);
                failed = true;
                goto out;
            }
        }
        std::chrono::steady_clock::time_point memcpy_begin = std::chrono::steady_clock::now();
        err = fns.cudaMemcpy(dst._get_raw_pointer(ptr_off + count, c), buffer, c, cudaMemcpyHostToDevice);      
        if (err != cudaSuccess) {
            std::printf("nogds_file_reader._thread: cudaMemcpy(%p, %p, %ld) failed, err=%d\n", dst._get_raw_pointer(ptr_off + count, c), buffer, count, err);
            failed = true;
            goto out;
        } else if (c <= 64 * 1024) {
            fns.cudaDeviceSynchronize();
        }
        count += c;
        if (debug_log) {
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            std::printf("[DEBUG] nogds_file_reader._thread: read (mmap=%d), fd=%d, offset=%ld, count=%ld, c=%ld, copy=%ld us, cuda_copy=%ld us\n",
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
        err = fns.cudaHostAlloc(&this->_s._read_buffer, this->_s._bbuf_size_kb * 1024 * this->_s._max_threads, 0);
        if (err != cudaSuccess) {
            std::printf("nogds_file_reader.submit_read: cudaHostAlloc(%lu) failed\n", this->_s._bbuf_size_kb * 1024 * this->_s._max_threads);
            return -1;
        }
        if (debug_log) {
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            std::printf("[DEBUG] nogds_file_reader.submit_read: cudaHostAlloc, size=%ld, elapsed=%ld us\n",
                this->_s._bbuf_size_kb * 1024, std::chrono::duration_cast<std::chrono::microseconds>(end - alloc_begin).count());
        }
    }
    std::thread *t = this->_threads[thread_id % this->_s._max_threads];
    if (t != nullptr) {
        t->join();
        delete(t);
    }
    t = new std::thread(nogds_file_reader::_thread, thread_id, fd, dst, offset, length, ptr_off, &this->_s);
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
        fns.cudaFreeHost(this->_s._read_buffer);
        this->_s._read_buffer = nullptr;
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
        std::printf("[DEBUG] ~nogds_file_reader: elapsed=%ld us\n",
            std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
    }
}

raw_gds_file_handle::raw_gds_file_handle(std::string filename) {
    CUfileHandle_t cf_handle;
    CUfileDescr_t cf_descr;
    CUfileError_t err;
    int fd;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    fd = open(filename.c_str(), O_RDONLY|O_DIRECT, 0644);
    if (fd < 0) {
        char msg[256];
        std::snprintf(msg, 256, "raw_gds_file_handle: open returned an error = %d", errno);
        throw std::runtime_error(msg);
    }
    std::memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
    cf_descr.handle.fd = fd;
    cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

    err = fns.cuFileHandleRegister(&cf_handle, &cf_descr);
    if (err.err != CU_FILE_SUCCESS) {
        close(fd);
        char msg[256];
        std::snprintf(msg, 256, "raw_gds_file_handle: cuFileHandleRegister returned an error = %d", err.err);
        throw std::runtime_error(msg);
    }
    if (debug_log) {
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::printf("[DEBUG] raw_gds_file_handle: fd=%d, cf_handle=%p, elapsed=%ld us\n", fd, cf_handle,
            std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
    }
    this->_cf_handle = cf_handle;
    this->_fd = fd;
}

raw_gds_file_handle::~raw_gds_file_handle() {
    if (this->_cf_handle != 0) {
        fns.cuFileHandleDeregister(this->_cf_handle);
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

void gds_file_reader::_thread(const int thread_id, const gds_file_handle &fh, const gds_device_buffer &dst, const uint64_t offset, const uint64_t length, const uint64_t ptr_off, const uint64_t file_length, thread_states_t *s) {
    ssize_t count = 0;
    void * devPtr_base = dst._get_raw_pointer(ptr_off, length);
    std::chrono::steady_clock::time_point begin, begin_notify;

    // NOTE: we cannot call register_buffer here since it apparently fails when cuFileRead runs in background.
    begin = std::chrono::steady_clock::now();
    while (uint64_t(count) < length && offset + uint64_t(count) < file_length) {
        ssize_t c;
        c = fns.cuFileRead(fh, devPtr_base, length - count, offset + count, count);
        if (debug_log) {
            std::printf("[DEBUG] gds_file_reader._thread: cuFileRead(%p, %p, length=%lu, off=%lu, ptr_off=%lu, count=%ld)=%ld\n", fh._get_cf_handle(), devPtr_base, length, offset, ptr_off, count, c);
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
        std::printf("[DEBUG] gds_file_reader._thread: fh=%p, offset=%lu, length=%lu, count=%ld, read=%ld us, notify=%ld us\n",
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
    t = new std::thread(_thread, id, fh, dst, offset, length, ptr_off, file_length, &this->_s);
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

// Bindings

PYBIND11_MODULE(__MOD_NAME__, m)
{
    m.def("is_cpu_mode", &is_cpu_mode);
    m.def("set_cpumode", &set_cpu_mode);
    m.def("set_debug_log", &set_debug_log);
    m.def("get_alignment_size", &get_alignment_size);
    m.def("init_gds", &init_gds);
    m.def("close_gds", &close_gds);
    m.def("get_device_pci_bus", &get_device_pci_bus);
    m.def("set_numa_node", &set_numa_node);
    m.def("read_buffer", &read_buffer);
    m.def("cpu_malloc", &cpu_malloc);
    m.def("cpu_free", &cpu_free);

    pybind11::class_<gds_device_buffer>(m, "gds_device_buffer")
        .def(pybind11::init<const uintptr_t, const uint64_t>())
        .def("cufile_register", &gds_device_buffer::cufile_register)
        .def("cufile_deregister", &gds_device_buffer::cufile_deregister)
        .def("memmove", &gds_device_buffer::memmove)
        .def("get_base_address", &gds_device_buffer::get_base_address);

    pybind11::class_<nogds_file_reader>(m, "nogds_file_reader")
        .def(pybind11::init<const bool, const uint64_t, const int>())
        .def("submit_read", &nogds_file_reader::submit_read)
        .def("wait_read", &nogds_file_reader::wait_read);

    pybind11::class_<gds_file_handle>(m, "gds_file_handle")
        .def(pybind11::init<std::string>());

    pybind11::class_<gds_file_reader>(m, "gds_file_reader")
        .def(pybind11::init<const int>())
        .def("submit_read", &gds_file_reader::submit_read)
        .def("wait_read", &gds_file_reader::wait_read);
}
