// SPDX-License-Identifier: Apache-2.0
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include "dstorage.h"
#include <mutex>
#include <string>
#include <vector>
#pragma once
#include "ext.hpp"

namespace py = pybind11;

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "dxguid.lib")

static constexpr UINT32 DS_STAGING_BUFFER_BYTES = 256u * 1024u * 1024u;

static const GUID IID_IDStorageFactory = {
    0x6924ea0c, 0xc3cd, 0x4826,
    {0xb1, 0x0a, 0xf6, 0x4f, 0x4e, 0xd9, 0x27, 0xc1}
};
static const GUID IID_IDStorageFile = {
    0x5de95e7b, 0x955a, 0x4868,
    {0xa7, 0x3c, 0x24, 0x3b, 0x29, 0xf4, 0xb8, 0xda}
};
static const GUID IID_IDStorageQueue = {
    0xcfdbd83f, 0x9e06, 0x4fda,
    {0x8e, 0xa5, 0x69, 0x04, 0x21, 0x37, 0xf4, 0x9b}
};

// DLL loading
typedef HRESULT (__stdcall *PFN_DStorageGetFactory)(REFIID riid, void** ppv);
static PFN_DStorageGetFactory g_pfnGetFactory = nullptr;

typedef HRESULT (__stdcall *PFN_DStorageSetConfiguration1)(DSTORAGE_CONFIGURATION1 const*);
static PFN_DStorageSetConfiguration1 g_pfnSetConfig1 = nullptr;

static bool LoadDirectStorage() {
    HMODULE hMod = LoadLibraryA("dstoragecore.dll");
    if (!hMod) return false;
    hMod = LoadLibraryA("dstorage.dll");
    if (!hMod) return false;
    g_pfnGetFactory = (PFN_DStorageGetFactory)GetProcAddress(hMod, "DStorageGetFactory");
    g_pfnSetConfig1 = (PFN_DStorageSetConfiguration1)GetProcAddress(hMod, "DStorageSetConfiguration1");
    return g_pfnGetFactory != nullptr;
}

// Global state, D3D12 device + DirectStorage factory
class GlobalDStorageState {
    static inline int s_device_id = 0;

public:
    static bool Initialize(int device_id, uintptr_t provided_device, const std::string& cudart_dll) {
        if (s_initialized) return true;
        std::lock_guard<std::mutex> lock(s_mutex);
        if (s_initialized) return true;

        if (!LoadDirectStorage()) {
            last_error_ = "Failed to load dstorage.dll or dstoragecore.dll";
            return false;
        }

        if (!cuda_fns.cudaSetDevice || !cuda_fns.cudaImportExternalMemory ||
            !cuda_fns.cudaExternalMemoryGetMappedBuffer || !cuda_fns.cudaDestroyExternalMemory) {
            last_error_ = "CUDA external memory functions not loaded";
            return false;
        }

        cudaError_t err = cuda_fns.cudaSetDevice(device_id);
        if (err != cudaSuccess) {
            last_error_ = "cudaSetDevice failed: " + std::to_string(err);
            return false;
        }
        s_device_id = device_id;

        if (provided_device) {
            s_device = reinterpret_cast<ID3D12Device*>(provided_device);
            s_device->AddRef();
        } else {
            IDXGIFactory1* factory = nullptr;
            HRESULT hr = CreateDXGIFactory1(IID_IDXGIFactory1, (void**)&factory);
            if (FAILED(hr)) { last_error_ = "CreateDXGIFactory1 failed"; return false; }

            IDXGIAdapter1* adapter = nullptr;
            for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
                DXGI_ADAPTER_DESC1 desc;
                adapter->GetDesc1(&desc);
                if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) { adapter->Release(); continue; }
                break;
            }
            factory->Release();
            if (!adapter) { last_error_ = "No hardware D3D12 adapter found"; return false; }
            HRESULT hr2 = D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_11_0, IID_ID3D12Device, (void**)&s_device);
            adapter->Release();
            if (FAILED(hr2)) { last_error_ = "D3D12CreateDevice failed"; return false; }
        }

        HRESULT hr = g_pfnGetFactory(IID_IDStorageFactory, (void**)&s_factory);
        if (FAILED(hr)) {
            last_error_ = "DStorageGetFactory failed";
            return false;
        }

        hr = s_factory->SetStagingBufferSize(DS_STAGING_BUFFER_BYTES);
        if (FAILED(hr)) {
            last_error_ = "SetStagingBufferSize(" +
                          std::to_string(DS_STAGING_BUFFER_BYTES) +
                          ") failed: hr=0x" + std::to_string(hr);
            return false;
        }

        if (g_pfnSetConfig1) {
            DSTORAGE_CONFIGURATION1 config = {};
            config.NumSubmitThreads = 8;
            g_pfnSetConfig1(&config);
        }

        s_initialized = true;
        return true;
    }

    static ID3D12Device*      GetDevice()       { return s_device; }
    static IDStorageFactory*  GetFactory()      { return s_factory; }
    static int                GetCudaDeviceId() { return s_device_id; }
    static const std::string& LastError()       { return last_error_; }

    static void Shutdown() {
        if (s_factory) { s_factory->Release(); s_factory = nullptr; }
        if (s_device)  { s_device->Release();  s_device  = nullptr; }
        s_initialized = false;
    }

private:
    static inline bool               s_initialized = false;
    static inline ID3D12Device*      s_device      = nullptr;
    static inline IDStorageFactory*  s_factory     = nullptr;
    static inline std::string        last_error_;
    static inline std::mutex         s_mutex;
};

// dstorage_file_handle, wraps IDStorageFile
class dstorage_file_handle {
public:
    bool open(const std::string& path_utf8) {
        int wlen = MultiByteToWideChar(CP_UTF8, 0, path_utf8.c_str(), -1, nullptr, 0);
        std::vector<WCHAR> wpath(wlen);
        MultiByteToWideChar(CP_UTF8, 0, path_utf8.c_str(), -1, wpath.data(), wlen);

        HRESULT hr = GlobalDStorageState::GetFactory()->OpenFile(
            wpath.data(), IID_IDStorageFile, (void**)&file_);
        return SUCCEEDED(hr);
    }

    void close() {
        if (file_) { file_->Close(); file_->Release(); file_ = nullptr; }
    }

    IDStorageFile* get() const { return file_; }

private:
    IDStorageFile* file_ = nullptr;
};

// dstorage_stream_reader, double-buffered DS staging to CUDA copy pipeline
class dstorage_stream_reader {
public:
    static constexpr uint64_t STAGE_SIZE = 64ULL * 1024 * 1024;

    dstorage_stream_reader() {
        auto* dev = GlobalDStorageState::GetDevice();
        auto* factory = GlobalDStorageState::GetFactory();
        if (!dev || !factory) {
            fprintf(stderr, "dstorage_stream_reader: GlobalDStorageState not initialized (dev=%p, factory=%p)\n", dev, factory);
            return;
        }
        cuda_fns.cudaSetDevice(GlobalDStorageState::GetCudaDeviceId());

        // Create the DirectStorage queue first
        DSTORAGE_QUEUE_DESC qdesc = {};
        qdesc.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
        qdesc.Capacity = 8192;
        qdesc.Priority = DSTORAGE_PRIORITY_HIGH;
        qdesc.Name = "fs_stream";
        qdesc.Device = dev;
        HRESULT hr = factory->CreateQueue(
            &qdesc, IID_IDStorageQueue, (void**)&queue_);
        if (FAILED(hr)) {
            fprintf(stderr, "dstorage_stream_reader: CreateQueue failed hr=0x%08X\n", (unsigned)hr);
            return;
        }

        hr = dev->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                              IID_ID3D12Fence, (void**)&fence_);
        if (FAILED(hr)) {
            fprintf(stderr, "dstorage_stream_reader: CreateFence failed hr=0x%08X\n", (unsigned)hr);
            return;
        }

        // Allocate two D3D12 staging buffers with CUDA interop
        D3D12_HEAP_PROPERTIES hp = {};
        hp.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC desc = {};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = STAGE_SIZE;
        desc.Height = 1;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

        for (int i = 0; i < 2; i++) {
            hr = dev->CreateCommittedResource(
                &hp, D3D12_HEAP_FLAG_SHARED, &desc,
                D3D12_RESOURCE_STATE_COMMON, nullptr,
                IID_ID3D12Resource, (void**)&stage_res_[i]);
            if (FAILED(hr)) {
                fprintf(stderr, "dstorage_stream_reader: CreateCommittedResource[%d] failed hr=0x%08X\n", i, (unsigned)hr);
                return;
            }

            hr = dev->CreateSharedHandle(stage_res_[i], nullptr, GENERIC_ALL,
                                         nullptr, &stage_handle_[i]);
            if (FAILED(hr)) {
                fprintf(stderr, "dstorage_stream_reader: CreateSharedHandle[%d] failed\n", i);
                return;
            }

            cudaExternalMemoryHandleDesc emd = {};
            emd.type = cudaExternalMemoryHandleTypeD3D12Resource;
            emd.handle.win32.handle = stage_handle_[i];
            emd.size = STAGE_SIZE;
            emd.flags = cudaExternalMemoryDedicated;
            cudaError_t cerr = cuda_fns.cudaImportExternalMemory(&stage_ext_mem_[i], &emd);
            if (cerr != cudaSuccess) {
                fprintf(stderr, "dstorage_stream_reader: cudaImportExternalMemory[%d] failed err=%d\n", i, cerr);
                return;
            }

            cudaExternalMemoryBufferDesc ebd = {};
            ebd.size = STAGE_SIZE;
            cerr = cuda_fns.cudaExternalMemoryGetMappedBuffer(&stage_cuda_ptr_[i], stage_ext_mem_[i], &ebd);
            if (cerr != cudaSuccess) {
                fprintf(stderr, "dstorage_stream_reader: cudaExternalMemoryGetMappedBuffer[%d] failed err=%d\n", i, cerr);
                return;
            }
        }

        ready_ = true;
    }

    ~dstorage_stream_reader() { close(); }

    bool is_ready() const { return ready_; }

    int64_t read_to_cuda(dstorage_file_handle& fh,
                         uintptr_t dst_cuda_ptr,
                         uint64_t file_offset,
                         uint64_t total_bytes) {
        if (!ready_) return -1;

        char* dst = reinterpret_cast<char*>(dst_cuda_ptr);
        uint64_t remaining = total_bytes;
        uint64_t src_off = file_offset;
        uint64_t dst_off = 0;
        int cur = 0;

        bool     prev_pending = false;
        int      prev_buf     = 0;
        uint64_t prev_size    = 0;
        uint64_t prev_dst_off = 0;
        uint64_t prev_fence   = 0;

        while (remaining > 0 || prev_pending) {
            // Kick off DS read for current chunk
            uint64_t chunk = 0;
            uint64_t cur_fence = 0;
            if (remaining > 0) {
                chunk = (remaining > STAGE_SIZE) ? STAGE_SIZE : remaining;

                DSTORAGE_REQUEST req = {};
                req.Options.CompressionFormat = DSTORAGE_COMPRESSION_FORMAT_NONE;
                req.Options.SourceType        = DSTORAGE_REQUEST_SOURCE_FILE;
                req.Options.DestinationType   = DSTORAGE_REQUEST_DESTINATION_BUFFER;
                req.Source.File.Source  = fh.get();
                req.Source.File.Offset = src_off;
                req.Source.File.Size   = static_cast<uint32_t>(chunk);
                req.Destination.Buffer.Resource = stage_res_[cur];
                req.Destination.Buffer.Offset   = 0;
                req.Destination.Buffer.Size     = static_cast<uint32_t>(chunk);
                req.UncompressedSize            = static_cast<uint32_t>(chunk);

                queue_->EnqueueRequest(&req);
                cur_fence = ++fence_val_;
                queue_->EnqueueSignal(fence_, cur_fence);
                queue_->Submit();

                src_off   += chunk;
                remaining -= chunk;
            }

            // While DS reads into cur, copy previous staging to final CUDA
            if (prev_pending) {
                wait_fence_internal(prev_fence);

                HANDLE errEvent = queue_->GetErrorEvent();
                if (errEvent && WaitForSingleObject(errEvent, 0) == WAIT_OBJECT_0) {
                    DSTORAGE_ERROR_RECORD rec = {};
                    queue_->RetrieveErrorRecord(&rec);
                    if (rec.FailureCount > 0) {
                        last_hresult_ = rec.FirstFailure.HResult;
                        return -2;
                    }
                }

                cudaError_t cerr = cuda_fns.cudaMemcpy(
                    dst + prev_dst_off,
                    stage_cuda_ptr_[prev_buf],
                    prev_size,
                    cudaMemcpyDeviceToDevice);
                if (cerr != cudaSuccess) {
                    fprintf(stderr, "dstorage_stream_reader: cudaMemcpy failed err=%d\n", cerr);
                    return -3;
                }
            }

            // Current becomes previous
            if (chunk > 0) {
                prev_pending = true;
                prev_buf     = cur;
                prev_size    = chunk;
                prev_dst_off = dst_off;
                prev_fence   = cur_fence;
                dst_off     += chunk;
                cur         ^= 1;
            } else {
                prev_pending = false;
            }
        }

        return static_cast<int64_t>(total_bytes);
    }

    int64_t last_hresult() const { return static_cast<int64_t>(last_hresult_); }

    void close() {
        for (int i = 0; i < 2; i++) {
            if (stage_cuda_ptr_[i]) {
                cuda_fns.cudaDestroyExternalMemory(stage_ext_mem_[i]);
                stage_cuda_ptr_[i] = nullptr;
                stage_ext_mem_[i]  = nullptr;
            }
            if (stage_res_[i])    { stage_res_[i]->Release();       stage_res_[i]    = nullptr; }
            if (stage_handle_[i]) { CloseHandle(stage_handle_[i]);  stage_handle_[i] = nullptr; }
        }
        if (queue_) { queue_->Close(); queue_->Release(); queue_ = nullptr; }
        if (fence_) { fence_->Release(); fence_ = nullptr; }
        ready_ = false;
    }

private:
    void wait_fence_internal(uint64_t fval) {
        if (fence_->GetCompletedValue() < fval) {
            HANDLE evt = CreateEventA(nullptr, FALSE, FALSE, nullptr);
            fence_->SetEventOnCompletion(fval, evt);
            WaitForSingleObject(evt, INFINITE);
            CloseHandle(evt);
        }
    }

    ID3D12Resource*      stage_res_[2]      = {};
    HANDLE               stage_handle_[2]   = {};
    cudaExternalMemory_t stage_ext_mem_[2]  = {};
    void*                stage_cuda_ptr_[2] = {};
    IDStorageQueue*      queue_             = nullptr;
    ID3D12Fence*         fence_             = nullptr;
    uint64_t             fence_val_         = 0;
    HRESULT              last_hresult_      = S_OK;
    bool                 ready_             = false;
};

void init_dstorage_bindings(py::module_& m) {
    m.def("init_dstorage", [](int device_id, uintptr_t d3d12_ptr, const std::string& cudart_dll) -> std::string {
        if (GlobalDStorageState::Initialize(device_id, d3d12_ptr, cudart_dll))
            return "ok";
        return GlobalDStorageState::LastError();
    }, py::arg("device_id") = 0, py::arg("d3d12_device_ptr") = 0, py::arg("cudart_dll") = "cudart64_12.dll");

    py::class_<dstorage_file_handle>(m, "dstorage_file_handle")
        .def(py::init<>())
        .def("open", &dstorage_file_handle::open)
        .def("close", &dstorage_file_handle::close);

    py::class_<dstorage_stream_reader>(m, "dstorage_stream_reader")
        .def(py::init<>())
        .def("is_ready", &dstorage_stream_reader::is_ready)
        .def("read_to_cuda", &dstorage_stream_reader::read_to_cuda,
             py::arg("fh"),
             py::arg("dst_cuda_ptr"),
             py::arg("file_offset"),
             py::arg("total_bytes"))
        .def("last_hresult", &dstorage_stream_reader::last_hresult)
        .def("close", &dstorage_stream_reader::close);
}
