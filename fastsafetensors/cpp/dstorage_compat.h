// SPDX-License-Identifier: Apache-2.0
/*
 * DirectStorage compatibility layer for fastsafetensors (Windows only)
 *
 * Provides NVMe -> GPU direct loading on Windows via DirectStorage + CUDA interop.
 * This is the Windows equivalent of cuFile/GDS on Linux.
 *
 * Architecture:
 *   NVMe -> DirectStorage -> D3D12 shared buffer -> cudaImportExternalMemory -> CUDA ptr
 *
 * Requirements:
 *   - Windows 10 1909+ (Windows 11 recommended)
 *   - NVMe SSD
 *   - NVIDIA GPU with CUDA support
 *   - DirectStorage runtime (dstorage.dll, dstoragecore.dll)
 *   - D3D12 runtime (d3d12.dll, dxgi.dll)
 */

#ifndef __DSTORAGE_COMPAT_H__
#define __DSTORAGE_COMPAT_H__

#ifdef _MSC_VER

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <mutex>
#include <thread>
#include <map>
#include <condition_variable>
#include <chrono>

// MSVC does not define ssize_t (it is a POSIX type)
#ifndef _SSIZE_T_DEFINED
#define _SSIZE_T_DEFINED
typedef intptr_t ssize_t;
#endif

// ---------------------------------------------------------------------------
// Minimal COM / D3D12 / DXGI / DirectStorage type stubs
//
// We dynamically load everything at runtime, so we only need the struct
// layouts and GUIDs that appear in our code.  Nothing here pulls in the
// Windows SDK D3D12 headers — the build stays header-light.
// ---------------------------------------------------------------------------

// Forward COM helpers
struct IUnknown_vtbl;

struct IUnknown_stub {
    IUnknown_vtbl* lpVtbl;
};

struct IUnknown_vtbl {
    HRESULT(__stdcall* QueryInterface)(IUnknown_stub* This, const IID& riid, void** ppvObject);
    ULONG  (__stdcall* AddRef)(IUnknown_stub* This);
    ULONG  (__stdcall* Release)(IUnknown_stub* This);
};

static inline ULONG safe_release(IUnknown_stub* p) {
    if (p) return p->lpVtbl->Release(p);
    return 0;
}

// ---------------------------------------------------------------------------
// Opaque interface pointers — we cast through void* when calling vtable
// entries loaded at runtime.  This avoids reproducing the full vtable
// layouts for ID3D12Device, IDXGIAdapter, IDStorageFactory, etc.
// ---------------------------------------------------------------------------
typedef void* DSHandle;  // opaque handle for DirectStorage objects

// ---------------------------------------------------------------------------
// DirectStorage enums / structs (minimal subset)
// ---------------------------------------------------------------------------

enum DSTORAGE_REQUEST_SOURCE_TYPE : uint8_t {
    DSTORAGE_REQUEST_SOURCE_FILE   = 0,
    DSTORAGE_REQUEST_SOURCE_MEMORY = 1,
};

enum DSTORAGE_COMPRESSION_FORMAT : uint8_t {
    DSTORAGE_COMPRESSION_FORMAT_NONE     = 0,
    DSTORAGE_COMPRESSION_FORMAT_GDEFLATE = 1,
};

enum DSTORAGE_REQUEST_DESTINATION_TYPE : uint8_t {
    DSTORAGE_REQUEST_DEST_MEMORY                  = 0,
    DSTORAGE_REQUEST_DEST_BUFFER                  = 1,
    DSTORAGE_REQUEST_DEST_TEXTURE_REGION          = 2,
    DSTORAGE_REQUEST_DEST_MULTIPLE_SUBRESOURCES   = 3,
    DSTORAGE_REQUEST_DEST_TILES                    = 4,
};

#pragma pack(push, 8)
struct DSTORAGE_REQUEST_OPTIONS {
    DSTORAGE_COMPRESSION_FORMAT          CompressionFormat;
    DSTORAGE_REQUEST_SOURCE_TYPE         SourceType;
    DSTORAGE_REQUEST_DESTINATION_TYPE    DestinationType;
    uint8_t                              Reserved1;
    uint32_t                             Reserved2;
};

struct DSTORAGE_SOURCE_FILE {
    DSHandle /* IDStorageFile* */ Source;
    uint64_t                     Offset;
    uint32_t                     Size;
};

struct DSTORAGE_SOURCE {
    union {
        DSTORAGE_SOURCE_FILE File;
        struct {
            void*    Source;
            uint32_t Size;
        } Memory;
    };
};

struct DSTORAGE_DESTINATION_MEMORY {
    void*    Buffer;
    uint32_t Size;
};

struct DSTORAGE_DESTINATION_BUFFER {
    DSHandle /* ID3D12Resource* */ Resource;
    uint64_t                      Offset;
    uint32_t                      Size;
};

struct DSTORAGE_DESTINATION {
    union {
        DSTORAGE_DESTINATION_MEMORY Memory;
        DSTORAGE_DESTINATION_BUFFER Buffer;
    };
};

struct DSTORAGE_REQUEST {
    DSTORAGE_REQUEST_OPTIONS Options;
    DSTORAGE_SOURCE          Source;
    DSTORAGE_DESTINATION     Destination;
    uint32_t                 UncompressedSize;
    uint64_t                 CancellationTag;
    const char*              Name;
};

struct DSTORAGE_QUEUE_DESC {
    DSTORAGE_REQUEST_SOURCE_TYPE SourceType;
    uint16_t                     Capacity;
    uint8_t                      Priority;       // DSTORAGE_PRIORITY_NORMAL = 0
    const char*                  Name;
    DSHandle /* ID3D12Device* */ Device;
};

// Status array for completion tracking
struct DSTORAGE_ERROR_RECORD {
    uint32_t FailureCount;
    // ... additional fields exist but we only check FailureCount
};
#pragma pack(pop)

// ---------------------------------------------------------------------------
// Function pointer types for dynamically loaded APIs
// ---------------------------------------------------------------------------

// dstorage.dll
typedef HRESULT (__stdcall *PFN_DStorageGetFactory)(const IID& riid, void** ppv);
typedef HRESULT (__stdcall *PFN_DStorageSetConfiguration)(void* conf);

// We call COM vtable methods indirectly, so we define helper wrappers
// rather than full vtable structs.

// ---------------------------------------------------------------------------
// dstorage_funcs_t — runtime-resolved DirectStorage + D3D12 function pointers
// ---------------------------------------------------------------------------
struct dstorage_funcs_t {
    // dstorage.dll
    PFN_DStorageGetFactory DStorageGetFactory;

    // d3d12.dll
    HRESULT (__stdcall *D3D12CreateDevice)(
        void* pAdapter, int MinimumFeatureLevel, const IID& riid, void** ppDevice);

    // dxgi.dll
    HRESULT (__stdcall *CreateDXGIFactory1)(const IID& riid, void** ppFactory);

    // cuda runtime (external memory interop)
    int (*cudaImportExternalMemory)(void** extMem, const void* desc);
    int (*cudaExternalMemoryGetMappedBuffer)(void** devPtr, void* extMem, const void* bufferDesc);
    int (*cudaDestroyExternalMemory)(void* extMem);
    int (*cudaImportExternalSemaphore)(void** extSem, const void* desc);
    int (*cudaWaitExternalSemaphoresAsync)(void** extSemArray, const void* paramsArray, unsigned int numExtSems, void* stream);
    int (*cudaDestroyExternalSemaphore)(void* extSem);
};

// ---------------------------------------------------------------------------
// dstorage_context — owns the D3D12 device, DXGI, and DirectStorage factory.
//                    One per process, initialized lazily.
// ---------------------------------------------------------------------------
class dstorage_context {
public:
    bool initialize(dstorage_funcs_t* fns, bool init_log);
    void shutdown();
    bool is_ready() const { return _ready; }

    DSHandle get_factory() const { return _factory; }
    DSHandle get_d3d12_device() const { return _device; }

    // Create a shared D3D12 committed resource (BUFFER) + CUDA import
    // Returns: CUDA device pointer, fills out the external memory handle
    void* create_shared_buffer(dstorage_funcs_t* fns, uint64_t size,
                               HANDLE* out_shared_handle,
                               void** out_cuda_ext_mem,
                               bool init_log);

private:
    bool     _ready   = false;
    DSHandle _factory = nullptr;  // IDStorageFactory*
    DSHandle _device  = nullptr;  // ID3D12Device*
    DSHandle _dxgi    = nullptr;  // IDXGIFactory1*
};

// ---------------------------------------------------------------------------
// dstorage_file_handle — wraps IDStorageFile for one safetensor file
// ---------------------------------------------------------------------------
class dstorage_file_handle {
public:
    dstorage_file_handle(const std::string& filename, dstorage_context* ctx);
    ~dstorage_file_handle();

    DSHandle get() const { return _file; }
    const std::string& filename() const { return _filename; }

private:
    DSHandle    _file = nullptr;  // IDStorageFile*
    std::string _filename;
    dstorage_context* _ctx;
};

// ---------------------------------------------------------------------------
// dstorage_staging_buffer — a shared D3D12 buffer with CUDA-mapped pointer
// ---------------------------------------------------------------------------
struct dstorage_staging_buffer {
    DSHandle d3d12_resource  = nullptr; // ID3D12Resource*
    HANDLE   shared_handle   = nullptr; // NT handle for CUDA import
    void*    cuda_ext_mem    = nullptr; // cudaExternalMemory_t
    void*    cuda_device_ptr = nullptr; // mapped CUDA pointer
    uint64_t size            = 0;
};

// ---------------------------------------------------------------------------
// dstorage_file_reader — the Windows equivalent of gds_file_reader
//
// Python interface mirrors gds_file_reader:
//   reader = dstorage_file_reader(device_id, max_threads)
//   tid    = reader.submit_read(file_handle, dst_buffer, offset, length, ptr_off, file_length)
//   result = reader.wait_read(tid)
//
// Internally:
//   1. Enqueues a DirectStorage read from the file into a D3D12 staging buffer
//   2. Signals a D3D12 fence on completion
//   3. Waits on the fence via CUDA external semaphore
//   4. cudaMemcpy from the staging buffer's CUDA-mapped ptr to the final
//      destination (the gds_device_buffer that the caller provides)
// ---------------------------------------------------------------------------

struct dstorage_thread_states_t {
    int                              _max_threads;
    std::mutex                       _result_lock;
    std::map<int, ssize_t>           _results;
    std::condition_variable          _result_cond;
    std::mutex                       _result_mutex;
};

class dstorage_file_reader {
public:
    dstorage_file_reader(int device_id, int max_threads,
                         dstorage_context* ctx, dstorage_funcs_t* fns);
    ~dstorage_file_reader();

    // submit_read enqueues a DirectStorage request and spawns a thread to
    // wait for completion and copy into the final CUDA buffer.
    //
    // Parameters match gds_file_reader::submit_read:
    //   fh          – dstorage_file_handle for the safetensor file
    //   dst         – target gds_device_buffer (CUDA memory)
    //   offset      – byte offset in the file
    //   length      – bytes to read
    //   ptr_off     – byte offset into dst buffer
    //   file_length – total file size (for bounds checking)
    //
    // Returns a thread id used in wait_read.
    int submit_read(const dstorage_file_handle& fh,
                    void* dst_cuda_ptr,
                    uint64_t offset, uint64_t length,
                    uint64_t ptr_off, uint64_t file_length);

    ssize_t wait_read(int thread_id);

private:
    static void _thread(int thread_id,
                        dstorage_file_reader* self,
                        const dstorage_file_handle& fh,
                        void* dst_cuda_ptr,
                        uint64_t offset, uint64_t length,
                        uint64_t ptr_off, uint64_t file_length);

    int                      _device_id;
    int                      _next_id = 0;
    dstorage_context*        _ctx;
    dstorage_funcs_t*        _fns;
    dstorage_thread_states_t _s;

    // Staging buffer pool (one per thread slot to avoid contention)
    dstorage_staging_buffer* _staging = nullptr;
    uint64_t                 _staging_size = 0;

    // DirectStorage queue + fence
    DSHandle                 _queue    = nullptr; // IDStorageQueue*
    DSHandle                 _fence    = nullptr; // ID3D12Fence*
    uint64_t                 _fence_value = 0;

    std::thread**            _threads  = nullptr;

    bool init_staging(uint64_t size_per_slot);
    bool init_queue();
};

// ---------------------------------------------------------------------------
// Top-level init/query functions (called from load_library_functions)
// ---------------------------------------------------------------------------

bool load_dstorage_functions(dstorage_funcs_t* fns, void* handle_cudart, bool init_log);
bool dstorage_available();

#endif // _MSC_VER
#endif // __DSTORAGE_COMPAT_H__