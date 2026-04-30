#include <windows.h>
#include <io.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>

#include "mman.h"

#ifndef FILE_MAP_EXECUTE
#define FILE_MAP_EXECUTE 0x0020
#endif

namespace {


int MapMmanError(DWORD err, int /*deferr*/) noexcept
{
    if (err == 0)
        return 0;
    // TODO: implement proper Windows -> errno mapping
    return static_cast<int>(err);
}

DWORD MapMmapProtPage(int prot) noexcept
{
    if (prot == PROT_NONE)
        return 0;

    if ((prot & PROT_EXEC) != 0)
    {
        return ((prot & PROT_WRITE) != 0)
                   ? PAGE_EXECUTE_READWRITE
                   : PAGE_EXECUTE_READ;
    }

    return ((prot & PROT_WRITE) != 0) ? PAGE_READWRITE : PAGE_READONLY;
}

DWORD MapMmapProtFile(int prot) noexcept
{
    if (prot == PROT_NONE)
        return 0;

    DWORD desiredAccess = 0;
    if ((prot & PROT_READ) != 0)
        desiredAccess |= FILE_MAP_READ;
    if ((prot & PROT_WRITE) != 0)
        desiredAccess |= FILE_MAP_WRITE;
    if ((prot & PROT_EXEC) != 0)
        desiredAccess |= FILE_MAP_EXECUTE;

    return desiredAccess;
}

// Split a 64-bit-capable offset into the high/low DWORD pair Win32 expects,
// in a way that avoids "shift count >= width of type" warnings when
// OffsetType is itself only 32 bits wide.
struct DwordPair
{
    DWORD low;
    DWORD high;
};

DwordPair SplitOffset(OffsetType value) noexcept
{
    if constexpr (sizeof(OffsetType) <= sizeof(DWORD))
    {
        return { static_cast<DWORD>(value), 0 };
    }
    else
    {
        const auto u = static_cast<std::uint64_t>(value);
        return {
            static_cast<DWORD>(u & 0xFFFFFFFFu),
            static_cast<DWORD>((u >> 32) & 0xFFFFFFFFu)
        };
    }
}

} // namespace

void* mmap(void* addr, std::size_t len, int prot, int flags, int fildes, OffsetType off)
{
    errno = 0;

    // Reject zero-length and unsupported protection combinations.
    if (len == 0 || prot == PROT_EXEC)
    {
        errno = EINVAL;
        return MAP_FAILED;
    }

    const DWORD protect       = MapMmapProtPage(prot);
    const DWORD desiredAccess = MapMmapProtFile(prot);

    const auto fileOffset = SplitOffset(off);
    const auto maxSize    = SplitOffset(off + static_cast<OffsetType>(len));

    HANDLE h = ((flags & MAP_ANONYMOUS) == 0)
                   ? reinterpret_cast<HANDLE>(_get_osfhandle(fildes))
                   : INVALID_HANDLE_VALUE;

    if ((flags & MAP_ANONYMOUS) == 0 && h == INVALID_HANDLE_VALUE)
    {
        errno = EBADF;
        return MAP_FAILED;
    }

    HANDLE fm = ::CreateFileMapping(h, nullptr, protect, maxSize.high, maxSize.low, nullptr);
    if (fm == nullptr)
    {
        errno = MapMmanError(::GetLastError(), EPERM);
        return MAP_FAILED;
    }

    void* map = ((flags & MAP_FIXED) == 0)
        ? ::MapViewOfFile(fm, desiredAccess, fileOffset.high, fileOffset.low, len)
        : ::MapViewOfFileEx(fm, desiredAccess, fileOffset.high, fileOffset.low, len, addr);

    ::CloseHandle(fm);

    if (map == nullptr)
    {
        errno = MapMmanError(::GetLastError(), EPERM);
        return MAP_FAILED;
    }

    return map;
}

int munmap(void* addr, std::size_t /*len*/)
{
    if (::UnmapViewOfFile(addr))
        return 0;

    errno = MapMmanError(::GetLastError(), EPERM);
    return -1;
}

int _mprotect(void* addr, std::size_t len, int prot)
{
    const DWORD newProtect = MapMmapProtPage(prot);
    DWORD oldProtect = 0;

    if (::VirtualProtect(addr, len, newProtect, &oldProtect))
        return 0;

    errno = MapMmanError(::GetLastError(), EPERM);
    return -1;
}

int msync(void* addr, std::size_t len, int /*flags*/)
{
    if (::FlushViewOfFile(addr, len))
        return 0;

    errno = MapMmanError(::GetLastError(), EPERM);
    return -1;
}

int mlock(const void* addr, std::size_t len)
{
    if (::VirtualLock(const_cast<LPVOID>(addr), len))
        return 0;

    errno = MapMmanError(::GetLastError(), EPERM);
    return -1;
}

int munlock(const void* addr, std::size_t len)
{
    if (::VirtualUnlock(const_cast<LPVOID>(addr), len))
        return 0;

    errno = MapMmanError(::GetLastError(), EPERM);
    return -1;
}