# SPDX-License-Identifier: Apache-2.0

import threading
from typing import TYPE_CHECKING, Any, Optional

from .common import init_logger
from .st_types import Device

if TYPE_CHECKING:
    from . import cpp as fstcpp
    from .frameworks import FrameworkOpBase

logger = init_logger(__name__)


class SharedDeviceAllocation:
    r"""Reference-counted owner of the device/host allocation backing DLPack tensors.

    A single ``gds_device_buffer`` holds the bytes for every tensor copied from
    one file. Tensors created through DLPack point at addresses *inside* that
    buffer but, historically, did not keep it alive: closing the owning
    ``FilesBufferOnDevice`` freed the buffer even while exported tensors still
    referenced it.

    ``SharedDeviceAllocation`` fixes that by giving the buffer shared ownership.
    References are held by:

    - the :class:`~fastsafetensors.tensor_factory.LazyTensorFactory` that created
      the allocation (the buffer-side reference), and
    - every exported DLPack tensor, through its ``manager_ctx`` (acquired when
      the tensor is created, released by the DLPack deleter when the consumer
      framework destroys the tensor storage).

    The backing memory is freed exactly once, via
    ``framework.free_tensor_memory``, when the final reference is released. This
    makes returned tensors safe to use after ``FilesBufferOnDevice.close()``:
    the physical memory survives until both the buffer and every exported tensor
    are gone.

    All operations are thread-safe. ``release()`` is idempotent past zero and the
    actual free runs outside the lock, so the DLPack deleter (which may run on an
    arbitrary framework thread) never deadlocks against buffer close.
    """

    # Process-wide live-allocation metrics. Count and total bytes of
    # SharedDeviceAllocation instances that currently own unfreed device/host
    # memory. Deterministic (updated under a lock at create/free), so tests and
    # diagnostics can assert exact fastsafetensors-owned memory independently of
    # framework allocator caches. See live_allocation_count()/live_allocation_bytes().
    _metrics_lock = threading.Lock()
    _live_count = 0
    _live_bytes = 0

    def __init__(
        self,
        gbuf: "fstcpp.gds_device_buffer",
        framework: "FrameworkOpBase",
        device: Device,
        owns_memory: bool = True,
    ):
        self._gbuf: Optional["fstcpp.gds_device_buffer"] = gbuf
        self._framework = framework
        self._device = device
        # DummyDeviceBuffer (example copier) wraps no real allocation; freeing it
        # would be meaningless. The factory passes owns_memory=False for it.
        self._owns_memory = owns_memory
        self._nbytes = gbuf.get_length() if owns_memory and gbuf is not None else 0
        self._refcount = 1  # the creating factory holds the initial reference
        self._lock = threading.Lock()
        if self._owns_memory:
            with SharedDeviceAllocation._metrics_lock:
                SharedDeviceAllocation._live_count += 1
                SharedDeviceAllocation._live_bytes += self._nbytes

    def get_base_address(self) -> int:
        """Return the base device address of the backing buffer.

        Exposed for the non-owning I/O path so copiers can address the storage
        without taking ownership. Raises if the memory has already been freed.
        """
        gbuf = self._gbuf
        if gbuf is None:
            raise RuntimeError(
                "SharedDeviceAllocation: backing buffer already released"
            )
        return gbuf.get_base_address()

    def acquire(self) -> "SharedDeviceAllocation":
        """Add a reference. Called for each exported DLPack tensor."""
        with self._lock:
            if self._refcount <= 0:
                raise RuntimeError(
                    "SharedDeviceAllocation: acquire() after the allocation was freed"
                )
            self._refcount += 1
        return self

    def release(self) -> None:
        """Drop a reference; free the backing memory when the last one goes.

        Idempotent once the refcount reaches zero: extra releases are ignored so
        repeated buffer close() calls cannot double-free.
        """
        gbuf_to_free: Optional["fstcpp.gds_device_buffer"] = None
        with self._lock:
            if self._refcount <= 0:
                return
            self._refcount -= 1
            if self._refcount == 0:
                gbuf_to_free = self._gbuf
                self._gbuf = None
        if gbuf_to_free is not None and self._owns_memory:
            self._framework.free_tensor_memory(gbuf_to_free, self._device)
            with SharedDeviceAllocation._metrics_lock:
                SharedDeviceAllocation._live_count -= 1
                SharedDeviceAllocation._live_bytes -= self._nbytes
            logger.debug(
                "SharedDeviceAllocation.release: freed buffer, addr=0x%x",
                gbuf_to_free.get_base_address(),
            )

    @property
    def live(self) -> bool:
        """True while the backing memory has not yet been freed."""
        return self._gbuf is not None

    def refcount(self) -> int:
        """Current reference count. For diagnostics and tests."""
        with self._lock:
            return self._refcount

    def __repr__(self) -> str:
        return (
            f"SharedDeviceAllocation(refcount={self._refcount}, "
            f"live={self.live}, owns_memory={self._owns_memory})"
        )


def live_allocation_count() -> int:
    """Number of fastsafetensors device/host allocations currently unfreed.

    Counts live ``SharedDeviceAllocation`` owners -- memory fastsafetensors
    still holds (either buffer-side or via exported tensors). This is distinct
    from bytes occupied by live framework tensors and from bytes retained in a
    framework allocator cache; returning an allocation to the framework
    allocator decrements this even if the allocator keeps the pages cached.
    """
    with SharedDeviceAllocation._metrics_lock:
        return SharedDeviceAllocation._live_count


def live_allocation_bytes() -> int:
    """Total bytes of fastsafetensors device/host allocations currently unfreed.

    See live_allocation_count() for the accounting boundary.
    """
    with SharedDeviceAllocation._metrics_lock:
        return SharedDeviceAllocation._live_bytes
