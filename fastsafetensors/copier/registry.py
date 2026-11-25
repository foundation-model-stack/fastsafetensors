# SPDX-License-Identifier: Apache-2.0

import functools
from typing import Any, Callable, Dict, Optional, Type

from ..common import SafeTensorsMetadata
from ..frameworks import FrameworkOpBase
from ..st_types import Device, DeviceType
from .base import CopierInterface

CopierType = str
CopierConstructFunc = Callable[
    [SafeTensorsMetadata, Device, FrameworkOpBase], CopierInterface
]
CopierConstructorFactory = Callable[..., CopierConstructFunc]

_copier_registry: Dict[CopierType, CopierConstructorFactory] = {}
_copier_class_registry: Dict[CopierType, Type[CopierInterface]] = {}
_copier_init_checks: Dict[CopierType, Callable[..., bool]] = {}
_copier_library_loaders: Dict[CopierType, Callable[..., Optional[bool]]] = {}


def register_copier_constructor(
    copier_type: CopierType,
    copier_class: Optional[Type[CopierInterface]] = None,
    *,
    load_library: Optional[Callable[..., Optional[bool]]] = None,
    init_and_check: Optional[Callable[..., bool]] = None,
):
    """Register a factory for *copier_type*.

    *copier_class* is the ``CopierInterface`` subclass the factory builds.
    Registering it lets callers reach the copier's class-level policy (e.g.
    ``chunk_transient_multiplier``) before any instance exists; see
    ``copier_class_of``.

    ``load_library`` runs before ``init_and_check`` and may return False when
    backend libraries are unavailable. None/True means loading succeeded.
    ``init_and_check`` optionally initializes and checks the requested device/options.
    It must not choose or construct a replacement backend.
    The registered factory only constructs the selected backend. Callers must
    run ``select_copier`` first to load, initialize and resolve the backend.
    """

    def decorator(factory_func: CopierConstructorFactory) -> CopierConstructorFactory:
        @functools.wraps(factory_func)
        def factory(*args: Any, **kwargs: Any) -> CopierConstructFunc:
            construct = factory_func(*args, **kwargs)
            # Preserve the actual copier tag if the constructor already has one.
            if getattr(construct, "copier_class", None) is None:
                try:
                    construct.copier_class = copier_class  # type: ignore[attr-defined]
                except AttributeError:
                    pass  # exotic callable that rejects attributes; tag is optional
            return construct

        _copier_registry[copier_type] = factory
        if load_library is not None:
            _copier_library_loaders[copier_type] = load_library
        else:
            _copier_library_loaders.pop(copier_type, None)
        if init_and_check is not None:
            _copier_init_checks[copier_type] = init_and_check
        else:
            _copier_init_checks.pop(copier_type, None)
        if copier_class is not None:
            _copier_class_registry[copier_type] = copier_class
        return factory

    return decorator


def get_copier_class(copier_type: CopierType) -> Type[CopierInterface]:
    """The ``CopierInterface`` subclass registered for *copier_type*.

    This is the statically requested copier. When a constructor is already in
    hand, prefer ``copier_class_of`` -- a factory may have fallen back to a
    different copier than the type name suggests.

    Falls back to ``CopierInterface`` itself when a factory was registered
    without its class, so class-level policy hits the interface defaults
    (which raise ``NotImplementedError``) rather than a wrong guess.
    """
    return _copier_class_registry.get(copier_type, CopierInterface)


def copier_class_of(construct: CopierConstructFunc) -> Type[CopierInterface]:
    """The ``CopierInterface`` subclass *construct* will actually build.

    Reads the tag applied by ``register_copier_constructor``, so it survives a
    factory delegating to another copier's factory. Unregistered constructors
    fall back to ``CopierInterface`` (whose class-level policy refuses).
    """
    return getattr(construct, "copier_class", None) or CopierInterface


def load_and_init_and_check_copier(
    copier_type: CopierType, device: Device, **kwargs
) -> bool:
    """Load backend libraries, then initialize/check without constructing a reader.

    Backend probes may initialize runtime libraries/drivers. Configuration and
    unexpected initialization errors propagate rather than triggering fallback.
    Registered backends without a probe are assumed available.
    """
    if copier_type not in _copier_registry:
        raise KeyError(f"Copier type '{copier_type}' is not registered")
    loader = _copier_library_loaders.get(copier_type)
    if loader is not None and loader(device, **kwargs) is False:
        return False
    probe = _copier_init_checks.get(copier_type)
    return probe(device, **kwargs) if probe is not None else True


def select_copier(copier_type: CopierType, device: Device, **kwargs) -> CopierType:
    """Resolve the final backend before constructing any copier resources."""
    if load_and_init_and_check_copier(copier_type, device, **kwargs):
        return copier_type
    if copier_type == "gds":
        from .unified import is_unified_memory_system

        if device.type != DeviceType.CPU and is_unified_memory_system(
            kwargs.get("framework")
        ):
            copier_type = "unified"
        else:
            copier_type = "nogds"
    elif copier_type in ("fgds", "dstorage"):
        copier_type = "nogds"
    else:
        raise RuntimeError(f"Copier '{copier_type}' is unavailable")
    if not load_and_init_and_check_copier(copier_type, device, **kwargs):
        raise RuntimeError(f"Copier '{copier_type}' is unavailable")
    return copier_type


def create_copier_constructor(
    copier_type: CopierType, device: Device, **kwargs
) -> CopierConstructFunc:
    """Construct an already selected and initialized backend without fallback."""
    if copier_type not in _copier_registry:
        raise KeyError(
            f"Copier type '{copier_type}' is not registered. "
            f"Available types: {list(_copier_registry.keys())}"
        )

    factory_func = _copier_registry[copier_type]
    return factory_func(device, **kwargs)
