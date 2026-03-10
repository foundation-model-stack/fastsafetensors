# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Dict

from ..common import SafeTensorsMetadata
from ..frameworks import FrameworkOpBase
from ..st_types import Device
from .base import CopierInterface

CopierType = str
CopierConstructFunc = Callable[
    [SafeTensorsMetadata, Device, FrameworkOpBase], CopierInterface
]
CopierConstructorFactory = Callable[..., CopierConstructFunc]

_copier_registry: Dict[CopierType, CopierConstructorFactory] = {}


def register_copier_constructor(copier_type: CopierType):
    def decorator(factory_func: CopierConstructorFactory) -> CopierConstructorFactory:
        _copier_registry[copier_type] = factory_func
        return factory_func

    return decorator


def create_copier_constructor(
    copier_type: CopierType, device: Device, **kwargs
) -> CopierConstructFunc:
    if copier_type not in _copier_registry:
        raise KeyError(
            f"Copier type '{copier_type}' is not registered. "
            f"Available types: {list(_copier_registry.keys())}"
        )

    factory_func = _copier_registry[copier_type]
    return factory_func(device, **kwargs)
