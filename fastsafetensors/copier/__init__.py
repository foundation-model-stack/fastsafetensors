# SPDX-License-Identifier: Apache-2.0
import sys

from .base import CopierInterface

if sys.platform == "win32":
    from .dstorage import DStorageFileCopier

from .gds import GdsFileCopier
from .nogds import NoGdsFileCopier
from .registry import (
    CopierConstructFunc,
    CopierType,
    create_copier_constructor,
    register_copier_constructor,
)
from .threefs import ThreeFSFileCopier
from .unified import UnifiedMemCopier
