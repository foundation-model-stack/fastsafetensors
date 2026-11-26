# SPDX-License-Identifier: Apache-2.0
import sys

from .base import CopierInterface

if sys.platform == "win32":
    from .dstorage import DStorageFileCopier

from .fgds import FgdsFileCopier
from .gds import GdsFileCopier
from .nogds import NoGdsFileCopier
from .registry import (
    CopierConstructFunc,
    CopierType,
    copier_class_of,
    create_copier_constructor,
    get_copier_class,
    load_and_init_and_check_copier,
    register_copier_constructor,
    select_copier,
)
from .threefs import ThreeFSFileCopier
from .unified import UnifiedMemCopier
