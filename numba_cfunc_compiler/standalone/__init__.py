"""Standalone container types for Numba (list, dict) with no NRT dependency."""

from numba_cfunc_compiler.standalone.dict import (
    StandaloneDictType,
    standalone_dict_from_voidptr,
    standalone_dict_length,
    standalone_dict_new,
    standalone_dict_to_voidptr,
)
from numba_cfunc_compiler.standalone.list import (
    StandaloneListType,
    standalone_list_from_voidptr,
    standalone_list_length,
    standalone_list_new,
    standalone_list_to_voidptr,
)

__all__ = [
    # List
    "StandaloneListType",
    "standalone_list_new",
    "standalone_list_from_voidptr",
    "standalone_list_to_voidptr",
    "standalone_list_length",
    # Dict
    "StandaloneDictType",
    "standalone_dict_new",
    "standalone_dict_from_voidptr",
    "standalone_dict_to_voidptr",
    "standalone_dict_length",
]
