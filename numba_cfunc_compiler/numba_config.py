"""Compatibility imports for the former combined configuration module.

New code should import from config, node_api, type_registry, or
compiler_constants according to the symbol's purpose.
"""

from numba_cfunc_compiler.compiler_constants import (
    LIFECYCLE_EXECUTE as LIFECYCLE_EXECUTE,
    LIFECYCLE_PARAM_NAME as LIFECYCLE_PARAM_NAME,
    LIFECYCLE_START as LIFECYCLE_START,
    LIFECYCLE_STOP as LIFECYCLE_STOP,
    OUTPUTS_ARRAY_NAME as OUTPUTS_ARRAY_NAME,
    STATE_ARRAY_NAME as STATE_ARRAY_NAME,
    TICKED_OUTPUTS_ARRAY_NAME as TICKED_OUTPUTS_ARRAY_NAME,
)
from numba_cfunc_compiler.config import NumbaConfig, get_numba_config, set_numba_config
from numba_cfunc_compiler.node_api import NumbaDict, NumbaList, State, create_new_dict, create_new_list, set_output
from numba_cfunc_compiler.type_registry import NumbaTypeInfo, NumbaTypeRegistry

__all__ = (
    "NumbaDict",
    "NumbaList",
    "NumbaConfig",
    "NumbaTypeInfo",
    "NumbaTypeRegistry",
    "State",
    "create_new_dict",
    "create_new_list",
    "get_numba_config",
    "set_numba_config",
    "set_output",
)
