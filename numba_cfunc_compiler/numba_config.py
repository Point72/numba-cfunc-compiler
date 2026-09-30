"""Compatibility imports for the former combined configuration module.

New code should import from config, node_api, type_registry, or
compiler_constants according to the symbol's purpose.
"""

from numba_cfunc_compiler import compiler_constants as _compiler_constants
from numba_cfunc_compiler.config import NumbaConfig, get_numba_config, set_numba_config
from numba_cfunc_compiler.node_api import NumbaDict, NumbaList, State, create_new_dict, create_new_list, set_output
from numba_cfunc_compiler.type_registry import NumbaTypeInfo, NumbaTypeRegistry

__all__ = (
    "NumbaConfig",
    "NumbaDict",
    "NumbaList",
    "NumbaTypeInfo",
    "NumbaTypeRegistry",
    "State",
    "create_new_dict",
    "create_new_list",
    "get_numba_config",
    "set_numba_config",
    "set_output",
)

LIFECYCLE_EXECUTE = _compiler_constants.LIFECYCLE_EXECUTE
LIFECYCLE_PARAM_NAME = _compiler_constants.LIFECYCLE_PARAM_NAME
LIFECYCLE_START = _compiler_constants.LIFECYCLE_START
LIFECYCLE_STOP = _compiler_constants.LIFECYCLE_STOP
OUTPUTS_ARRAY_NAME = _compiler_constants.OUTPUTS_ARRAY_NAME
STATE_ARRAY_NAME = _compiler_constants.STATE_ARRAY_NAME
TICKED_OUTPUTS_ARRAY_NAME = _compiler_constants.TICKED_OUTPUTS_ARRAY_NAME
