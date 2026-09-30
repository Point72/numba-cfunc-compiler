from dataclasses import dataclass

from numba_cfunc_compiler.types.registry import NumbaTypeRegistry

CONTAINER_STATE_INIT = 0


@dataclass(frozen=True)
class ListTypeMarker:
    """Type marker for NumbaList types used in function signatures and state."""

    element_type: type

    def __post_init__(self):
        allowed = NumbaTypeRegistry.get_list_element_types()
        if self.element_type not in allowed:
            raise TypeError(f"Unsupported List element type: {self.element_type}. Supported: {[t.__name__ for t in allowed]}")


@dataclass(frozen=True)
class DictTypeMarker:
    """Type marker for NumbaDict types used in function signatures and state."""

    key_type: type
    value_type: type

    def __post_init__(self):
        allowed_keys = NumbaTypeRegistry.get_dict_key_types()
        allowed_vals = NumbaTypeRegistry.get_dict_value_types()
        if self.key_type not in allowed_keys:
            raise TypeError(f"Unsupported Dict key type: {self.key_type}. Supported: {[t.__name__ for t in allowed_keys]}")
        if self.value_type not in allowed_vals:
            raise TypeError(f"Unsupported Dict value type: {self.value_type}. Supported: {[t.__name__ for t in allowed_vals]}")
