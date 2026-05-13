import ast
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from numba_cfunc_compiler.numba_config import (
    STATE_ARRAY_NAME,
    NumbaTypeRegistry,
)
from numba_cfunc_compiler.utils.ast import AST

__all__ = [
    # Type markers and sentinels
    "ListTypeMarker",
    "DictTypeMarker",
    "UnknownNumbaType",
    "UnknownNumbaValue",
    "NoneType",
    # Variable types
    "VariableType",
    "ContainerType",
    "UnknownType",
    # Parameter/state info
    "ParameterInfo",
    "StateVariableInfo",
    # Analysis results
    "InputAnalysis",
    "StateAnalysis",
    "OutputAnalysis",
    # Constants
    "CONTAINER_STATE_INIT",
]

# Sentinel value for container state initialization (created lazily on first use)
CONTAINER_STATE_INIT = 0


@dataclass(frozen=True)
class ParameterInfo:
    """Info about a parsed input parameter.

    Args:
        expected_type: The expected Python type for this parameter.
        category (str): Client-defined category (e.g., "signal", "passive", "constant").
            Defaults to "constant".
    """

    expected_type: Any
    category: str = "constant"


@dataclass(frozen=True)
class StateVariableInfo:
    """Info about a state variable in a numba_node function."""

    name: str
    initial_value: Any
    state_type: Any


@dataclass
class InputAnalysis:
    """Parsed input parameters grouped by category.

    Args:
        parameters: Dict mapping param name to (validated_value, ParameterInfo).
    """

    parameters: Dict[str, tuple] = field(default_factory=dict)  # name -> (value, ParameterInfo)

    def get_by_category(self, category: str) -> Dict[str, Any]:
        """Get parameters by category. Returns {name: value}."""
        return {name: value for name, (value, info) in self.parameters.items() if info.category == category}

    def get_params_by_category(self, category: str) -> Dict[str, tuple]:
        """Get full parameter info by category. Returns {name: (value, ParameterInfo)}."""
        return {name: (value, info) for name, (value, info) in self.parameters.items() if info.category == category}


@dataclass
class StateAnalysis:
    """Summarized info about all state variables of a numba_node function."""

    state_vars: Dict[str, StateVariableInfo] = field(default_factory=dict)

    def sorted_by_size(self) -> List[StateVariableInfo]:
        """Sort state variables by size (largest first), then by name."""
        from numba_cfunc_compiler.type_factory import TypeFactory

        return sorted(
            self.state_vars.values(),
            key=lambda info: (-TypeFactory.get_type_size(info.state_type), info.name),
        )


@dataclass(frozen=True)
class OutputAnalysis:
    """Summarized info about all output parameters of a numba_node function."""

    output_types: List[type]
    named_outputs: Optional[Dict[str, type]] = None


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


class UnknownNumbaType:
    """Sentinel for unknown numba types."""

    pass


class UnknownNumbaValue:
    """Sentinel for unknown values (used when value is not provided)."""

    pass


class NoneType:
    """Sentinel for None type."""

    pass


@dataclass(frozen=True)
class VariableType(ABC):
    """
    VariableType represents the type of a variable.

    value: the python type of the variable
    runtime_value: the actual runtime value, known for constants.
                   For example, if we have a constant float 1.0, value is float, and runtime_value is 1.0
    """

    value: Any
    runtime_value: Any

    def is_opaque_pointer(self) -> bool:
        return False

    def read_constant(self, local_var_name: str) -> ast.Assign:
        return AST.assignment(local_var_name, ast.Constant(value=self.runtime_value))

    def prepare_voidptr_read(self, source: Any) -> "VariableType":
        return self

    def get_methods(self) -> list:
        return []

    def accepts_value_type(self, value_type: type) -> bool:
        """Used for return value checking"""
        return self.value == value_type

    @abstractmethod
    def get_numba_type_name(self) -> str:
        raise NotImplementedError(f"get_numba_type_name method not implemented for {self.runtime_value}")

    @classmethod
    def is_type_supported(cls, var_type: Any) -> bool:
        """
        Check if this class handles the given type.

        This is the single source of truth for type matching.
        Subclasses must override this method.
        """
        return False

    @classmethod
    def get_type_size(cls, var_type: Any) -> int:
        """Get size in bytes for a type, derived from its numba type name.

        Creates a temporary instance to obtain the numba type name, then
        looks up the size in NumbaTypeRegistry.  Subclasses only need to
        override this if their storage size differs from the numba type
        (which is unusual).
        """
        from numba_cfunc_compiler.numba_config import NumbaTypeRegistry

        instance = cls.from_type(var_type, UnknownNumbaValue())
        if instance is None:
            raise ValueError(f"{cls.__name__} cannot determine size for {var_type}")
        return NumbaTypeRegistry.get_size_for_numba_name(instance.get_numba_type_name())

    @classmethod
    def from_type(cls, var_type: Any, value: Any) -> Optional["VariableType"]:
        """Create a VariableType instance from a Python type and optional value."""
        if cls.is_type_supported(var_type):
            return cls(var_type, value)
        return None

    @classmethod
    def try_lower_assignment(cls, node: "ast.Assign", rhs: "ast.AST", call_globals: dict) -> Optional[tuple[list, "VariableType"]]:
        """
        Try to lower/transform an assignment AST node.

        Used for special assignments like `x = datetime(2020, 1, 1)` that need
        to be transformed (e.g., to nanoseconds).

        Returns (replacement_statements, variable_type) or None if not handled.
        """
        return None

    @classmethod
    def try_parse_input(cls, param: inspect.Parameter, ann: Any) -> Optional["ParameterInfo"]:
        """
        Try to parse an input parameter annotation.
        """
        return None

    @classmethod
    def validate_input(cls, param_name: str, value: Any, expected_type: Any) -> Any:
        """
        Validate and potentially transform an input value.

        Called after try_parse_input succeeds. Can transform the value
        (e.g., datetime to nanoseconds).
        """
        if not isinstance(value, expected_type):
            raise TypeError(f"Argument '{param_name}' expected {expected_type}, got {type(value)}")
        return value

    @classmethod
    def try_parse_state(cls, node: ast.AnnAssign, var_name: str, globalns: dict) -> Optional["StateVariableInfo"]:
        """
        Try to parse a state variable declaration.
        """
        return None


@dataclass(frozen=True)
class ContainerType(VariableType):
    """Base class for container types (List, Dict)."""

    def is_opaque_pointer(self) -> bool:
        return True

    def init_statements(self, var_name: str, loaded_value_target, state_slot_target=None) -> list[ast.stmt]:
        """Create statements to initialize a new container and store its voidptr."""
        stmts = list(self.create_new_container(var_name))
        voidptr_val = AST.function_call(
            self._to_voidptr_func_name(),
            ast.Name(id=var_name, ctx=ast.Load()),
        )
        stmts.append(AST.assignment(loaded_value_target, voidptr_val))
        if state_slot_target is not None:
            stmts.append(AST.assignment(state_slot_target, voidptr_val))
        return stmts

    def _to_voidptr_func_name(self) -> str:
        raise NotImplementedError

    def create_new_container(self, var_name: str) -> List[ast.AST]:
        raise NotImplementedError

    def from_voidptr(self, local_var_name: str, var_name: str, loaded_value: ast.AST) -> ast.AST:
        raise NotImplementedError

    def post_load_statements(self, local_var_name: str, var_name: str, loaded_value: ast.AST) -> list[ast.stmt]:
        """Cast loaded voidptr to typed container."""
        return [self.from_voidptr(local_var_name, var_name, loaded_value)]

    @staticmethod
    def emit_container_state_init(standalone_state_vars, state_array_name: str = STATE_ARRAY_NAME) -> list[ast.stmt]:
        """Create new containers for state vars (called at LIFECYCLE_START)."""
        if not standalone_state_vars:
            return []
        statements: list[ast.stmt] = []
        for v in standalone_state_vars:
            state_slot = AST.array_access(state_array_name, v.array_idx)
            loaded_name = f"_gc_standalone_loaded_{v.array_idx}"
            statements.extend(v.type.init_statements(v.name, loaded_name, state_slot))
        return statements

    @staticmethod
    def emit_container_state_load(standalone_state_vars, state_array_name: str = STATE_ARRAY_NAME) -> list[ast.stmt]:
        """Load and reconstruct typed containers from state voidptrs."""
        if not standalone_state_vars:
            return []
        statements: list[ast.stmt] = []
        for v in standalone_state_vars:
            loaded_name = f"_gc_standalone_loaded_{v.array_idx}"
            state_slot = AST.array_access(state_array_name, v.array_idx)
            statements.append(AST.assignment(loaded_name, state_slot))
            statements.extend(
                v.type.post_load_statements(
                    v.local_variable_name(),
                    v.name,
                    ast.Name(id=loaded_name, ctx=ast.Load()),
                )
            )
        return statements


@dataclass(frozen=True)
class UnknownType(VariableType):
    """Represents an unknown or unsupported type."""

    def get_numba_type_name(self) -> str:
        raise ValueError(f"UnknownType cannot be used for type {self.runtime_value}")

    @classmethod
    def from_type(cls, var_type: Any, value: Any) -> Optional["UnknownType"]:
        # UnknownType is the fallback - it always matches but should be tried last
        return None  # Never auto-match; TypeFactory creates it as fallback
