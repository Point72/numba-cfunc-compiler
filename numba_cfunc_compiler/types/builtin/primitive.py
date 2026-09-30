"""Call register() to register Primitive support (int, float, bool)."""

import ast
import inspect
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, ClassVar, Optional

from numba import types as numba_types

from numba_cfunc_compiler.core.analysis import ParameterInfo
from numba_cfunc_compiler.types.base import HostType
from numba_cfunc_compiler.types.factory import HostTypeFactory
from numba_cfunc_compiler.types.registry import NumbaTypeInfo, NumbaTypeRegistry


@dataclass(frozen=True)
class PrimitiveHostType(HostType):
    """Handles primitive types: int, float, bool."""

    _PRIMITIVE_TYPES = (int, float, bool)
    _CONSTANT_INPUT_TYPES: tuple[type, ...] = (int, float, bool, datetime, timedelta)
    _STATE_TYPE_NAMES: ClassVar[dict[str, type]] = {"int": int, "float": float, "bool": bool}

    def get_numba_type_name(self) -> str:
        return NumbaTypeRegistry.resolve_numba_name(self.value)

    def lower_loaded_value(self, value: ast.AST) -> ast.AST:
        if self.value is bool:
            return ast.Compare(left=value, ops=[ast.NotEq()], comparators=[ast.Constant(value=0)])
        return value

    def get_state_payload(self):
        from numba_cfunc_compiler.types.native.state import copy_state_payload

        representations = {
            int: (numba_types.int64, numba_types.int64, 8, 8),
            float: (numba_types.float64, numba_types.float64, 8, 8),
            bool: (numba_types.boolean, numba_types.int8, 1, 1),
        }
        if self.value not in representations:
            raise TypeError(f"Unsupported primitive state type {self.value!r}")
        payload, storage, size, alignment = representations[self.value]
        return copy_state_payload((self.value.__module__, self.value.__qualname__), payload, storage, size, alignment)

    @classmethod
    def is_type_supported(cls, var_type: Any) -> bool:
        return var_type in cls._PRIMITIVE_TYPES

    @classmethod
    def from_type(cls, var_type: Any) -> Optional["PrimitiveHostType"]:
        """Create PrimitiveHostType from a declared Python type."""
        if cls.is_type_supported(var_type):
            return cls(var_type)
        return None

    @classmethod
    def try_parse_input(cls, param: inspect.Parameter, ann: Any) -> ParameterInfo | None:
        """Parse constant input annotations: int, float, bool, datetime, timedelta."""
        if ann in cls._CONSTANT_INPUT_TYPES:
            return ParameterInfo(expected_type=ann)
        return None

    @classmethod
    def validate_input(cls, param_name: str, value: Any, expected_type: Any) -> Any:
        """Validate and convert input values. Converts datetime/timedelta to nanoseconds."""
        if not isinstance(value, expected_type):
            raise TypeError(f"Argument '{param_name}' expected {expected_type}, got {type(value)}")

        # Convert datetime/timedelta to nanoseconds for Numba
        if expected_type is datetime:
            from numba_cfunc_compiler.types.builtin.datetime import DateTimeHostType

            return DateTimeHostType.to_nanos(value)
        if expected_type is timedelta:
            from numba_cfunc_compiler.types.builtin.timedelta import TimeDeltaHostType

            return TimeDeltaHostType.to_nanos(value)

        return value

    @classmethod
    def try_parse_state(cls, node: ast.AnnAssign, var_name: str, globalns: dict) -> tuple[Any, Any] | None:
        """Parse State[int], State[float], State[bool] declarations."""
        slice_node = node.annotation.slice

        if not isinstance(slice_node, ast.Name) or slice_node.id not in cls._STATE_TYPE_NAMES:
            return None

        state_type = cls._STATE_TYPE_NAMES[slice_node.id]

        if not isinstance(node.value, ast.Constant):
            raise TypeError(f"State '{var_name}' must have a literal initial value")

        initial_value = node.value.value
        initial_type = type(initial_value)

        # State storage is allocated by the host from the concrete Python value,
        # while generated code reads it using the declared State type. Keep those
        # representations identical, allowing only the safe numeric widening that
        # is already supported for inputs.
        if state_type is float and initial_type is int:
            initial_value = float(initial_value)
        elif initial_type is not state_type:
            raise TypeError(f"State '{var_name}' expected an initial value of type {state_type.__name__}, got {initial_type.__name__}")

        return initial_value, state_type


def register():
    """Register primitive type support."""
    HostTypeFactory.register(PrimitiveHostType)
    NumbaTypeRegistry.register_type(
        NumbaTypeInfo(
            python_type=int,
            numba_name="int64",
            numba_type=numba_types.int64,
            size=8,
            type_name="int",
        )
    )
    NumbaTypeRegistry.register_type(
        NumbaTypeInfo(
            python_type=float,
            numba_name="float64",
            numba_type=numba_types.float64,
            size=8,
            type_name="float",
        )
    )
    NumbaTypeRegistry.register_type(
        NumbaTypeInfo(
            python_type=bool,
            numba_name="int8",
            numba_type=numba_types.int8,
            size=1,
            type_name="bool",
        )
    )

    # Register internal types (no Python equivalent, no type_name)
    NumbaTypeRegistry.register_type(
        NumbaTypeInfo(
            python_type=None,
            numba_name="int16",
            numba_type=numba_types.int16,
            size=2,
        )
    )
    NumbaTypeRegistry.register_type(
        NumbaTypeInfo(
            python_type=None,
            numba_name="voidptr",
            numba_type=numba_types.voidptr,
            size=8,
        )
    )
