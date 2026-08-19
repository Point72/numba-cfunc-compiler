"""Call register() to register Primitive support (int, float, bool)."""

import ast
import inspect
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, ClassVar, Optional

from numba import types as numba_types

from numba_cfunc_compiler.models import (
    ParameterInfo,
    StateVariableInfo,
    UnknownNumbaValue,
    VariableType,
)
from numba_cfunc_compiler.numba_config import NumbaTypeInfo, NumbaTypeRegistry
from numba_cfunc_compiler.type_factory import TypeFactory


@dataclass(frozen=True)
class PrimitiveType(VariableType):
    """Handles primitive types: int, float, bool."""

    _PRIMITIVE_TYPES = (int, float, bool)
    _CONSTANT_INPUT_TYPES: tuple[type, ...] = (int, float, bool, datetime, timedelta)
    _STATE_TYPE_NAMES: ClassVar[dict[str, type]] = {"int": int, "float": float, "bool": bool}

    def get_numba_type_name(self) -> str:
        return NumbaTypeRegistry.resolve_numba_name(self.value)

    @classmethod
    def is_type_supported(cls, var_type: Any) -> bool:
        return var_type in cls._PRIMITIVE_TYPES

    @classmethod
    def from_type(cls, var_type: Any, value: Any) -> Optional["PrimitiveType"]:
        """Create PrimitiveType from a Python type or primitive value."""
        if cls.is_type_supported(var_type):
            return cls(var_type, value)
        # Fallback to runtime value type
        if not isinstance(value, UnknownNumbaValue) and cls.is_type_supported(type(value)):
            return cls(type(value), value)
        return None

    @classmethod
    def try_parse_input(cls, param: inspect.Parameter, ann: Any) -> ParameterInfo | None:
        """Parse constant input annotations: int, float, bool, datetime, timedelta."""
        if ann in cls._CONSTANT_INPUT_TYPES:
            return ParameterInfo(expected_type=ann)  # defaults to category="constant"
        return None

    @classmethod
    def validate_input(cls, param_name: str, value: Any, expected_type: Any) -> Any:
        """Validate and convert input values. Converts datetime/timedelta to nanoseconds."""
        if not isinstance(value, expected_type):
            raise TypeError(f"Argument '{param_name}' expected {expected_type}, got {type(value)}")

        # Convert datetime/timedelta to nanoseconds for Numba
        if expected_type is datetime:
            from numba_cfunc_compiler.defaults.datetime_support import DateTimeType

            return DateTimeType.to_nanos(value)
        if expected_type is timedelta:
            from numba_cfunc_compiler.defaults.timedelta_support import TimeDeltaType

            return TimeDeltaType.to_nanos(value)

        return value

    @classmethod
    def try_parse_state(cls, node: ast.AnnAssign, var_name: str, globalns: dict) -> StateVariableInfo | None:
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

        return StateVariableInfo(var_name, initial_value, state_type)


def register():
    """Register primitive type support."""
    TypeFactory.register(PrimitiveType)
    NumbaTypeRegistry.register_type(
        NumbaTypeInfo(
            python_type=int,
            numba_name="int64",
            numba_type=numba_types.int64,
            size=8,
            is_numeric=True,
            is_primitive=True,
            type_name="int",
        )
    )
    NumbaTypeRegistry.register_type(
        NumbaTypeInfo(
            python_type=float,
            numba_name="float64",
            numba_type=numba_types.float64,
            size=8,
            is_numeric=True,
            is_primitive=True,
            type_name="float",
        )
    )
    NumbaTypeRegistry.register_type(
        NumbaTypeInfo(
            python_type=bool,
            numba_name="int8",
            numba_type=numba_types.int8,
            size=1,
            is_numeric=True,
            is_primitive=True,
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
            is_numeric=True,
            is_primitive=False,
        )
    )
    NumbaTypeRegistry.register_type(
        NumbaTypeInfo(
            python_type=None,
            numba_name="voidptr",
            numba_type=numba_types.voidptr,
            size=8,
            is_numeric=False,
            is_primitive=False,
        )
    )
