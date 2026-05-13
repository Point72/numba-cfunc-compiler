"""Call register() to register TimeDelta support."""

import ast
from dataclasses import dataclass
from datetime import timedelta as _PyTimedelta
from typing import Any, Optional

from numba import types as numba_types

from numba_cfunc_compiler.models import StateVariableInfo, UnknownNumbaValue, VariableType
from numba_cfunc_compiler.numba_config import NumbaTypeInfo, NumbaTypeRegistry
from numba_cfunc_compiler.type_factory import TypeFactory
from numba_cfunc_compiler.utils.ast import AST
from numba_cfunc_compiler.utils.types import TypeHelper


@dataclass(frozen=True)
class TimeDeltaType(VariableType):
    """Handles timedelta types, stored as nanoseconds (int64)."""

    def get_numba_type_name(self) -> str:
        return "int64"

    @staticmethod
    def to_nanos(val: _PyTimedelta) -> int:
        """Convert timedelta to nanoseconds."""
        return int(val.total_seconds() * 1e9)

    @classmethod
    def is_type_supported(cls, var_type: Any) -> bool:
        return var_type is _PyTimedelta

    @classmethod
    def from_type(cls, var_type: Any, value: Any) -> Optional["TimeDeltaType"]:
        """Create TimeDeltaType from a Python type or timedelta instance."""
        if cls.is_type_supported(var_type):
            return cls(_PyTimedelta, value)
        if not isinstance(value, UnknownNumbaValue) and isinstance(value, _PyTimedelta):
            return cls(_PyTimedelta, value)
        return None

    @classmethod
    def try_lower_assignment(cls, node: ast.Assign, rhs: ast.AST, call_globals: dict) -> Optional[tuple[list, "TimeDeltaType"]]:
        """Lower: x = timedelta(seconds=5) → x = <nanoseconds>"""
        if not isinstance(rhs, ast.Call):
            return None
        if not isinstance(node.targets[0], ast.Name):
            return None

        func_name = TypeHelper.get_time_func_name(rhs)
        if func_name != "timedelta":
            return None

        val, _ = TypeHelper.eval_time_constructor(rhs)
        if val is None or not isinstance(val, _PyTimedelta):
            return None

        var_name = node.targets[0].id
        nanos = cls.to_nanos(val)
        var_type = cls(_PyTimedelta, val)

        return AST.assignment(var_name, ast.Constant(value=nanos)), var_type

    @classmethod
    def try_parse_state(cls, node: ast.AnnAssign, var_name: str, globalns: dict) -> Optional[StateVariableInfo]:
        """Parse State[timedelta] declarations."""
        slice_node = node.annotation.slice

        if not isinstance(slice_node, ast.Name) or slice_node.id != "timedelta":
            return None

        initial_value = cls._parse_state_init(node.value, var_name)
        return StateVariableInfo(var_name, initial_value, _PyTimedelta)

    @classmethod
    def _parse_state_init(cls, value_node: ast.AST, var_name: str) -> int:
        """Parse and convert the initialization value to nanoseconds."""
        # Allow explicit nanos as numeric literals
        if isinstance(value_node, ast.Constant) and isinstance(value_node.value, (int, float)):
            return int(value_node.value)

        # Handle constructor calls like timedelta(...)
        if isinstance(value_node, ast.Call):
            val, func_name = TypeHelper.eval_time_constructor(value_node)
            if val is not None and isinstance(val, _PyTimedelta):
                return cls.to_nanos(val)

        raise TypeError(f"Invalid initializer for state '{var_name}': must be a numeric literal (nanoseconds) or timedelta() constructor call")


def register():
    """Register timedelta type support."""
    from numba_cfunc_compiler.ast_handlers import ast_handler

    TypeFactory.register(TimeDeltaType)
    NumbaTypeRegistry.register_type(
        NumbaTypeInfo(
            python_type=_PyTimedelta,
            numba_name="int64",
            numba_type=numba_types.int64,
            size=8,
            is_numeric=True,
            is_primitive=False,
            type_name="timedelta",
        )
    )

    # Register AST handler for timedelta constructor lowering in expressions
    @ast_handler("Call", pre=True)
    def _timedelta_call_handler(converter, node: ast.Call):
        """Lower timedelta constructor calls to nanoseconds constants."""
        if TypeHelper.get_time_func_name(node) != "timedelta":
            return None
        return TypeHelper.lower_time_constructor(node)
