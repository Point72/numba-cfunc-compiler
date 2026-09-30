"""Call register() to register TimeDelta support."""

import ast
from dataclasses import dataclass
from datetime import timedelta as _PyTimedelta
from typing import Any, Optional

from numba import types as numba_types

from numba_cfunc_compiler.types.base import HostType
from numba_cfunc_compiler.types.builtin.time import evaluate_time_constructor, lower_time_constructor, time_constructor_name
from numba_cfunc_compiler.types.factory import HostTypeFactory
from numba_cfunc_compiler.types.registry import NumbaTypeInfo, NumbaTypeRegistry


@dataclass(frozen=True)
class TimeDeltaHostType(HostType):
    """Handles timedelta types, stored as nanoseconds (int64)."""

    def get_numba_type_name(self) -> str:
        return "int64"

    def get_state_payload(self):
        from numba_cfunc_compiler.types.native.state import copy_state_payload

        return copy_state_payload(("datetime", "timedelta"), numba_types.int64, numba_types.int64, 8, 8)

    @staticmethod
    def to_nanos(val: _PyTimedelta) -> int:
        """Convert timedelta to nanoseconds."""
        return int(val.total_seconds() * 1e9)

    @classmethod
    def is_type_supported(cls, var_type: Any) -> bool:
        return var_type is _PyTimedelta

    @classmethod
    def from_type(cls, var_type: Any) -> Optional["TimeDeltaHostType"]:
        """Create TimeDeltaHostType from a declared Python type."""
        if cls.is_type_supported(var_type):
            return cls(_PyTimedelta)
        return None

    @classmethod
    def try_parse_state(cls, node: ast.AnnAssign, var_name: str, globalns: dict) -> tuple[Any, Any] | None:
        """Parse State[timedelta] declarations."""
        slice_node = node.annotation.slice

        if not isinstance(slice_node, ast.Name) or slice_node.id != "timedelta":
            return None

        initial_value = cls._parse_state_init(node.value, var_name)
        return initial_value, _PyTimedelta

    @classmethod
    def _parse_state_init(cls, value_node: ast.AST, var_name: str) -> int:
        """Parse and convert the initialization value to nanoseconds."""
        # Allow explicit nanos as numeric literals
        if isinstance(value_node, ast.Constant) and isinstance(value_node.value, (int, float)):
            return int(value_node.value)

        # Handle constructor calls like timedelta(...)
        if isinstance(value_node, ast.Call):
            val = evaluate_time_constructor(value_node)
            if isinstance(val, _PyTimedelta):
                return cls.to_nanos(val)

        raise TypeError(f"Invalid initializer for state '{var_name}': must be a numeric literal (nanoseconds) or timedelta() constructor call")


def register():
    """Register timedelta type support."""
    from numba_cfunc_compiler.extension.ast import ast_handler

    HostTypeFactory.register(TimeDeltaHostType)
    NumbaTypeRegistry.register_type(
        NumbaTypeInfo(
            python_type=_PyTimedelta,
            numba_name="int64",
            numba_type=numba_types.int64,
            size=8,
            type_name="timedelta",
        )
    )

    # Register AST handler for timedelta constructor lowering in expressions
    @ast_handler("Call", pre=True)
    def _timedelta_call_handler(converter, node: ast.Call):
        """Lower timedelta constructor calls to nanoseconds constants."""
        if time_constructor_name(node) != "timedelta":
            return None
        return lower_time_constructor(node)
