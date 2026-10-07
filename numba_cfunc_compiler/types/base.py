"""Host-side extension base for value families."""

from __future__ import annotations

import ast
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from numba_cfunc_compiler.core.analysis import ParameterInfo
from numba_cfunc_compiler.core.names import bind_state_name, state_slot_name
from numba_cfunc_compiler.types.binding import ConstantPlan, StatePlan
from numba_cfunc_compiler.utils.ast import AST


@dataclass(frozen=True)
class HostType(ABC):
    """Implement one family of host annotations and native values.

    Instances describe a Python type or type marker. Runtime constant values
    are passed separately to :meth:`constant_plan`.
    """

    value: Any

    def is_opaque_pointer(self) -> bool:
        return False

    def constant_plan(self, name: str, value: Any, call_globals: dict, slot_index: int) -> ConstantPlan:
        """Build a constant; report slots claimed from ``slot_index`` in the plan."""
        return ConstantPlan((AST.assignment(name, ast.Constant(value=value)),), ast.Constant(value=value))

    def slot_read(self, name: str, slot: ast.expr, variable_factory) -> ast.stmt | list[ast.stmt]:
        return AST.assignment(name, AST.cast_from_voidptr(slot, self.get_numba_type_name()))

    def lower_loaded_value(self, value: ast.expr) -> ast.expr:
        return value

    def state_plan(self, variable) -> StatePlan:
        """Bind a state name to its host cell before the lifecycle branches."""

        slot_name = state_slot_name(variable.name)
        return StatePlan(
            before=(
                AST.assignment(slot_name, AST.array_access(variable.get_storage_location(), variable.array_idx)),
                AST.assignment(variable.name, AST.function_call(bind_state_name(variable.name), ast.Name(id=slot_name, ctx=ast.Load()))),
            )
        )

    @abstractmethod
    def get_state_payload(self):
        """Return native and host storage details, also used by input/output bindings."""

    def get_numba_type_name(self) -> str:
        raise NotImplementedError(f"{type(self).__name__} does not support plain pointer reads")

    @classmethod
    @abstractmethod
    def is_type_supported(cls, var_type: Any) -> bool:
        """Whether this family owns a Python type or marker."""

    @classmethod
    def from_type(cls, var_type: Any) -> HostType | None:
        if cls.is_type_supported(var_type):
            return cls(var_type)
        return None

    @classmethod
    def try_parse_input(cls, param: inspect.Parameter, ann: Any) -> ParameterInfo | None:
        return None

    @classmethod
    def validate_input(cls, param_name: str, value: Any, expected_type: Any) -> Any:
        if not isinstance(value, expected_type):
            raise TypeError(f"Argument '{param_name}' expected {expected_type}, got {type(value)}")
        return value

    @classmethod
    def try_parse_state(cls, node: ast.AnnAssign, var_name: str, globalns: dict) -> tuple[Any, Any] | None:
        """Return (initial value, declared type), or None if not handled."""
        return None
