"""Context-scoped registration and resolution of HostType subclasses."""

import ast
import inspect
from typing import Any

from numba_cfunc_compiler.core.analysis import ParameterInfo
from numba_cfunc_compiler.core.context import CompilationContext
from numba_cfunc_compiler.types.base import HostType
from numba_cfunc_compiler.types.binding import TypeBinding

__all__ = ["HostTypeFactory"]


class HostTypeFactory:
    """Resolve a registered family once for each declared host value."""

    @classmethod
    def register(cls, type_class: type[HostType], priority: int | None = None) -> None:
        context = CompilationContext.current()
        if priority is None:
            context.type_classes.append(type_class)
        else:
            context.type_classes.insert(priority, type_class)

    @classmethod
    def resolve(cls, var_type: Any) -> TypeBinding:
        for type_class in CompilationContext.current().type_classes:
            result = type_class.from_type(var_type)
            if result is not None:
                return TypeBinding(result, result.get_state_payload())
        raise TypeError(f"No registered HostType supports {var_type!r}")

    @classmethod
    def lower_local_assignment(cls, node: ast.Assign, call_globals: dict) -> list[ast.stmt] | None:
        rhs = node.value
        if not isinstance(rhs, ast.Call) or not isinstance(rhs.func, ast.Name):
            return None
        for type_class in CompilationContext.current().type_classes:
            result = type_class.lower_local_assignment(node, rhs, call_globals)
            if result is not None:
                return result
        return None

    @classmethod
    def try_parse_input(cls, param: inspect.Parameter, ann: Any) -> tuple[type[HostType], ParameterInfo] | None:
        for type_class in CompilationContext.current().type_classes:
            result = type_class.try_parse_input(param, ann)
            if result is not None:
                return type_class, result
        return None

    @classmethod
    def try_parse_state(cls, node: ast.AnnAssign, var_name: str, globalns: dict) -> tuple[Any, Any] | None:
        for type_class in CompilationContext.current().type_classes:
            result = type_class.try_parse_state(node, var_name, globalns)
            if result is not None:
                return result
        return None
