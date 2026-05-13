import ast
import inspect
from typing import Any, Optional, Tuple, Type

from numba_cfunc_compiler.compilation_context import CompilationContext
from numba_cfunc_compiler.models import (
    NoneType,
    ParameterInfo,
    StateVariableInfo,
    UnknownNumbaType,
    UnknownNumbaValue,
    UnknownType,
    VariableType,
)

__all__ = [
    "TypeFactory",
]


class TypeFactory:
    """Factory for creating VariableType instances. Types are tried in registration order."""

    @classmethod
    def register(cls, type_class: Type[VariableType], priority: Optional[int] = None) -> None:
        """Register a type class. Lower priority = tried first."""
        tc = CompilationContext.current().type_classes
        if priority is None:
            tc.append(type_class)
        else:
            tc.insert(priority, type_class)

    @classmethod
    def clear(cls) -> None:
        CompilationContext.current().type_classes.clear()

    @classmethod
    def get_type(cls, var_type: Any, value: Any = None) -> VariableType:
        if var_type is None or var_type is type(None):
            return UnknownType(UnknownNumbaType(), NoneType())
        for type_class in CompilationContext.current().type_classes:
            result = type_class.from_type(var_type, value)
            if result is not None:
                return result
        return UnknownType(UnknownNumbaType(), var_type)

    @classmethod
    def get_type_from_ast(cls, node: ast.AST) -> VariableType:
        from numba_cfunc_compiler.defaults.primitive_support import PrimitiveType

        if isinstance(node, ast.Constant):
            return cls.get_type(type(node.value), node.value)
        if isinstance(node, ast.Compare):
            return PrimitiveType(bool, UnknownNumbaValue())
        return UnknownType(UnknownNumbaType(), node)

    @classmethod
    def get_type_size(cls, var_type: Any) -> int:
        for type_class in CompilationContext.current().type_classes:
            if type_class.is_type_supported(var_type):
                return type_class.get_type_size(var_type)
        raise ValueError(f"No registered type class supports type: {var_type}")

    @classmethod
    def try_lower_assignment(cls, node: ast.Assign, rhs: ast.AST, call_globals: dict) -> Optional[tuple[list, VariableType]]:
        """Try to lower/transform an assignment by querying registered type classes."""
        for type_class in CompilationContext.current().type_classes:
            result = type_class.try_lower_assignment(node, rhs, call_globals)
            if result is not None:
                return result
        return None

    @classmethod
    def try_parse_input(cls, param: inspect.Parameter, ann: Any) -> Optional[Tuple[Type[VariableType], ParameterInfo]]:
        for type_class in CompilationContext.current().type_classes:
            result = type_class.try_parse_input(param, ann)
            if result is not None:
                return (type_class, result)
        return None

    @classmethod
    def try_parse_state(cls, node: ast.AnnAssign, var_name: str, globalns: dict) -> Optional[StateVariableInfo]:
        for type_class in CompilationContext.current().type_classes:
            result = type_class.try_parse_state(node, var_name, globalns)
            if result is not None:
                return result
        return None
