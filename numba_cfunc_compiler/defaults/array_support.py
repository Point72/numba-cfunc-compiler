"""Compiler support for fixed-size primitive arrays."""

import ast
from dataclasses import dataclass
from typing import Any

from numba_cfunc_compiler.models import ArrayTypeMarker, StateVariableInfo, VariableType
from numba_cfunc_compiler.type_factory import TypeFactory
from numba_cfunc_compiler.type_registry import NumbaTypeRegistry
from numba_cfunc_compiler.utils.ast import AST


@dataclass(frozen=True)
class NumbaArrayType(VariableType):
    """Host-backed state or stack-backed local array of primitive values."""

    def get_numba_type_name(self) -> str:
        return "voidptr"

    def is_opaque_pointer(self) -> bool:
        return True

    @classmethod
    def is_type_supported(cls, var_type: Any) -> bool:
        return isinstance(var_type, ArrayTypeMarker)

    @classmethod
    def get_type_size(cls, var_type: ArrayTypeMarker) -> int:
        return var_type.length * NumbaTypeRegistry.get_size(var_type.element_type)

    @staticmethod
    def _parse_new_array(call: ast.AST, globalns: dict) -> ArrayTypeMarker | None:
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id != "create_new_array":
            return None
        if len(call.args) != 2 or call.keywords:
            raise TypeError("create_new_array expects exactly two positional arguments: element type and length")

        elem_node, length_node = call.args
        if not isinstance(elem_node, ast.Name):
            raise TypeError("Array element type must be a type name")
        allowed = {t.__name__: t for t in (int, float, bool)}
        if elem_node.id not in allowed:
            raise TypeError(f"Unsupported array element type: {elem_node.id}")

        if isinstance(length_node, ast.Constant):
            length = length_node.value
        elif isinstance(length_node, ast.Name):
            length = globalns.get(length_node.id)
        else:
            length = None
        if type(length) is not int:
            raise TypeError("Array length must be a positive compile-time integer")
        return ArrayTypeMarker(allowed[elem_node.id], length)

    @classmethod
    def try_lower_assignment(cls, node: ast.Assign, rhs: ast.AST, call_globals: dict):
        marker = cls._parse_new_array(rhs, call_globals)
        if marker is None or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            return None
        name = node.targets[0].id
        dtype_name = NumbaTypeRegistry.resolve_numba_name(marker.element_type)
        stmt = AST.assignment(
            name,
            AST.function_call("standalone_array_new", ast.Constant(dtype_name), ast.Constant(marker.length)),
        )
        return [stmt], cls(marker, None)

    @classmethod
    def try_parse_state(cls, node: ast.AnnAssign, var_name: str, globalns: dict) -> StateVariableInfo | None:
        marker = cls._parse_new_array(node.value, globalns)
        if marker is None:
            return None
        if not isinstance(node.annotation.slice, ast.Name) or node.annotation.slice.id != "NumbaArray":
            raise TypeError(f"State '{var_name}' must use State[NumbaArray]")
        # Host raw-buffer state is zero-initialized, with the byte size in metadata.
        return StateVariableInfo(var_name, 0, marker)

    def read_from_voidptr(self, local_name: str, loaded_value: ast.AST) -> ast.Assign:
        dtype_name = NumbaTypeRegistry.resolve_numba_name(self.value.element_type)
        return AST.assignment(
            local_name,
            AST.function_call(
                "standalone_array_from_voidptr",
                loaded_value,
                ast.Constant(dtype_name),
                ast.Constant(self.value.length),
            ),
        )


_array_for_counter = 0


def _static_index(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        inner = _static_index(node.operand)
        if inner is not None:
            return inner if isinstance(node.op, ast.UAdd) else -inner
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
        left = _static_index(node.left)
        right = _static_index(node.right)
        if left is not None and right is not None:
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            return left * right
    return None


def handle_array_subscript(converter, node: ast.Subscript):
    """Reject invalid compile-time indices before Numba erases literal types."""
    if not isinstance(node.value, ast.Name):
        return None
    var = converter.variable_factory.from_name(node.value.id)
    if var is None or not isinstance(var.type, NumbaArrayType):
        return None
    index = _static_index(node.slice)
    if index is not None and not -var.type.value.length <= index < var.type.value.length:
        raise IndexError(f"Constant array index {index} is out of range for length {var.type.value.length}")
    return None


def handle_array_for(converter, node: ast.For):
    """Lower array iteration to a range loop over its compile-time length."""
    if not isinstance(node.iter, ast.Name):
        return None
    var = converter.variable_factory.from_name(node.iter.id)
    if var is None or not isinstance(var.type, NumbaArrayType):
        return None

    global _array_for_counter
    index_name = f"_ai{_array_for_counter}"
    _array_for_counter += 1
    array_ref = var.get()
    item_assign = ast.Assign(
        targets=[node.target],
        value=ast.Subscript(value=array_ref, slice=ast.Name(id=index_name, ctx=ast.Load()), ctx=ast.Load()),
    )
    body = []
    for stmt in node.body:
        result = converter.visit(stmt)
        if isinstance(result, list):
            body.extend(result)
        elif result is not None:
            body.append(result)
    else_body = []
    for stmt in node.orelse:
        result = converter.visit(stmt)
        if isinstance(result, list):
            else_body.extend(result)
        elif result is not None:
            else_body.append(result)
    replacement = ast.For(
        target=ast.Name(id=index_name, ctx=ast.Store()),
        iter=AST.function_call("range", ast.Constant(var.type.value.length)),
        body=[item_assign] + body,
        orelse=else_body,
    )
    return ast.fix_missing_locations(replacement)


def register():
    from numba_cfunc_compiler.ast_handlers import ASTHandlerRegistry, HandlerPhase

    TypeFactory.register(NumbaArrayType)
    ASTHandlerRegistry.register("Subscript", handle_array_subscript, HandlerPhase.PRE)
    ASTHandlerRegistry.register("For", handle_array_for, HandlerPhase.PRE)
