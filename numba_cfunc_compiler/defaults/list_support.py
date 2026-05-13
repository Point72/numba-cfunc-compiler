"""Call register() to register NumbaList support."""

import ast
import inspect
from dataclasses import dataclass
from typing import Any, List, Optional, get_args, get_origin

from numba_cfunc_compiler.models import (
    CONTAINER_STATE_INIT,
    ContainerType,
    ListTypeMarker,
    ParameterInfo,
    StateVariableInfo,
)
from numba_cfunc_compiler.numba_config import NumbaList, NumbaTypeRegistry
from numba_cfunc_compiler.type_factory import TypeFactory
from numba_cfunc_compiler.utils.ast import AST


@dataclass(frozen=True)
class NumbaListType(ContainerType):
    """Standalone list type backed by NB_List (NRT-free). Supports append, pop, clear."""

    def get_numba_type_name(self) -> str:
        return "voidptr"

    def get_methods(self):
        from numba_cfunc_compiler.method_factory import NativeMethod

        return [NativeMethod("append"), NativeMethod("pop"), NativeMethod("clear")]

    def _to_voidptr_func_name(self) -> str:
        return "standalone_list_to_voidptr"

    def _elem_type_name(self) -> str:
        return NumbaTypeRegistry.resolve_numba_name(self.value.element_type)

    def create_new_container(self, var_name: str) -> List[ast.AST]:
        item_size = NumbaTypeRegistry.get_size(self.value.element_type)
        return [
            AST.assignment(
                f"{var_name}_ptr",
                AST.function_call(
                    "standalone_list_new",
                    ast.Constant(value=item_size),
                    ast.Constant(value=0),
                ),
            ),
            AST.assignment(
                var_name,
                AST.function_call(
                    "standalone_list_from_voidptr",
                    ast.Name(id=f"{var_name}_ptr", ctx=ast.Load()),
                    ast.Constant(value=self._elem_type_name()),
                ),
            ),
        ]

    def from_voidptr(self, local_var_name: str, var_name: str, loaded_value: ast.AST) -> ast.AST:
        """Cast voidptr back to typed standalone list."""
        return AST.assignment(
            local_var_name,
            AST.function_call(
                "standalone_list_from_voidptr",
                loaded_value,
                ast.Constant(value=self._elem_type_name()),
            ),
        )

    def read_constant(self, local_var_name: str):
        """Create a standalone list and populate it with constant values."""
        values = list(self.runtime_value)
        if not isinstance(self.value, ListTypeMarker):
            raise TypeError(f"Expected ListTypeMarker, got {type(self.value)}")

        # Create the list using the type's create_new_container method
        stmts = self.create_new_container(local_var_name)

        # Append each value
        for v in values:
            stmts.append(
                ast.Expr(
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id=local_var_name, ctx=ast.Load()),
                            attr="append",
                            ctx=ast.Load(),
                        ),
                        args=[ast.Constant(value=v)],
                        keywords=[],
                    )
                )
            )

        return stmts

    @classmethod
    def is_type_supported(cls, var_type: Any) -> bool:
        return isinstance(var_type, ListTypeMarker)

    @classmethod
    def create_local_from_type_name(cls, var_name: str, elem_type_name: str):
        """Create AST statements for initializing a local list variable."""
        allowed = {t.__name__: t for t in NumbaTypeRegistry.get_list_element_types()}
        if elem_type_name not in allowed:
            raise TypeError(f"Unsupported List element type: {elem_type_name}. Supported: {list(allowed.keys())}")
        var_type = cls(ListTypeMarker(allowed[elem_type_name]), None)
        return var_type.create_new_container(var_name), var_type

    @classmethod
    def try_lower_assignment(cls, node: ast.Assign, rhs: ast.AST, call_globals: dict) -> Optional[tuple[list, "NumbaListType"]]:
        """Lower: l = create_new_list(int) → standalone list initialization"""
        if not isinstance(rhs, ast.Call):
            return None
        if not isinstance(rhs.func, ast.Name):
            return None
        if rhs.func.id != "create_new_list":
            return None
        if not isinstance(node.targets[0], ast.Name):
            return None

        var_name = node.targets[0].id

        if len(rhs.args) != 1:
            raise TypeError(f"create_new_list expects exactly 1 argument (element type), got {len(rhs.args)}")

        elem_type_node = rhs.args[0]
        if isinstance(elem_type_node, ast.Name):
            elem_type_name = elem_type_node.id
        else:
            raise TypeError(f"Unsupported List element type: {ast.dump(elem_type_node)}")

        stmts, var_type = cls.create_local_from_type_name(var_name, elem_type_name)
        return stmts, var_type

    @classmethod
    def _is_create_new_list_call(cls, value_node: ast.AST) -> bool:
        """Check if the value node is a create_new_list(...) call."""
        return isinstance(value_node, ast.Call) and isinstance(value_node.func, ast.Name) and value_node.func.id == "create_new_list"

    @classmethod
    def try_parse_state(cls, node: ast.AnnAssign, var_name: str, globalns: dict) -> Optional[StateVariableInfo]:
        """Parse State[NumbaList] = create_new_list(elem_type) declarations."""
        if not cls._is_create_new_list_call(node.value):
            return None

        call_node = node.value
        if len(call_node.args) != 1:
            raise TypeError(f"create_new_list expects exactly 1 argument (element type) for state '{var_name}'")

        elem_type_node = call_node.args[0]
        if not isinstance(elem_type_node, ast.Name):
            raise TypeError(f"List element type must be a type name for state '{var_name}'")

        elem_type_name = elem_type_node.id
        allowed = NumbaTypeRegistry.get_list_element_types()
        allowed_names = {t.__name__: t for t in allowed}
        if elem_type_name not in allowed_names:
            raise TypeError(f"Unsupported List element type '{elem_type_name}' for state '{var_name}'. Supported: {list(allowed_names.keys())}")

        state_type = ListTypeMarker(allowed_names[elem_type_name])
        return StateVariableInfo(var_name, CONTAINER_STATE_INIT, state_type)

    @classmethod
    def try_parse_input(cls, param: inspect.Parameter, ann: Any) -> Optional[ParameterInfo]:
        """Parse NumbaList[element_type] constant input annotations."""
        origin = get_origin(ann)
        if origin is not NumbaList:
            return None

        args = get_args(ann)
        if len(args) != 1:
            raise TypeError(f"NumbaList requires exactly 1 type argument, got {len(args)}")

        elem_type = args[0]
        allowed = NumbaTypeRegistry.get_list_element_types()
        if elem_type not in allowed:
            raise TypeError(f"Unsupported NumbaList element type: {elem_type}. Supported: {[t.__name__ for t in allowed]}")

        return ParameterInfo(expected_type=ListTypeMarker(elem_type))  # defaults to category="constant"

    @classmethod
    def validate_input(cls, param_name: str, value: Any, expected_type: Any) -> Any:
        """Validate and return a NumbaList constant input value."""
        if not isinstance(expected_type, ListTypeMarker):
            raise TypeError(f"Expected ListTypeMarker, got {type(expected_type)}")

        if not isinstance(value, (list, tuple)):
            raise TypeError(f"Argument '{param_name}' expected list or tuple, got {type(value).__name__}")

        elem_type = expected_type.element_type
        for i, item in enumerate(value):
            if not isinstance(item, elem_type):
                raise TypeError(f"Argument '{param_name}' element {i}: expected {elem_type.__name__}, got {type(item).__name__}")

        return list(value)


_list_for_counter = 0


def handle_list_for(converter, node):
    """
    Rewrite ``for`` loops over NumbaListType at the AST level.

    Transforms:
        for x in lst:
            body

    Into:
        for _li0 in range(len(lst)):
            x = lst[_li0]
            body
    """
    it = node.iter
    if not isinstance(it, ast.Name):
        return None

    var = converter.variable_factory.from_name(it.id)
    if var is None or not isinstance(var.type, NumbaListType):
        return None

    global _list_for_counter
    uid = _list_for_counter
    _list_for_counter += 1

    list_ref = var.get()
    index_name = f"_li{uid}"

    index_node = ast.Name(id=index_name, ctx=ast.Load())
    elem_assign = ast.Assign(
        targets=[node.target],
        value=ast.Subscript(
            value=list_ref,
            slice=index_node,
            ctx=ast.Load(),
        ),
    )

    visited_body = []
    for stmt in node.body:
        result = converter.visit(stmt)
        if isinstance(result, list):
            visited_body.extend(result)
        elif result is not None:
            visited_body.append(result)

    new_for = ast.For(
        target=ast.Name(id=index_name, ctx=ast.Store()),
        iter=ast.Call(
            func=ast.Name(id="range", ctx=ast.Load()),
            args=[
                ast.Call(
                    func=ast.Name(id="len", ctx=ast.Load()),
                    args=[list_ref],
                    keywords=[],
                )
            ],
            keywords=[],
        ),
        body=[elem_assign] + visited_body,
        orelse=[],
    )
    ast.fix_missing_locations(new_for)

    return new_for


def register():
    """Register NumbaList type support."""
    from numba_cfunc_compiler.ast_handlers import ASTHandlerRegistry, HandlerPhase

    TypeFactory.register(NumbaListType)
    ASTHandlerRegistry.register("For", handle_list_for, HandlerPhase.PRE)
