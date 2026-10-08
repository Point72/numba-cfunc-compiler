"""Host annotation and storage adapter for NumbaList."""

import ast
import inspect
from dataclasses import dataclass
from typing import Any, get_args, get_origin

from numba_cfunc_compiler.api import NumbaList
from numba_cfunc_compiler.core.analysis import ParameterInfo
from numba_cfunc_compiler.core.names import container_ptr_name, state_init_name
from numba_cfunc_compiler.extension.ast import ast_handler
from numba_cfunc_compiler.types.base import HostType
from numba_cfunc_compiler.types.binding import ConstantPlan
from numba_cfunc_compiler.types.builtin.container import (
    container_constant_plan,
    container_state_plan,
    reject_container_constructor,
    require_container_state_initializer,
)
from numba_cfunc_compiler.types.factory import HostTypeFactory
from numba_cfunc_compiler.types.markers import CONTAINER_STATE_INIT, ListTypeMarker
from numba_cfunc_compiler.types.registry import NumbaTypeRegistry
from numba_cfunc_compiler.utils.ast import AST


@dataclass(frozen=True)
class ListHostType(HostType):
    """Standalone list type backed by NB_List (NRT-free). Supports append, pop, clear."""

    def get_numba_type_name(self) -> str:
        return "voidptr"

    def get_state_payload(self):
        from numba_cfunc_compiler.types.builtin.list.native import ListValueType
        from numba_cfunc_compiler.types.native.state import borrowed_container_payload

        dtype = NumbaTypeRegistry.resolve_to_numba_type(self.value.element_type)
        return borrowed_container_payload("NumbaList", ListValueType(dtype), ("list", dtype.name))

    def to_voidptr_func_name(self) -> str:
        return "standalone_list_to_voidptr"

    def free_func_name(self) -> str:
        return "standalone_list_free"

    def _elem_type_name(self) -> str:
        return NumbaTypeRegistry.resolve_numba_name(self.value.element_type)

    def create_new_container(self, var_name: str) -> list[ast.AST]:
        item_size = NumbaTypeRegistry.get_size(self.value.element_type)
        ptr_name = container_ptr_name(var_name)
        return [
            AST.assignment(
                ptr_name,
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
                    ast.Name(id=ptr_name, ctx=ast.Load()),
                    ast.Constant(value=self._elem_type_name()),
                    ast.Constant(value=False),
                ),
            ),
        ]

    def from_voidptr(self, local_var_name: str, var_name: str, loaded_value: ast.AST, readonly: bool = False) -> ast.AST:
        """Cast voidptr back to typed standalone list."""
        return AST.assignment(
            local_var_name,
            AST.function_call(
                "standalone_list_from_voidptr",
                loaded_value,
                ast.Constant(value=self._elem_type_name()),
                ast.Constant(value=readonly),
            ),
        )

    def slot_read(self, name: str, slot: ast.expr, variable_factory) -> ast.stmt:
        return self.from_voidptr(name, name, slot)

    def state_plan(self, variable):
        return container_state_plan(self, variable)

    def is_opaque_pointer(self) -> bool:
        return True

    def constant_plan(self, local_var_name: str, value: Any, call_globals: dict, slot_index: int) -> ConstantPlan:
        """Populate a list once in START and bind it read-only thereafter."""
        values = list(value)
        if not isinstance(self.value, ListTypeMarker):
            raise TypeError(f"Expected ListTypeMarker, got {type(self.value)}")

        init_name = state_init_name(local_var_name)
        populate = []
        for v in values:
            populate.append(
                ast.Expr(
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id=init_name, ctx=ast.Load()),
                            attr="append",
                            ctx=ast.Load(),
                        ),
                        args=[ast.Constant(value=v)],
                        keywords=[],
                    )
                )
            )

        return container_constant_plan(self, local_var_name, slot_index, populate)

    @classmethod
    def is_type_supported(cls, var_type: Any) -> bool:
        return isinstance(var_type, ListTypeMarker)

    @classmethod
    def _is_create_new_list_call(cls, value_node: ast.AST) -> bool:
        """Check if the value node is a create_new_list(...) call."""
        return isinstance(value_node, ast.Call) and isinstance(value_node.func, ast.Name) and value_node.func.id == "create_new_list"

    @classmethod
    def try_parse_state(cls, node: ast.AnnAssign, var_name: str, globalns: dict) -> tuple[Any, Any] | None:
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
        return CONTAINER_STATE_INIT, state_type

    @classmethod
    def try_parse_input(cls, param: inspect.Parameter, ann: Any) -> ParameterInfo | None:
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

        return ParameterInfo(expected_type=ListTypeMarker(elem_type))

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


def register() -> None:
    """Register the built-in list host adapter."""
    HostTypeFactory.register(ListHostType)

    @ast_handler("AnnAssign", pre=True)
    def _list_state_initializer(converter, node: ast.AnnAssign):
        if ListHostType._is_create_new_list_call(node.value):
            require_container_state_initializer(node, "NumbaList", body=converter.current_body)

    @ast_handler("Call", post=True)
    def _reject_local_list(converter, node: ast.Call, result):
        if ListHostType._is_create_new_list_call(node):
            reject_container_constructor(node, "NumbaList")
        return result
