"""Host declarations and storage for fixed size primitive arrays."""

import ast
import inspect
from dataclasses import dataclass
from typing import Any, get_args, get_origin

from numba_cfunc_compiler.api import NumbaArray
from numba_cfunc_compiler.core.analysis import ParameterInfo
from numba_cfunc_compiler.core.names import state_init_name
from numba_cfunc_compiler.types.base import HostType
from numba_cfunc_compiler.types.binding import ConstantPlan, StatePlan
from numba_cfunc_compiler.types.builtin.container import ensure_no_local
from numba_cfunc_compiler.types.factory import HostTypeFactory
from numba_cfunc_compiler.types.markers import ArrayTypeMarker
from numba_cfunc_compiler.types.registry import NumbaTypeRegistry
from numba_cfunc_compiler.utils.ast import AST


@dataclass(frozen=True)
class ArrayHostType(HostType):
    """A host-owned state buffer or read-only constant input."""

    @classmethod
    def is_type_supported(cls, var_type: Any) -> bool:
        return isinstance(var_type, ArrayTypeMarker)

    def get_numba_type_name(self) -> str:
        return "voidptr"

    def get_state_payload(self):
        from numba_cfunc_compiler.types.builtin.array.native import StandaloneArrayType, standalone_array_view
        from numba_cfunc_compiler.types.native.state import borrowed_container_payload

        dtype = NumbaTypeRegistry.resolve_to_numba_type(self.value.element_type)
        size = NumbaTypeRegistry.get_size(self.value.element_type)
        native_type = StandaloneArrayType(dtype, self.value.length)
        return borrowed_container_payload(
            "NumbaArray",
            native_type,
            ("array", dtype.name, self.value.length),
            host_size=size * self.value.length,
            alignment=size,
            bind_function=standalone_array_view(native_type),
        )

    def state_plan(self, variable) -> StatePlan:
        plan = super().state_plan(variable)
        return StatePlan(before=plan.before, metadata={"struct_state_indices": (variable.array_idx,), "struct_state_sizes": (self.value.byte_size,)})

    def constant_plan(self, name: str, value: Any, call_globals: dict, slot_index: int) -> ConstantPlan:
        """Build a read-only array from validated values in each callback frame."""
        dtype_name = NumbaTypeRegistry.resolve_numba_name(self.value.element_type)
        init_name = state_init_name(name)
        setup = [AST.assignment(init_name, AST.function_call("standalone_array_new", ast.Constant(dtype_name), ast.Constant(self.value.length)))]
        setup.extend(
            ast.Assign(
                targets=[ast.Subscript(value=ast.Name(id=init_name, ctx=ast.Load()), slice=ast.Constant(index), ctx=ast.Store())],
                value=ast.Constant(item),
            )
            for index, item in enumerate(value)
        )
        setup.append(AST.assignment(name, AST.function_call("standalone_array_readonly", ast.Name(id=init_name, ctx=ast.Load()))))
        return ConstantPlan(tuple(setup), ast.Name(id=name, ctx=ast.Load()))

    @classmethod
    def try_parse_input(cls, param: inspect.Parameter, ann: Any) -> ParameterInfo | None:
        if get_origin(ann) is not NumbaArray:
            return None
        args = get_args(ann)
        if len(args) != 2:
            raise TypeError("NumbaArray input requires an element type and length: NumbaArray[int, 8]")
        return ParameterInfo(expected_type=ArrayTypeMarker(*args))

    @classmethod
    def validate_input(cls, param_name: str, value: Any, expected_type: Any) -> Any:
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"Argument '{param_name}' expected list or tuple, got {type(value).__name__}")
        if len(value) != expected_type.length:
            raise TypeError(f"Argument '{param_name}' expected {expected_type.length} elements, got {len(value)}")
        for index, item in enumerate(value):
            if not isinstance(item, expected_type.element_type):
                raise TypeError(f"Argument '{param_name}' element {index}: expected {expected_type.element_type.__name__}, got {type(item).__name__}")
        return tuple(value)

    @staticmethod
    def _is_create_new_array_call(node: ast.AST) -> bool:
        return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "create_new_array"

    @staticmethod
    def _parse_new_array(call: ast.AST, globalns: dict) -> ArrayTypeMarker | None:
        if not ArrayHostType._is_create_new_array_call(call):
            return None
        if len(call.args) != 2 or call.keywords:
            raise TypeError("create_new_array expects exactly two positional arguments: element type and length")
        elem_node, length_node = call.args
        allowed = {t.__name__: t for t in (int, float, bool)}
        if not isinstance(elem_node, ast.Name) or elem_node.id not in allowed:
            raise TypeError("Array element type must be int, float, or bool")
        if isinstance(length_node, ast.Constant):
            length = length_node.value
        elif isinstance(length_node, ast.Name):
            length = globalns.get(length_node.id)
        else:
            length = None
        return ArrayTypeMarker(allowed[elem_node.id], length)

    @classmethod
    def try_parse_state(cls, node: ast.AnnAssign, var_name: str, globalns: dict) -> tuple[Any, Any] | None:
        marker = cls._parse_new_array(node.value, globalns)
        if marker is None:
            return None
        if not isinstance(node.annotation.slice, ast.Name) or node.annotation.slice.id != "NumbaArray":
            raise TypeError(f"State '{var_name}' must use State[NumbaArray]")
        return 0, marker


def register() -> None:
    HostTypeFactory.register(ArrayHostType)
    ensure_no_local("create_new_array", "NumbaArray")
