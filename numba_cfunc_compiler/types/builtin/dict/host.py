"""Host annotation and storage adapter for NumbaDict."""

import ast
import inspect
from dataclasses import dataclass
from typing import Any, get_args, get_origin

from numba_cfunc_compiler.api import NumbaDict
from numba_cfunc_compiler.core.analysis import ParameterInfo
from numba_cfunc_compiler.core.names import container_ptr_name, state_init_name
from numba_cfunc_compiler.types.base import HostType
from numba_cfunc_compiler.types.binding import ConstantPlan
from numba_cfunc_compiler.types.builtin.container import (
    container_constant_plan,
    container_state_plan,
    ensure_no_local,
)
from numba_cfunc_compiler.types.factory import HostTypeFactory
from numba_cfunc_compiler.types.markers import CONTAINER_STATE_INIT, DictTypeMarker
from numba_cfunc_compiler.types.registry import NumbaTypeRegistry
from numba_cfunc_compiler.utils.ast import AST


@dataclass(frozen=True)
class DictHostType(HostType):
    """Standalone dict type backed by NB_Dict (NRT-free). Supports get, pop, clear, items, keys."""

    def get_numba_type_name(self) -> str:
        return "voidptr"

    def get_state_payload(self):
        from numba_cfunc_compiler.types.builtin.dict.native import DictValueType
        from numba_cfunc_compiler.types.native.state import borrowed_container_payload

        key_type = NumbaTypeRegistry.resolve_to_numba_type(self.value.key_type)
        value_type = NumbaTypeRegistry.resolve_to_numba_type(self.value.value_type)
        return borrowed_container_payload("NumbaDict", DictValueType(key_type, value_type), ("dict", key_type.name, value_type.name))

    def to_voidptr_func_name(self) -> str:
        return "standalone_dict_to_voidptr"

    def free_func_name(self) -> str:
        return "standalone_dict_free"

    def _key_type_name(self) -> str:
        return NumbaTypeRegistry.resolve_numba_name(self.value.key_type)

    def _val_type_name(self) -> str:
        return NumbaTypeRegistry.resolve_numba_name(self.value.value_type)

    def create_new_container(self, var_name: str) -> list[ast.AST]:
        key_size = NumbaTypeRegistry.get_size(self.value.key_type)
        val_size = NumbaTypeRegistry.get_size(self.value.value_type)
        ptr_name = container_ptr_name(var_name)
        return [
            AST.assignment(
                ptr_name,
                AST.function_call(
                    "standalone_dict_new",
                    ast.Constant(value=key_size),
                    ast.Constant(value=val_size),
                ),
            ),
            AST.assignment(
                var_name,
                AST.function_call(
                    "standalone_dict_from_voidptr",
                    ast.Name(id=ptr_name, ctx=ast.Load()),
                    ast.Constant(value=self._key_type_name()),
                    ast.Constant(value=self._val_type_name()),
                    ast.Constant(value=False),
                ),
            ),
        ]

    def from_voidptr(self, local_var_name: str, var_name: str, loaded_value: ast.AST, readonly: bool = False) -> ast.AST:
        """Cast voidptr back to typed standalone dict."""
        return AST.assignment(
            local_var_name,
            AST.function_call(
                "standalone_dict_from_voidptr",
                loaded_value,
                ast.Constant(value=self._key_type_name()),
                ast.Constant(value=self._val_type_name()),
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
        """Populate a dict once in START and bind it read-only thereafter."""
        items = list(value.items())
        if not isinstance(self.value, DictTypeMarker):
            raise TypeError(f"Expected DictTypeMarker, got {type(self.value)}")

        init_name = state_init_name(local_var_name)
        populate = []
        for k, v in items:
            populate.append(
                ast.Assign(
                    targets=[
                        ast.Subscript(
                            value=ast.Name(id=init_name, ctx=ast.Load()),
                            slice=ast.Constant(value=k),
                            ctx=ast.Store(),
                        )
                    ],
                    value=ast.Constant(value=v),
                )
            )

        return container_constant_plan(self, local_var_name, slot_index, populate)

    @classmethod
    def is_type_supported(cls, var_type: Any) -> bool:
        return isinstance(var_type, DictTypeMarker)

    @classmethod
    def _is_create_new_dict_call(cls, value_node: ast.AST) -> bool:
        """Check if the value node is a create_new_dict(...) call."""
        return isinstance(value_node, ast.Call) and isinstance(value_node.func, ast.Name) and value_node.func.id == "create_new_dict"

    @classmethod
    def try_parse_state(cls, node: ast.AnnAssign, var_name: str, globalns: dict) -> tuple[Any, Any] | None:
        """Parse State[NumbaDict] = create_new_dict(key_type, val_type) declarations."""
        if not cls._is_create_new_dict_call(node.value):
            return None

        call_node = node.value
        if len(call_node.args) != 2:
            raise TypeError(f"create_new_dict expects exactly 2 arguments (key_type, value_type) for state '{var_name}'")

        key_type_node = call_node.args[0]
        val_type_node = call_node.args[1]

        if not isinstance(key_type_node, ast.Name) or not isinstance(val_type_node, ast.Name):
            raise TypeError(f"Dict key/value types must be type names for state '{var_name}'")

        key_type_name = key_type_node.id
        val_type_name = val_type_node.id

        allowed_keys = {t.__name__: t for t in NumbaTypeRegistry.get_dict_key_types()}
        allowed_vals = {t.__name__: t for t in NumbaTypeRegistry.get_dict_value_types()}
        if key_type_name not in allowed_keys:
            raise TypeError(f"Unsupported Dict key type '{key_type_name}' for state '{var_name}'. Supported: {list(allowed_keys.keys())}")
        if val_type_name not in allowed_vals:
            raise TypeError(f"Unsupported Dict value type '{val_type_name}' for state '{var_name}'. Supported: {list(allowed_vals.keys())}")

        state_type = DictTypeMarker(allowed_keys[key_type_name], allowed_vals[val_type_name])
        return CONTAINER_STATE_INIT, state_type

    @classmethod
    def try_parse_input(cls, param: inspect.Parameter, ann: Any) -> ParameterInfo | None:
        """Parse NumbaDict[key_type, value_type] constant input annotations."""
        origin = get_origin(ann)
        if origin is not NumbaDict:
            return None

        args = get_args(ann)
        if len(args) != 2:
            raise TypeError(f"NumbaDict requires exactly 2 type arguments, got {len(args)}")

        key_type, val_type = args
        allowed_keys = NumbaTypeRegistry.get_dict_key_types()
        allowed_vals = NumbaTypeRegistry.get_dict_value_types()

        if key_type not in allowed_keys:
            raise TypeError(f"Unsupported NumbaDict key type: {key_type}. Supported: {[t.__name__ for t in allowed_keys]}")
        if val_type not in allowed_vals:
            raise TypeError(f"Unsupported NumbaDict value type: {val_type}. Supported: {[t.__name__ for t in allowed_vals]}")

        return ParameterInfo(expected_type=DictTypeMarker(key_type, val_type))

    @classmethod
    def validate_input(cls, param_name: str, value: Any, expected_type: Any) -> Any:
        """Validate and return a NumbaDict constant input value."""
        if not isinstance(expected_type, DictTypeMarker):
            raise TypeError(f"Expected DictTypeMarker, got {type(expected_type)}")

        if not isinstance(value, dict):
            raise TypeError(f"Argument '{param_name}' expected dict, got {type(value).__name__}")

        key_type = expected_type.key_type
        val_type = expected_type.value_type

        for k, v in value.items():
            if not isinstance(k, key_type):
                raise TypeError(f"Argument '{param_name}' key {k!r}: expected {key_type.__name__}, got {type(k).__name__}")
            if not isinstance(v, val_type):
                raise TypeError(f"Argument '{param_name}' value for key {k!r}: expected {val_type.__name__}, got {type(v).__name__}")

        return dict(value)


def register() -> None:
    """Register the built-in dictionary host adapter."""
    HostTypeFactory.register(DictHostType)
    ensure_no_local("create_new_dict", "NumbaDict")
