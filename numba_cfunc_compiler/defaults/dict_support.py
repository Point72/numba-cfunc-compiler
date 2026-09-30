"""Call register() to register NumbaDict support."""

import ast
import inspect
from dataclasses import dataclass
from typing import Any, get_args, get_origin

from numba_cfunc_compiler.models import (
    CONTAINER_STATE_INIT,
    ContainerType,
    DictTypeMarker,
    ParameterInfo,
    StateVariableInfo,
)
from numba_cfunc_compiler.node_api import NumbaDict
from numba_cfunc_compiler.type_factory import TypeFactory
from numba_cfunc_compiler.type_registry import NumbaTypeRegistry
from numba_cfunc_compiler.utils.ast import AST


@dataclass(frozen=True)
class NumbaDictType(ContainerType):
    """Standalone dict type backed by NB_Dict (NRT-free). Supports get, pop, clear, items, keys."""

    def get_numba_type_name(self) -> str:
        return "voidptr"

    def _to_voidptr_func_name(self) -> str:
        return "standalone_dict_to_voidptr"

    def _free_func_name(self) -> str:
        return "standalone_dict_free"

    def _key_type_name(self) -> str:
        return NumbaTypeRegistry.resolve_numba_name(self.value.key_type)

    def _val_type_name(self) -> str:
        return NumbaTypeRegistry.resolve_numba_name(self.value.value_type)

    def create_new_container(self, var_name: str) -> list[ast.AST]:
        key_size = NumbaTypeRegistry.get_size(self.value.key_type)
        val_size = NumbaTypeRegistry.get_size(self.value.value_type)
        return [
            AST.assignment(
                f"{var_name}_ptr",
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
                    ast.Name(id=f"{var_name}_ptr", ctx=ast.Load()),
                    ast.Constant(value=self._key_type_name()),
                    ast.Constant(value=self._val_type_name()),
                ),
            ),
        ]

    def from_voidptr(self, local_var_name: str, var_name: str, loaded_value: ast.AST) -> ast.AST:
        """Cast voidptr back to typed standalone dict."""
        return AST.assignment(
            local_var_name,
            AST.function_call(
                "standalone_dict_from_voidptr",
                loaded_value,
                ast.Constant(value=self._key_type_name()),
                ast.Constant(value=self._val_type_name()),
            ),
        )

    def read_constant(self, local_var_name: str):
        """Create a standalone dict and populate it with constant values."""
        items = list(self.runtime_value.items())
        if not isinstance(self.value, DictTypeMarker):
            raise TypeError(f"Expected DictTypeMarker, got {type(self.value)}")

        # Create the dict using the type's create_new_container method
        stmts = self.create_new_container(local_var_name)

        # Set each key-value pair
        for k, v in items:
            stmts.append(
                ast.Assign(
                    targets=[
                        ast.Subscript(
                            value=ast.Name(id=local_var_name, ctx=ast.Load()),
                            slice=ast.Constant(value=k),
                            ctx=ast.Store(),
                        )
                    ],
                    value=ast.Constant(value=v),
                )
            )

        return stmts

    @classmethod
    def is_type_supported(cls, var_type: Any) -> bool:
        return isinstance(var_type, DictTypeMarker)

    @classmethod
    def create_local_from_type_names(cls, var_name: str, key_type_name: str, val_type_name: str):
        """Create AST statements for initializing a local dict variable."""
        allowed_keys = {t.__name__: t for t in NumbaTypeRegistry.get_dict_key_types()}
        allowed_vals = {t.__name__: t for t in NumbaTypeRegistry.get_dict_value_types()}
        if key_type_name not in allowed_keys:
            raise TypeError(f"Unsupported Dict key type: {key_type_name}. Supported: {list(allowed_keys.keys())}")
        if val_type_name not in allowed_vals:
            raise TypeError(f"Unsupported Dict value type: {val_type_name}. Supported: {list(allowed_vals.keys())}")
        var_type = cls(
            DictTypeMarker(allowed_keys[key_type_name], allowed_vals[val_type_name]),
            None,
        )
        return var_type.create_new_container(var_name), var_type

    @classmethod
    def try_lower_assignment(cls, node: ast.Assign, rhs: ast.AST, call_globals: dict) -> list[ast.stmt] | None:
        """Lower: d = create_new_dict(int, float) → standalone dict initialization"""
        if not isinstance(rhs, ast.Call):
            return None
        if not isinstance(rhs.func, ast.Name):
            return None
        if rhs.func.id != "create_new_dict":
            return None
        if not isinstance(node.targets[0], ast.Name):
            return None

        var_name = node.targets[0].id

        if len(rhs.args) != 2:
            raise TypeError(f"create_new_dict expects exactly 2 arguments (key_type, value_type), got {len(rhs.args)}")

        key_type_node = rhs.args[0]
        val_type_node = rhs.args[1]

        def get_type_name(type_node):
            if isinstance(type_node, ast.Name):
                return type_node.id
            raise TypeError(f"Unsupported type: {ast.dump(type_node)}")

        key_type_name = get_type_name(key_type_node)
        val_type_name = get_type_name(val_type_node)

        stmts, _ = cls.create_local_from_type_names(var_name, key_type_name, val_type_name)
        return stmts

    @classmethod
    def _is_create_new_dict_call(cls, value_node: ast.AST) -> bool:
        """Check if the value node is a create_new_dict(...) call."""
        return isinstance(value_node, ast.Call) and isinstance(value_node.func, ast.Name) and value_node.func.id == "create_new_dict"

    @classmethod
    def try_parse_state(cls, node: ast.AnnAssign, var_name: str, globalns: dict) -> StateVariableInfo | None:
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
        return StateVariableInfo(var_name, CONTAINER_STATE_INIT, state_type)

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

        return ParameterInfo(
            expected_type=DictTypeMarker(key_type, val_type)  # defaults to category="constant"
        )

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


def register():
    """Register standalone dict host storage support."""
    TypeFactory.register(NumbaDictType)
