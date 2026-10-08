"""Host struct metadata for Numba typed views."""

import ast
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from numba import types

from numba_cfunc_compiler.types.base import HostType
from numba_cfunc_compiler.types.binding import StatePlan
from numba_cfunc_compiler.types.builtin.struct.native import StructField, StructLayout
from numba_cfunc_compiler.utils.ast import AST


@dataclass(frozen=True)
class StructFieldInfo:
    """Name, byte offset, Numba type name, and byte size of a host field."""

    name: str
    offset: int
    numba_type_name: str
    size: int
    python_type: Any = None
    indirect: bool = False
    is_enum_set: bool = False


@dataclass(frozen=True)
class StructHostType(HostType):
    """Host struct descriptor; source reads produce Numba typed views."""

    fields: dict[str, StructFieldInfo] = None
    size: int = 0

    def get_state_payload(self):
        from numba_cfunc_compiler.types.native.state import borrowed_struct_payload

        return borrowed_struct_payload(self.get_typed_layout())

    @classmethod
    def try_parse_state(cls, node: ast.AnnAssign, var_name: str, globalns: dict) -> tuple[Any, Any] | None:
        annotation = node.annotation.slice
        if not isinstance(annotation, ast.Name):
            return None
        state_type = globalns.get(annotation.id)
        if not cls.is_type_supported(state_type):
            return None
        if not isinstance(node.value, ast.Constant) or node.value.value is not None:
            raise TypeError(f"State[{annotation.id}] '{var_name}' requires a host-supported initializer")
        return 0, state_type

    def get_typed_layout(self) -> StructLayout:
        if self.fields is None:
            raise TypeError("StructHostType has no field metadata")
        if not isinstance(self.value, type):
            raise TypeError("Struct views require a Python struct type")
        numeric_types = {
            name: getattr(types, name) for name in ("int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "float32", "float64")
        }
        fields = []
        for name, field in sorted(self.fields.items(), key=lambda item: (item[1].offset, item[0])):
            if name != field.name:
                raise ValueError(f"Struct field key '{name}' does not match metadata name '{field.name}'")
            if field.python_type is not None and isinstance(field.python_type, type) and issubclass(field.python_type, Enum):
                from numba_cfunc_compiler.types.builtin.enum.host import enum_binding

                binding = enum_binding(field.python_type)
                numba_type = binding.set_type if field.is_enum_set else binding.value_type
                typed_field = StructField(name, field.offset, numba_type, field.size, field.indirect)
                fields.append(typed_field)
                continue
            if field.numba_type_name == "voidptr":
                continue
            numba_type = numeric_types.get(field.numba_type_name)
            if numba_type is None:
                raise TypeError(f"Struct field '{name}' has unsupported type '{field.numba_type_name}'")
            typed_field = StructField(name, field.offset, numba_type, field.size)
            if field.size != typed_field.size:
                raise ValueError(f"Struct field '{name}' declares {field.size} bytes, expected {typed_field.size}")
            fields.append(typed_field)
        name = f"{self.value.__module__}.{self.value.__qualname__}"
        return StructLayout(name, self.size, tuple(fields))

    def get_numba_type_name(self) -> str:
        return "voidptr"

    def is_opaque_pointer(self) -> bool:
        return True

    def slot_read(self, name: str, slot: ast.expr, variable_factory) -> ast.stmt:
        view_name = variable_factory.bind_struct_view(self)
        return AST.assignment(name, AST.function_call(view_name, slot))

    def state_plan(self, variable) -> StatePlan:
        plan = super().state_plan(variable)
        return StatePlan(before=plan.before, metadata={"struct_state_indices": (variable.array_idx,), "struct_state_sizes": (self.size,)})

    @classmethod
    def get_struct_fields(cls, var_type: type) -> dict[str, StructFieldInfo]:
        return {}

    @classmethod
    def get_struct_size(cls, var_type: type) -> int:
        return 0

    @classmethod
    def from_type(cls, var_type: Any) -> Optional["StructHostType"]:
        if cls.is_type_supported(var_type):
            return cls(value=var_type, fields=cls.get_struct_fields(var_type), size=cls.get_struct_size(var_type))
        return None
