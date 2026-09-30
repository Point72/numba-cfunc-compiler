"""Host struct metadata for Numba typed views."""

from dataclasses import dataclass
from typing import Any, Optional

from numba import types

from numba_cfunc_compiler.models import UnknownNumbaValue, VariableType
from numba_cfunc_compiler.standalone.struct import StructField, StructLayout


@dataclass(frozen=True)
class StructFieldInfo:
    """Name, byte offset, Numba type name, and byte size of a host field."""

    name: str
    offset: int
    numba_type_name: str
    size: int


@dataclass(frozen=True)
class StructType(VariableType):
    """Host struct descriptor; source reads produce Numba typed views."""

    fields: dict[str, StructFieldInfo] = None
    size: int = 0

    def get_typed_layout(self) -> StructLayout:
        if self.fields is None:
            raise TypeError("StructType has no field metadata")
        if not isinstance(self.value, type):
            raise TypeError("Struct views require a Python struct type")
        numeric_types = {
            name: getattr(types, name) for name in ("int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "float32", "float64")
        }
        fields = []
        for name, field in sorted(self.fields.items(), key=lambda item: (item[1].offset, item[0])):
            if name != field.name:
                raise ValueError(f"Struct field key '{name}' does not match metadata name '{field.name}'")
            if field.numba_type_name == "voidptr":
                continue
            numba_type = numeric_types.get(field.numba_type_name)
            if numba_type is None:
                raise TypeError(f"Struct field '{name}' has unsupported type '{field.numba_type_name}'")
            typed_field = StructField(name, field.offset, numba_type)
            if field.size != typed_field.size:
                raise ValueError(f"Struct field '{name}' declares {field.size} bytes, expected {typed_field.size}")
            fields.append(typed_field)
        name = f"{self.value.__module__}.{self.value.__qualname__}"
        return StructLayout(name, self.size, tuple(fields))

    def get_numba_type_name(self) -> str:
        return "voidptr"

    def is_opaque_pointer(self) -> bool:
        return True

    def get_size(self) -> int:
        return self.size

    @classmethod
    def _get_struct_fields(cls, var_type: type) -> dict[str, StructFieldInfo]:
        return {}

    @classmethod
    def _get_struct_size(cls, var_type: type) -> int:
        return 0

    @classmethod
    def from_type(cls, var_type: Any, value: Any) -> Optional["StructType"]:
        if cls.is_type_supported(var_type):
            return cls(value=var_type, runtime_value=value, fields=cls._get_struct_fields(var_type), size=cls._get_struct_size(var_type))
        if not isinstance(value, UnknownNumbaValue) and value is not None:
            runtime_type = type(value)
            if cls.is_type_supported(runtime_type):
                return cls(
                    value=runtime_type,
                    runtime_value=value,
                    fields=cls._get_struct_fields(runtime_type),
                    size=cls._get_struct_size(runtime_type),
                )
        return None


def is_struct_type(var_type) -> bool:
    return isinstance(var_type, StructType)


def register():
    """Struct descriptors are registered by host integrations."""
