"""Borrowed, layout-specific views of host-owned structs in Numba code.

Opted-in ``StructHostType`` registrations use these views at node source reads.
Direct callers can also construct a typed view from a ``voidptr``.
"""

import ctypes
import hashlib
import json
from dataclasses import dataclass
from functools import cache

from llvmlite import ir
from numba import types
from numba.core import cgutils
from numba.core.typing.templates import AttributeTemplate
from numba.extending import infer_getattr, intrinsic, lower_getattr_generic, lower_setattr_generic, models, register_model

__all__ = ["StructField", "StructLayout", "StructPtrType", "struct_copy", "struct_ptr_type", "struct_view"]

_FIELD_SIZES = {
    types.int8: 1,
    types.uint8: 1,
    types.int16: 2,
    types.uint16: 2,
    types.int32: 4,
    types.uint32: 4,
    types.int64: 8,
    types.uint64: 8,
    types.float32: 4,
    types.float64: 8,
}


@dataclass(frozen=True)
class StructField:
    """A primitive field at a byte offset in host-owned storage."""

    name: str
    offset: int
    numba_type: types.Type
    physical_size: int | None = None
    indirect: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise ValueError("Struct field name must be a Python identifier")
        if type(self.offset) is not int or self.offset < 0:
            raise ValueError(f"Struct field '{self.name}' offset must be a non-negative integer")
        from numba_cfunc_compiler.types.builtin.enum.native import EnumSetValueType, EnumValueType

        if self.numba_type not in _FIELD_SIZES and not isinstance(self.numba_type, (EnumValueType, EnumSetValueType)):
            raise TypeError(f"Struct field '{self.name}' has unsupported native type {self.numba_type}")
        if self.indirect and not isinstance(self.numba_type, EnumSetValueType):
            raise TypeError("Indirect struct fields require an enum-set value type")
        expected = ctypes.sizeof(ctypes.c_void_p) if self.indirect else getattr(self.numba_type, "storage_type", self.numba_type)
        expected_size = expected if isinstance(expected, int) else _FIELD_SIZES.get(expected, getattr(expected, "bitwidth", 0) // 8)
        if self.physical_size is not None and self.physical_size != expected_size:
            narrow_enum = (
                isinstance(self.numba_type, EnumValueType)
                and self.numba_type.storage_type == types.uint16
                and type(self.physical_size) is int
                and self.physical_size == 1
            )
            if not narrow_enum:
                raise ValueError(f"Struct field '{self.name}' has size {self.physical_size}, expected {expected_size}")

    @property
    def size(self) -> int:
        if self.physical_size is not None:
            return self.physical_size
        if self.indirect:
            return ctypes.sizeof(ctypes.c_void_p)
        native = getattr(self.numba_type, "storage_type", self.numba_type)
        return _FIELD_SIZES.get(native, getattr(native, "bitwidth", 0) // 8)


@dataclass(frozen=True)
class StructLayout:
    """Immutable native layout; its identity is part of Numba's type key."""

    name: str
    size: int
    fields: tuple[StructField, ...]
    version: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Struct layout name must be a non-empty string")
        if type(self.size) is not int or self.size <= 0:
            raise ValueError("Struct layout size must be a positive integer")
        if type(self.version) is not int or self.version <= 0:
            raise ValueError("Struct layout version must be a positive integer")
        if not isinstance(self.fields, tuple) or not self.fields:
            raise ValueError("Struct layout must have at least one field")

        names: set[str] = set()
        occupied: list[tuple[int, int]] = []
        for field in self.fields:
            if not isinstance(field, StructField):
                raise TypeError("Struct layout fields must be StructField instances")
            if field.name in names:
                raise ValueError(f"Duplicate struct field '{field.name}'")
            names.add(field.name)
            end = field.offset + field.size
            if end > self.size:
                raise ValueError(f"Struct field '{field.name}' exceeds layout size {self.size}")
            if any(field.offset < other_end and other_start < end for other_start, other_end in occupied):
                raise ValueError(f"Struct field '{field.name}' overlaps another field")
            occupied.append((field.offset, end))

    @property
    def key(self) -> tuple:
        return self.name, self.size, self.version, tuple((f.name, f.offset, f.numba_type.name, f.size, f.indirect) for f in self.fields)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.key, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def get_field(self, name: str) -> StructField | None:
        for field in self.fields:
            if field.name == name:
                return field
        return None


class StructPtrType(types.Type):
    """A borrowed ``voidptr`` whose field layout is known to Numba."""

    def __init__(self, layout: StructLayout):
        self.layout = layout
        super().__init__(name=f"StructPtr[{layout.name}:{layout.fingerprint[:12]}]")

    @property
    def key(self):
        return self.layout.key


@register_model(StructPtrType)
class StructPtrModel(models.PrimitiveModel):
    def __init__(self, dmm, fe_type):
        super().__init__(dmm, fe_type, ir.IntType(8).as_pointer())


@cache
def struct_ptr_type(layout: StructLayout) -> StructPtrType:
    return StructPtrType(layout)


@cache
def struct_view(layout: StructLayout):
    """Return a Numba intrinsic that gives a ``voidptr`` this layout's type."""
    result_type = struct_ptr_type(layout)

    @intrinsic
    def view(typingctx, raw_ptr):
        if raw_ptr != types.voidptr:
            return None
        sig = result_type(raw_ptr)

        def codegen(context, builder, signature, args):
            return builder.bitcast(args[0], context.get_value_type(result_type))

        return sig, codegen

    return view


@cache
def struct_copy(layout: StructLayout):
    """Copy a matching borrowed view into host-owned output storage."""
    source_type = struct_ptr_type(layout)

    @intrinsic
    def copy(typingctx, dst, src):
        if dst != types.voidptr or src != source_type:
            return None
        sig = types.void(dst, src)

        def codegen(context, builder, signature, args):
            cgutils.memcpy(builder, args[0], args[1], context.get_constant(types.intp, layout.size))
            return context.get_dummy_value()

        return sig, codegen

    return copy


@infer_getattr
class StructPtrAttributes(AttributeTemplate):
    key = StructPtrType

    def generic_resolve(self, typ, attr):
        field = typ.layout.get_field(attr)
        return field.numba_type if field is not None else None


def _field_pointer(context, builder, typ: StructPtrType, value, attr: str):
    field = typ.layout.get_field(attr)
    if field is None:
        raise KeyError(f"Struct {typ.layout.name} has no field '{attr}'")
    byte_pointer = builder.bitcast(value, ir.IntType(8).as_pointer())
    field_pointer = builder.gep(byte_pointer, [context.get_constant(types.intp, field.offset)])
    if field.indirect:
        field_pointer = builder.load(builder.bitcast(field_pointer, ir.IntType(8).as_pointer().as_pointer()), align=1)
    native_type = context.get_value_type(field.numba_type)
    from numba_cfunc_compiler.types.builtin.enum.native import EnumValueType

    if isinstance(field.numba_type, EnumValueType) and field.size == 1 and native_type.width == 16:
        native_type = ir.IntType(8)
    return builder.bitcast(field_pointer, native_type.as_pointer()), field


@lower_getattr_generic(StructPtrType)
def struct_ptr_getattr(context, builder, typ, value, attr):
    pointer, field = _field_pointer(context, builder, typ, value, attr)
    loaded = builder.load(pointer, align=1)
    value_type = context.get_value_type(field.numba_type)
    if loaded.type != value_type:
        return builder.zext(loaded, value_type)
    return loaded


@lower_setattr_generic(StructPtrType)
def struct_ptr_setattr(context, builder, sig, args, attr):
    typ, value_type = sig.args
    pointer, field = _field_pointer(context, builder, typ, args[0], attr)
    value = context.cast(builder, args[1], value_type, field.numba_type)
    if value.type != pointer.type.pointee:
        value = builder.trunc(value, pointer.type.pointee)
    builder.store(value, pointer, align=1)
