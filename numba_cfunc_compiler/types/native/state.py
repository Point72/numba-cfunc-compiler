"""Native types and bind/store operations for host-owned state cells."""

import ctypes
import operator
from dataclasses import dataclass
from functools import cache

from llvmlite import ir as llvm_ir
from numba import types
from numba.core.errors import TypingError
from numba.extending import intrinsic, lower_builtin, lower_cast, models, overload, register_model

from numba_cfunc_compiler.types.builtin.struct.native import StructLayout, struct_ptr_type, struct_view
from numba_cfunc_compiler.types.native.input import InputValueType, input_payload
from numba_cfunc_compiler.types.policy import ValueSemantics


class CopyStateValueType(types.Type):
    """A copied state value paired with its host cell address."""

    def __init__(self, nominal_key: tuple, payload_type: types.Type, storage_type: types.Type, size: int, alignment: int):
        self.nominal_key = nominal_key
        self.payload_type = payload_type
        self.storage_type = storage_type
        self.size = size
        self.alignment = alignment
        super().__init__(f"StateValue[{nominal_key!r}:copy]")

    @property
    def key(self):
        return (self.nominal_key, self.payload_type, self.storage_type, self.size, self.alignment)

    def unify(self, typingctx, other):
        if other == self.payload_type or (isinstance(other, (InputValueType, CopyStateValueType)) and self.payload_type == other.payload_type):
            return self.payload_type
        return None


@register_model(CopyStateValueType)
class CopyStateValueModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        super().__init__(dmm, fe_type, [("slot", types.voidptr), ("value", fe_type.payload_type)])


@lower_cast(CopyStateValueType, types.Boolean)
@lower_cast(CopyStateValueType, types.Type)
def lower_copy_state_value_cast(context, builder, from_type, to_type, value):
    if to_type != from_type.payload_type:
        raise TypingError(f"Cannot convert {from_type} to {to_type}")
    return context.make_helper(builder, from_type, value=value).value


@lower_builtin(bool, CopyStateValueType)
def copy_state_value_bool(context, builder, signature, args):
    value_type = signature.args[0]
    value = context.make_helper(builder, value_type, value=args[0]).value
    return context.cast(builder, value, value_type.payload_type, types.boolean)


@overload(operator.truth)
def copy_state_truth(value):
    if not isinstance(value, CopyStateValueType):
        return None

    def impl(value):
        return bool(input_payload(value))

    return impl


@dataclass(frozen=True)
class StatePayload:
    """Native representation and host-cell operations for a resolved value type.

    TypeBinding also uses this descriptor to derive input and output boundary
    types. For declared state, the host owns storage with ``host_size`` and
    ``alignment``; ``bind_function`` and ``store_factory`` provide Numba
    operations for reading and assigning it.
    """

    label: str
    semantics: ValueSemantics
    native_type: types.Type
    host_size: int
    alignment: int
    layout_key: tuple
    replacement: bool
    bind_function: object
    store_factory: object
    mutable: bool = False

    def __post_init__(self):
        if not isinstance(self.semantics, ValueSemantics):
            raise TypeError(f"State semantics must be a ValueSemantics member, got {self.semantics!r}")
        if self.host_size <= 0 or self.alignment <= 0:
            raise ValueError("State host size and alignment must be positive")
        if self.alignment & (self.alignment - 1):
            raise ValueError("State host alignment must be a power of two")
        if not callable(self.bind_function) or not callable(self.store_factory):
            raise TypeError("State binding and store operations must be callable")

    @property
    def key(self):
        return (
            self.label,
            self.semantics.value,
            self.native_type.name,
            self.host_size,
            self.alignment,
            self.layout_key,
            self.replacement,
            self.mutable,
        )

    def store_function(self, name: str):
        return self.store_factory(name)


@cache
def _copy_type(nominal_key: tuple, payload_type: types.Type, storage_type: types.Type, size: int, alignment: int):
    return CopyStateValueType(nominal_key, payload_type, storage_type, size, alignment)


@cache
def _copy_bind(value_type: CopyStateValueType):
    """Build an intrinsic that loads a host cell into a typed snapshot."""

    @intrinsic
    def bind(typingctx, slot):
        if slot != types.voidptr:
            return None
        sig = value_type(slot)

        def codegen(context, builder, signature, args):
            pointer = builder.bitcast(args[0], context.get_value_type(value_type.storage_type).as_pointer())
            stored = builder.load(pointer, align=value_type.alignment)
            result = context.make_helper(builder, value_type)
            result.slot = args[0]
            result.value = context.cast(builder, stored, value_type.storage_type, value_type.payload_type)
            return result._getvalue()

        return sig, codegen

    return bind


@cache
def _copy_store(value_type: CopyStateValueType, name: str):
    """Build an intrinsic that writes a host cell and returns a new snapshot."""

    @intrinsic
    def store(typingctx, slot, rhs):
        if slot != types.voidptr:
            return None
        if (
            rhs != value_type
            and rhs != value_type.payload_type
            and not (isinstance(rhs, InputValueType) and rhs.payload_type == value_type.payload_type)
        ):
            raise TypingError(f"cannot store {rhs} in State[{value_type.nominal_key[-1]}] '{name}': expected {value_type.payload_type}")
        sig = value_type(slot, rhs)

        def codegen(context, builder, signature, args):
            if rhs == value_type.payload_type:
                value = args[1]
            else:
                value = context.make_helper(builder, rhs, value=args[1]).value
            stored = context.cast(builder, value, value_type.payload_type, value_type.storage_type)
            pointer = builder.bitcast(args[0], context.get_value_type(value_type.storage_type).as_pointer())
            builder.store(stored, pointer, align=value_type.alignment)
            result = context.make_helper(builder, value_type)
            result.slot = args[0]
            result.value = value
            return result._getvalue()

        return sig, codegen

    return store


@cache
def copy_state_payload(nominal_key: tuple, payload_type: types.Type, storage_type: types.Type, size: int, alignment: int) -> StatePayload:
    """Describe a copied scalar with matching native and host storage types.

    Boolean values may use an ``int8`` host cell.
    """
    if payload_type != storage_type and not (payload_type == types.boolean and storage_type == types.int8):
        raise TypeError("Copied state requires identical payload and storage types, except boolean/int8")
    if getattr(storage_type, "bitwidth", 0) != size * 8:
        raise ValueError(f"Copied state storage {storage_type} does not occupy {size} bytes")
    value_type = _copy_type(nominal_key, payload_type, storage_type, size, alignment)

    def store_factory(name):
        return _copy_store(value_type, name)

    return StatePayload(
        label=str(nominal_key[-1]),
        semantics=ValueSemantics.COPY,
        native_type=value_type,
        host_size=size,
        alignment=alignment,
        layout_key=(payload_type.name, storage_type.name),
        replacement=True,
        bind_function=_copy_bind(value_type),
        store_factory=store_factory,
    )


@cache
def nominal_copy_state_payload(nominal_key: tuple, payload_type: types.Type, storage_type: types.Type, size: int, alignment: int) -> StatePayload:
    """Describe a copied nominal value with a Numba-supported storage cast.

    Numba must support casts in both directions between ``storage_type`` and
    ``payload_type`` when they differ.
    """
    if getattr(storage_type, "bitwidth", getattr(storage_type, "bit_width", 0)) != size * 8:
        raise ValueError(f"Nominal state storage {storage_type} does not occupy {size} bytes")
    value_type = _copy_type(nominal_key, payload_type, storage_type, size, alignment)

    def store_factory(name):
        return _copy_store(value_type, name)

    return StatePayload(
        label=str(nominal_key[-1]),
        semantics=ValueSemantics.COPY,
        native_type=value_type,
        host_size=size,
        alignment=alignment,
        layout_key=(payload_type.name, storage_type.name),
        replacement=True,
        bind_function=_copy_bind(value_type),
        store_factory=store_factory,
    )


@cache
def _struct_store(layout: StructLayout, name: str):
    view_type = struct_ptr_type(layout)

    @intrinsic
    def store(typingctx, slot, rhs):
        if slot != types.voidptr:
            return None
        if rhs != view_type and not (isinstance(rhs, InputValueType) and rhs.payload_type == view_type):
            raise TypingError(f"cannot store {rhs} in State[{layout.name}] '{name}': expected {view_type}")
        sig = view_type(slot, rhs)

        def codegen(context, builder, signature, args):
            byte_ptr = llvm_ir.IntType(8).as_pointer()
            size_type = context.get_value_type(types.intp)
            memmove = builder.module.declare_intrinsic("llvm.memmove", [byte_ptr, byte_ptr, size_type])
            dst = builder.bitcast(args[0], byte_ptr)
            source = context.make_helper(builder, rhs, value=args[1]).value if isinstance(rhs, InputValueType) else args[1]
            src = builder.bitcast(source, byte_ptr)
            builder.call(memmove, [dst, src, context.get_constant(types.intp, layout.size), llvm_ir.Constant(llvm_ir.IntType(1), 0)])
            return builder.bitcast(args[0], context.get_value_type(view_type))

        return sig, codegen

    return store


@cache
def borrowed_struct_payload(layout: StructLayout) -> StatePayload:
    """Describe a struct view whose replacement copies into the host cell."""

    def store_factory(name):
        return _struct_store(layout, name)

    return StatePayload(
        label=layout.name,
        semantics=ValueSemantics.BORROWED_VIEW,
        native_type=struct_ptr_type(layout),
        host_size=layout.size,
        alignment=1,
        layout_key=layout.key,
        replacement=True,
        bind_function=struct_view(layout),
        store_factory=store_factory,
        mutable=True,
    )


@cache
def _container_marker(native_type: types.Type):
    @intrinsic
    def bind(typingctx, value):
        if value != native_type:
            return None
        sig = native_type(value)

        def codegen(context, builder, signature, args):
            return args[0]

        return sig, codegen

    return bind


@cache
def _no_replacement(label: str, name: str):
    @intrinsic
    def store(typingctx, slot, rhs):
        raise TypingError(f"State[{label}] '{name}' does not support whole-object replacement")

    return store


@cache
def borrowed_container_payload(label: str, native_type: types.Type, layout_key: tuple) -> StatePayload:
    """Describe a host-owned container that permits mutation, not replacement."""

    def store_factory(name):
        return _no_replacement(label, name)

    return StatePayload(
        label=label,
        semantics=ValueSemantics.BORROWED_VIEW,
        native_type=native_type,
        host_size=ctypes.sizeof(ctypes.c_void_p),
        alignment=ctypes.alignment(ctypes.c_void_p),
        layout_key=layout_key,
        replacement=False,
        bind_function=_container_marker(native_type),
        store_factory=store_factory,
    )


@cache
def borrowed_ref_payload(label: str, native_type: types.Type, layout_key: tuple) -> StatePayload:
    """Describe an external pointer; the host must keep its target alive."""
    from numba_cfunc_compiler.types.native.ffi import FFIRefType, ffi_ref_from_voidptr

    if not isinstance(native_type, FFIRefType):
        raise TypeError("Borrowed FFI references require an FFIRefType")

    def store_factory(name):
        return _no_replacement(label, name)

    return StatePayload(
        label=label,
        semantics=ValueSemantics.BORROWED_VIEW,
        native_type=native_type,
        host_size=ctypes.sizeof(ctypes.c_void_p),
        alignment=ctypes.alignment(ctypes.c_void_p),
        layout_key=layout_key,
        replacement=False,
        bind_function=ffi_ref_from_voidptr(native_type),
        store_factory=store_factory,
    )
