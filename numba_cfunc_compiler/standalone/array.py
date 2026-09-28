"""NRT-free fixed arrays with direct LLVM element access."""

import operator

from llvmlite import ir
from numba.core import types
from numba.core.datamodel import models
from numba.extending import intrinsic, overload, register_model

from numba_cfunc_compiler.standalone.utils import get_llvm_type_for_numba_dtype, get_or_declare_function, i8ptr, i64
from numba_cfunc_compiler.type_registry import NumbaTypeRegistry


class StandaloneArrayType(types.Type):
    """An element pointer whose dtype and length are part of its Numba type."""

    def __init__(self, dtype, length):
        self.dtype = dtype
        self.length = length
        super().__init__(name=f"StandaloneArray[{dtype}, {length}]")

    @property
    def key(self):
        return self.dtype, self.length


@register_model(StandaloneArrayType)
class StandaloneArrayModel(models.PrimitiveModel):
    def __init__(self, dmm, fe_type):
        super().__init__(dmm, fe_type, i8ptr())


def _array_type(dtype_ty, length_ty):
    if not isinstance(dtype_ty, types.Literal) or not isinstance(length_ty, types.IntegerLiteral):
        return None
    dtype_map = NumbaTypeRegistry.get_numba_type_map((int, float, bool))
    dtype = dtype_map.get(dtype_ty.literal_value)
    length = length_ty.literal_value
    if dtype is None or type(length) is not int or length <= 0:
        return None
    return StandaloneArrayType(dtype, length)


@intrinsic
def standalone_array_new(typingctx, dtype_ty, length_ty):
    """Allocate a zeroed fixed array on the callback stack."""
    array_ty = _array_type(dtype_ty, length_ty)
    if array_ty is None:
        return None
    sig = array_ty(dtype_ty, length_ty)

    def codegen(context, builder, signature, args):
        item_ty = get_llvm_type_for_numba_dtype(array_ty.dtype)
        storage_ty = ir.ArrayType(item_ty, array_ty.length)
        with builder.goto_entry_block():
            storage = builder.alloca(storage_ty, name="fixed_array")
        builder.store(ir.Constant(storage_ty, None), storage)
        return builder.bitcast(storage, i8ptr())

    return sig, codegen


@intrinsic
def standalone_array_from_voidptr(typingctx, ptr_ty, dtype_ty, length_ty):
    """Give a host-owned raw state buffer its fixed array type."""
    array_ty = _array_type(dtype_ty, length_ty)
    if ptr_ty != types.voidptr or array_ty is None:
        return None
    sig = array_ty(ptr_ty, dtype_ty, length_ty)

    def codegen(context, builder, signature, args):
        return args[0]

    return sig, codegen


def _element_pointer(builder, array_ptr, index, array_ty, index_ty):
    """Return an element pointer after at most one dynamic range check."""
    if isinstance(index_ty, types.IntegerLiteral):
        position = index_ty.literal_value
        if position < 0:
            position += array_ty.length
        # The typing function rejects invalid literals before code generation.
        offset = ir.Constant(i64(), position)
    else:
        index_i64 = index
        if index.type.width < 64:
            index_i64 = builder.sext(index, i64()) if index_ty.signed else builder.zext(index, i64())
        if index_ty.signed:
            negative = builder.icmp_signed("<", index_i64, ir.Constant(i64(), 0))
            shifted = builder.add(index_i64, ir.Constant(i64(), array_ty.length))
            offset = builder.select(negative, shifted, index_i64)
        else:
            offset = index_i64
        valid = builder.icmp_unsigned("<", offset, ir.Constant(i64(), array_ty.length))
        with builder.if_then(builder.not_(valid)):
            trap = get_or_declare_function(builder.module, "llvm.trap", ir.FunctionType(ir.VoidType(), []))
            builder.call(trap, [])
            builder.unreachable()

    item_ty = get_llvm_type_for_numba_dtype(array_ty.dtype)
    typed_ptr = builder.bitcast(array_ptr, item_ty.as_pointer())
    return builder.gep(typed_ptr, [offset])


@intrinsic
def standalone_array_getitem(typingctx, array_ty, index_ty):
    if not isinstance(array_ty, StandaloneArrayType) or not isinstance(index_ty, types.Integer):
        return None
    if isinstance(index_ty, types.IntegerLiteral) and not -array_ty.length <= index_ty.literal_value < array_ty.length:
        raise IndexError(f"Constant array index {index_ty.literal_value} is out of range for length {array_ty.length}")
    sig = array_ty.dtype(array_ty, index_ty)

    def codegen(context, builder, signature, args):
        ptr = _element_pointer(builder, args[0], args[1], array_ty, index_ty)
        return builder.load(ptr)

    return sig, codegen


@intrinsic
def standalone_array_setitem(typingctx, array_ty, index_ty, value_ty):
    if not isinstance(array_ty, StandaloneArrayType) or not isinstance(index_ty, types.Integer):
        return None
    if not isinstance(value_ty, (types.Boolean, types.Integer, types.Float)):
        return None
    if isinstance(index_ty, types.IntegerLiteral) and not -array_ty.length <= index_ty.literal_value < array_ty.length:
        raise IndexError(f"Constant array index {index_ty.literal_value} is out of range for length {array_ty.length}")
    sig = types.void(array_ty, index_ty, value_ty)

    def codegen(context, builder, signature, args):
        ptr = _element_pointer(builder, args[0], args[1], array_ty, index_ty)
        value = context.cast(builder, args[2], value_ty, array_ty.dtype)
        builder.store(value, ptr)
        return context.get_dummy_value()

    return sig, codegen


@overload(len)
def overload_len_standalone_array(array):
    if isinstance(array, StandaloneArrayType):
        length = array.length

        def impl(array):
            return length

        return impl


@overload(operator.getitem, prefer_literal=True)
def overload_getitem_standalone_array(array, index):
    if isinstance(array, StandaloneArrayType) and isinstance(index, types.Integer):
        if isinstance(index, types.IntegerLiteral) and not -array.length <= index.literal_value < array.length:
            raise IndexError(f"Constant array index {index.literal_value} is out of range for length {array.length}")

        def impl(array, index):
            return standalone_array_getitem(array, index)

        return impl


@overload(operator.setitem, prefer_literal=True)
def overload_setitem_standalone_array(array, index, value):
    if isinstance(array, StandaloneArrayType) and isinstance(index, types.Integer):
        if isinstance(index, types.IntegerLiteral) and not -array.length <= index.literal_value < array.length:
            raise IndexError(f"Constant array index {index.literal_value} is out of range for length {array.length}")

        def impl(array, index, value):
            standalone_array_setitem(array, index, value)

        return impl
