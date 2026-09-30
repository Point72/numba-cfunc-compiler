"""
Standalone list type backed by listobject.c.

Supported operations:
- len(lst) - get length
- lst[i] - get item at index (supports negative indexing)
- lst[i] = x - set item at index (supports negative indexing)
- lst.append(x) - append item
- lst.pop() - remove and return last item
- lst.pop(i) - remove and return item at index
- lst.clear() - remove all items

Memory is managed via state slots, with allocation on first use.
"""

import operator

from llvmlite import ir
from numba.core import cgutils, types
from numba.core.datamodel import models
from numba.core.imputils import RefType, impl_ret_borrowed, iternext_impl
from numba.extending import intrinsic, lower_builtin, overload, overload_method, register_model

from numba_cfunc_compiler.types.native.input import payload_type_of, unwrap_source
from numba_cfunc_compiler.types.native.llvm import (
    convert_to_i8,
    convert_to_i64,
    get_llvm_type_for_numba_dtype,
    get_or_declare_function,
    i8ptr,
    i32,
    i64,
)
from numba_cfunc_compiler.types.registry import NumbaTypeRegistry


# Typing for ListValueType
class ListValueType(types.IterableType):
    """
    Custom Numba type for NRT-free lists backed by NB_List*.

    The list stores elements of a single primitive type (int64, float64, int8).
    """

    def __init__(self, dtype):
        self.dtype = dtype  # Element type: types.int64, types.float64, types.int8
        self.item_size = self._get_item_size(dtype)
        super().__init__(name=f"StandaloneList[{dtype}]")

    @staticmethod
    def _get_item_size(dtype):
        """Get the size in bytes for the element type."""
        try:
            return NumbaTypeRegistry.get_size_for_numba_type(dtype)
        except KeyError:
            raise TypeError(f"Unsupported element type for StandaloneList: {dtype}")

    @property
    def key(self):
        return self.dtype

    @property
    def iterator_type(self):
        return StandaloneListIteratorType(self)


class StandaloneListIteratorType(types.SimpleIteratorType):
    def __init__(self, parent):
        self.parent = parent
        super().__init__(f"iter[{parent}]", parent.dtype)

    @property
    def key(self):
        return self.parent


# LLVM Data Model
@register_model(ListValueType)
class StandaloneListModel(models.PrimitiveModel):
    """
    Data model: StandaloneList is represented as an opaque pointer (i8*).
    This pointer points to an NB_List struct allocated by numba_list_new().
    """

    def __init__(self, dmm, fe_type):
        be_type = i8ptr()  # i8* (voidptr)
        super().__init__(dmm, fe_type, be_type)


@register_model(StandaloneListIteratorType)
class StandaloneListIteratorModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        super().__init__(dmm, fe_type, [("parent", fe_type.parent), ("index", types.EphemeralPointer(types.intp))])


@lower_builtin("getiter", ListValueType)
def lower_list_getiter(context, builder, sig, args):
    iterator = context.make_helper(builder, sig.return_type)
    iterator.parent = args[0]
    index_ptr = cgutils.alloca_once(builder, i64())
    builder.store(ir.Constant(i64(), 0), index_ptr)
    iterator.index = index_ptr
    return impl_ret_borrowed(context, builder, sig.return_type, iterator._getvalue())


@lower_builtin("iternext", StandaloneListIteratorType)
@iternext_impl(RefType.BORROWED)
def lower_list_iternext(context, builder, sig, args, result):
    iterator = context.make_helper(builder, sig.args[0], args[0])
    index = builder.load(iterator.index)
    length_fn = get_or_declare_function(builder.module, "numba_list_length", ir.FunctionType(i64(), [i8ptr()]))
    length = builder.call(length_fn, [iterator.parent])
    valid = builder.icmp_signed("<", index, length)
    result.set_valid(valid)
    with builder.if_then(valid):
        item = _make_getitem_codegen(sig.args[0].parent.dtype)(context, builder, sig, [iterator.parent, index])
        result.yield_(item)
        builder.store(builder.add(index, ir.Constant(i64(), 1)), iterator.index)


# Intrinsics
def _make_append_codegen(dtype):
    """Create a codegen function for list append with the given dtype."""
    lltype = get_llvm_type_for_numba_dtype(dtype)

    def codegen(context, builder, signature, args):
        lst_ptr, value = args

        fnty = ir.FunctionType(i32(), [i8ptr(), i8ptr()])
        fn = get_or_declare_function(builder.module, "numba_list_append", fnty)

        # Convert and store value
        if dtype == types.int64:
            converted = convert_to_i64(builder, value)
        elif dtype == types.int8:
            converted = convert_to_i8(builder, value)
        else:
            converted = value

        value_ptr = builder.alloca(lltype)
        builder.store(converted, value_ptr)
        item_ptr = builder.bitcast(value_ptr, i8ptr())

        return builder.call(fn, [lst_ptr, item_ptr])

    return codegen


def _make_getitem_codegen(dtype):
    """Create a codegen function for list getitem with the given dtype."""
    lltype = get_llvm_type_for_numba_dtype(dtype)

    def codegen(context, builder, signature, args):
        lst_ptr, index = args

        fnty = ir.FunctionType(i32(), [i8ptr(), i64(), i8ptr()])
        fn = get_or_declare_function(builder.module, "numba_list_getitem", fnty)

        out_ptr = builder.alloca(lltype)
        out_char_ptr = builder.bitcast(out_ptr, i8ptr())
        index_i64 = convert_to_i64(builder, index)

        builder.call(fn, [lst_ptr, index_i64, out_char_ptr])
        return builder.load(out_ptr)

    return codegen


def _make_setitem_codegen(dtype):
    """Create a codegen function for list setitem with the given dtype."""
    lltype = get_llvm_type_for_numba_dtype(dtype)

    def codegen(context, builder, signature, args):
        lst_ptr, index, value = args

        fnty = ir.FunctionType(i32(), [i8ptr(), i64(), i8ptr()])
        fn = get_or_declare_function(builder.module, "numba_list_setitem", fnty)

        index_i64 = convert_to_i64(builder, index)

        # Convert and store value
        if dtype == types.int64:
            converted = convert_to_i64(builder, value)
        elif dtype == types.int8:
            converted = convert_to_i8(builder, value)
        else:
            converted = value

        value_ptr = builder.alloca(lltype)
        builder.store(converted, value_ptr)
        item_ptr = builder.bitcast(value_ptr, i8ptr())

        return builder.call(fn, [lst_ptr, index_i64, item_ptr])

    return codegen


@intrinsic
def standalone_list_new(typingctx, item_size_ty, allocated_ty):
    """
    Allocate a new NB_List.

    int numba_list_new(NB_List **out, Py_ssize_t item_size, Py_ssize_t allocated)

    Returns: NB_List* as voidptr (i8*)
    """
    if isinstance(item_size_ty, types.Integer) and isinstance(allocated_ty, types.Integer):
        sig = types.voidptr(item_size_ty, allocated_ty)

        def codegen(context, builder, signature, args):
            item_size, allocated = args

            # Declare: int numba_list_new(NB_List **out, Py_ssize_t item_size, Py_ssize_t allocated)
            i8ptrptr = i8ptr().as_pointer()

            fnty = ir.FunctionType(i32(), [i8ptrptr, i64(), i64()])
            fn = get_or_declare_function(builder.module, "numba_list_new", fnty)

            # Allocate space for the output pointer
            out_ptr = builder.alloca(i8ptr())
            builder.store(ir.Constant(i8ptr(), None), out_ptr)

            # Convert item_size and allocated to i64 if needed
            item_size_i64 = builder.sext(item_size, i64()) if item_size.type.width < 64 else item_size
            allocated_i64 = builder.sext(allocated, i64()) if allocated.type.width < 64 else allocated

            # Call numba_list_new
            builder.call(fn, [out_ptr, item_size_i64, allocated_i64])

            # Load and return the list pointer
            return builder.load(out_ptr)

        return sig, codegen


@intrinsic
def standalone_list_free(typingctx, lst_ty):
    """Release an NB_List allocated by :func:`standalone_list_new`."""
    if isinstance(lst_ty, ListValueType):
        sig = types.void(lst_ty)

        def codegen(context, builder, signature, args):
            [lst_ptr] = args
            fnty = ir.FunctionType(ir.VoidType(), [i8ptr()])
            fn = get_or_declare_function(builder.module, "numba_list_free", fnty)
            builder.call(fn, [lst_ptr])
            return context.get_dummy_value()

        return sig, codegen


@intrinsic
def standalone_list_length(typingctx, lst_ty):
    """
    Get the length of a StandaloneList.

    Py_ssize_t numba_list_length(NB_List *lp)
    """
    if isinstance(lst_ty, ListValueType):
        sig = types.int64(lst_ty)

        def codegen(context, builder, signature, args):
            [lst_ptr] = args

            # Declare: Py_ssize_t numba_list_length(NB_List *lp)
            fnty = ir.FunctionType(i64(), [i8ptr()])
            fn = get_or_declare_function(builder.module, "numba_list_length", fnty)

            return builder.call(fn, [lst_ptr])

        return sig, codegen


@intrinsic
def standalone_list_append(typingctx, lst_ty, value_ty):
    """Append a value to a StandaloneList."""
    if isinstance(lst_ty, ListValueType) and isinstance(value_ty, (types.Integer, types.Float, types.Boolean)):
        return types.int32(lst_ty, value_ty), _make_append_codegen(lst_ty.dtype)


@intrinsic
def standalone_list_getitem(typingctx, lst_ty, index_ty):
    """Get an item from a StandaloneList."""
    if isinstance(lst_ty, ListValueType) and isinstance(index_ty, types.Integer):
        return lst_ty.dtype(lst_ty, index_ty), _make_getitem_codegen(lst_ty.dtype)


@intrinsic
def standalone_list_from_voidptr(typingctx, voidptr_ty, dtype_ty):
    """
    Cast a voidptr (from state slot) to a typed ListValueType.

    This is needed to convert the raw pointer stored in state back to a typed list.
    """
    if voidptr_ty == types.voidptr and isinstance(dtype_ty, types.Literal):
        dtype_name = dtype_ty.literal_value
        elem_type_map = NumbaTypeRegistry.get_numba_type_map(NumbaTypeRegistry.get_list_element_types())

        if dtype_name not in elem_type_map:
            return None

        result_type = ListValueType(elem_type_map[dtype_name])
        sig = result_type(voidptr_ty, dtype_ty)

        def codegen(context, builder, signature, args):
            [voidptr, _] = args
            # Just return the pointer as-is; the type change is at Numba level only
            return voidptr

        return sig, codegen


@intrinsic
def standalone_list_to_voidptr(typingctx, lst_ty):
    """
    Cast a ListValueType back to voidptr for storage in state slot.
    """
    if isinstance(lst_ty, ListValueType):
        sig = types.voidptr(lst_ty)

        def codegen(context, builder, signature, args):
            [lst_ptr] = args
            return lst_ptr

        return sig, codegen


@intrinsic
def standalone_list_setitem(typingctx, lst_ty, index_ty, value_ty):
    """Set an item in a StandaloneList."""
    if isinstance(lst_ty, ListValueType) and isinstance(index_ty, types.Integer) and isinstance(value_ty, (types.Integer, types.Float)):
        return types.int32(lst_ty, index_ty, value_ty), _make_setitem_codegen(lst_ty.dtype)


@intrinsic
def standalone_list_delitem(typingctx, lst_ty, index_ty):
    """
    Delete an item from a StandaloneList.

    int numba_list_delitem(NB_List *lp, Py_ssize_t index)
    """
    if isinstance(lst_ty, ListValueType) and isinstance(index_ty, types.Integer):
        sig = types.int32(lst_ty, index_ty)

        def codegen(context, builder, signature, args):
            lst_ptr, index = args

            fnty = ir.FunctionType(i32(), [i8ptr(), i64()])
            fn = get_or_declare_function(builder.module, "numba_list_delitem", fnty)

            # Convert index to i64
            index_i64 = builder.sext(index, i64()) if index.type.width < 64 else index

            return builder.call(fn, [lst_ptr, index_i64])

        return sig, codegen


@intrinsic
def standalone_list_resize(typingctx, lst_ty, newsize_ty):
    """
    Resize a StandaloneList (used for clear).

    int numba_list_resize(NB_List *lp, Py_ssize_t newsize)
    """
    if isinstance(lst_ty, ListValueType) and isinstance(newsize_ty, types.Integer):
        sig = types.int32(lst_ty, newsize_ty)

        def codegen(context, builder, signature, args):
            lst_ptr, newsize = args

            fnty = ir.FunctionType(i32(), [i8ptr(), i64()])
            fn = get_or_declare_function(builder.module, "numba_list_resize", fnty)

            # Convert newsize to i64
            newsize_i64 = builder.sext(newsize, i64()) if newsize.type.width < 64 else newsize

            return builder.call(fn, [lst_ptr, newsize_i64])

        return sig, codegen


@intrinsic
def _normalize_index(typingctx, index_ty, length_ty):
    """
    Normalize a potentially negative index to a positive one.
    If index < 0, returns index + length, otherwise returns index.
    """
    if isinstance(index_ty, types.Integer) and isinstance(length_ty, types.Integer):
        sig = types.int64(index_ty, length_ty)

        def codegen(context, builder, signature, args):
            index, length = args

            # Convert to i64
            index_i64 = builder.sext(index, i64()) if index.type.width < 64 else index
            length_i64 = builder.sext(length, i64()) if length.type.width < 64 else length

            zero = ir.Constant(i64(), 0)
            is_negative = builder.icmp_signed("<", index_i64, zero)

            # If negative, add length
            adjusted = builder.add(index_i64, length_i64)
            result = builder.select(is_negative, adjusted, index_i64)

            return result

        return sig, codegen


# Overloads (python syntax support)
@overload(len)
def overload_len_standalone_list(lst):
    """Overload len() for ListValueType."""
    if isinstance(lst, ListValueType):

        def impl(lst):
            return standalone_list_length(lst)

        return impl


# method overloads
@overload_method(ListValueType, "append")
def overload_list_append(lst, value):
    if isinstance(payload_type_of(value), (types.Integer, types.Float, types.Boolean)):

        def impl(lst, value):
            standalone_list_append(lst, unwrap_source(value))

        return impl


@overload_method(ListValueType, "pop")
def overload_list_pop(lst, index=None):
    if isinstance(index, (types.Omitted, types.NoneType)) or index is None:

        def impl(lst, index=None):
            idx = standalone_list_length(lst) - 1
            val = standalone_list_getitem(lst, idx)
            standalone_list_delitem(lst, idx)
            return val

        return impl
    elif isinstance(payload_type_of(index), types.Integer):

        def impl(lst, index=None):
            idx = _normalize_index(unwrap_source(index), standalone_list_length(lst))
            val = standalone_list_getitem(lst, idx)
            standalone_list_delitem(lst, idx)
            return val

        return impl


@overload_method(ListValueType, "clear")
def overload_list_clear(lst):
    def impl(lst):
        standalone_list_resize(lst, 0)

    return impl


@overload(operator.getitem)
def overload_getitem_standalone_list(lst, index):
    """Overload lst[i] for ListValueType with negative index support."""
    if isinstance(lst, ListValueType) and isinstance(payload_type_of(index), types.Integer):

        def impl(lst, index):
            idx = _normalize_index(unwrap_source(index), standalone_list_length(lst))
            return standalone_list_getitem(lst, idx)

        return impl


@overload(operator.setitem)
def overload_setitem_standalone_list(lst, index, value):
    """Overload lst[i] = x for ListValueType with negative index support."""
    if (
        isinstance(lst, ListValueType)
        and isinstance(payload_type_of(index), types.Integer)
        and isinstance(payload_type_of(value), (types.Integer, types.Float, types.Boolean))
    ):

        def impl(lst, index, value):
            idx = _normalize_index(unwrap_source(index), standalone_list_length(lst))
            standalone_list_setitem(lst, idx, unwrap_source(value))

        return impl
