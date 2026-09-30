"""
Standalone dict type backed by dictobject.c.

Supported operations:
- len(d) - get length
- d[key] - get value at key
- d[key] = value - set value at key
- key in d - containment check
- d.get(key, default) - get with default
- d.pop(key) - remove and return value
- d.clear() - remove all items

Iteration is typed and lowered by Numba for keys and items.

Memory is managed via state slots, with allocation on first use.
"""

import operator

from llvmlite import ir
from numba.core import cgutils, types
from numba.core.datamodel import models
from numba.core.imputils import RefType, impl_ret_borrowed, iternext_impl
from numba.extending import intrinsic, lower_builtin, overload, overload_method, register_model

from numba_cfunc_compiler.standalone.utils import (
    get_llvm_type_for_numba_dtype as _get_llvm_type,
    get_or_declare_function,
    i8,
    i8ptr,
    i32,
    i64,
    prepare_int_key_for_lookup as _prepare_key,
    store_value_to_buffer,
)
from numba_cfunc_compiler.type_registry import NumbaTypeRegistry

# Default minimum size for dict (matches D_MINSIZE in dictobject.c)
D_MINSIZE = 8


# Typing for StandaloneDictType
class StandaloneDictType(types.IterableType):
    """
    Custom Numba type for NRT-free dicts backed by NB_Dict*.

    The dict stores key-value pairs of primitive types (int64, float64, int8).
    """

    def __init__(self, key_type, value_type):
        self.key_type = key_type  # Key type: types.int64, types.int8
        self.value_type = value_type  # Value type: types.int64, types.float64, types.int8
        self.key_size = self._get_type_size(key_type)
        self.value_size = self._get_type_size(value_type)
        super().__init__(name=f"StandaloneDict[{key_type}, {value_type}]")

    @property
    def iterator_type(self):
        return StandaloneDictIteratorType(self, "keys")

    @staticmethod
    def _get_type_size(dtype):
        """Get the size in bytes for the type."""
        try:
            return NumbaTypeRegistry.get_size_for_numba_type(dtype)
        except KeyError:
            raise TypeError(f"Unsupported type for StandaloneDict: {dtype}")

    @property
    def key(self):
        return (self.key_type, self.value_type)


class StandaloneDictIterableType(types.IterableType):
    def __init__(self, parent, mode):
        self.parent = parent
        self.mode = mode
        super().__init__(f"{mode}[{parent}]")

    @property
    def key(self):
        return (self.parent, self.mode)

    @property
    def iterator_type(self):
        return StandaloneDictIteratorType(self.parent, self.mode)


class StandaloneDictIteratorType(types.SimpleIteratorType):
    def __init__(self, parent, mode):
        self.parent = parent
        self.mode = mode
        yield_type = types.Tuple((parent.key_type, parent.value_type)) if mode == "items" else parent.key_type
        super().__init__(f"iter[{mode}[{parent}]]", yield_type)

    @property
    def key(self):
        return (self.parent, self.mode)


# LLVM Data Model
@register_model(StandaloneDictType)
class StandaloneDictModel(models.PrimitiveModel):
    """
    Data model: StandaloneDict is represented as an opaque pointer (i8*).
    This pointer points to an NB_Dict struct allocated by numba_dict_new().
    """

    def __init__(self, dmm, fe_type):
        be_type = i8ptr()  # i8* (voidptr)
        super().__init__(dmm, fe_type, be_type)


@register_model(StandaloneDictIterableType)
class StandaloneDictIterableModel(models.PrimitiveModel):
    def __init__(self, dmm, fe_type):
        super().__init__(dmm, fe_type, i8ptr())


@register_model(StandaloneDictIteratorType)
class StandaloneDictIteratorModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        super().__init__(dmm, fe_type, [("state", types.voidptr), ("parent", fe_type.parent)])


@lower_builtin("getiter", StandaloneDictType)
@lower_builtin("getiter", StandaloneDictIterableType)
def lower_dict_getiter(context, builder, sig, args):
    iterator = context.make_helper(builder, sig.return_type)
    iterator.parent = args[0]
    state = cgutils.alloca_once(builder, ir.ArrayType(i8(), 32), zfill=True)
    iterator.state = builder.bitcast(state, i8ptr())
    init_fn = get_or_declare_function(builder.module, "numba_dict_iter", ir.FunctionType(ir.VoidType(), [i8ptr(), i8ptr()]))
    builder.call(init_fn, [iterator.state, args[0]])
    return impl_ret_borrowed(context, builder, sig.return_type, iterator._getvalue())


@lower_builtin("iternext", StandaloneDictIteratorType)
@iternext_impl(RefType.BORROWED)
def lower_dict_iternext(context, builder, sig, args, result):
    iterator_type = sig.args[0]
    iterator = context.make_helper(builder, iterator_type, args[0])
    next_fn = get_or_declare_function(
        builder.module,
        "numba_dict_iter_next",
        ir.FunctionType(i32(), [i8ptr(), i8ptr().as_pointer(), i8ptr().as_pointer()]),
    )
    key_ptr = builder.alloca(i8ptr())
    value_ptr = builder.alloca(i8ptr())
    status = builder.call(next_fn, [iterator.state, key_ptr, value_ptr])
    valid = builder.icmp_signed("==", status, ir.Constant(i32(), 0))
    result.set_valid(valid)
    with builder.if_then(valid):
        key = _load_typed_value(builder, builder.load(key_ptr), iterator_type.parent.key_type)
        if iterator_type.mode == "items":
            value = _load_typed_value(builder, builder.load(value_ptr), iterator_type.parent.value_type)
            result.yield_(context.make_tuple(builder, iterator_type.yield_type, [key, value]))
        else:
            result.yield_(key)


@intrinsic
def standalone_dict_items_view(typingctx, dict_type):
    if isinstance(dict_type, StandaloneDictType):
        view_type = StandaloneDictIterableType(dict_type, "items")
        return view_type(dict_type), lambda context, builder, signature, args: args[0]


@intrinsic
def standalone_dict_keys_view(typingctx, dict_type):
    if isinstance(dict_type, StandaloneDictType):
        view_type = StandaloneDictIterableType(dict_type, "keys")
        return view_type(dict_type), lambda context, builder, signature, args: args[0]


@overload_method(StandaloneDictType, "items")
def overload_dict_items(d):
    def impl(d):
        return standalone_dict_items_view(d)

    return impl


@overload_method(StandaloneDictType, "keys")
def overload_dict_keys(d):
    def impl(d):
        return standalone_dict_keys_view(d)

    return impl


def _call_dict_lookup(builder, dict_ptr, key, key_type, val_lltype):
    """
    Call numba_dict_lookup and return (index, value_ptr, val_char_ptr).

    This handles key preparation, value buffer allocation, and the lookup call.
    """
    fnty = ir.FunctionType(i64(), [i8ptr(), i8ptr(), i64(), i8ptr()])
    fn = get_or_declare_function(builder.module, "numba_dict_lookup", fnty)

    key_char_ptr, hash_val = _prepare_key(builder, key, key_type)

    val_ptr = builder.alloca(val_lltype)
    val_char_ptr = builder.bitcast(val_ptr, i8ptr())

    ix = builder.call(fn, [dict_ptr, key_char_ptr, hash_val, val_char_ptr])

    return ix, val_ptr, hash_val


def _call_dict_insert(builder, dict_ptr, key, value, key_type, val_type):
    """
    Call numba_dict_insert_ez.

    Returns the result of the insert call.
    """
    fnty = ir.FunctionType(i32(), [i8ptr(), i8ptr(), i64(), i8ptr()])
    fn = get_or_declare_function(builder.module, "numba_dict_insert_ez", fnty)

    key_char_ptr, hash_val = _prepare_key(builder, key, key_type)
    _, val_char_ptr = store_value_to_buffer(builder, value, val_type)

    return builder.call(fn, [dict_ptr, key_char_ptr, hash_val, val_char_ptr])


@intrinsic
def standalone_dict_new(typingctx, key_size_ty, val_size_ty):
    """
    Allocate a new NB_Dict.

    int numba_dict_new(NB_Dict **out, Py_ssize_t size, Py_ssize_t key_size, Py_ssize_t val_size)

    Returns: NB_Dict* as voidptr (i8*)
    """
    if isinstance(key_size_ty, types.Integer) and isinstance(val_size_ty, types.Integer):
        sig = types.voidptr(key_size_ty, val_size_ty)

        def codegen(context, builder, signature, args):
            key_size, val_size = args

            i8ptrptr = i8ptr().as_pointer()

            # int numba_dict_new(NB_Dict **out, Py_ssize_t size, Py_ssize_t key_size, Py_ssize_t val_size)
            fnty = ir.FunctionType(i32(), [i8ptrptr, i64(), i64(), i64()])
            fn = get_or_declare_function(builder.module, "numba_dict_new", fnty)

            # Allocate space for the output pointer
            out_ptr = builder.alloca(i8ptr())
            builder.store(ir.Constant(i8ptr(), None), out_ptr)

            # Convert sizes to i64 if needed
            key_size_i64 = builder.sext(key_size, i64()) if key_size.type.width < 64 else key_size
            val_size_i64 = builder.sext(val_size, i64()) if val_size.type.width < 64 else val_size

            # Call numba_dict_new with D_MINSIZE
            min_size = ir.Constant(i64(), D_MINSIZE)
            builder.call(fn, [out_ptr, min_size, key_size_i64, val_size_i64])

            # Load and return the dict pointer
            return builder.load(out_ptr)

        return sig, codegen


@intrinsic
def standalone_dict_free(typingctx, dict_ty):
    """Release an NB_Dict allocated by :func:`standalone_dict_new`."""
    if isinstance(dict_ty, StandaloneDictType):
        sig = types.void(dict_ty)

        def codegen(context, builder, signature, args):
            [dict_ptr] = args
            fnty = ir.FunctionType(ir.VoidType(), [i8ptr()])
            fn = get_or_declare_function(builder.module, "numba_dict_free", fnty)
            builder.call(fn, [dict_ptr])
            return context.get_dummy_value()

        return sig, codegen


@intrinsic
def standalone_dict_length(typingctx, dict_ty):
    """
    Get the length of a StandaloneDict.

    Py_ssize_t numba_dict_length(NB_Dict *d)
    """
    if isinstance(dict_ty, StandaloneDictType):
        sig = types.int64(dict_ty)

        def codegen(context, builder, signature, args):
            [dict_ptr] = args

            fnty = ir.FunctionType(i64(), [i8ptr()])
            fn = get_or_declare_function(builder.module, "numba_dict_length", fnty)

            return builder.call(fn, [dict_ptr])

        return sig, codegen


@intrinsic
def standalone_dict_lookup(typingctx, dict_ty, key_ty):
    """Lookup a key in a StandaloneDict and return the value."""
    if isinstance(dict_ty, StandaloneDictType) and isinstance(key_ty, types.Integer):
        sig = dict_ty.value_type(dict_ty, key_ty)

        def codegen(context, builder, signature, args):
            dict_ptr, key = args
            dt = signature.args[0]
            val_lltype = _get_llvm_type(dt.value_type)
            _, val_ptr, _ = _call_dict_lookup(builder, dict_ptr, key, dt.key_type, val_lltype)
            return builder.load(val_ptr)

        return sig, codegen


@intrinsic
def standalone_dict_contains(typingctx, dict_ty, key_ty):
    """Check if a key exists in the dict. Returns 1 if found, 0 if not."""
    if isinstance(dict_ty, StandaloneDictType) and isinstance(key_ty, types.Integer):
        sig = types.int64(dict_ty, key_ty)

        def codegen(context, builder, signature, args):
            dict_ptr, key = args
            dt = signature.args[0]
            val_lltype = _get_llvm_type(dt.value_type)
            ix, _, _ = _call_dict_lookup(builder, dict_ptr, key, dt.key_type, val_lltype)
            zero = ir.Constant(i64(), 0)
            found = builder.icmp_signed(">=", ix, zero)
            return builder.zext(found, i64())

        return sig, codegen


@intrinsic
def standalone_dict_insert(typingctx, dict_ty, key_ty, val_ty):
    """Insert a key-value pair into the dict."""
    if isinstance(dict_ty, StandaloneDictType) and isinstance(key_ty, types.Integer) and isinstance(val_ty, (types.Integer, types.Float)):
        sig = types.int32(dict_ty, key_ty, val_ty)

        def codegen(context, builder, signature, args):
            dict_ptr, key, value = args
            dt = signature.args[0]
            return _call_dict_insert(builder, dict_ptr, key, value, dt.key_type, dt.value_type)

        return sig, codegen


@intrinsic
def standalone_dict_get(typingctx, dict_ty, key_ty, default_ty):
    """Lookup key; return value if found, else default (single lookup)."""
    if isinstance(dict_ty, StandaloneDictType) and isinstance(key_ty, types.Integer):
        sig = dict_ty.value_type(dict_ty, key_ty, default_ty)

        def codegen(context, builder, signature, args):
            d, key, default = args
            dt = signature.args[0]
            val_lltype = _get_llvm_type(dt.value_type)
            ix, val_ptr, _ = _call_dict_lookup(builder, d, key, dt.key_type, val_lltype)
            zero = ir.Constant(i64(), 0)
            found = builder.icmp_signed(">=", ix, zero)
            value = builder.load(val_ptr)
            return builder.select(found, value, default)

        return sig, codegen


@intrinsic
def standalone_dict_pop(typingctx, dict_ty, key_ty):
    """Lookup key, delete it, and return the value (single lookup)."""
    if isinstance(dict_ty, StandaloneDictType) and isinstance(key_ty, types.Integer):
        sig = dict_ty.value_type(dict_ty, key_ty)

        def codegen(context, builder, signature, args):
            d, key = args
            dt = signature.args[0]
            val_lltype = _get_llvm_type(dt.value_type)
            ix, val_ptr, hash_val = _call_dict_lookup(builder, d, key, dt.key_type, val_lltype)
            del_fnty = ir.FunctionType(i32(), [i8ptr(), i64(), i64()])
            del_fn = get_or_declare_function(builder.module, "numba_dict_delitem", del_fnty)
            builder.call(del_fn, [d, hash_val, ix])
            return builder.load(val_ptr)

        return sig, codegen


@intrinsic
def standalone_dict_popitem(typingctx, dict_ty):
    """
    Pop an arbitrary item from the dict.
    Returns the key (value is discarded).
    """
    if isinstance(dict_ty, StandaloneDictType):
        sig = dict_ty.key_type(dict_ty)

        def codegen(context, builder, signature, args):
            [dict_ptr] = args
            dt = signature.args[0]

            fnty = ir.FunctionType(i32(), [i8ptr(), i8ptr(), i8ptr()])
            fn = get_or_declare_function(builder.module, "numba_dict_popitem", fnty)

            key_lltype = _get_llvm_type(dt.key_type)
            val_lltype = _get_llvm_type(dt.value_type)

            key_ptr = builder.alloca(key_lltype)
            key_char_ptr = builder.bitcast(key_ptr, i8ptr())
            val_ptr = builder.alloca(val_lltype)
            val_char_ptr = builder.bitcast(val_ptr, i8ptr())

            builder.call(fn, [dict_ptr, key_char_ptr, val_char_ptr])
            return builder.load(key_ptr)

        return sig, codegen


@intrinsic
def standalone_dict_clear(typingctx, dict_ty):
    """Clear all items from a StandaloneDict."""
    if isinstance(dict_ty, StandaloneDictType):
        sig = types.int32(dict_ty)

        def codegen(context, builder, signature, args):
            [dict_ptr] = args

            fnty = ir.FunctionType(i32(), [i8ptr()])
            fn = get_or_declare_function(builder.module, "numba_dict_clear", fnty)

            return builder.call(fn, [dict_ptr])

        return sig, codegen


@intrinsic
def standalone_dict_from_voidptr(typingctx, voidptr_ty, key_type_ty, val_type_ty):
    """
    Cast a voidptr (from state slot) to a typed StandaloneDictType.
    """
    if voidptr_ty == types.voidptr and isinstance(key_type_ty, types.Literal) and isinstance(val_type_ty, types.Literal):
        key_type_name = key_type_ty.literal_value
        val_type_name = val_type_ty.literal_value

        key_map = NumbaTypeRegistry.get_numba_type_map(NumbaTypeRegistry.get_dict_key_types())
        val_map = NumbaTypeRegistry.get_numba_type_map(NumbaTypeRegistry.get_dict_value_types())

        if key_type_name in key_map and val_type_name in val_map:
            result_type = StandaloneDictType(key_map[key_type_name], val_map[val_type_name])
            sig = result_type(voidptr_ty, key_type_ty, val_type_ty)

            def codegen(context, builder, signature, args):
                [voidptr, _, _] = args
                return voidptr

            return sig, codegen


@intrinsic
def standalone_dict_to_voidptr(typingctx, dict_ty):
    """
    Cast a StandaloneDictType back to voidptr for storage in state slot.
    """
    if isinstance(dict_ty, StandaloneDictType):
        sig = types.voidptr(dict_ty)

        def codegen(context, builder, signature, args):
            [dict_ptr] = args
            return dict_ptr

        return sig, codegen


def _load_typed_value(builder, ptr, numba_type):
    """Load a value from a C dict cell using its Numba element type."""
    lltype = _get_llvm_type(numba_type)
    return builder.load(builder.bitcast(ptr, lltype.as_pointer()))


# overloads
@overload(len)
def overload_len_standalone_dict(d):
    """Overload len() for StandaloneDictType."""
    if isinstance(d, StandaloneDictType):

        def impl(d):
            return standalone_dict_length(d)

        return impl


@overload(operator.getitem)
def overload_getitem_standalone_dict(d, key):
    """Overload d[key] for StandaloneDictType."""
    if isinstance(d, StandaloneDictType) and isinstance(key, types.Integer):

        def impl(d, key):
            return standalone_dict_lookup(d, key)

        return impl


@overload(operator.setitem)
def overload_setitem_standalone_dict(d, key, value):
    """Overload d[key] = value for StandaloneDictType."""
    if isinstance(d, StandaloneDictType) and isinstance(key, types.Integer) and isinstance(value, (types.Integer, types.Float)):

        def impl(d, key, value):
            standalone_dict_insert(d, key, value)

        return impl


@overload(operator.contains)
def overload_contains_standalone_dict(d, key):
    """Overload 'key in d' for StandaloneDictType."""
    if isinstance(d, StandaloneDictType) and isinstance(key, types.Integer):

        def impl(d, key):
            return standalone_dict_contains(d, key) != 0

        return impl


# method overloads
@overload_method(StandaloneDictType, "get")
def overload_dict_get(d, key, default):
    if isinstance(key, types.Integer):

        def impl(d, key, default):
            return standalone_dict_get(d, key, default)

        return impl


@overload_method(StandaloneDictType, "pop")
def overload_dict_pop(d, key):
    if isinstance(key, types.Integer):

        def impl(d, key):
            return standalone_dict_pop(d, key)

        return impl


@overload_method(StandaloneDictType, "clear")
def overload_dict_clear(d):
    def impl(d):
        standalone_dict_clear(d)

    return impl


@overload_method(StandaloneDictType, "contains")
def overload_dict_contains_method(d, key):
    if isinstance(key, types.Integer):

        def impl(d, key):
            return standalone_dict_contains(d, key) != 0

        return impl
