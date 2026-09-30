"""Nominal pointer values and direct, typed external method calls."""

from functools import cache

from llvmlite import ir
from numba import types
from numba.core import cgutils
from numba.core.imputils import RefType, impl_ret_borrowed, iternext_impl
from numba.extending import intrinsic, lower_builtin, models, register_model


class FFIRefType(types.Type):
    """A borrowed external object pointer with a nominal method set."""

    def __init__(self, name: str):
        self.ref_name = name
        super().__init__(f"FFIRef[{name}]")

    @property
    def key(self):
        return self.ref_name


class FFISideRefType(types.Type):
    """A side pointer and the selector required by side-specific C symbols."""

    def __init__(self, name: str):
        self.ref_name = name
        super().__init__(f"FFISideRef[{name}]")

    @property
    def key(self):
        return self.ref_name


class FFIRangeType(types.IterableType):
    def __init__(self, element_type: FFIRefType, next_symbol: str):
        self.element_type = element_type
        self.next_symbol = next_symbol
        super().__init__(f"FFIRange[{element_type}:{next_symbol}]")

    @property
    def key(self):
        return self.element_type, self.next_symbol

    @property
    def iterator_type(self):
        return FFIRangeIteratorType(self)


class FFIRangeIteratorType(types.SimpleIteratorType):
    def __init__(self, parent: FFIRangeType):
        self.parent = parent
        super().__init__(f"iter[{parent}]", parent.element_type)

    @property
    def key(self):
        return self.parent


@register_model(FFIRefType)
class FFIRefModel(models.PrimitiveModel):
    def __init__(self, dmm, fe_type):
        super().__init__(dmm, fe_type, ir.IntType(8).as_pointer())


@register_model(FFISideRefType)
class FFISideRefModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        super().__init__(dmm, fe_type, [("pointer", types.voidptr), ("bid", types.boolean)])


@register_model(FFIRangeType)
class FFIRangeModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        super().__init__(dmm, fe_type, [("begin", fe_type.element_type), ("end", fe_type.element_type)])


@register_model(FFIRangeIteratorType)
class FFIRangeIteratorModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        super().__init__(dmm, fe_type, [("current", types.EphemeralPointer(fe_type.parent.element_type)), ("end", fe_type.parent.element_type)])


@cache
def ffi_ref_type(name: str):
    if not name.isidentifier():
        raise ValueError("FFI reference requires an identifier")
    return FFIRefType(name)


@cache
def ffi_side_ref_type(name: str):
    if not name.isidentifier():
        raise ValueError("FFI side reference requires an identifier")
    return FFISideRefType(name)


@cache
def ffi_side_from_voidptr(side_type: FFISideRefType):
    @intrinsic
    def wrap(typingctx, pointer, bid):
        if pointer != types.voidptr or not isinstance(bid, types.Boolean):
            return None
        sig = side_type(pointer, bid)

        def codegen(context, builder, signature, args):
            result = context.make_helper(builder, side_type)
            result.pointer = args[0]
            result.bid = args[1]
            return result._getvalue()

        return sig, codegen

    return wrap


@cache
def ffi_ref_from_voidptr(ref_type: FFIRefType):
    @intrinsic
    def wrap(typingctx, pointer):
        if pointer != types.voidptr:
            return None
        sig = ref_type(pointer)

        def codegen(context, builder, signature, args):
            return builder.bitcast(args[0], context.get_value_type(ref_type))

        return sig, codegen

    return wrap


def declare_symbol(builder, symbol, native_sig):
    existing = builder.module.globals.get(symbol)
    if existing is not None:
        if existing.function_type != native_sig:
            raise TypeError(f"FFI symbol '{symbol}' has a conflicting native signature")
        return existing
    function = ir.Function(builder.module, native_sig, name=symbol)
    function.attributes.add("nounwind")
    return function


@cache
def ffi_range_type(element_type, next_symbol):
    return FFIRangeType(element_type, next_symbol)


@cache
def make_ffi_range(range_type):
    @intrinsic
    def make(typingctx, begin, end):
        if begin != range_type.element_type or end != range_type.element_type:
            return None
        sig = range_type(begin, end)

        def codegen(context, builder, signature, args):
            result = context.make_helper(builder, range_type)
            result.begin, result.end = args
            return result._getvalue()

        return sig, codegen

    return make


@lower_builtin("getiter", FFIRangeType)
def lower_ffi_range_getiter(context, builder, sig, args):
    source = context.make_helper(builder, sig.args[0], value=args[0])
    iterator = context.make_helper(builder, sig.return_type)
    current = cgutils.alloca_once(builder, context.get_value_type(sig.args[0].element_type))
    builder.store(source.begin, current)
    iterator.current = current
    iterator.end = source.end
    return impl_ret_borrowed(context, builder, sig.return_type, iterator._getvalue())


@lower_builtin("iternext", FFIRangeIteratorType)
@iternext_impl(RefType.BORROWED)
def lower_ffi_range_iternext(context, builder, sig, args, result):
    iterator = context.make_helper(builder, sig.args[0], args[0])
    current = builder.load(iterator.current)
    valid = builder.icmp_unsigned("!=", current, iterator.end)
    result.set_valid(valid)
    with builder.if_then(valid):
        result.yield_(current)
        next_symbol = sig.args[0].parent.next_symbol
        native_sig = ir.FunctionType(ir.IntType(8).as_pointer(), [ir.IntType(8).as_pointer()])
        next_function = declare_symbol(builder, next_symbol, native_sig)
        builder.store(builder.call(next_function, [current]), iterator.current)
