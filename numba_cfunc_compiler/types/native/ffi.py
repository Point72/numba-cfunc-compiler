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


class FFITaggedRefType(types.Type):
    """A borrowed pointer and an integer selecting a native API variant."""

    def __init__(self, name: str, cases: tuple[str, ...]):
        self.ref_name = name
        self.cases = cases
        super().__init__(f"FFITaggedRef[{name}:{','.join(cases)}]")

    @property
    def key(self):
        return self.ref_name, self.cases

    def tag(self, name: str) -> int:
        try:
            return self.cases.index(name)
        except ValueError as exc:
            raise ValueError(f"Unknown case {name!r} for {self}") from exc


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


@register_model(FFITaggedRefType)
class FFITaggedRefModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        super().__init__(dmm, fe_type, [("pointer", types.voidptr), ("tag", types.int32)])


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
def ffi_tagged_ref_type(name: str, cases: tuple[str, ...]):
    if (
        not isinstance(name, str)
        or not name.isidentifier()
        or not isinstance(cases, tuple)
        or not cases
        or len(cases) >= 2**31
        or any(not isinstance(case, str) or not case.isidentifier() for case in cases)
        or len(set(cases)) != len(cases)
    ):
        raise ValueError("Tagged FFI reference requires an identifier and distinct, ordered case names")
    return FFITaggedRefType(name, cases)


@cache
def ffi_tagged_from_voidptr(ref_type: FFITaggedRefType):
    if not isinstance(ref_type, FFITaggedRefType):
        raise TypeError("Expected a tagged FFI reference type")

    @intrinsic(prefer_literal=True)
    def wrap(typingctx, pointer, tag):
        if pointer != types.voidptr or not isinstance(tag, types.Integer):
            return None
        if isinstance(tag, types.IntegerLiteral) and not 0 <= tag.literal_value < len(ref_type.cases):
            raise ValueError(f"Invalid literal tag {tag.literal_value} for {ref_type}")
        sig = ref_type(pointer, tag)

        def codegen(context, builder, signature, args):
            result = context.make_helper(builder, ref_type)
            result.pointer = args[0]
            if len(ref_type.cases) < 1 << args[1].type.width:
                invalid = builder.icmp_unsigned(">=", args[1], ir.Constant(args[1].type, len(ref_type.cases)))
                with builder.if_then(invalid):
                    trap = declare_symbol(builder, "llvm.trap", ir.FunctionType(ir.VoidType(), ()))
                    builder.call(trap, ())
                    builder.unreachable()
            result.tag = context.cast(builder, args[1], tag, types.int32)
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
