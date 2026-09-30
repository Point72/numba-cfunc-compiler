from functools import cache

from llvmlite import ir
from numba import types
from numba.core import cgutils
from numba.core.errors import TypingError
from numba.extending import intrinsic, overload_method

from numba_cfunc_compiler.types.native.ffi import FFIRefType, FFISideRefType, declare_symbol, ffi_range_type, make_ffi_range
from numba_cfunc_compiler.types.native.input import InputValueType, input_payload, unwrap_source_args

_METHODS = {}
_ITERATORS = {}
_REF_DISPATCH = set()
_DEPENDENCIES = {}


def register_ffi_dependency(parent, child):
    """Include methods of a returned native reference in a parent's cache key."""
    if not isinstance(parent, (FFIRefType, FFISideRefType)) or not isinstance(child, (FFIRefType, FFISideRefType)):
        raise TypeError("FFI dependencies require nominal reference types")
    _DEPENDENCIES.setdefault(parent, set()).add(child)


def ffi_binding_fingerprint(native_type):
    """Native symbols and iterator signatures reachable from an FFI value."""

    def serialize(item):
        if isinstance(item, types.Type):
            return item.name
        if isinstance(item, tuple):
            return tuple(serialize(part) for part in item)
        return item

    if not isinstance(native_type, (FFIRefType, FFISideRefType)):
        return ()
    seen = set()
    pending = [native_type]
    entries = []
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        for child in _DEPENDENCIES.get(current, ()):
            entries.append((current.name, "dependency", child.name, ()))
            pending.append(child)
        for registry_name, registry in (("method", _METHODS), ("iterator", _ITERATORS)):
            for (owner, name), spec in registry.items():
                if owner != current:
                    continue
                entries.append((current.name, registry_name, name, tuple(serialize(item) for item in spec)))
                pending.extend(item for item in spec if isinstance(item, (FFIRefType, FFISideRefType)))
    return tuple(sorted(entries))


def _llvm_type(context, numba_type):
    if isinstance(numba_type, FFIRefType):
        return ir.IntType(8).as_pointer()
    return context.get_value_type(numba_type)


@cache
def _ref_call_intrinsic(ref_type, name):
    symbol, return_type, arg_types = _METHODS[(ref_type, name)]

    @intrinsic
    def call_native(typingctx, receiver, args_tuple):
        actual_types = tuple(args_tuple) if isinstance(args_tuple, types.BaseTuple) else None
        if receiver != ref_type or actual_types != arg_types:
            raise TypingError(f"{ref_type}.{name} requires {arg_types}, got {actual_types}")
        sig = return_type(receiver, args_tuple)

        def codegen(context, builder, signature, values):
            native_sig = ir.FunctionType(
                _llvm_type(context, return_type), [_llvm_type(context, ref_type), *[_llvm_type(context, typ) for typ in arg_types]]
            )
            function = declare_symbol(builder, symbol, native_sig)
            native_args = [values[0], *(builder.extract_value(values[1], idx) for idx in range(len(arg_types)))]
            return builder.call(function, native_args)

        return sig, codegen

    return call_native


def register_ffi_method(ref_type: FFIRefType, name: str, symbol: str, return_type: types.Type, arg_types: tuple[types.Type, ...] = ()):
    """Install a Numba method that calls one immutable native symbol."""
    if not isinstance(ref_type, FFIRefType) or not name.isidentifier() or not symbol:
        raise ValueError("Invalid FFI method registration")
    key = ref_type, name
    spec = symbol, return_type, arg_types
    if key in _METHODS:
        if _METHODS[key] != spec:
            raise ValueError(f"Conflicting FFI method registration for {ref_type}.{name}")
        return
    _METHODS[key] = spec
    if name in _REF_DISPATCH:
        return

    @overload_method(FFIRefType, name)
    def ref_method(value, *args):
        if isinstance(value, FFIRefType) and (value, name) in _METHODS:
            call_native = _ref_call_intrinsic(value, name)

            def impl(value, *args):
                return call_native(value, unwrap_source_args(args))

            return impl
        return None

    @overload_method(InputValueType, name)
    def input_method(value, *args):
        if isinstance(value, InputValueType) and (value.payload_type, name) in _METHODS:
            call_native = _ref_call_intrinsic(value.payload_type, name)

            def impl(value, *args):
                return call_native(input_payload(value), unwrap_source_args(args))

            return impl
        return None

    _REF_DISPATCH.add(name)


@cache
def _side_call_intrinsic(side_type, name):
    bid_symbol, ask_symbol, return_type, arg_types = _METHODS[(side_type, name)]

    @intrinsic
    def call_native(typingctx, receiver, args_tuple):
        actual = tuple(args_tuple) if isinstance(args_tuple, types.BaseTuple) else None
        if receiver != side_type or actual != arg_types:
            raise TypingError(f"{side_type}.{name} requires {arg_types}, got {actual}")
        sig = return_type(receiver, args_tuple)

        def codegen(context, builder, signature, values):
            side = context.make_helper(builder, side_type, value=values[0])
            native_sig = ir.FunctionType(
                _llvm_type(context, return_type), [ir.IntType(8).as_pointer(), *[_llvm_type(context, typ) for typ in arg_types]]
            )
            bid_function = declare_symbol(builder, bid_symbol, native_sig)
            ask_function = declare_symbol(builder, ask_symbol, native_sig)
            native_args = [side.pointer, *(builder.extract_value(values[1], idx) for idx in range(len(arg_types)))]
            result = cgutils.alloca_once(builder, _llvm_type(context, return_type))
            is_bid = builder.icmp_unsigned("!=", side.bid, ir.Constant(side.bid.type, 0))
            with builder.if_else(is_bid) as (bid_case, ask_case):
                with bid_case:
                    builder.store(builder.call(bid_function, native_args), result)
                with ask_case:
                    builder.store(builder.call(ask_function, native_args), result)
            return builder.load(result)

        return sig, codegen

    return call_native


def register_ffi_side_method(
    side_type: FFISideRefType, name: str, bid_symbol: str, ask_symbol: str, return_type: types.Type, arg_types: tuple[types.Type, ...] = ()
):
    """Dispatch a side method using the selector carried by the native value."""
    if not isinstance(side_type, FFISideRefType) or not name.isidentifier() or not bid_symbol or not ask_symbol:
        raise ValueError("Invalid side FFI method registration")
    key = side_type, name
    spec = bid_symbol, ask_symbol, return_type, arg_types
    if key in _METHODS:
        if _METHODS[key] != spec:
            raise ValueError(f"Conflicting FFI method registration for {side_type}.{name}")
        return
    _METHODS[key] = spec

    @overload_method(FFISideRefType, name)
    def side_method(value, *args):
        if value == side_type:
            call_native = _side_call_intrinsic(side_type, name)

            def impl(value, *args):
                return call_native(value, unwrap_source_args(args))

            return impl
        return None


def register_ffi_iterator_method(ref_type, name, begin_method, end_method, element_type, next_symbol):
    """Expose a begin/end C API as a Numba iterable of nominal references."""
    if not isinstance(ref_type, (FFIRefType, FFISideRefType)) or not isinstance(element_type, FFIRefType):
        raise TypeError("FFI iterator methods require nominal reference types")
    if not all(part.isidentifier() for part in (name, begin_method, end_method, next_symbol)):
        raise ValueError("FFI iterator method names must be identifiers")
    key = ref_type, name
    spec = begin_method, end_method, element_type, next_symbol
    if key in _ITERATORS:
        if _ITERATORS[key] != spec:
            raise ValueError(f"Conflicting FFI iterator registration for {ref_type}.{name}")
        return
    range_type = ffi_range_type(element_type, next_symbol)
    make = make_ffi_range(range_type)
    intrinsic_for = _ref_call_intrinsic if isinstance(ref_type, FFIRefType) else _side_call_intrinsic
    begin_call = intrinsic_for(ref_type, begin_method)
    end_call = intrinsic_for(ref_type, end_method)

    @overload_method(type(ref_type), name)
    def iterator_method(value):
        if value == ref_type:

            def impl(value):
                return make(begin_call(value, ()), end_call(value, ()))

            return impl
        return None

    _ITERATORS[key] = spec
