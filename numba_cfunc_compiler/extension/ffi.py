from functools import cache

from llvmlite import ir
from numba import types
from numba.core import cgutils
from numba.core.errors import TypingError
from numba.extending import intrinsic, overload_method

from numba_cfunc_compiler.types.native.ffi import (
    FFIRefType,
    FFITaggedRefType,
    declare_symbol,
    ffi_range_type,
    make_ffi_range,
)
from numba_cfunc_compiler.types.native.input import InputValueType, input_payload, unwrap_source_args

# (reference type, method name) -> (symbols in case order, return type, argument types)
_METHODS = {}
_ITERATORS = {}
_INSTALLED_METHOD_OVERLOADS = set()
_DEPENDENCIES = {}
_REFERENCE_TYPES = (FFIRefType, FFITaggedRefType)


def register_ffi_dependency(parent, child):
    """Include methods of a returned native reference in a parent's cache key."""
    if not isinstance(parent, _REFERENCE_TYPES) or not isinstance(child, _REFERENCE_TYPES):
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

    if not isinstance(native_type, _REFERENCE_TYPES):
        return ()
    seen = set()
    pending = [native_type]
    entries = []
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        if isinstance(current, FFITaggedRefType):
            entries.append((current.name, "cases", current.cases, ()))
        for child in _DEPENDENCIES.get(current, ()):
            entries.append((current.name, "dependency", child.name, ()))
            pending.append(child)
        for registry_name, registry in (("method", _METHODS), ("iterator", _ITERATORS)):
            for (owner, name), spec in registry.items():
                if owner != current:
                    continue
                entries.append((current.name, registry_name, name, tuple(serialize(item) for item in spec)))
                pending.extend(item for item in spec if isinstance(item, _REFERENCE_TYPES))
    return tuple(sorted(entries))


def _llvm_type(context, numba_type):
    if isinstance(numba_type, FFIRefType):
        return ir.IntType(8).as_pointer()
    return context.get_value_type(numba_type)


def _method_signature(context, return_type, arg_types):
    return ir.FunctionType(_llvm_type(context, return_type), [ir.IntType(8).as_pointer(), *[_llvm_type(context, typ) for typ in arg_types]])


def _check_method_args(ref_type, name, receiver, args_tuple, arg_types):
    actual = tuple(args_tuple) if isinstance(args_tuple, types.BaseTuple) else None
    if receiver != ref_type or actual != arg_types:
        raise TypingError(f"{ref_type}.{name} requires {arg_types}, got {actual}")


def _register_method_spec(ref_type, name, spec):
    """Store one immutable method definition; report whether it is new."""
    key = ref_type, name
    if key in _METHODS:
        if _METHODS[key] != spec:
            raise ValueError(f"Conflicting FFI method registration for {ref_type}.{name}")
        return False
    _METHODS[key] = spec
    return True


def _install_method_overload(owner_type, name, intrinsic_for):
    """Install one Numba typing rule per reference family and method name."""
    key = owner_type, name
    if key in _INSTALLED_METHOD_OVERLOADS:
        return False

    @overload_method(owner_type, name)
    def method(value, *args):
        if isinstance(value, owner_type) and (value, name) in _METHODS:
            call_native = intrinsic_for(value, name)

            def impl(value, *args):
                return call_native(value, unwrap_source_args(args))

            return impl
        return None

    _INSTALLED_METHOD_OVERLOADS.add(key)
    return True


@cache
def _ref_call_intrinsic(ref_type, name):
    symbols, return_type, arg_types = _METHODS[(ref_type, name)]
    symbol = symbols[0]

    @intrinsic
    def call_native(typingctx, receiver, args_tuple):
        _check_method_args(ref_type, name, receiver, args_tuple, arg_types)
        sig = return_type(receiver, args_tuple)

        def codegen(context, builder, signature, values):
            native_sig = _method_signature(context, return_type, arg_types)
            function = declare_symbol(builder, symbol, native_sig)
            native_args = [values[0], *(builder.extract_value(values[1], idx) for idx in range(len(arg_types)))]
            return builder.call(function, native_args)

        return sig, codegen

    return call_native


def register_ffi_method(ref_type: FFIRefType, name: str, symbol: str, return_type: types.Type, arg_types: tuple[types.Type, ...] = ()):
    """Install a Numba method that calls one immutable native symbol."""
    if not isinstance(ref_type, FFIRefType) or not name.isidentifier() or not symbol:
        raise ValueError("Invalid FFI method registration")
    if not _register_method_spec(ref_type, name, ((symbol,), return_type, arg_types)):
        return
    if not _install_method_overload(FFIRefType, name, _ref_call_intrinsic):
        return

    @overload_method(InputValueType, name)
    def input_method(value, *args):
        if isinstance(value, InputValueType) and (value.payload_type, name) in _METHODS:
            call_native = _ref_call_intrinsic(value.payload_type, name)

            def impl(value, *args):
                return call_native(input_payload(value), unwrap_source_args(args))

            return impl
        return None


@cache
def _tagged_call_intrinsic(ref_type, name):
    symbols, return_type, arg_types = _METHODS[(ref_type, name)]

    @intrinsic
    def call_native(typingctx, receiver, args_tuple):
        _check_method_args(ref_type, name, receiver, args_tuple, arg_types)
        sig = return_type(receiver, args_tuple)

        def codegen(context, builder, signature, values):
            reference = context.make_helper(builder, ref_type, value=values[0])
            native_sig = _method_signature(context, return_type, arg_types)
            functions = [declare_symbol(builder, symbol, native_sig) for symbol in symbols]
            native_args = [reference.pointer, *(builder.extract_value(values[1], idx) for idx in range(len(arg_types)))]
            tag_value = reference.tag
            result = None if return_type == types.void else cgutils.alloca_once(builder, _llvm_type(context, return_type))
            failure = builder.append_basic_block("invalid_ffi_tag")
            done = builder.append_basic_block("tagged_ffi_done")
            dispatch = builder.switch(tag_value, failure)
            for tag, function in enumerate(functions):
                case = builder.append_basic_block(f"tagged_ffi_case_{tag}")
                dispatch.add_case(ir.Constant(tag_value.type, tag), case)
                builder.position_at_end(case)
                called = builder.call(function, native_args)
                if result is not None:
                    builder.store(called, result)
                builder.branch(done)
            builder.position_at_end(failure)
            builder.call(declare_symbol(builder, "llvm.trap", ir.FunctionType(ir.VoidType(), ())), ())
            builder.unreachable()
            builder.position_at_end(done)
            return context.get_dummy_value() if result is None else builder.load(result)

        return sig, codegen

    return call_native


def _validate_tagged_method(ref_type, name, symbols, return_type, arg_types):
    """Check a tagged declaration and return its symbols in tag order."""
    if not isinstance(ref_type, FFITaggedRefType) or not isinstance(name, str) or not name.isidentifier():
        raise ValueError("Invalid tagged FFI method registration")
    if (
        not isinstance(symbols, dict)
        or set(symbols) != set(ref_type.cases)
        or any(not isinstance(symbol, str) or not symbol for symbol in symbols.values())
    ):
        raise ValueError("Invalid tagged FFI method registration")
    if not isinstance(return_type, types.Type) or not isinstance(arg_types, tuple) or any(not isinstance(arg, types.Type) for arg in arg_types):
        raise ValueError("Invalid tagged FFI method registration")
    return tuple(symbols[case] for case in ref_type.cases)


def register_ffi_tagged_method(
    ref_type: FFITaggedRefType,
    name: str,
    symbols: dict[str, str],
    return_type: types.Type,
    arg_types: tuple[types.Type, ...] = (),
):
    """Dispatch a method by the case tag carried with a borrowed pointer."""
    ordered_symbols = _validate_tagged_method(ref_type, name, symbols, return_type, arg_types)
    if not _register_method_spec(ref_type, name, (ordered_symbols, return_type, arg_types)):
        return
    _install_method_overload(FFITaggedRefType, name, _tagged_call_intrinsic)


def register_ffi_iterator_method(ref_type, name, begin_method, end_method, element_type, next_symbol):
    """Expose a begin/end C API as a Numba iterable of nominal references."""
    if not isinstance(ref_type, _REFERENCE_TYPES) or not isinstance(element_type, FFIRefType):
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
    intrinsic_for = _ref_call_intrinsic if isinstance(ref_type, FFIRefType) else _tagged_call_intrinsic
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
