"""Opt-in struct views through the generated node callback ABI."""

import ctypes
import inspect
from typing import get_args, get_origin

import pytest
from numba.core.errors import TypingError
from numba.extending import overload_method, register_jitable

from numba_cfunc_compiler.compilation_context import CompilationContext
from numba_cfunc_compiler.defaults import register_all
from numba_cfunc_compiler.defaults.struct_support import StructFieldInfo, StructType
from numba_cfunc_compiler.function_analyzer import FunctionAnalyzer
from numba_cfunc_compiler.node_api import set_output
from numba_cfunc_compiler.numba_core import _build_semantic_key, create_compiled_func
from numba_cfunc_compiler.source_registry import SourceRegistry
from numba_cfunc_compiler.standalone.struct import StructPtrType
from numba_cfunc_compiler.tests.harness import (
    _CFUNC_T,
    LIFECYCLE_EXECUTE,
    Signal,
    _SignalCategory,
    _SignalInputHandler,
    _SingleSignalOutputHandler,
    numba_node,
)
from numba_cfunc_compiler.type_factory import TypeFactory


class Quote(ctypes.Structure):
    _fields_ = [("price", ctypes.c_double), ("count", ctypes.c_int64)]


class ReorderedQuote(ctypes.Structure):
    _fields_ = [("count", ctypes.c_int64), ("price", ctypes.c_double)]


class OtherQuote(ctypes.Structure):
    _fields_ = [("price", ctypes.c_double), ("count", ctypes.c_int64)]


class TypedQuoteType(StructType):
    typed_view = True
    storage_type = Quote

    @classmethod
    def is_type_supported(cls, var_type):
        return var_type is Quote

    @classmethod
    def _get_struct_fields(cls, var_type):
        storage = cls.storage_type
        return {
            "price": StructFieldInfo("price", storage.price.offset, "float64", 8),
            "count": StructFieldInfo("count", storage.count.offset, "int64", 8),
        }

    @classmethod
    def _get_struct_size(cls, var_type):
        return ctypes.sizeof(cls.storage_type)

    @classmethod
    def get_type_size(cls, var_type):
        return cls._get_struct_size(var_type)


class ReorderedTypedQuoteType(TypedQuoteType):
    storage_type = ReorderedQuote


class OtherTypedQuoteType(TypedQuoteType):
    storage_type = OtherQuote

    @classmethod
    def is_type_supported(cls, var_type):
        return var_type is OtherQuote


class LegacyQuoteType(TypedQuoteType):
    typed_view = False


@register_jitable(inline="always")
def quote_adjustment(quote):
    return quote.price + quote.count


@register_jitable(inline="always")
def identity_quote(quote):
    return quote


@overload_method(StructPtrType, "adjusted_price")
def typed_adjusted_price(quote, offset):
    if quote.layout.get_field("price") is None:
        return None

    def impl(quote, offset):
        return quote.price + offset

    return impl


@numba_node
def update_quote(q: Signal[Quote]) -> Signal[Quote]:
    alias = q
    alias.price += 1.5
    alias.count = alias.count + 2
    return alias


@numba_node
def update_legacy_quote(q: Signal[Quote]) -> Signal[Quote]:
    q.price = q.price + 1.5
    q.count = q.count + 2
    return q


@numba_node
def quote_value(q: Signal[Quote]) -> Signal[float]:
    return quote_adjustment(q)


@numba_node
def quote_price(q: Signal[Quote]) -> Signal[float]:
    return q.price


@numba_node
def quote_method(q: Signal[Quote]) -> Signal[float]:
    return q.adjusted_price(2.0)


@numba_node
def quote_from_helper(q: Signal[Quote]) -> Signal[Quote]:
    return identity_quote(q)


@numba_node
def set_quote_from_helper(q: Signal[Quote]) -> Signal[Quote]:
    set_output("output_0", identity_quote(q))


@numba_node
def choose_quote(first: Signal[Quote], second: Signal[Quote], choose_first: Signal[bool]) -> Signal[Quote]:
    return first if choose_first else second


@numba_node
def rebound_alias(first: Signal[Quote], second: Signal[Quote]) -> Signal[Quote]:
    alias = first
    alias = second
    alias.price = alias.price + 5.0
    return first


@numba_node
def incompatible_alias(first: Signal[Quote], second: Signal[OtherQuote]) -> Signal[Quote]:
    alias = first
    alias = second
    return alias


@numba_node
def optional_quote(q: Signal[Quote], emit: Signal[bool]) -> Signal[Quote]:
    if emit:
        return q
    return None


@numba_node
def wrong_layout(q: Signal[OtherQuote]) -> Signal[Quote]:
    return q


@numba_node
def wrong_scalar(q: Signal[Quote]) -> Signal[Quote]:
    return q.count


def _compile(func, struct_class):
    with CompilationContext():
        register_all()
        SourceRegistry.register(_SignalCategory())
        FunctionAnalyzer.register_input_handler(_SignalInputHandler())
        FunctionAnalyzer.register_output_handler(_SingleSignalOutputHandler())
        struct_classes = struct_class if isinstance(struct_class, tuple) else (struct_class,)
        for type_class in struct_classes:
            TypeFactory.register(type_class, priority=0)
        inputs = []
        for parameter in inspect.signature(func).parameters.values():
            if get_origin(parameter.annotation) is Signal:
                inputs.append(Signal(typ=get_args(parameter.annotation)[0]))
        return create_compiled_func(
            func,
            *inputs,
            extract_python_type_fn=lambda signal: signal.get_type(),
            call_globals={"quote_adjustment": quote_adjustment, "identity_quote": identity_quote},
        )


def _execute(result, source, output):
    callback = _CFUNC_T(result.compiled_func.address)
    sources = source if isinstance(source, tuple) else (source,)
    inputs = (ctypes.c_void_p * len(sources))(*(ctypes.addressof(value) for value in sources))
    outputs = (ctypes.c_void_p * 1)(ctypes.addressof(output))
    state = (ctypes.c_void_p * 1)()
    output_ticked = (ctypes.c_int8 * 1)()
    input_ticked = (ctypes.c_int8 * len(sources))(*([1] * len(sources)))
    input_valid = (ctypes.c_int8 * len(sources))(*([1] * len(sources)))
    callback(outputs, output_ticked, state, LIFECYCLE_EXECUTE, inputs, input_ticked, input_valid)
    return bool(output_ticked[0])


def test_typed_struct_field_store_alias_and_output_copy():
    result = _compile(update_quote, TypedQuoteType)
    source = Quote(3.25, 4)
    output = Quote(-1.0, -1)

    assert _execute(result, source, output)
    assert (source.price, source.count) == (4.75, 6)
    assert (output.price, output.count) == (4.75, 6)
    assert "NRT_MemInfo_alloc" not in result.compiled_func.inspect_llvm()


def test_numba_types_struct_returned_by_helper_and_conditional():
    helper_result = _compile(quote_from_helper, TypedQuoteType)
    source = Quote(6.25, 7)
    output = Quote()
    assert _execute(helper_result, source, output)
    assert (output.price, output.count) == (6.25, 7)

    named_result = _compile(set_quote_from_helper, TypedQuoteType)
    assert _execute(named_result, source, output)
    assert (output.price, output.count) == (6.25, 7)

    conditional_result = _compile(choose_quote, TypedQuoteType)
    first = Quote(1.5, 2)
    second = Quote(9.5, 10)
    assert _execute(conditional_result, (first, second, ctypes.c_int8(1)), output)
    assert (output.price, output.count) == (1.5, 2)
    assert _execute(conditional_result, (first, second, ctypes.c_int8(0)), output)
    assert (output.price, output.count) == (9.5, 10)


def test_numba_owns_typed_struct_local_aliases():
    result = _compile(rebound_alias, TypedQuoteType)
    first = Quote(1.0, 2)
    second = Quote(3.0, 4)
    output = Quote()
    assert _execute(result, (first, second), output)
    assert (first.price, second.price) == (1.0, 8.0)
    assert (output.price, output.count) == (1.0, 2)

    with pytest.raises(TypingError):
        _compile(incompatible_alias, (TypedQuoteType, OtherTypedQuoteType))


def test_typed_struct_optional_return_preserves_output_tick():
    result = _compile(optional_quote, TypedQuoteType)
    source = Quote(4.5, 6)
    output = Quote(-1.0, -1)
    assert not _execute(result, (source, ctypes.c_int8(0)), output)
    assert (output.price, output.count) == (-1.0, -1)
    assert _execute(result, (source, ctypes.c_int8(1)), output)
    assert (output.price, output.count) == (4.5, 6)


def test_numba_rejects_incompatible_struct_and_scalar_outputs():
    with pytest.raises(TypingError, match="copy"):
        _compile(wrong_layout, (TypedQuoteType, OtherTypedQuoteType))
    with pytest.raises(TypingError, match="copy"):
        _compile(wrong_scalar, TypedQuoteType)


def test_numba_types_helper_return_and_field_expression():
    result = _compile(quote_value, TypedQuoteType)
    source = Quote(2.5, 3)
    output = ctypes.c_double()
    assert _execute(result, source, output)
    assert output.value == 5.5

    method_result = _compile(quote_method, TypedQuoteType)
    assert _execute(method_result, source, output)
    assert output.value == 4.5


def test_layout_changes_native_behavior_and_semantic_key_across_contexts():
    first = _compile(quote_price, TypedQuoteType)
    second = _compile(quote_price, TypedQuoteType)
    reordered = _compile(quote_price, ReorderedTypedQuoteType)
    assert first.semantic_key == second.semantic_key
    assert first.semantic_key != reordered.semantic_key

    output = ctypes.c_double()
    assert _execute(first, Quote(3.5, 9), output)
    assert output.value == 3.5
    assert _execute(reordered, ReorderedQuote(9, 7.25), output)
    assert output.value == 7.25

    first_layout = TypedQuoteType.from_type(Quote, None).get_typed_layout()
    second_layout = ReorderedTypedQuoteType.from_type(Quote, None).get_typed_layout()
    assert _build_semantic_key("same", "sig", "opts", {first_layout.fingerprint: first_layout}) != _build_semantic_key(
        "same", "sig", "opts", {second_layout.fingerprint: second_layout}
    )


def test_legacy_struct_keeps_existing_field_and_output_lowering():
    result = _compile(update_legacy_quote, LegacyQuoteType)
    source = Quote(1.0, 1)
    output = Quote()
    assert _execute(result, source, output)
    assert (output.price, output.count) == (2.5, 3)


def test_typed_layout_rejects_bad_width_and_unsupported_field():
    class BadWidth(TypedQuoteType):
        @classmethod
        def _get_struct_fields(cls, var_type):
            return {"price": StructFieldInfo("price", 0, "float64", 4)}

    with pytest.raises(ValueError, match="declares 4 bytes"):
        BadWidth.from_type(Quote, None).get_typed_layout()

    class BadType(TypedQuoteType):
        @classmethod
        def _get_struct_fields(cls, var_type):
            return {"price": StructFieldInfo("price", 0, "boolean", 1)}

    with pytest.raises(TypeError, match="unsupported typed-view type"):
        BadType.from_type(Quote, None).get_typed_layout()


@numba_node
def invalid_quote(q: Signal[Quote]) -> Signal[float]:
    return q.missing


def test_invalid_typed_field_fails_in_numba():
    with pytest.raises(TypingError):
        _compile(invalid_quote, TypedQuoteType)
