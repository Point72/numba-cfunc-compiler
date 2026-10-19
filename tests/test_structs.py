"""Consolidated integration tests for structs."""

import ctypes
import inspect
from enum import Enum
from typing import get_args, get_origin

import pytest
from numba import cfunc, types
from numba.core.errors import TypingError
from numba.extending import overload_method, register_jitable

from numba_cfunc_compiler.api import set_output
from numba_cfunc_compiler.core.analysis import FunctionAnalyzer
from numba_cfunc_compiler.core.compile import build_semantic_key, create_compiled_func
from numba_cfunc_compiler.core.context import CompilationContext
from numba_cfunc_compiler.core.defaults import register_all
from numba_cfunc_compiler.extension.callback_components import ComponentRegistry
from numba_cfunc_compiler.extension.methods import forward_value_method
from numba_cfunc_compiler.types.builtin.enum.host import register_enum_family
from numba_cfunc_compiler.types.builtin.struct.host import StructFieldInfo, StructHostType
from numba_cfunc_compiler.types.builtin.struct.native import StructField, StructLayout, StructPtrType, struct_ptr_type, struct_view
from numba_cfunc_compiler.types.factory import HostTypeFactory
from tests.harness import (
    CFUNC_T,
    LIFECYCLE_EXECUTE,
    Signal,
    SignalComponent,
    SignalInputHandler,
    SingleSignalOutputHandler,
    numba_node,
)


class Quote(ctypes.Structure):
    _fields_ = [("price", ctypes.c_double), ("count", ctypes.c_int64)]


class ReorderedQuote(ctypes.Structure):
    _fields_ = [("count", ctypes.c_int64), ("price", ctypes.c_double)]


class OtherQuote(ctypes.Structure):
    _fields_ = [("price", ctypes.c_double), ("count", ctypes.c_int64)]


class PackedQuote(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("tag", ctypes.c_int8), ("price", ctypes.c_double)]


class TypedQuoteType(StructHostType):
    storage_type = Quote

    @classmethod
    def is_type_supported(cls, var_type):
        return var_type is Quote

    @classmethod
    def get_struct_fields(cls, var_type):
        storage = cls.storage_type
        return {
            "price": StructFieldInfo("price", storage.price.offset, "float64", 8),
            "count": StructFieldInfo("count", storage.count.offset, "int64", 8),
        }

    @classmethod
    def get_struct_size(cls, var_type):
        return ctypes.sizeof(cls.storage_type)


class ReorderedTypedQuoteType(TypedQuoteType):
    storage_type = ReorderedQuote


class OtherTypedQuoteType(TypedQuoteType):
    storage_type = OtherQuote

    @classmethod
    def is_type_supported(cls, var_type):
        return var_type is OtherQuote


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


forward_value_method(struct_ptr_type(TypedQuoteType.from_type(Quote).get_typed_layout()), "adjusted_price", ("offset",))


@numba_node
def update_quote(q: Signal[Quote]) -> Signal[Quote]:
    alias = q
    alias.price += 1.5
    alias.count = alias.count + 2
    return alias


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
        ComponentRegistry.register(SignalComponent())
        FunctionAnalyzer.register_input_handler(SignalInputHandler())
        FunctionAnalyzer.register_output_handler(SingleSignalOutputHandler())
        struct_classes = struct_class if isinstance(struct_class, tuple) else (struct_class,)
        for type_class in struct_classes:
            HostTypeFactory.register(type_class, priority=0)
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
    callback = CFUNC_T(result.compiled_func.address)
    sources = source if isinstance(source, tuple) else (source,)
    inputs = (ctypes.c_void_p * len(sources))(*(ctypes.addressof(value) for value in sources))
    outputs = (ctypes.c_void_p * 1)(ctypes.addressof(output))
    state = (ctypes.c_void_p * 1)()
    output_ticked = (ctypes.c_int8 * 1)()
    input_ticked = (ctypes.c_int8 * len(sources))(*([1] * len(sources)))
    input_valid = (ctypes.c_int8 * len(sources))(*([1] * len(sources)))
    callback(outputs, output_ticked, state, LIFECYCLE_EXECUTE, inputs, input_ticked, input_valid)
    return bool(output_ticked[0])


@numba_node
def invalid_quote(q: Signal[Quote]) -> Signal[float]:
    return q.missing


def test_struct_field_aliases():
    result = _compile(update_quote, TypedQuoteType)
    source = Quote(3.25, 4)
    output = Quote(-1.0, -1)

    assert _execute(result, source, output)
    assert (source.price, source.count) == (4.75, 6)
    assert (output.price, output.count) == (4.75, 6)
    assert "NRT_MemInfo_alloc" not in result.compiled_func.inspect_llvm()

    source = Quote(6.25, 7)
    for func in (quote_from_helper, set_quote_from_helper):
        output = Quote()
        assert _execute(_compile(func, TypedQuoteType), source, output), func.__name__
        assert (output.price, output.count) == (6.25, 7), func.__name__

    conditional_result = _compile(choose_quote, TypedQuoteType)
    first = Quote(1.5, 2)
    second = Quote(9.5, 10)
    assert _execute(conditional_result, (first, second, ctypes.c_int8(1)), output)
    assert (output.price, output.count) == (1.5, 2)
    assert _execute(conditional_result, (first, second, ctypes.c_int8(0)), output)
    assert (output.price, output.count) == (9.5, 10)

    result = _compile(rebound_alias, TypedQuoteType)
    first = Quote(1.0, 2)
    second = Quote(3.0, 4)
    output = Quote()
    assert _execute(result, (first, second), output)
    assert (first.price, second.price) == (1.0, 8.0)
    assert (output.price, output.count) == (1.0, 2)

    with pytest.raises(TypingError):
        _compile(incompatible_alias, (TypedQuoteType, OtherTypedQuoteType))

    result = _compile(optional_quote, TypedQuoteType)
    source = Quote(4.5, 6)
    output = Quote(-1.0, -1)
    assert not _execute(result, (source, ctypes.c_int8(0)), output)
    assert (output.price, output.count) == (-1.0, -1)
    assert _execute(result, (source, ctypes.c_int8(1)), output)
    assert (output.price, output.count) == (4.5, 6)

    source = Quote(2.5, 3)
    for func, expected in ((quote_value, 5.5), (quote_method, 4.5)):
        output = ctypes.c_double()
        assert _execute(_compile(func, TypedQuoteType), source, output), func.__name__
        assert output.value == expected, func.__name__

    packed_view = struct_view(
        StructLayout(
            "PackedQuote",
            ctypes.sizeof(PackedQuote),
            (
                StructField("tag", PackedQuote.tag.offset, types.int8),
                StructField("price", PackedQuote.price.offset, types.float64),
            ),
        )
    )

    @cfunc(types.float64(types.voidptr), _nrt=False)
    def update_packed(pointer):
        packed = packed_view(pointer)
        packed.price = packed.price + packed.tag
        return packed.price

    packed = PackedQuote(2, 1.5)
    assert update_packed.ctypes(ctypes.c_void_p(ctypes.addressof(packed))) == 3.5
    assert packed.price == 3.5


class Venue(Enum):
    A = 1
    B = 2


class Trade(ctypes.Structure):
    _fields_ = [("venue", ctypes.c_int16), ("conditions", ctypes.c_void_p)]


def test_enum_struct_fields():
    binding = register_enum_family(Venue, abi_values={"A": 3, "B": 7})
    layout = StructLayout(
        "TestTradeEnumFields",
        ctypes.sizeof(Trade),
        (
            StructField("venue", Trade.venue.offset, binding.value_type, 2),
            StructField("conditions", Trade.conditions.offset, binding.set_type, ctypes.sizeof(ctypes.c_void_p), True),
        ),
    )
    view = struct_view(layout)
    literal_a = binding.literal("A")
    make_set = binding.set_builder

    @cfunc(types.void(types.voidptr, types.CPointer(types.int8)), _nrt=False)
    def callback(pointer, output):
        trade = view(pointer)
        venue = trade.venue
        conditions = trade.conditions
        required = make_set((literal_a(),))
        output[0] = venue.isin(conditions)
        output[1] = conditions.contains_all(required)
        output[2] = conditions.contains_any(required)

    mask = (ctypes.c_uint64 * 2)(1 << 3, 0)
    trade = Trade(3, ctypes.addressof(mask))
    output = (ctypes.c_int8 * 3)()
    callback.ctypes(ctypes.c_void_p(ctypes.addressof(trade)), output)
    assert tuple(output) == (1, 1, 1)
    trade.venue = 7
    callback.ctypes(ctypes.c_void_p(ctypes.addressof(trade)), output)
    assert tuple(output) == (0, 1, 1)
    mask[0] = 0
    callback.ctypes(ctypes.c_void_p(ctypes.addressof(trade)), output)
    assert tuple(output) == (0, 0, 0)


def test_layout_identity():
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

    first_layout = TypedQuoteType.from_type(Quote).get_typed_layout()
    second_layout = ReorderedTypedQuoteType.from_type(Quote).get_typed_layout()
    assert build_semantic_key("same", "sig", "opts", {first_layout.fingerprint: first_layout}) != build_semantic_key(
        "same", "sig", "opts", {second_layout.fingerprint: second_layout}
    )


def test_invalid_layouts():
    class BadWidth(TypedQuoteType):
        @classmethod
        def get_struct_fields(cls, var_type):
            return {"price": StructFieldInfo("price", 0, "float64", 4)}

    with pytest.raises(ValueError, match="has size 4, expected 8"):
        BadWidth.from_type(Quote).get_typed_layout()

    class BadType(TypedQuoteType):
        @classmethod
        def get_struct_fields(cls, var_type):
            return {"price": StructFieldInfo("price", 0, "boolean", 1)}

    with pytest.raises(TypeError, match="unsupported type"):
        BadType.from_type(Quote).get_typed_layout()

    with pytest.raises(TypingError, match="cannot store value"):
        _compile(wrong_layout, (TypedQuoteType, OtherTypedQuoteType))
    with pytest.raises(TypingError, match="cannot store value"):
        _compile(wrong_scalar, TypedQuoteType)

    with pytest.raises(TypingError):
        _compile(invalid_quote, TypedQuoteType)
