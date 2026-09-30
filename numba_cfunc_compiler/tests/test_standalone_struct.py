"""Direct cfunc checks for the borrowed struct-pointer prototype."""

import ctypes

import pytest
from numba import cfunc, types
from numba.core.errors import TypingError
from numba.extending import register_jitable

from numba_cfunc_compiler.standalone.struct import StructField, StructLayout, struct_ptr_type, struct_view


class Quote(ctypes.Structure):
    _fields_ = [("price", ctypes.c_double), ("count", ctypes.c_int64)]


class ReorderedQuote(ctypes.Structure):
    _fields_ = [("count", ctypes.c_int64), ("price", ctypes.c_double)]


class PackedQuote(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("tag", ctypes.c_int8), ("price", ctypes.c_double)]


QUOTE_LAYOUT = StructLayout(
    "Quote",
    ctypes.sizeof(Quote),
    (
        StructField("price", Quote.price.offset, types.float64),
        StructField("count", Quote.count.offset, types.int64),
    ),
)
REORDERED_LAYOUT = StructLayout(
    "Quote",
    ctypes.sizeof(ReorderedQuote),
    (
        StructField("price", ReorderedQuote.price.offset, types.float64),
        StructField("count", ReorderedQuote.count.offset, types.int64),
    ),
)
PACKED_LAYOUT = StructLayout(
    "PackedQuote",
    ctypes.sizeof(PackedQuote),
    (
        StructField("tag", PackedQuote.tag.offset, types.int8),
        StructField("price", PackedQuote.price.offset, types.float64),
    ),
)

QUOTE_VIEW = struct_view(QUOTE_LAYOUT)
REORDERED_VIEW = struct_view(REORDERED_LAYOUT)
PACKED_VIEW = struct_view(PACKED_LAYOUT)


@register_jitable(inline="always")
def adjust_price(quote, amount):
    quote.price += amount
    return quote.price + quote.count


@register_jitable(inline="always")
def invalid_price_expression(quote):
    return quote.price + quote.missing


def _ptr(value):
    return ctypes.c_void_p(ctypes.addressof(value))


def test_struct_view_reads_writes_and_crosses_helper():
    @cfunc(types.float64(types.voidptr, types.float64), _nrt=False)
    def update(raw_ptr, amount):
        quote = QUOTE_VIEW(raw_ptr)
        alias = quote
        return adjust_price(alias, amount)

    value = Quote(3.5, 4)
    assert update.ctypes(_ptr(value), 1.25) == 8.75
    assert value.price == 4.75
    assert value.count == 4
    llvm_ir = update.inspect_llvm()
    assert "NRT_MemInfo_alloc" not in llvm_ir
    assert "NRT_incref" not in llvm_ir


def test_struct_view_preserves_layout_identity_and_branch_join():
    assert struct_ptr_type(QUOTE_LAYOUT) is struct_ptr_type(QUOTE_LAYOUT)
    assert struct_ptr_type(QUOTE_LAYOUT) != struct_ptr_type(REORDERED_LAYOUT)
    assert QUOTE_LAYOUT.fingerprint != REORDERED_LAYOUT.fingerprint

    @cfunc(types.float64(types.voidptr), _nrt=False)
    def read_reordered(raw_ptr):
        return REORDERED_VIEW(raw_ptr).price

    value = ReorderedQuote(9, 2.25)
    assert read_reordered.ctypes(_ptr(value)) == 2.25

    @cfunc(types.float64(types.voidptr, types.voidptr, types.boolean), _nrt=False)
    def choose(first, second, use_first):
        if use_first:
            quote = QUOTE_VIEW(first)
        else:
            quote = QUOTE_VIEW(second)
        return quote.price

    first = Quote(5.0, 1)
    second = Quote(7.0, 2)
    assert choose.ctypes(_ptr(first), _ptr(second), True) == 5.0
    assert choose.ctypes(_ptr(first), _ptr(second), False) == 7.0

    with pytest.raises(TypingError):

        @cfunc(types.float64(types.voidptr, types.voidptr, types.boolean), _nrt=False)
        def incompatible(first, second, use_first):
            if use_first:
                quote = QUOTE_VIEW(first)
            else:
                quote = REORDERED_VIEW(second)
            return quote.price


def test_unaligned_host_field_is_loaded_and_stored_safely():
    @cfunc(types.float64(types.voidptr), _nrt=False)
    def update(raw_ptr):
        quote = PACKED_VIEW(raw_ptr)
        quote.price = quote.price + quote.tag
        return quote.price

    value = PackedQuote(2, 1.5)
    assert update.ctypes(_ptr(value)) == 3.5
    assert value.price == 3.5


def test_layout_validation_and_missing_fields():
    with pytest.raises(ValueError, match="exceeds layout"):
        StructLayout("Bad", 4, (StructField("value", 0, types.int64),))
    with pytest.raises(ValueError, match="overlaps"):
        StructLayout("Bad", 16, (StructField("a", 0, types.int64), StructField("b", 4, types.int64)))
    with pytest.raises(TypeError, match="unsupported native type"):
        StructField("value", 0, types.voidptr)

    with pytest.raises(TypingError):

        @cfunc(types.float64(types.voidptr), _nrt=False)
        def missing_field(raw_ptr):
            return QUOTE_VIEW(raw_ptr).missing

    with pytest.raises(TypingError):

        @cfunc(types.float64(types.voidptr), _nrt=False)
        def invalid_helper(raw_ptr):
            return invalid_price_expression(QUOTE_VIEW(raw_ptr))
