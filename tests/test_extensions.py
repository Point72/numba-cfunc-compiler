"""Consolidated integration tests for extensions."""

import ast
import ctypes
from enum import Enum

import llvmlite.binding as llvm
import pytest
from numba import cfunc, njit, types
from numba.core.errors import TypingError

from numba_cfunc_compiler.core.context import CompilationContext
from numba_cfunc_compiler.extension.ast import ASTHandlerRegistry, HandlerPhase, HandlerResult, ast_handler, with_handlers
from numba_cfunc_compiler.extension.bindings import register_value_type, value_binding
from numba_cfunc_compiler.extension.ffi import (
    ffi_binding_fingerprint,
    register_ffi_dependency,
    register_ffi_iterator_method,
    register_ffi_method,
    register_ffi_side_method,
)
from numba_cfunc_compiler.extension.methods import method_registration_key
from numba_cfunc_compiler.types.builtin.enum.host import register_enum_family
from numba_cfunc_compiler.types.builtin.enum.native import enum_code
from numba_cfunc_compiler.types.native.ffi import ffi_ref_from_voidptr, ffi_ref_type, ffi_side_from_voidptr, ffi_side_ref_type
from numba_cfunc_compiler.types.native.input import bind_input
from numba_cfunc_compiler.types.policy import ValueSemantics
from tests.harness import ACTIVE_SOURCE


class Venue(Enum):
    A = 1
    B = 2


class Side(Enum):
    A = 1
    B = 2


class SignedMode(Enum):
    DOWN = -1
    FLAT = 0
    UP = 1


def test_handlers():
    with CompilationContext():
        assert ASTHandlerRegistry.get_handlers("Name", HandlerPhase.PRE) == []
        calls = []

        def first(converter, node):
            calls.append("first")

        def second(converter, node):
            calls.append("second")
            return HandlerResult(ast.Name(id="handled", ctx=ast.Load()), [ast.Pass()])

        ASTHandlerRegistry.register("Name", second, HandlerPhase.PRE, priority=10)
        ASTHandlerRegistry.register("Name", first, HandlerPhase.PRE, priority=0)
        result, side_effects = ASTHandlerRegistry.run_pre_handlers("Name", None, ast.Name(id="x", ctx=ast.Load()))
        assert calls == ["first", "second"]
        assert result.id == "handled"
        assert len(side_effects) == 1

        def post_one(converter, node, result):
            return ast.Name(id=f"{result.id}_one", ctx=ast.Load())

        def post_two(converter, node, result):
            return HandlerResult(ast.Name(id=f"{result.id}_two", ctx=ast.Load()), [ast.Pass()])

        ASTHandlerRegistry.register("Name", post_one, HandlerPhase.POST, priority=0)
        ASTHandlerRegistry.register("Name", post_two, HandlerPhase.POST, priority=1)
        result, side_effects = ASTHandlerRegistry.run_post_handlers(
            "Name", None, ast.Name(id="x", ctx=ast.Load()), ast.Name(id="base", ctx=ast.Load()), []
        )
        assert result.id == "base_one_two"
        assert len(side_effects) == 1

        class Visitor:
            @with_handlers("Name")
            def visit_Name(self, node):
                return ast.Name(id="default", ctx=ast.Load())

        wrapped = Visitor().visit_Name(ast.Name(id="x", ctx=ast.Load()))
        assert isinstance(wrapped, list)
        assert wrapped[-1].id == "handled"
        ASTHandlerRegistry.clear("Name")
        assert Visitor().visit_Name(ast.Name(id="x", ctx=ast.Load())).id == "default"
        ASTHandlerRegistry.clear()
        with pytest.raises(ValueError, match="Must specify"):
            ast_handler("Name")
        with pytest.raises(ValueError, match="Cannot specify"):
            ast_handler("Name", pre=True, post=True)

        @ast_handler("Name", pre=True)
        def decorator_registered(converter, node):
            return ast.Name(id="decorated", ctx=ast.Load())

        result, _ = ASTHandlerRegistry.run_pre_handlers("Name", None, ast.Name(id="x", ctx=ast.Load()))
        assert result.id == "decorated"


def test_enum_sources_and_family_mismatch():
    venue = register_enum_family(Venue, abi_values={"A": 3, "B": 7})
    side = register_enum_family(Side, abi_values={"A": 3, "B": 7})
    venue_a = venue.literal("A")
    venue_b = venue.literal("B")
    side_a = side.literal("A")
    venue_set = venue.set_builder
    side_set = side.set_builder

    @venue.method("is_a")
    def is_a(value):
        return enum_code(value) == 3

    @side.method("is_side_a")
    def is_side_a(value):
        return enum_code(value) == 3

    @venue.overload_method("matches")
    def matches(value, other):
        if other != venue.value_type:
            return None

        def impl(value, other):
            return enum_code(value) == enum_code(other)

        return impl

    source_type = ACTIVE_SOURCE.type_for(venue.value_type, venue.storage_type, ValueSemantics.COPY, venue.state_payload().key)
    bind = bind_input(source_type)

    @cfunc(types.void(types.CPointer(types.voidptr), types.CPointer(types.int8), types.CPointer(types.int8), types.CPointer(types.int8)), _nrt=False)
    def callback(inputs, valid, ticked, output):
        source = bind(inputs[0], (valid, ticked, 0))
        alias = source
        selected = venue_set((alias, venue_b()))
        output[0] = alias.valid()
        output[1] = alias.ticked()
        output[2] = alias.isin(selected)
        output[3] = alias.is_a()
        output[4] = alias.matches(venue_a())

    value = ctypes.c_int16(3)
    inputs = (ctypes.c_void_p * 1)(ctypes.addressof(value))
    valid = (ctypes.c_int8 * 1)(1)
    ticked = (ctypes.c_int8 * 1)(1)
    output = (ctypes.c_int8 * 5)()
    callback.ctypes(inputs, valid, ticked, output)
    assert tuple(output) == (1, 1, 1, 1, 1)
    assert venue.value_type != side.value_type
    assert method_registration_key(venue.value_type) != method_registration_key(side.value_type)

    @njit
    def valid_set():
        selected = venue_a()
        allowed = venue_set((selected, venue_b()))
        empty = venue_set(())
        return selected.isin(allowed) and allowed.contains_any(venue_set((venue_b(),))) and allowed.contains_all(empty)

    assert valid_set()

    @njit
    def wrong_family_method():
        return venue_a().matches(side_a())

    with pytest.raises(TypingError, match="matches"):
        wrong_family_method()

    @njit
    def wrong_family_set():
        return venue_a().isin(side_set((side_a(),)))

    with pytest.raises(TypingError, match="isin"):
        wrong_family_set()

    signed = register_enum_family(SignedMode, abi_values={"DOWN": -1, "FLAT": 0, "UP": 1}, scalar_storage="int64", set_bits=None)
    down = signed.literal("DOWN")

    @njit
    def signed_code():
        return enum_code(down())

    assert signed_code() == -1


def test_ffi_alias_methods():
    ref_type = ffi_ref_type("TestCString")
    symbol = "ncc_test_cstring_length"
    # The current process does not export strlen on every platform.
    length_callback = ctypes.CFUNCTYPE(ctypes.c_size_t, ctypes.c_void_p)(lambda pointer: len(ctypes.string_at(pointer)))
    llvm.add_symbol(symbol, ctypes.cast(length_callback, ctypes.c_void_p).value)
    register_ffi_method(ref_type, "length", symbol, types.uintp)
    wrap = ffi_ref_from_voidptr(ref_type)

    @cfunc(types.void(types.voidptr, types.CPointer(types.uintp)), _nrt=False)
    def callback(raw, output):
        reference = wrap(raw)
        alias = reference
        output[0] = alias.length()

    storage = ctypes.create_string_buffer(b"typed ffi")
    result = ctypes.c_size_t()
    callback.ctypes(ctypes.cast(storage, ctypes.c_void_p), ctypes.pointer(result))
    assert result.value == len(b"typed ffi")

    first_type = ffi_ref_type("ScoreFirst")
    second_type = ffi_ref_type("ScoreSecond")
    side_type = ffi_side_ref_type("ScoreSide")
    callbacks = []

    def bind_symbol(name, function, *args):
        callback = ctypes.CFUNCTYPE(ctypes.c_int64, ctypes.c_void_p, *args)(function)
        callbacks.append(callback)
        llvm.add_symbol(name, ctypes.cast(callback, ctypes.c_void_p).value)

    bind_symbol("ncc_score_first", lambda _, value: value + 10, ctypes.c_int64)
    bind_symbol("ncc_score_second", lambda _, left, right: left * 10 + right, ctypes.c_int64, ctypes.c_int64)
    bind_symbol("ncc_score_bid", lambda _, value: value + 100, ctypes.c_int64)
    bind_symbol("ncc_score_ask", lambda _, value: value - 100, ctypes.c_int64)
    register_ffi_method(first_type, "score", "ncc_score_first", types.int64, (types.int64,))
    register_ffi_method(second_type, "score", "ncc_score_second", types.int64, (types.int64, types.int64))
    register_ffi_side_method(side_type, "score", "ncc_score_bid", "ncc_score_ask", types.int64, (types.int64,))
    first_source_type = ACTIVE_SOURCE.type_for(first_type, first_type, ValueSemantics.BORROWED_VIEW, ("score", "first"))
    number_source_type = ACTIVE_SOURCE.type_for(types.int64, types.int64, ValueSemantics.COPY, ("score", "number"))
    bind_first = bind_input(first_source_type)
    bind_number = bind_input(number_source_type)
    wrap_second = ffi_ref_from_voidptr(second_type)
    wrap_first = ffi_ref_from_voidptr(first_type)
    make_side = ffi_side_from_voidptr(side_type)

    @cfunc(types.void(types.voidptr, types.voidptr, types.CPointer(types.int8), types.CPointer(types.int64)), _nrt=False)
    def callback(raw, number_slot, flags, output):
        first = bind_first(raw, (flags, flags, 0))
        number = bind_number(number_slot, (flags, flags, 0))
        second = wrap_second(raw)
        side = make_side(raw, True)
        output[0] = first.score(number) + second.score(3, 4) + side.score(number)

    number = ctypes.c_int64(5)
    flags = (ctypes.c_int8 * 1)(1)
    output = ctypes.c_int64()
    callback.ctypes(ctypes.c_void_p(1), ctypes.c_void_p(ctypes.addressof(number)), flags, ctypes.pointer(output))
    assert output.value == 154

    with pytest.raises(TypingError, match="requires"):

        @cfunc(types.int64(types.voidptr), _nrt=False)
        def wrong_arity(raw):
            return wrap_first(raw).score(1, 2)


def test_ffi_iterators():
    book_type = ffi_ref_type("TestBook")
    side_type = ffi_side_ref_type("TestSide")
    item_type = ffi_ref_type("TestItem")
    callbacks = []

    def symbol(name, return_type, *argument_types):
        def decorate(function):
            callback = ctypes.CFUNCTYPE(return_type, *argument_types)(function)
            callbacks.append(callback)
            llvm.add_symbol(name, ctypes.cast(callback, ctypes.c_void_p).value)
            return callback

        return decorate

    @symbol("ncc_test_bid_side", ctypes.c_void_p, ctypes.c_void_p)
    def bid_side(_):
        return 42

    @symbol("ncc_test_ask_side", ctypes.c_void_p, ctypes.c_void_p)
    def ask_side(_):
        return 43

    @symbol("ncc_test_bid_begin", ctypes.c_void_p, ctypes.c_void_p)
    def bid_begin(_):
        return 1

    @symbol("ncc_test_ask_begin", ctypes.c_void_p, ctypes.c_void_p)
    def ask_begin(_):
        return 2

    @symbol("ncc_test_end", ctypes.c_void_p, ctypes.c_void_p)
    def end(_):
        return None

    @symbol("ncc_test_next", ctypes.c_void_p, ctypes.c_void_p)
    def next_item(pointer):
        return pointer + 1 if pointer < 2 else None

    @symbol("ncc_test_item_value", ctypes.c_int64, ctypes.c_void_p)
    def item_value(pointer):
        return pointer

    register_ffi_method(book_type, "bid_raw", "ncc_test_bid_side", types.voidptr)
    register_ffi_method(book_type, "ask_raw", "ncc_test_ask_side", types.voidptr)
    register_ffi_side_method(side_type, "begin_raw", "ncc_test_bid_begin", "ncc_test_ask_begin", item_type)
    register_ffi_side_method(side_type, "end_raw", "ncc_test_end", "ncc_test_end", item_type)
    register_ffi_method(item_type, "value", "ncc_test_item_value", types.int64)
    register_ffi_iterator_method(side_type, "items", "begin_raw", "end_raw", item_type, "ncc_test_next")
    register_ffi_dependency(book_type, side_type)
    fingerprint = ffi_binding_fingerprint(book_type)
    assert any(entry[:3] == (book_type.name, "dependency", side_type.name) for entry in fingerprint)
    assert any(entry[:3] == (side_type.name, "iterator", "items") for entry in fingerprint)
    assert any(entry[:3] == (item_type.name, "method", "value") for entry in fingerprint)
    wrap_book = ffi_ref_from_voidptr(book_type)
    make_side = ffi_side_from_voidptr(side_type)

    @cfunc(types.void(types.voidptr, types.CPointer(types.int64)), _nrt=False)
    def callback(raw, output):
        book = wrap_book(raw)
        bid = make_side(book.bid_raw(), True)
        ask = make_side(book.ask_raw(), False)
        total = 0
        for item in bid.items():
            total += item.value()
        for item in ask.items():
            total += item.value()
        output[0] = total

    output = ctypes.c_int64()
    callback.ctypes(None, ctypes.pointer(output))
    assert output.value == 5


def test_value_methods():
    class CopiedValue:
        pass

    class HostReference:
        pass

    with pytest.raises(TypeError, match="ValueSemantics member"):
        register_value_type(CopiedValue, numba_type=types.int32, storage=types.int32, semantics="copy")
    binding = register_value_type(CopiedValue, numba_type=types.int32, storage=types.int32, semantics=ValueSemantics.COPY)
    assert value_binding(CopiedValue) is binding
    assert register_value_type(CopiedValue, numba_type=types.int32, storage=types.int32, semantics=ValueSemantics.COPY) is binding
    assert binding.state_payload().host_size == 4
    with pytest.raises(ValueError, match="different value binding"):
        register_value_type(CopiedValue, numba_type=types.int64, storage=types.int64, semantics=ValueSemantics.COPY)
    with pytest.raises(ValueError, match="owner/lifetime"):
        register_value_type(HostReference, numba_type=ffi_ref_type("UnownedReference"), storage=("opaque",), semantics=ValueSemantics.BORROWED_VIEW)

    ref_type = ffi_ref_type("BindingHostReference")
    binding = register_value_type(
        HostReference,
        numba_type=ref_type,
        storage=("host pointer", 1),
        semantics=ValueSemantics.BORROWED_VIEW,
        mutable=False,
        owner="callback duration",
    )

    @binding.method("ready")
    def ready(value):
        return True

    assert binding.method("ready")(ready) is ready

    def other_ready(value):
        return False

    with pytest.raises(ValueError, match="already registered"):
        binding.method("ready")(other_ready)

    binding.method("valid")(ready)
    assert method_registration_key(ref_type) == (("ready", (), "method"), ("valid", (), "method"))

    payload = binding.state_payload()
    source_type = ACTIVE_SOURCE.type_for(ref_type, ref_type, ValueSemantics.BORROWED_VIEW, payload.key, payload.mutable)
    bind = bind_input(source_type)

    @cfunc(types.void(types.voidptr, types.CPointer(types.int8), types.CPointer(types.int8)), _nrt=False)
    def callback(raw, flags, output):
        source = bind(raw, (flags, flags, 0))
        alias = source
        output[0] = alias.valid() and alias.ready()

    storage = ctypes.c_int64(12)
    flags = (ctypes.c_int8 * 1)(1)
    output = (ctypes.c_int8 * 1)()
    callback.ctypes(ctypes.c_void_p(ctypes.addressof(storage)), flags, output)
    assert output[0] == 1
    flags[0] = 0
    callback.ctypes(ctypes.c_void_p(ctypes.addressof(storage)), flags, output)
    assert output[0] == 0
