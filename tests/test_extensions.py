"""Consolidated integration tests for extensions."""

import ast
import ctypes
from enum import Enum

import llvmlite.binding as llvm
import pytest
from numba import cfunc, njit, types
from numba.core.errors import TypingError

from numba_cfunc_compiler.core.context import CompilationContext
from numba_cfunc_compiler.core.converter import NumbaASTConverter
from numba_cfunc_compiler.core.defaults import register_all
from numba_cfunc_compiler.core.variable_factory import VariableFactory
from numba_cfunc_compiler.extension.ast import ASTHandlerRegistry, HandlerPhase, HandlerResult, LifecycleBody, ast_handler, with_handlers
from numba_cfunc_compiler.extension.bindings import register_value_type, value_binding
from numba_cfunc_compiler.extension.ffi import (
    ffi_binding_fingerprint,
    register_ffi_dependency,
    register_ffi_iterator_method,
    register_ffi_method,
    register_ffi_tagged_method,
)
from numba_cfunc_compiler.extension.methods import method_registration_key
from numba_cfunc_compiler.types.builtin.enum.host import register_enum_family
from numba_cfunc_compiler.types.builtin.enum.native import enum_code
from numba_cfunc_compiler.types.native.ffi import (
    ffi_ref_from_voidptr,
    ffi_ref_type,
    ffi_tagged_from_voidptr,
    ffi_tagged_ref_type,
)
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


@pytest.fixture
def bind_ffi_symbol():
    callbacks = []

    def bind(name, return_type, function, *argument_types):
        callback = ctypes.CFUNCTYPE(return_type, ctypes.c_void_p, *argument_types)(function)
        callbacks.append(callback)
        llvm.add_symbol(name, ctypes.cast(callback, ctypes.c_void_p).value)
        return callback

    return bind


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


def test_ast_handlers_receive_lifecycle_body():
    with CompilationContext():
        register_all()
        observed = []

        @ast_handler("Expr", pre=True)
        def record_body(converter, node):
            observed.append((node.value.value, converter.current_body))

        tree = ast.parse("def callback():\n    3").body[0]
        converter = NumbaASTConverter(
            tree,
            VariableFactory(),
            start_body=ast.parse("1").body,
            stop_body=ast.parse("2").body,
        )
        assert converter.current_body is None
        converter.visit(tree)
        assert dict(observed) == {1: LifecycleBody.START, 2: LifecycleBody.STOP, 3: LifecycleBody.EXECUTE}
        assert converter.current_body is None


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


def test_ffi_alias_methods(bind_ffi_symbol):
    ref_type = ffi_ref_type("TestCString")
    symbol = "ncc_test_cstring_length"
    # The current process does not export strlen on every platform.
    bind_ffi_symbol(symbol, ctypes.c_size_t, lambda pointer: len(ctypes.string_at(pointer)))
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
    bind_ffi_symbol("ncc_score_first", ctypes.c_int64, lambda _, value: value + 10, ctypes.c_int64)
    bind_ffi_symbol("ncc_score_second", ctypes.c_int64, lambda _, left, right: left * 10 + right, ctypes.c_int64, ctypes.c_int64)
    register_ffi_method(first_type, "score", "ncc_score_first", types.int64, (types.int64,))
    register_ffi_method(second_type, "score", "ncc_score_second", types.int64, (types.int64, types.int64))
    register_ffi_method(first_type, "score", "ncc_score_first", types.int64, (types.int64,))
    with pytest.raises(ValueError, match="Conflicting"):
        register_ffi_method(first_type, "score", "ncc_score_second", types.int64, (types.int64,))
    first_source_type = ACTIVE_SOURCE.type_for(first_type, first_type, ValueSemantics.BORROWED_VIEW, ("score", "first"))
    number_source_type = ACTIVE_SOURCE.type_for(types.int64, types.int64, ValueSemantics.COPY, ("score", "number"))
    bind_first = bind_input(first_source_type)
    bind_number = bind_input(number_source_type)
    wrap_second = ffi_ref_from_voidptr(second_type)
    wrap_first = ffi_ref_from_voidptr(first_type)

    @cfunc(types.void(types.voidptr, types.voidptr, types.CPointer(types.int8), types.CPointer(types.int64)), _nrt=False)
    def callback(raw, number_slot, flags, output):
        first = bind_first(raw, (flags, flags, 0))
        number = bind_number(number_slot, (flags, flags, 0))
        second = wrap_second(raw)
        output[0] = first.score(number) + second.score(3, 4)

    number = ctypes.c_int64(5)
    flags = (ctypes.c_int8 * 1)(1)
    output = ctypes.c_int64()
    callback.ctypes(ctypes.c_void_p(1), ctypes.c_void_p(ctypes.addressof(number)), flags, ctypes.pointer(output))
    assert output.value == 49

    with pytest.raises(TypingError, match="requires"):

        @cfunc(types.int64(types.voidptr), _nrt=False)
        def wrong_arity(raw):
            return wrap_first(raw).score(1, 2)


def test_ffi_iterators(bind_ffi_symbol):
    owner_type = ffi_ref_type("TestRangeOwner")
    item_type = ffi_ref_type("TestRangeItem")
    bind_ffi_symbol("ncc_direct_begin", ctypes.c_void_p, lambda _: 1)
    bind_ffi_symbol("ncc_direct_end", ctypes.c_void_p, lambda _: None)
    bind_ffi_symbol("ncc_direct_next", ctypes.c_void_p, lambda pointer: pointer + 1 if pointer < 2 else None)
    bind_ffi_symbol("ncc_direct_value", ctypes.c_int64, lambda pointer: pointer)

    register_ffi_method(owner_type, "begin_raw", "ncc_direct_begin", item_type)
    register_ffi_method(owner_type, "end_raw", "ncc_direct_end", item_type)
    register_ffi_method(item_type, "value", "ncc_direct_value", types.int64)
    register_ffi_iterator_method(owner_type, "items", "begin_raw", "end_raw", item_type, "ncc_direct_next")
    register_ffi_dependency(owner_type, item_type)
    fingerprint = ffi_binding_fingerprint(owner_type)
    assert any(entry[:3] == (owner_type.name, "iterator", "items") for entry in fingerprint)
    assert any(entry[:3] == (item_type.name, "method", "value") for entry in fingerprint)
    assert any(entry[1:4] == ("method", "value", (("ncc_direct_value",), types.int64.name, ())) for entry in fingerprint)
    wrap_owner = ffi_ref_from_voidptr(owner_type)

    @cfunc(types.void(types.voidptr, types.CPointer(types.int64)), _nrt=False)
    def callback(raw, output):
        total = 0
        for item in wrap_owner(raw).items():
            total += item.value()
        output[0] = total

    output = ctypes.c_int64()
    callback.ctypes(None, ctypes.pointer(output))
    assert output.value == 3


def test_tagged_ffi_methods_and_fingerprint(bind_ffi_symbol):
    ref_type = ffi_tagged_ref_type("TaggedScore", ("low", "medium", "high"))
    assert ref_type.tag("low") == 0
    assert ref_type.tag("high") == 2
    assert ref_type == ffi_tagged_ref_type("TaggedScore", ("low", "medium", "high"))
    assert ref_type != ffi_tagged_ref_type("TaggedScore", ("high", "medium", "low"))
    with pytest.raises(ValueError, match="Unknown case"):
        ref_type.tag("missing")
    with pytest.raises(ValueError, match="distinct"):
        ffi_tagged_ref_type("DuplicateCases", ("low", "low"))

    bind_ffi_symbol("ncc_tag_low", ctypes.c_int64, lambda _, value: value + 1, ctypes.c_int64)
    bind_ffi_symbol("ncc_tag_medium", ctypes.c_int64, lambda _, value: value + 10, ctypes.c_int64)
    bind_ffi_symbol("ncc_tag_high", ctypes.c_int64, lambda _, value: value + 100, ctypes.c_int64)
    bind_ffi_symbol("ncc_tag_shared", ctypes.c_int64, lambda _, value: value * 2, ctypes.c_int64)

    score = {"low": "ncc_tag_low", "medium": "ncc_tag_medium", "high": "ncc_tag_high"}
    shared = dict.fromkeys(ref_type.cases, "ncc_tag_shared")
    register_ffi_tagged_method(ref_type, "score", score, types.int64, (types.int64,))
    register_ffi_tagged_method(ref_type, "shared", shared, types.int64, (types.int64,))
    register_ffi_tagged_method(ref_type, "score", dict(reversed(list(score.items()))), types.int64, (types.int64,))
    with pytest.raises(ValueError, match="Invalid tagged"):
        register_ffi_tagged_method(ref_type, "missing", {"low": "ncc_tag_low"}, types.int64)
    with pytest.raises(ValueError, match="Conflicting"):
        register_ffi_tagged_method(ref_type, "score", shared, types.int64, (types.int64,))

    wrap = ffi_tagged_from_voidptr(ref_type)

    @cfunc(types.int64(types.voidptr, types.int32, types.int64), _nrt=False)
    def callback(raw, tag, value):
        reference = wrap(raw, tag)
        return reference.score(value) + reference.shared(value)

    assert [callback.ctypes(None, tag, 5) for tag in range(3)] == [16, 25, 115]
    assert "llvm.trap" in callback.inspect_llvm()

    low_tag = ref_type.tag("low")
    high_tag = ref_type.tag("high")

    @cfunc(types.int64(types.voidptr, types.boolean), _nrt=False)
    def select_case(raw, low):
        if low:
            reference = wrap(raw, low_tag)
        else:
            reference = wrap(raw, high_tag)
        return reference.score(5)

    assert select_case.ctypes(None, True) == 6
    assert select_case.ctypes(None, False) == 105

    @njit
    def bad_literal(raw):
        return wrap(raw, 3)

    with pytest.raises((TypingError, ValueError), match="Invalid literal tag"):
        bad_literal.compile((types.voidptr,))

    @njit
    def bad_argument(raw):
        return wrap(raw, 0).score(1.5)

    with pytest.raises(TypingError, match="requires"):
        bad_argument.compile((types.voidptr,))

    fingerprint = ffi_binding_fingerprint(ref_type)
    root_type = ffi_ref_type("TaggedScoreRoot")
    register_ffi_dependency(root_type, ref_type)
    assert any(entry[1:3] == ("cases", ref_type.cases) for entry in ffi_binding_fingerprint(root_type))
    reordered = ffi_tagged_ref_type("TaggedScore", ("high", "medium", "low"))
    register_ffi_tagged_method(reordered, "score", score, types.int64, (types.int64,))
    wrap_reordered = ffi_tagged_from_voidptr(reordered)

    @cfunc(types.int64(types.voidptr), _nrt=False)
    def reordered_score(raw):
        return wrap_reordered(raw, 0).score(5)

    assert reordered_score.ctypes(None) == 105
    changed = ffi_tagged_ref_type("TaggedScoreChanged", ref_type.cases)
    register_ffi_tagged_method(changed, "score", {**score, "high": "ncc_tag_low"}, types.int64, (types.int64,))

    def entry(fingerprint, kind, name):
        return next(item for item in fingerprint if item[1:3] == (kind, name))

    assert entry(fingerprint, "method", "score")[3][0] == ("ncc_tag_low", "ncc_tag_medium", "ncc_tag_high")
    assert entry(fingerprint, "cases", ref_type.cases)[2] != entry(ffi_binding_fingerprint(reordered), "cases", reordered.cases)[2]
    assert entry(fingerprint, "method", "score")[3][0] != entry(ffi_binding_fingerprint(changed), "method", "score")[3][0]


def test_tagged_ffi_iterator_and_void_method(bind_ffi_symbol):
    ref_type = ffi_tagged_ref_type("TaggedItems", ("one", "two", "three"))
    item_type = ffi_ref_type("TaggedItem")
    for tag in range(3):
        bind_ffi_symbol(f"ncc_tag_begin_{tag}", ctypes.c_void_p, lambda _, tag=tag: tag + 1)
    bind_ffi_symbol("ncc_tag_end", ctypes.c_void_p, lambda _: None)
    bind_ffi_symbol("ncc_tag_next", ctypes.c_void_p, lambda _: None)
    bind_ffi_symbol("ncc_tag_item_value", ctypes.c_int64, lambda pointer: pointer)
    bind_ffi_symbol(
        "ncc_tag_store", None, lambda pointer, value: ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int64)).__setitem__(0, value), ctypes.c_int64
    )

    begin = {case: f"ncc_tag_begin_{tag}" for tag, case in enumerate(ref_type.cases)}
    end = dict.fromkeys(ref_type.cases, "ncc_tag_end")
    register_ffi_tagged_method(ref_type, "begin", begin, item_type)
    register_ffi_tagged_method(ref_type, "end", end, item_type)
    register_ffi_tagged_method(ref_type, "store", dict.fromkeys(ref_type.cases, "ncc_tag_store"), types.void, (types.int64,))
    register_ffi_method(item_type, "value", "ncc_tag_item_value", types.int64)
    register_ffi_iterator_method(ref_type, "items", "begin", "end", item_type, "ncc_tag_next")
    register_ffi_dependency(ref_type, item_type)
    assert any(entry[1:3] == ("iterator", "items") for entry in ffi_binding_fingerprint(ref_type))
    wrap = ffi_tagged_from_voidptr(ref_type)

    @cfunc(types.int64(types.voidptr, types.int32), _nrt=False)
    def callback(raw, tag):
        reference = wrap(raw, tag)
        reference.store(42)
        total = 0
        for item in reference.items():
            total += item.value()
        return total

    value = ctypes.c_int64(0)
    pointer = ctypes.cast(ctypes.pointer(value), ctypes.c_void_p)
    assert [callback.ctypes(pointer, tag) for tag in range(3)] == [1, 2, 3]
    assert value.value == 42


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
