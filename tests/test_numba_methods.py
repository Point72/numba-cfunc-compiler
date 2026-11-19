"""Helpers compiled in the calling node's Numba context."""

import ctypes
import enum
import types

import pytest

from numba_cfunc_compiler import numba_methods
from numba_cfunc_compiler.api import NumbaList, State, create_new_list, numba_method
from numba_cfunc_compiler.types.builtin.enum.host import register_enum_family
from tests.harness import CompiledNode, Signal, compile_function, numba_node
from tests.test_structs import Quote, TypedQuoteType, _compile, _execute


@numba_method
def advance(total, value):
    value += 1
    return total + value


@numba_node
def accumulate_with_method(x: Signal[int]) -> Signal[int]:
    total: State[int] = 0
    total = advance(total, x)
    return total


@numba_method
def append_and_count(values, value):
    values.append(value)
    return len(values)


@numba_node
def method_mutates_list(x: Signal[int]) -> Signal[int]:
    values: State[NumbaList] = create_new_list(int)
    return append_and_count(values, x)


@numba_method
def plus_two(value):
    return value + 2


@numba_method(force_inline=False)
def plus_three(value):
    return value + 3


@numba_method(force_inline=False)
def plus_two_without_inline(value):
    return value + 2


@numba_method
def keyword_only_plus(value, *, offset):
    return value + offset


@numba_method
def call_plus_two(value):
    return plus_two(value)


METHOD_MODULE = types.ModuleType("method_test_module")
METHOD_MODULE.apply = plus_two
INJECTED_METHOD = None


@numba_node
def nested_method(x: Signal[int]) -> Signal[int]:
    local = x
    return call_plus_two(local)


@numba_node
def repeated_method(x: Signal[int], y: Signal[float]) -> Signal[float]:
    return plus_two(x) + plus_two(y)


@numba_node
def invalid_second_method_call(x: Signal[int]) -> Signal[int]:
    first = plus_two(x)
    return first + plus_two(x + 1)


@numba_node
def module_method(x: Signal[int]) -> Signal[int]:
    return METHOD_MODULE.apply(x)


@numba_node
def injected_method(x: Signal[int]) -> Signal[int]:
    return INJECTED_METHOD(x)


def closure_method(method):
    @numba_node
    def call_method(x: Signal[int]) -> Signal[int]:
        return method(x)

    return call_method


@numba_node
def keyword_argument_method(x: Signal[int]) -> Signal[int]:
    return plus_two(value=x)


@numba_node
def keyword_only_method(x: Signal[int]) -> Signal[int]:
    return keyword_only_plus(x, offset=x)


@numba_node
def expression_argument(x: Signal[int]) -> Signal[int]:
    return plus_two(x + 1)


OFFSET = 2


@numba_method
def captured_global(value):
    return value + OFFSET


@numba_node
def invalid_capture(x: Signal[int]) -> Signal[int]:
    return captured_global(x)


@numba_method
def recursive(value):
    return recursive(value)


@numba_node
def invalid_recursion(x: Signal[int]) -> Signal[int]:
    return recursive(x)


@numba_method
def default_argument(value, offset=1):
    return value + offset


@numba_node
def invalid_default(x: Signal[int]) -> Signal[int]:
    return default_argument(x)


@numba_method
def adjust_quote(quote):
    quote.count += 1
    return quote.price


@numba_node
def method_struct_fields(quote: Signal[Quote]) -> Signal[float]:
    return adjust_quote(quote)


class Adjustment(enum.Enum):
    BONUS = 3


@numba_method
def add_adjustment(value):
    return value > 0 and Adjustment.BONUS == Adjustment.BONUS


@numba_node
def method_enum_literal(x: Signal[int]) -> Signal[bool]:
    return add_adjustment(x)


def test_scalar_state_and_python_callable():
    assert advance(2, 3) == 6
    assert numba_method(advance) is advance
    node = CompiledNode(compile_function(accumulate_with_method), input_types=[int]).start()
    assert node.execute([2])[0] == 3
    assert node.execute([4])[0] == 8


def test_explicit_mutable_state():
    node = CompiledNode(compile_function(method_mutates_list), input_types=[int]).start()
    assert node.execute([10])[0] == 1
    assert node.execute([20])[0] == 2


def test_native_struct_and_enum_types():
    result = _compile(method_struct_fields, TypedQuoteType)
    source = Quote(12.5, 3)
    output = ctypes.c_double()
    assert _execute(result, source, output)
    assert source.count == 4
    assert output.value == 12.5

    register_enum_family(Adjustment, abi_values={"BONUS": 3})
    node = CompiledNode(compile_function(method_enum_literal), input_types=[int]).start()
    assert node.execute([4])[0] is True


def test_nested_and_module_calls_change_semantic_key():
    nested = CompiledNode(compile_function(nested_method), input_types=[int]).start()
    assert nested.execute([5])[0] == 7
    original = compile_function(module_method)
    assert CompiledNode(original, input_types=[int]).start().execute([5])[0] == 7
    METHOD_MODULE.apply = plus_three
    try:
        changed = compile_function(module_method)
    finally:
        METHOD_MODULE.apply = plus_two
    assert changed.semantic_key != original.semantic_key
    assert CompiledNode(changed, input_types=[int]).start().execute([5])[0] == 8
    METHOD_MODULE.apply = plus_two_without_inline
    try:
        no_inline = compile_function(module_method)
    finally:
        METHOD_MODULE.apply = plus_two
    assert no_inline.semantic_key != original.semantic_key
    assert CompiledNode(no_inline, input_types=[int]).start().execute([5])[0] == 7


def test_repeated_helper_is_lowered_once_but_each_call_is_validated(monkeypatch):
    original = numba_methods._function_ast
    lowered = []

    def record_lowering(helper):
        lowered.append(helper)
        return original(helper)

    monkeypatch.setattr(numba_methods, "_function_ast", record_lowering)
    result = compile_function(repeated_method)
    assert lowered == [plus_two]
    node = CompiledNode(result, input_types=[int, float]).start()
    assert node.execute([3, 1.5])[0] == 8.5

    with pytest.raises(TypeError, match="must be a named variable"):
        compile_function(invalid_second_method_call)


def test_call_globals_and_closure_helpers_change_semantic_key():
    injected_two = compile_function(injected_method, call_globals={"INJECTED_METHOD": plus_two})
    injected_three = compile_function(injected_method, call_globals={"INJECTED_METHOD": plus_three})
    assert injected_two.semantic_key != injected_three.semantic_key
    assert CompiledNode(injected_two, input_types=[int]).start().execute([5])[0] == 7
    assert CompiledNode(injected_three, input_types=[int]).start().execute([5])[0] == 8

    closure_two = compile_function(closure_method(plus_two))
    closure_three = compile_function(closure_method(plus_three))
    assert closure_two.semantic_key != closure_three.semantic_key
    assert CompiledNode(closure_two, input_types=[int]).start().execute([5])[0] == 7
    assert CompiledNode(closure_three, input_types=[int]).start().execute([5])[0] == 8


def test_keyword_calls_reject_keyword_only_helper_parameters():
    node = CompiledNode(compile_function(keyword_argument_method), input_types=[int]).start()
    assert node.execute([5])[0] == 7
    with pytest.raises(TypeError, match="keyword_only_plus.*keyword-only parameters \\(offset\\); Numba cannot call"):
        compile_function(keyword_only_method)


def test_invalid_calls():
    with pytest.raises(TypeError, match="must be a named variable"):
        compile_function(expression_argument)
    with pytest.raises(TypeError, match="captures 'OFFSET'"):
        compile_function(invalid_capture)
    with pytest.raises(TypeError, match="Recursive @numba_method"):
        compile_function(invalid_recursion)
    with pytest.raises(TypeError, match="cannot declare default arguments"):
        compile_function(invalid_default)
    with pytest.raises(TypeError, match="force_inline must be a bool"):
        numba_method(force_inline=1)
