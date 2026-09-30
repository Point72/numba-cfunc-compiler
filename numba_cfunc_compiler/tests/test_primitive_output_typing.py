"""Primitive output checks at the generated callback boundary."""

import pytest
from numba.core.errors import TypingError
from numba.extending import register_jitable

from numba_cfunc_compiler.node_api import set_output
from numba_cfunc_compiler.numba_core import create_compiled_func
from numba_cfunc_compiler.tests.harness import CompiledNode, Signal, compile_function, numba_node, setup_standalone_context


@register_jitable(inline="always")
def integer_helper(value):
    return value + 1


@register_jitable(inline="always")
def float_helper(value):
    return value + 0.5


@numba_node
def int_from_int(x: Signal[int]) -> Signal[int]:
    return x + 2


@numba_node
def float_from_float(x: Signal[float]) -> Signal[float]:
    return x + 0.25


@numba_node
def bool_from_compare(x: Signal[int]) -> Signal[bool]:
    return x > 0


@numba_node
def int_from_float_literal(x: Signal[int]) -> Signal[int]:
    return 1.25


@numba_node
def float_from_int_literal(x: Signal[float]) -> Signal[float]:
    return 1


@numba_node
def int_from_bool_literal(x: Signal[int]) -> Signal[int]:
    return True


@numba_node
def int_from_float_expression(x: Signal[float]) -> Signal[int]:
    return x + 0.25


@numba_node
def float_from_int_expression(x: Signal[int]) -> Signal[float]:
    return x + 1


@numba_node
def int_from_bool_expression(x: Signal[int]) -> Signal[int]:
    return (x > 0) or False


@numba_node
def set_int_from_expression(x: Signal[int]) -> Signal[int]:
    set_output("output_0", x + 1)


@numba_node
def set_int_from_float_expression(x: Signal[float]) -> Signal[int]:
    set_output("output_0", x + 0.25)


@numba_node
def bool_identity(flag: Signal[bool]) -> Signal[bool]:
    return flag


@numba_node
def bool_alias(flag: Signal[bool]) -> Signal[bool]:
    local = flag
    return local


@numba_node
def int_from_bool_source(flag: Signal[bool]) -> Signal[int]:
    return flag


@numba_node
def int_from_bool_alias(flag: Signal[bool]) -> Signal[int]:
    local = flag
    return local


@numba_node
def rebound_int_local(x: Signal[int]) -> Signal[int]:
    local = x
    local = 9
    return local


@numba_node
def rebound_bool_local(flag: Signal[bool]) -> Signal[bool]:
    local = flag
    local = False
    return local


@numba_node
def int_from_helper(x: Signal[int]) -> Signal[int]:
    return integer_helper(x)


@numba_node
def int_from_float_helper(x: Signal[int]) -> Signal[int]:
    return float_helper(x)


def _compile_helper(func):
    with setup_standalone_context():
        return create_compiled_func(
            func,
            Signal(typ=int),
            extract_python_type_fn=lambda signal: signal.get_type(),
            call_globals={"integer_helper": integer_helper, "float_helper": float_helper},
        )


def test_matching_primitive_outputs_and_source_bool_representation():
    assert CompiledNode(compile_function(int_from_int), [int]).execute([3]) == (5, True)
    assert CompiledNode(compile_function(float_from_float), [float]).execute([1.5]) == (1.75, True)
    assert CompiledNode(compile_function(bool_from_compare), [int]).execute([2]) == (True, True)
    assert CompiledNode(compile_function(set_int_from_expression), [int]).execute([4]) == (5, True)
    assert CompiledNode(compile_function(bool_identity), [bool]).execute([True]) == (True, True)
    assert CompiledNode(compile_function(bool_alias), [bool]).execute([True]) == (True, True)
    assert CompiledNode(compile_function(rebound_int_local), [int]).execute([3]) == (9, True)
    assert CompiledNode(compile_function(rebound_bool_local), [bool]).execute([True]) == (False, True)
    assert CompiledNode(_compile_helper(int_from_helper), [int]).execute([4]) == (5, True)


def test_primitive_output_type_checks_use_numba_expression_type():
    for func in (
        int_from_float_literal,
        float_from_int_literal,
        int_from_bool_literal,
        int_from_float_expression,
        float_from_int_expression,
        int_from_bool_expression,
        set_int_from_float_expression,
        int_from_bool_source,
        int_from_bool_alias,
    ):
        with pytest.raises(TypingError, match="Primitive .* output cannot store value"):
            compile_function(func)

    with pytest.raises(TypingError, match="Primitive int output cannot store value of type float64"):
        _compile_helper(int_from_float_helper)
