"""Fixed arrays across the native callback boundary."""

import pytest
from numba.core.errors import TypingError

from numba_cfunc_compiler.api import NumbaArray, State, create_new_array
from tests.harness import CompiledNode, Signal, compile_function, numba_node


@numba_node
def state_array(index: Signal[int], value: Signal[int]) -> Signal[int]:
    values: State[NumbaArray] = create_new_array(int, 4)
    values[index] = values[index] + value
    return values[-1] + values[index] + len(values)


@numba_node
def constant_array(value: Signal[int], values: NumbaArray[int, 3]) -> Signal[int]:
    total = 0
    for item in values:
        total += item
    return total + value + len(values)


@numba_node
def mutate_constant_array(value: Signal[int], values: NumbaArray[int, 3]) -> Signal[int]:
    alias = values
    alias[0] = value
    return alias[0]


@numba_node
def local_array(value: Signal[int]) -> Signal[int]:
    values = create_new_array(int, 3)
    return value + len(values)


@numba_node
def bool_array(value: Signal[bool]) -> Signal[bool]:
    values: State[NumbaArray] = create_new_array(bool, 2)
    values[1] = value
    return values[-1]


@numba_node
def float_array(value: Signal[float]) -> Signal[float]:
    values: State[NumbaArray] = create_new_array(float, 2)
    values[0] = value
    return values[0]


@numba_node
def bad_index(value: Signal[int]) -> Signal[int]:
    values: State[NumbaArray] = create_new_array(int, 2)
    return values[2] + value


@numba_node
def bad_folded_index(value: Signal[int]) -> Signal[int]:
    values: State[NumbaArray] = create_new_array(int, 2)
    return values[1 + 1] + value


@numba_node
def bad_length(value: Signal[int]) -> Signal[int]:
    values: State[NumbaArray] = create_new_array(int, 0)
    return value + len(values)


def test_state_array_persists_and_uses_raw_storage():
    result = compile_function(state_array)
    assert result.struct_state_indices == (0,)
    assert result.struct_state_sizes == (32,)
    node = CompiledNode(result, [int, int]).start()
    assert node.execute([1, 5]) == (9, True)
    assert node.execute([1, 7]) == (16, True)
    assert node.execute([-1, 3]) == (10, True)
    assert node.execute([1, 0]) == (19, True)
    node.stop()


def test_constant_array_iterates_and_is_readonly():
    node = CompiledNode(compile_function(constant_array, values=[1, 2, 3]), [int]).start()
    assert node.execute([10]) == (19, True)
    assert node.execute([20]) == (29, True)
    node.stop()

    with pytest.raises(TypingError):
        compile_function(mutate_constant_array, values=[1, 2, 3])


def test_array_input_validation_and_local_rejection():
    with pytest.raises(TypeError, match="State\\[NumbaArray\\].*line"):
        compile_function(local_array)
    with pytest.raises(TypeError, match="expected 3 elements"):
        compile_function(constant_array, values=[1, 2])
    with pytest.raises(TypeError, match="element 1: expected int"):
        compile_function(constant_array, values=[1, "bad", 3])


def test_bool_and_float_arrays():
    bool_node = CompiledNode(compile_function(bool_array), [bool]).start()
    float_node = CompiledNode(compile_function(float_array), [float]).start()
    assert bool_node.execute([True]) == (True, True)
    assert bool_node.execute([False]) == (False, True)
    assert float_node.execute([1.5]) == (1.5, True)
    bool_node.stop()
    float_node.stop()


def test_invalid_static_bounds_and_length_are_rejected():
    with pytest.raises(Exception, match="out of range"):
        compile_function(bad_index)
    with pytest.raises(Exception, match="out of range"):
        compile_function(bad_folded_index)
    with pytest.raises(TypeError, match="positive compile-time integer"):
        compile_function(bad_length)
