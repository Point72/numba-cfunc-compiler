"""Fixed arrays across the native callback boundary."""

import ctypes

import pytest
from numba.core.errors import TypingError

from numba_cfunc_compiler.api import NumbaArray, State, create_new_array
from numba_cfunc_compiler.types.builtin.array.native import standalone_array_clear
from numba_cfunc_compiler.types.factory import HostTypeFactory
from tests.harness import CompiledNode, Signal, compile_function, numba_node, setup_standalone_context
from tests.helpers import CtypesStructHostType


class ArrayReadout(ctypes.Structure):
    _fields_ = [
        ("int_sum", ctypes.c_int64),
        ("float_sum", ctypes.c_double),
        ("bool_count", ctypes.c_int64),
        ("int_length", ctypes.c_int64),
        ("float_length", ctypes.c_int64),
        ("bool_length", ctypes.c_int64),
    ]


class ArrayReadoutHostType(CtypesStructHostType):
    storage_type = ArrayReadout


@numba_node
def state_arrays(index: Signal[int], value: Signal[int]) -> Signal[float]:
    ints: State[NumbaArray] = create_new_array(int, 4)
    floats: State[NumbaArray] = create_new_array(float, 2)
    bools: State[NumbaArray] = create_new_array(bool, 2)
    ints[index] = ints[index] + value
    floats[0] = value + 0.5
    bools[1] = value > 0
    return ints[-1] + ints[index] + len(ints) + floats[0] + int(bools[-1])


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
def clear_array(value: Signal[int]) -> Signal[ArrayReadout]:
    ints: State[NumbaArray] = create_new_array(int, 3)
    floats: State[NumbaArray] = create_new_array(float, 2)
    bools: State[NumbaArray] = create_new_array(bool, 2)
    readout: State[ArrayReadout] = None
    int_alias = ints
    if value < 0:
        int_alias.clear()
        floats.clear()
        bools.clear()
    else:
        ints[0] = value
        ints[1] = value + 1
        ints[2] = value + 2
        floats[0] = value + 0.25
        floats[1] = value + 0.5
        bools[0] = True
        bools[1] = True
    readout.int_sum = ints[0] + ints[1] + ints[2]
    readout.float_sum = floats[0] + floats[1]
    readout.bool_count = int(bools[0]) + int(bools[1])
    readout.int_length = len(ints)
    readout.float_length = len(floats)
    readout.bool_length = len(bools)
    return readout


@numba_node
def clear_constant_array(value: Signal[int], values: NumbaArray[int, 3]) -> Signal[int]:
    alias = values
    alias.clear()
    return value + len(values)


@numba_node
def clear_constant_array_intrinsic(value: Signal[int], values: NumbaArray[int, 3]) -> Signal[int]:
    standalone_array_clear(values)
    return value + len(values)


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


def test_state_arrays():
    result = compile_function(state_arrays)
    assert result.struct_state_indices == (0, 1, 2)
    assert result.struct_state_sizes == (32, 16, 2)
    node = CompiledNode(result, [int, int]).start()
    assert node.execute([1, 5]) == (15.5, True)
    assert node.execute([1, 7]) == (24.5, True)
    assert node.execute([-1, 3]) == (14.5, True)
    assert node.execute([1, 0]) == (19.5, True)
    node.stop()


def test_constant_array_iterates_and_is_readonly():
    node = CompiledNode(compile_function(constant_array, values=[1, 2, 3]), [int]).start()
    assert node.execute([10]) == (19, True)
    assert node.execute([20]) == (29, True)
    node.stop()

    with pytest.raises(TypingError):
        compile_function(mutate_constant_array, values=[1, 2, 3])


def test_invalid_arrays_are_rejected():
    with pytest.raises(TypeError, match="State\\[NumbaArray\\].*line"):
        compile_function(local_array)
    with pytest.raises(TypeError, match="expected 3 elements"):
        compile_function(constant_array, values=[1, 2])
    with pytest.raises(TypeError, match="element 1: expected int"):
        compile_function(constant_array, values=[1, "bad", 3])
    with pytest.raises(Exception, match="out of range"):
        compile_function(bad_index)
    with pytest.raises(Exception, match="out of range"):
        compile_function(bad_folded_index)
    with pytest.raises(TypeError, match="positive compile-time integer"):
        compile_function(bad_length)


def test_clear_array():
    with setup_standalone_context():
        HostTypeFactory.register(ArrayReadoutHostType, priority=0)
    node = CompiledNode(compile_function(clear_array), [int]).start()
    for value, totals in ((2, (9, 4.75, 2)), (5, (18, 10.75, 2)), (-1, (0, 0.0, 0)), (-1, (0, 0.0, 0)), (1, (6, 2.75, 2))):
        readout, ticked = node.execute([value])
        assert ticked
        assert (readout.int_sum, readout.float_sum, readout.bool_count) == totals
        assert (readout.int_length, readout.float_length, readout.bool_length) == (3, 2, 2)
    node.stop()

    for invalid_node in (clear_constant_array, clear_constant_array_intrinsic):
        with pytest.raises(TypingError):
            compile_function(invalid_node, values=[1, 2, 3])
