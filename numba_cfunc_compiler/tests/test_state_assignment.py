"""State assignment and alias behavior through the native callback ABI."""

import ast
import ctypes
from datetime import datetime, timedelta

import pytest
from numba import types
from numba.core.errors import TypingError

from numba_cfunc_compiler.compilation_context import CompilationContext
from numba_cfunc_compiler.defaults import register_all
from numba_cfunc_compiler.defaults.struct_support import StructFieldInfo, StructType
from numba_cfunc_compiler.function_analyzer import FunctionAnalyzer
from numba_cfunc_compiler.node_api import NumbaList, State, create_new_list
from numba_cfunc_compiler.numba_core import create_compiled_func
from numba_cfunc_compiler.source_registry import SourceRegistry
from numba_cfunc_compiler.state_values import copy_state_payload
from numba_cfunc_compiler.tests.harness import (
    _CFUNC_T,
    LIFECYCLE_EXECUTE,
    CompiledNode,
    Signal,
    _SignalCategory,
    _SignalInputHandler,
    _SingleSignalOutputHandler,
    compile_function,
    numba_node,
    setup_standalone_context,
)
from numba_cfunc_compiler.type_factory import TypeFactory


@numba_node
def copied_counter(change: Signal[int]) -> Signal[int]:
    counter: State[int] = 0
    observed = counter
    if change > 0:
        counter = counter + change
    return observed


@numba_node
def current_counter(change: Signal[int]) -> Signal[int]:
    counter: State[int] = 0
    counter += change
    return counter


@numba_node
def chained_counter(change: Signal[int]) -> Signal[int]:
    counter: State[int] = 0
    counter = other = change
    return other + counter


@numba_node
def chained_raw_value(change: Signal[int]) -> Signal[int]:
    counter: State[int] = 0
    counter = other = change  # noqa: F841 - verify the other target keeps the raw RHS
    other += 1
    return other


@numba_node
def walrus_direct_value(change: Signal[int]) -> Signal[int]:
    counter: State[int] = 0
    other = (counter := change + 1)  # noqa: F841 - verify the expression result keeps the raw RHS
    other += 1
    return other


@numba_node
def unpacked_counter(change: Signal[int]) -> Signal[int]:
    counter: State[int] = 0
    counter, other = (change, change + 1)
    return other + counter


@numba_node
def loop_counter(change: Signal[int]) -> Signal[int]:
    counter: State[int] = 0
    for counter in range(change):
        pass
    return counter


@numba_node
def while_counter(change: Signal[int]) -> Signal[int]:
    counter: State[int] = 0
    while counter < change:
        counter += 1
    return counter


@numba_node
def walrus_counter(change: Signal[int]) -> Signal[int]:
    counter: State[int] = 0
    if (counter := change) > 0:
        return counter
    return counter


@numba_node
def walrus_constant(change: Signal[int]) -> Signal[int]:
    counter: State[int] = 0
    result = (counter := 5) + change  # noqa: F841 - the host state store is the effect
    return result


@numba_node
def walrus_call(change: Signal[int]) -> Signal[int]:
    counter: State[int] = 0
    result = (counter := abs(change)) + counter
    return result


@numba_node
def walrus_twice(change: Signal[int]) -> Signal[int]:
    counter: State[int] = 0
    result = (counter := change + 1) + (counter := change + 2)
    return result + counter


@numba_node
def walrus_local(change: Signal[int]) -> Signal[int]:
    counter: State[int] = 0
    result = (intermediate := change + 1) + (counter := intermediate * 2)  # noqa: F841 - host state store
    return result


@numba_node
def walrus_return(change: Signal[int]) -> Signal[int]:
    counter: State[int] = 0
    return (counter := change + 1)  # noqa: F841 - the state store is the effect


@numba_node
def branch_and_early_return(change: Signal[int]) -> Signal[int]:
    counter: State[int] = 0
    if change > 0:
        counter = change
        return counter
    return counter


@numba_node
def alias_rebinding(change: Signal[int]) -> Signal[int]:
    counter: State[int] = 0
    counter = change
    alias = counter
    alias = counter
    return alias


@numba_node
def chained_side_effect(change: Signal[int]) -> Signal[int]:
    values: State[NumbaList] = create_new_list(int)
    counter: State[int] = 0
    values.append(change)
    values.append(change + 1)
    counter = other = values.pop()
    return len(values) * 100 + counter + other


@numba_node
def copied_float(change: Signal[float]) -> Signal[float]:
    total: State[float] = 0.0
    old = total
    total = total + change
    return old


@numba_node
def copied_bool(change: Signal[int]) -> Signal[bool]:
    flag: State[bool] = False
    old = flag
    flag = change > 0
    return old


@numba_node
def copied_time_values(change: Signal[int]) -> Signal[int]:
    stamp: State[datetime] = 1
    span: State[timedelta] = 2
    stamp += change
    span += change
    return stamp + span


@numba_node
def chained_distinct_time_states(change: Signal[int]) -> Signal[int]:
    stamp: State[datetime] = 0
    span: State[timedelta] = 0
    stamp = span = change
    return stamp + span


@numba_node
def wrong_state_type(change: Signal[int]) -> Signal[int]:
    counter: State[int] = 0
    counter = 1.5
    return counter + change


@numba_node
def wrong_boolean_state_type(change: Signal[int]) -> Signal[int]:
    counter: State[int] = 0
    counter = change > 0
    return counter


@numba_node
def changed_alias_type(change: Signal[int]) -> Signal[int]:
    counter: State[int] = 0
    alias = counter
    alias = alias + change
    return alias


@numba_node
def changed_numeric_type(change: Signal[int]) -> Signal[float]:
    local = change
    local = 1.5
    return local


@numba_node
def same_local_type(change: Signal[int]) -> Signal[int]:
    local = change
    local = local + 1
    return local


@numba_node
def branch_local_type(change: Signal[int]) -> Signal[int]:
    if change > 0:
        local = change
    else:
        local = change + 1
    return local


@numba_node
def changed_branch_type(change: Signal[int]) -> Signal[float]:
    if change > 0:
        local = change
    else:
        local = 1.5
    return local


@numba_node
def literal_local_type(change: Signal[int]) -> Signal[int]:
    local = 1
    local = 2
    return local + change


@numba_node
def no_container_replacement(change: Signal[int]) -> Signal[int]:
    values: State[NumbaList] = create_new_list(int)
    alias = values
    values = alias
    return change


@numba_node
def mutate_list_alias(change: Signal[int]) -> Signal[int]:
    values: State[NumbaList] = create_new_list(int)
    alias = values
    alias.append(change)
    return values[0]


@numba_node
def deleted_state(change: Signal[int]) -> Signal[int]:
    counter: State[int] = 0
    del counter
    return change


@numba_node
def comprehension_capture(change: Signal[int]) -> Signal[int]:
    counter: State[int] = 0  # noqa: F841 - exercise comprehension name shadowing
    return sum(counter for counter in range(change))


@numba_node
def pattern_capture(change: Signal[int]) -> Signal[int]:
    counter: State[int] = 0
    match change:
        case counter:
            return counter


def test_copy_snapshots_and_assignment_time_persistence():
    node = CompiledNode(compile_function(copied_counter), input_types=[int]).start()
    assert node.execute([3]) == (0, True)
    assert node.execute([4]) == (3, True)
    assert node.execute([0]) == (7, True)
    assert node._state_store[0] == 7


def test_state_assignments_run_in_start_and_stop_phases():
    with setup_standalone_context():
        result = create_compiled_func(
            copied_counter,
            Signal(typ=int),
            extract_python_type_fn=lambda signal: signal.get_type(),
            start_body=ast.parse("counter = 10").body,
            stop_body=ast.parse("counter = 20").body,
        )
    node = CompiledNode(result, input_types=[int]).start()
    assert node._state_store[0] == 10
    assert node.execute([0]) == (10, True)
    node.stop()
    assert node._state_store[0] == 20


def test_augmented_and_other_binding_forms():
    node = CompiledNode(compile_function(current_counter), input_types=[int]).start()
    assert node.execute([3]) == (3, True)
    assert node.execute([4]) == (7, True)

    for func, initial, expected, stored in (
        (chained_counter, 4, 8, 4),
        (unpacked_counter, 4, 9, 4),
        (loop_counter, 4, 3, 3),
        (while_counter, 4, 4, 4),
        (walrus_counter, 4, 4, 4),
    ):
        node = CompiledNode(compile_function(func), input_types=[int]).start()
        assert node.execute([initial]) == (expected, True)
        assert node._state_store[0] == stored

    empty = CompiledNode(compile_function(loop_counter), input_types=[int]).start()
    assert empty.execute([0]) == (0, True)
    assert empty._state_store[0] == 0
    for func, expected, stored in ((chained_raw_value, 5, 4), (walrus_direct_value, 6, 5)):
        node = CompiledNode(compile_function(func), input_types=[int]).start()
        assert node.execute([4]) == (expected, True)
        assert node._state_store[0] == stored


def test_walrus_expression_values_and_branch_control_flow():
    for func, value, expected, stored in (
        (walrus_constant, 4, 9, 5),
        (walrus_call, -4, 8, 4),
        (walrus_twice, 4, 17, 6),
        (walrus_local, 4, 15, 10),
        (walrus_return, 4, 5, 5),
    ):
        node = CompiledNode(compile_function(func), input_types=[int]).start()
        assert node.execute([value]) == (expected, True)
        assert node._state_store[0] == stored

    branch = CompiledNode(compile_function(branch_and_early_return), input_types=[int]).start()
    assert branch.execute([-1]) == (0, True)
    assert branch.execute([3]) == (3, True)
    assert branch.execute([-1]) == (3, True)

    aliases = CompiledNode(compile_function(alias_rebinding), input_types=[int]).start()
    assert aliases.execute([4]) == (4, True)
    assert aliases._state_store[0] == 4

    side_effect = CompiledNode(compile_function(chained_side_effect), input_types=[int]).start()
    assert side_effect.execute([4]) == (110, True)
    assert side_effect._state_store[0] == 5
    side_effect.stop()


def test_copy_float_and_bool_values():
    floats = CompiledNode(compile_function(copied_float), input_types=[float]).start()
    assert floats.execute([1.25]) == (0.0, True)
    assert floats.execute([2.5]) == (1.25, True)
    flags = CompiledNode(compile_function(copied_bool), input_types=[int]).start()
    assert flags.execute([1]) == (False, True)
    assert flags.execute([0]) == (True, True)
    time_result = compile_function(copied_time_values)
    times = CompiledNode(time_result, input_types=[int])
    for index, value in enumerate(time_result.state_values):
        times._state_store[index] = value
    times.start()
    assert times.execute([3]) == (9, True)
    assert times.execute([1]) == (11, True)
    chained = CompiledNode(compile_function(chained_distinct_time_states), input_types=[int]).start()
    assert chained.execute([3]) == (6, True)
    assert tuple(chained._state_store) == (3, 3)


def test_state_and_local_types_are_stable():
    with pytest.raises(TypingError, match="cannot store float64 in State\\[int\\] 'counter'"):
        compile_function(wrong_state_type)
    with pytest.raises(TypingError, match="cannot store bool in State\\[int\\] 'counter'"):
        compile_function(wrong_boolean_state_type)
    with pytest.raises(TypingError, match="Local 'alias' changes type"):
        compile_function(changed_alias_type)
    with pytest.raises(TypingError, match="Local 'local' changes type"):
        compile_function(changed_numeric_type)
    node = CompiledNode(compile_function(same_local_type), input_types=[int]).start()
    assert node.execute([2]) == (3, True)
    branch = CompiledNode(compile_function(branch_local_type), input_types=[int]).start()
    assert branch.execute([2]) == (2, True)
    assert branch.execute([-2]) == (-1, True)
    literal = CompiledNode(compile_function(literal_local_type), input_types=[int]).start()
    assert literal.execute([2]) == (4, True)
    with pytest.raises(TypingError, match="[Ll]ocal.*type|Cannot unify"):
        compile_function(changed_branch_type)


def test_container_replacement_and_deletion_fail_at_compile_time():
    values = CompiledNode(compile_function(mutate_list_alias), input_types=[int]).start()
    assert values.execute([4]) == (4, True)
    assert values.execute([5]) == (4, True)
    values.stop()

    with pytest.raises(TypingError, match="does not support whole-object replacement"):
        compile_function(no_container_replacement)
    with pytest.raises(TypeError, match="Cannot delete State 'counter' at line"):
        compile_function(deleted_state)
    with pytest.raises(TypeError, match="State 'counter' cannot be bound in a comprehension"):
        compile_function(comprehension_capture)
    with pytest.raises(TypeError, match="State 'counter' cannot be a pattern capture"):
        compile_function(pattern_capture)


class Quote(ctypes.Structure):
    _fields_ = [("price", ctypes.c_double), ("count", ctypes.c_int64)]


class OtherQuote(ctypes.Structure):
    _fields_ = [("count", ctypes.c_int64), ("price", ctypes.c_double)]


class QuoteType(StructType):
    @classmethod
    def is_type_supported(cls, var_type):
        return var_type is Quote

    @classmethod
    def _get_struct_fields(cls, var_type):
        return {
            "price": StructFieldInfo("price", Quote.price.offset, "float64", 8),
            "count": StructFieldInfo("count", Quote.count.offset, "int64", 8),
        }

    @classmethod
    def _get_struct_size(cls, var_type):
        return ctypes.sizeof(Quote)

    @classmethod
    def get_type_size(cls, var_type):
        return ctypes.sizeof(Quote)


class OtherQuoteType(QuoteType):
    @classmethod
    def is_type_supported(cls, var_type):
        return var_type is OtherQuote

    @classmethod
    def _get_struct_fields(cls, var_type):
        return {
            "price": StructFieldInfo("price", OtherQuote.price.offset, "float64", 8),
            "count": StructFieldInfo("count", OtherQuote.count.offset, "int64", 8),
        }


@numba_node
def replace_quote(replacement: Signal[Quote]) -> Signal[float]:
    trade: State[Quote] = None
    observed = trade
    alias = trade
    alias.price = 7.0
    trade = replacement
    return observed.price


@numba_node
def mutate_quote_alias(replacement: Signal[Quote]) -> Signal[float]:
    trade: State[Quote] = None
    observed = trade
    alias = trade
    alias.price = 7.0
    return observed.price


@numba_node
def self_replace_quote(replacement: Signal[Quote]) -> Signal[float]:
    trade: State[Quote] = None
    trade = trade  # noqa: PLW0127 - exercise overlapping self-replacement
    return trade.price


@numba_node
def wrong_quote_layout(replacement: Signal[OtherQuote]) -> Signal[float]:
    trade: State[Quote] = None
    trade = replacement
    return trade.price


def _compile_quote(func, type_classes=(QuoteType,)):
    with CompilationContext():
        register_all()
        SourceRegistry.register(_SignalCategory())
        FunctionAnalyzer.register_input_handler(_SignalInputHandler())
        FunctionAnalyzer.register_output_handler(_SingleSignalOutputHandler())
        for type_class in type_classes:
            TypeFactory.register(type_class, priority=0)
        input_type = OtherQuote if func is wrong_quote_layout else Quote
        return create_compiled_func(func, Signal(typ=input_type), extract_python_type_fn=lambda signal: signal.get_type())


def _run_quote(result, state_value, input_value):
    callback = _CFUNC_T(result.compiled_func.address)
    output = ctypes.c_double()
    outputs = (ctypes.c_void_p * 1)(ctypes.addressof(output))
    ticked = (ctypes.c_int8 * 1)()
    state = (ctypes.c_void_p * 1)(ctypes.addressof(state_value))
    inputs = (ctypes.c_void_p * 1)(ctypes.addressof(input_value))
    active = (ctypes.c_int8 * 1)(1)
    callback(outputs, ticked, state, LIFECYCLE_EXECUTE, inputs, active, active)
    return output.value, bool(ticked[0])


def test_borrowed_view_replacement_preserves_alias_cell():
    mutation = _compile_quote(mutate_quote_alias)
    mutated_state = Quote(1.0, 2)
    assert _run_quote(mutation, mutated_state, Quote()) == (7.0, True)
    assert (mutated_state.price, mutated_state.count) == (7.0, 2)

    result = _compile_quote(replace_quote)
    state = Quote(1.0, 2)
    replacement = Quote(9.0, 5)
    assert _run_quote(result, state, replacement) == (9.0, True)
    assert (state.price, state.count) == (9.0, 5)
    assert (replacement.price, replacement.count) == (9.0, 5)


def test_borrowed_view_self_replacement_and_layout_check():
    result = _compile_quote(self_replace_quote)
    state = Quote(2.5, 4)
    assert _run_quote(result, state, Quote()) == (2.5, True)
    assert (state.price, state.count) == (2.5, 4)
    with pytest.raises(TypingError, match="cannot store.*State\\["):
        _compile_quote(wrong_quote_layout, (QuoteType, OtherQuoteType))


def test_borrowed_view_overlapping_cells_and_callback_lifetime():
    result = _compile_quote(replace_quote)
    buffer = ctypes.create_string_buffer(ctypes.sizeof(Quote) + ctypes.sizeof(ctypes.c_int64))
    state = Quote.from_buffer(buffer, 0)
    replacement = Quote.from_buffer(buffer, ctypes.sizeof(ctypes.c_int64))
    state.price, state.count = 1.0, 2
    replacement.price, replacement.count = 9.0, 5
    assert _run_quote(result, state, replacement) == (9.0, True)
    assert (state.price, state.count) == (9.0, 5)

    another_state = Quote(3.0, 1)
    another_replacement = Quote(11.0, 8)
    assert _run_quote(result, another_state, another_replacement) == (11.0, True)
    assert (another_state.price, another_state.count) == (11.0, 8)
    assert (state.price, state.count) == (9.0, 5)


def test_copy_registration_rejects_native_layout_narrowing():
    with pytest.raises(TypeError, match="identical payload and storage types"):
        copy_state_payload(("bad",), types.int64, types.int32, 4, 4)
    with pytest.raises(ValueError, match="does not occupy 4 bytes"):
        copy_state_payload(("bad",), types.int64, types.int64, 4, 4)
