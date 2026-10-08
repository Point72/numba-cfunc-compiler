"""Consolidated integration tests for state."""

import ast
import ctypes
from datetime import datetime, timedelta
from enum import Enum

import pytest
from numba import types
from numba.core.errors import TypingError

from numba_cfunc_compiler.api import NumbaList, State, create_new_list
from numba_cfunc_compiler.core.analysis import FunctionAnalyzer
from numba_cfunc_compiler.core.compile import create_compiled_func
from numba_cfunc_compiler.core.context import CompilationContext
from numba_cfunc_compiler.core.defaults import register_all
from numba_cfunc_compiler.extension.callback_components import ComponentRegistry
from numba_cfunc_compiler.types.base import HostType
from numba_cfunc_compiler.types.builtin.enum.host import register_enum_family
from numba_cfunc_compiler.types.factory import HostTypeFactory
from numba_cfunc_compiler.types.native.state import copy_state_payload
from tests.harness import (
    CFUNC_T,
    LIFECYCLE_EXECUTE,
    CompiledNode,
    Signal,
    SignalComponent,
    SignalInputHandler,
    SingleSignalOutputHandler,
    compile_function,
    numba_node,
    setup_standalone_context,
)
from tests.helpers import CtypesStructHostType


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
def copied_counter_plus_half(change: Signal[int]) -> Signal[float]:
    counter: State[int] = 0
    counter = change
    return counter + 0.5


@numba_node
def copied_counter_equals_half(change: Signal[int]) -> Signal[bool]:
    counter: State[int] = 0
    counter = change
    return counter == 0.5


@numba_node
def chained_counter(change: Signal[int]) -> Signal[int]:
    counter: State[int] = 0
    counter = other = change
    return other + counter


@numba_node
def chained_raw_value(change: Signal[int]) -> Signal[int]:
    counter: State[int] = 0
    counter = other = change  # noqa: F841 - exercise chained state binding
    incremented = other + 1
    return incremented


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
def state_bool_condition(change: Signal[int]) -> Signal[int]:
    initialized: State[bool] = False
    if not initialized:
        initialized = True
        return 1
    return 2


@numba_node
def state_bool_toggle(change: Signal[int]) -> Signal[int]:
    flag: State[bool] = False
    flag = not flag
    return 1 if flag else 2


@numba_node
def choose_datetime(value: Signal[datetime], choose_input: Signal[bool]) -> Signal[datetime]:
    previous: State[datetime] = 0
    if choose_input:
        result = value
    else:
        result = previous
    previous = value
    return result


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
    local = change + 0
    local = local + 1
    return local


@numba_node
def branch_local_type(change: Signal[int]) -> Signal[int]:
    if change > 0:
        local = change + 0
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
def chained_locals(change: Signal[int]) -> Signal[int]:
    first = second = change + 1
    return first + second


@numba_node
def annotated_local(change: Signal[int]) -> Signal[int]:
    local: int = change + 1
    return local


@numba_node
def rebound_input(change: Signal[int]) -> Signal[int]:
    change = change + 1
    return change


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


class Quote(ctypes.Structure):
    _fields_ = [("price", ctypes.c_double), ("count", ctypes.c_int64)]


class OtherQuote(ctypes.Structure):
    _fields_ = [("count", ctypes.c_int64), ("price", ctypes.c_double)]


class QuoteType(CtypesStructHostType):
    storage_type = Quote


class OtherQuoteType(QuoteType):
    storage_type = OtherQuote


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
        ComponentRegistry.register(SignalComponent())
        FunctionAnalyzer.register_input_handler(SignalInputHandler())
        FunctionAnalyzer.register_output_handler(SingleSignalOutputHandler())
        for type_class in type_classes:
            HostTypeFactory.register(type_class, priority=0)
        input_type = OtherQuote if func is wrong_quote_layout else Quote
        return create_compiled_func(func, Signal(typ=input_type), extract_python_type_fn=lambda signal: signal.get_type())


def _run_quote(result, state_value, input_value):
    callback = CFUNC_T(result.compiled_func.address)
    output = ctypes.c_double()
    outputs = (ctypes.c_void_p * 1)(ctypes.addressof(output))
    ticked = (ctypes.c_int8 * 1)()
    state = (ctypes.c_void_p * 1)(ctypes.addressof(state_value))
    inputs = (ctypes.c_void_p * 1)(ctypes.addressof(input_value))
    active = (ctypes.c_int8 * 1)(1)
    callback(outputs, ticked, state, LIFECYCLE_EXECUTE, inputs, active, active)
    return output.value, bool(ticked[0])


class Venue(Enum):
    A = 1
    B = 2


class EnumSet:
    pass


class VenueStorage(HostType):
    @classmethod
    def is_type_supported(cls, var_type):
        return var_type is Venue

    def get_numba_type_name(self):
        return "int16"

    def get_state_payload(self):
        return register_enum_family(Venue, abi_values={"A": 3, "B": 7}).state_payload()

    @classmethod
    def try_parse_state(cls, node, var_name, globalns):
        if getattr(node.annotation.slice, "id", None) != "Venue":
            return None
        member = node.value.attr
        return getattr(Venue, member), Venue


@numba_node
def enum_state(value: Signal[Venue]) -> Signal[bool]:
    held: State[Venue] = Venue.A
    old = held
    held = value
    return old.isin(EnumSet[Venue]([Venue.A])) and held.isin(EnumSet[Venue]([Venue.B]))


def test_copy_snapshots():
    node = CompiledNode(compile_function(copied_counter), input_types=[int]).start()
    assert node.execute([3]) == (0, True)
    assert node.execute([4]) == (3, True)
    assert node.execute([0]) == (7, True)
    assert node._state_store[0] == 7

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

    floats = CompiledNode(compile_function(copied_float), input_types=[float]).start()
    assert floats.execute([1.25]) == (0.0, True)
    assert floats.execute([2.5]) == (1.25, True)
    flags = CompiledNode(compile_function(copied_bool), input_types=[int]).start()
    assert flags.execute([1]) == (False, True)
    assert flags.execute([0]) == (True, True)
    conditions = CompiledNode(compile_function(state_bool_condition), input_types=[int]).start()
    assert conditions.execute([0]) == (1, True)
    assert conditions.execute([0]) == (2, True)
    toggled = CompiledNode(compile_function(state_bool_toggle), input_types=[int]).start()
    assert toggled.execute([0]) == (1, True)
    assert toggled.execute([0]) == (2, True)
    chosen = CompiledNode(compile_function(choose_datetime), input_types=[int, bool]).start()
    assert chosen.execute([10, True]) == (10, True)
    assert chosen.execute([20, False]) == (10, True)
    time_result = compile_function(copied_time_values)
    times = CompiledNode(time_result, input_types=[int])
    for index, value in enumerate(time_result.state_values):
        times._state_store[index] = value
    times.start()
    assert times.execute([3]) == (9, True)
    assert times.execute([1]) == (11, True)


def test_copy_state_mixed_numeric_operations():
    arithmetic = CompiledNode(compile_function(copied_counter_plus_half), input_types=[int]).start()
    assert arithmetic.execute([3]) == (3.5, True)

    comparison = CompiledNode(compile_function(copied_counter_equals_half), input_types=[int]).start()
    assert comparison.execute([0]) == (False, True)


def test_control_flow():
    node = CompiledNode(compile_function(current_counter), input_types=[int]).start()
    assert node.execute([3]) == (3, True)
    assert node.execute([4]) == (7, True)

    looping = CompiledNode(compile_function(while_counter), input_types=[int]).start()
    assert looping.execute([4]) == (4, True)
    assert looping._state_store[0] == 4

    branch = CompiledNode(compile_function(branch_and_early_return), input_types=[int]).start()
    assert branch.execute([-1]) == (0, True)
    assert branch.execute([3]) == (3, True)
    assert branch.execute([-1]) == (3, True)

    aliases = CompiledNode(compile_function(alias_rebinding), input_types=[int]).start()
    assert aliases.execute([4]) == (4, True)
    assert aliases._state_store[0] == 4


def test_unsupported_state_binding_forms():
    for func in (chained_counter, chained_raw_value, chained_side_effect, chained_distinct_time_states, unpacked_counter):
        with pytest.raises(TypeError, match="requires a single-name assignment"):
            compile_function(func)
    with pytest.raises(TypeError, match="requires a single-name assignment"):
        compile_function(loop_counter)


@pytest.mark.parametrize(
    ("source", "error"),
    [
        ("with holder as counter:\n    pass", "requires a single-name assignment"),
        ("try:\n    pass\nexcept Exception as counter:\n    pass", "cannot be an exception target"),
        ("global counter", "cannot be declared global or nonlocal"),
        ("counter: int = 1", "requires a single-name assignment"),
        ("callback = lambda value: value", "Lambda expressions in nodes are unsupported"),
    ],
)
def test_unsupported_lifecycle_bindings(source, error):
    with setup_standalone_context(), pytest.raises(TypeError, match=error):
        create_compiled_func(
            current_counter,
            Signal(typ=int),
            extract_python_type_fn=lambda signal: signal.get_type(),
            start_body=ast.parse(source).body,
        )


def test_borrowed_aliases():
    mutation = _compile_quote(mutate_quote_alias)
    assert mutation.struct_state_indices == (0,)
    assert mutation.struct_state_sizes == (ctypes.sizeof(Quote),)
    mutated_state = Quote(1.0, 2)
    assert _run_quote(mutation, mutated_state, Quote()) == (7.0, True)
    assert (mutated_state.price, mutated_state.count) == (7.0, 2)

    result = _compile_quote(replace_quote)
    state = Quote(1.0, 2)
    replacement = Quote(9.0, 5)
    assert _run_quote(result, state, replacement) == (9.0, True)
    assert (state.price, state.count) == (9.0, 5)
    assert (replacement.price, replacement.count) == (9.0, 5)

    result = _compile_quote(self_replace_quote)
    state = Quote(2.5, 4)
    assert _run_quote(result, state, Quote()) == (2.5, True)
    assert (state.price, state.count) == (2.5, 4)
    with pytest.raises(TypingError, match="cannot store.*State\\["):
        _compile_quote(wrong_quote_layout, (QuoteType, OtherQuoteType))

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


def test_enum_snapshots():
    register_enum_family(Venue, abi_values={"A": 3, "B": 7})
    with CompilationContext():
        register_all()
        HostTypeFactory.register(VenueStorage, priority=0)
        ComponentRegistry.register(SignalComponent())
        FunctionAnalyzer.register_input_handler(SignalInputHandler())
        FunctionAnalyzer.register_output_handler(SingleSignalOutputHandler())
        result = create_compiled_func(enum_state, Signal(typ=Venue), extract_python_type_fn=lambda signal: signal.get_type())

    callback_type = ctypes.CFUNCTYPE(
        None,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_int8),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int8,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_int8),
        ctypes.POINTER(ctypes.c_int8),
    )
    callback = callback_type(result.compiled_func.address)
    input_value = ctypes.c_int16(7)
    state_value = ctypes.c_int16(3)
    output_value = ctypes.c_int8()
    inputs = (ctypes.c_void_p * 1)(ctypes.addressof(input_value))
    outputs = (ctypes.c_void_p * 1)(ctypes.addressof(output_value))
    state = (ctypes.c_void_p * 1)(ctypes.addressof(state_value))
    output_ticked = (ctypes.c_int8 * 1)()
    flags = (ctypes.c_int8 * 1)(1)
    callback(outputs, output_ticked, state, 0, inputs, flags, flags)
    assert (state_value.value, output_value.value, output_ticked[0]) == (7, 1, 1)


def test_container_lifecycle():
    result = compile_function(mutate_list_alias)
    assert result.nrt_state_indices == (0,)
    values = CompiledNode(result, input_types=[int]).start()
    assert values.execute([4]) == (4, True)
    assert values.execute([5]) == (4, True)
    values.stop()

    with pytest.raises(TypingError, match="does not support whole-object replacement"):
        compile_function(no_container_replacement)
    with pytest.raises(TypeError, match="Cannot delete State 'counter' at line"):
        compile_function(deleted_state)
    with pytest.raises(TypeError, match="State 'counter' requires a single-name assignment"):
        compile_function(comprehension_capture)
    with pytest.raises(TypeError, match="State 'counter' cannot be a pattern capture"):
        compile_function(pattern_capture)


def test_invalid_assignments():
    with pytest.raises(TypingError, match="cannot store float64 in State\\[int\\] 'counter'"):
        compile_function(wrong_state_type)
    with pytest.raises(TypingError, match="cannot store bool in State\\[int\\] 'counter'"):
        compile_function(wrong_boolean_state_type)
    alias = CompiledNode(compile_function(changed_alias_type), input_types=[int]).start()
    assert alias.execute([2]) == (2, True)
    assert alias.execute([3]) == (3, True)
    assert alias._state_store[0] == 0
    numeric = CompiledNode(compile_function(changed_numeric_type), input_types=[int]).start()
    assert numeric.execute([2]) == (1.5, True)
    node = CompiledNode(compile_function(same_local_type), input_types=[int]).start()
    assert node.execute([2]) == (3, True)
    branch = CompiledNode(compile_function(branch_local_type), input_types=[int]).start()
    assert branch.execute([2]) == (2, True)
    assert branch.execute([-2]) == (-1, True)
    literal = CompiledNode(compile_function(literal_local_type), input_types=[int]).start()
    assert literal.execute([2]) == (4, True)
    assert CompiledNode(compile_function(chained_locals), input_types=[int]).execute([2]) == (6, True)
    assert CompiledNode(compile_function(annotated_local), input_types=[int]).execute([2]) == (3, True)
    assert CompiledNode(compile_function(rebound_input), input_types=[int]).execute([2]) == (3, True)
    with pytest.raises(TypingError, match="[Ll]ocal.*type|Cannot unify"):
        compile_function(changed_branch_type)

    with pytest.raises(TypeError, match="identical payload and storage types"):
        copy_state_payload(("bad",), types.int64, types.int32, 4, 4)
    with pytest.raises(ValueError, match="does not occupy 4 bytes"):
        copy_state_payload(("bad",), types.int64, types.int64, 4, 4)
