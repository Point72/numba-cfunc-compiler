"""Compiled list and dict operations across the native callback boundary."""

import ast

import pytest
from numba.core.errors import TypingError

from numba_cfunc_compiler.api import NumbaDict, NumbaList, State, create_new_dict, create_new_list
from numba_cfunc_compiler.core.compile import create_compiled_func
from tests.harness import CompiledNode, Signal, compile_function, numba_node, setup_standalone_context


@numba_node
def dict_contains(x: Signal[int]) -> Signal[int]:
    # Mirrors csp's test_dict_contains, the case that segfaulted on Windows.
    seen: State[NumbaDict] = create_new_dict(int, int)
    found = 1 if x in seen else 0
    seen[x] = x * 10
    return found


@numba_node
def dict_float_accumulate(key: Signal[int], val: Signal[float]) -> Signal[float]:
    sums: State[NumbaDict] = create_new_dict(int, float)
    sums[key] = sums.get(key, 0.0) + val
    return sums[key]


@numba_node
def list_append_len(x: Signal[int]) -> Signal[int]:
    xs: State[NumbaList] = create_new_list(int)
    xs.append(x)
    return len(xs)


@numba_node
def list_pop_value(x: Signal[int]) -> Signal[int]:
    xs: State[NumbaList] = create_new_list(int)
    xs.append(x)
    removed = xs.pop()
    return removed + 1


@numba_node
def list_alias_iteration(x: Signal[int]) -> Signal[int]:
    xs: State[NumbaList] = create_new_list(int)
    xs.append(x)
    alias = xs
    total = 0
    for value in alias:
        total += value
    return total


@numba_node
def dict_alias_iteration(x: Signal[int]) -> Signal[int]:
    values: State[NumbaDict] = create_new_dict(int, int)
    values[x] = x * 10
    alias = values
    total = 0
    for key, value in alias.items():
        total += key + value
    for key in alias.keys():  # noqa: SIM118 - explicitly exercise the keys() iterator
        total += key
    for key in alias:
        total += key
    return total


@numba_node
def dict_mutations(x: Signal[int]) -> Signal[int]:
    values: State[NumbaDict] = create_new_dict(int, int)
    previous = values.get(x, -1)
    values[x] = x * 10
    removed = values.pop(x)
    values[x] = removed + 1
    if len(values) > 1:
        values.clear()
    return previous + removed + len(values)


@numba_node
def list_mutations(x: Signal[int]) -> Signal[int]:
    values: State[NumbaList] = create_new_list(int)
    values.append(x)
    if len(values) > 1:
        values[0] = values[0] + x
    removed = values.pop(0)
    values.append(removed)
    result = values[0] + values[-1] + len(values)
    if len(values) > 2:
        values.clear()
    return result


@numba_node
def local_list(x: Signal[int]) -> Signal[int]:
    values = create_new_list(int)
    return len(values) + x


@numba_node
def local_dict(x: Signal[int]) -> Signal[int]:
    mapping = create_new_dict(int, int)
    return len(mapping) + x


@numba_node
def annotated_local_list(x: Signal[int]) -> Signal[int]:
    values: NumbaList[int] = create_new_list(int)
    return len(values) + x


@numba_node
def chained_local_dict(x: Signal[int]) -> Signal[int]:
    first = second = create_new_dict(int, int)
    return len(first) + len(second) + x


@numba_node
def conditional_local_list(x: Signal[int]) -> Signal[int]:
    if x > 0:
        values = create_new_list(int)
        return len(values)
    return x


@numba_node
def expression_local_dict(x: Signal[int]) -> Signal[int]:
    return x + len(create_new_dict(int, int))


@numba_node
def mismatched_state_constructor(x: Signal[int]) -> Signal[int]:
    values: State[NumbaList] = create_new_dict(int, int)
    return len(values) + x


@numba_node
def list_constant_total(x: Signal[int], values: NumbaList[int]) -> Signal[int]:
    alias = values
    total = x + len(alias)
    for value in alias:
        total += value
    return total


@numba_node
def dict_constant_lookup(x: Signal[int], values: NumbaDict[int, int]) -> Signal[int]:
    alias = values
    return alias.get(x, -1) + len(alias)


@numba_node
def invalid_rebound_list_method(x: Signal[int]) -> Signal[int]:
    xs: State[NumbaList] = create_new_list(int)
    alias = xs
    alias = 1
    alias.append(x)
    return x


def test_container_operations():
    seen = CompiledNode(compile_function(dict_contains), [int]).start()
    assert [seen.execute([value])[0] for value in (1, 2, 1, 3, 2)] == [0, 0, 1, 0, 1]
    dict_aliases = CompiledNode(compile_function(dict_alias_iteration), [int]).start()
    assert [dict_aliases.execute([value])[0] for value in (2, 3)] == [26, 65]
    sums = CompiledNode(compile_function(dict_float_accumulate), [int, float]).start()
    assert [sums.execute([key, value])[0] for key, value in ((1, 10.0), (2, 20.0), (1, 30.0))] == [10.0, 20.0, 40.0]
    dict_mutation_node = CompiledNode(compile_function(dict_mutations), [int]).start()
    assert [dict_mutation_node.execute([value])[0] for value in (2, 2, 3)] == [20, 42, 29]

    lengths = CompiledNode(compile_function(list_append_len), [int]).start()
    assert [lengths.execute([value])[0] for value in (10, 20, 30)] == [1, 2, 3]
    list_aliases = CompiledNode(compile_function(list_alias_iteration), [int]).start()
    assert [list_aliases.execute([value])[0] for value in (2, 3)] == [2, 5]
    popped = CompiledNode(compile_function(list_pop_value), [int]).start()
    assert popped.execute([10]) == (11, True)
    list_mutation_node = CompiledNode(compile_function(list_mutations), [int]).start()
    assert [list_mutation_node.execute([value])[0] for value in (2, 3, 4, 1)] == [5, 10, 15, 3]
    with pytest.raises(TypingError):
        compile_function(invalid_rebound_list_method)

    for node in (seen, dict_aliases, sums, dict_mutation_node, lengths, list_aliases, popped, list_mutation_node):
        node.stop()
        assert node._state[0] is None


@pytest.mark.parametrize(
    ("node", "constructor", "state_type"),
    [
        (local_list, "create_new_list", "NumbaList"),
        (local_dict, "create_new_dict", "NumbaDict"),
        (annotated_local_list, "create_new_list", "NumbaList"),
        (chained_local_dict, "create_new_dict", "NumbaDict"),
        (conditional_local_list, "create_new_list", "NumbaList"),
        (expression_local_dict, "create_new_dict", "NumbaDict"),
        (mismatched_state_constructor, "create_new_dict", "NumbaDict"),
    ],
)
def test_container_constructors_require_matching_state(node, constructor, state_type):
    with pytest.raises(TypeError, match=rf"{constructor}\(\).*State\[{state_type}\].*line \d+"):
        compile_function(node)


@pytest.mark.parametrize("body_name", ["start_body", "stop_body"])
@pytest.mark.parametrize(
    ("constructor", "arguments", "state_type"), [("create_new_list", "int", "NumbaList"), ("create_new_dict", "int, int", "NumbaDict")]
)
def test_container_constructors_are_rejected_in_lifecycle_bodies(body_name, constructor, arguments, state_type):
    with (
        setup_standalone_context(),
        pytest.raises(TypeError, match=rf"{constructor}\(\).*State\[{state_type}\].*line \d+"),
    ):
        create_compiled_func(
            list_append_len,
            Signal(typ=int),
            extract_python_type_fn=lambda signal: signal.get_type(),
            **{body_name: ast.parse(f"local = {constructor}({arguments})").body},
        )


def test_constant_container_inputs():
    for values, expected in (([2, 3], (10, True)), ((2, 3), (10, True)), ([], (3, True))):
        node = CompiledNode(compile_function(list_constant_total, values=values), [int])
        assert node.execute([3]) == expected

    for values, expected in (({3: 30, 5: 50}, (32, True)), ({}, (-1, True))):
        node = CompiledNode(compile_function(dict_constant_lookup, values=values), [int])
        assert node.execute([3]) == expected
        if values:
            assert node.execute([4]) == (1, True)

    with pytest.raises(TypeError, match="values.*expected list or tuple"):
        compile_function(list_constant_total, values=1)
    with pytest.raises(TypeError, match="values.*element 1.*expected int"):
        compile_function(list_constant_total, values=[1, "wrong"])
    with pytest.raises(TypeError, match="values.*expected dict"):
        compile_function(dict_constant_lookup, values=[(1, 2)])
    with pytest.raises(TypeError, match="values.*key.*expected int"):
        compile_function(dict_constant_lookup, values={"wrong": 2})
    with pytest.raises(TypeError, match="values.*value.*expected int"):
        compile_function(dict_constant_lookup, values={1: "wrong"})
