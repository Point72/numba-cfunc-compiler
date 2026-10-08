"""Compiled list and dict operations across the native callback boundary."""

import ast
import inspect
import textwrap

import pytest
from numba import njit, types
from numba.core.errors import TypingError
from numba.extending import register_jitable

from numba_cfunc_compiler.api import NumbaDict, NumbaList, State, create_new_dict, create_new_list
from numba_cfunc_compiler.core.compile import create_compiled_func
from numba_cfunc_compiler.extension.input_sources import register_input_source
from numba_cfunc_compiler.types.builtin.dict.native import standalone_dict_free, standalone_dict_insert, standalone_dict_to_voidptr
from numba_cfunc_compiler.types.builtin.list.native import standalone_list_append, standalone_list_free, standalone_list_to_voidptr
from numba_cfunc_compiler.types.factory import HostTypeFactory
from numba_cfunc_compiler.types.markers import DictTypeMarker, ListTypeMarker
from numba_cfunc_compiler.types.native.input import bind_input, input_payload
from tests.harness import CompiledNode, Signal, compile_function, numba_node, setup_standalone_context


def _compile_body(template, body, *args, call_globals=None):
    """Compile a short source fragment against an existing node signature."""
    signature = inspect.signature(template)
    parameters = ", ".join(signature.parameters)
    tree = ast.parse(f"def case({parameters}):\n{textwrap.indent(body, '    ')}\n    return x\n").body[0]
    with setup_standalone_context():
        return create_compiled_func(
            tree,
            *args,
            signature=signature,
            func_globals=globals(),
            extract_python_type_fn=lambda signal: signal.get_type(),
            call_globals=call_globals,
        )


@numba_node
def dict_contains(x: Signal[int]) -> Signal[int]:
    # Mirrors csp's test_dict_contains, the case that segfaulted on Windows.
    seen: State[NumbaDict] = create_new_dict(int, int)
    found = 1 if x in seen else 0
    seen[x] = x * 10
    return found


@numba_node
def list_append_len(x: Signal[int]) -> Signal[int]:
    xs: State[NumbaList] = create_new_list(int)
    xs.append(x)
    return len(xs)


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
def dict_constant_items(x: Signal[int], values: NumbaDict[int, int]) -> Signal[int]:
    total = x
    for key, value in values.items():
        total += key + value
    return total


@numba_node
def float_list_constant_total(x: Signal[float], values: NumbaList[float]) -> Signal[float]:
    total = x
    for value in values:
        total += value
    return total


@register_jitable
def _append_in_helper(values, value):
    values.append(value)


@numba_node
def list_constant_with_state(x: Signal[int], values: NumbaList[int]) -> Signal[int]:
    total: State[int] = 0
    total += values[0] + x
    return total


@numba_node
def two_constant_containers(x: Signal[int], items: NumbaList[int], mapping: NumbaDict[int, int]) -> Signal[int]:
    return items[0] + mapping.get(x, 0)


def test_container_operations():
    seen = CompiledNode(compile_function(dict_contains), [int]).start()
    assert [seen.execute([value])[0] for value in (1, 2, 1, 3, 2)] == [0, 0, 1, 0, 1]
    dict_aliases = CompiledNode(compile_function(dict_alias_iteration), [int]).start()
    assert [dict_aliases.execute([value])[0] for value in (2, 3)] == [26, 65]
    dict_mutation_node = CompiledNode(compile_function(dict_mutations), [int]).start()
    assert [dict_mutation_node.execute([value])[0] for value in (2, 2, 3)] == [20, 42, 29]

    lengths = CompiledNode(compile_function(list_append_len), [int]).start()
    assert [lengths.execute([value])[0] for value in (10, 20, 30)] == [1, 2, 3]
    list_mutation_node = CompiledNode(compile_function(list_mutations), [int]).start()
    assert [list_mutation_node.execute([value])[0] for value in (2, 3, 4, 1)] == [5, 10, 15, 3]

    for node in (seen, dict_aliases, dict_mutation_node, lengths, list_mutation_node):
        node.stop()
        assert node._state[0] is None


@pytest.mark.parametrize(
    ("body", "constructor", "state_type"),
    [
        ("values = create_new_list(int)", "create_new_list", "NumbaList"),
        ("first = second = create_new_dict(int, int)", "create_new_dict", "NumbaDict"),
        ("if x > 0:\n    values = create_new_list(int)", "create_new_list", "NumbaList"),
        ("return x + len(create_new_dict(int, int))", "create_new_dict", "NumbaDict"),
        ("values: State[NumbaList] = create_new_dict(int, int)", "create_new_dict", "NumbaDict"),
    ],
)
def test_container_constructors_require_matching_state(body, constructor, state_type):
    with pytest.raises(TypeError, match=rf"{constructor}\(\).*State\[{state_type}\].*line \d+"):
        _compile_body(list_append_len, body, Signal(typ=int))


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
        result = compile_function(list_constant_total, values=values)
        assert result.state_values == (0,)
        assert result.constant_container_indices == (0,)
        node = CompiledNode(result, [int]).start()
        pointer = node._state[0]
        assert pointer
        assert node.execute([3]) == expected
        assert node.execute([4])[1]
        assert node._state[0] == pointer
        node.stop()
        assert node._state[0] is None

    for values, expected in (({3: 30, 5: 50}, (32, True)), ({}, (-1, True))):
        result = compile_function(dict_constant_lookup, values=values)
        assert result.state_values == (0,)
        assert result.constant_container_indices == (0,)
        node = CompiledNode(result, [int]).start()
        pointer = node._state[0]
        assert pointer
        assert node.execute([3]) == expected
        if values:
            assert node.execute([4]) == (1, True)
        assert node._state[0] == pointer
        node.stop()
        assert node._state[0] is None

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

    dict_node = CompiledNode(compile_function(dict_constant_items, values={1: 10, 2: 20}), [int]).start()
    assert dict_node.execute([3]) == (36, True)
    dict_node.stop()
    float_node = CompiledNode(compile_function(float_list_constant_total, values=[0.5, 1.5]), [float]).start()
    assert float_node.execute([2.0]) == (4.0, True)
    float_node.stop()


@pytest.mark.parametrize(
    ("template", "body", "values"),
    [
        (list_constant_total, "alias = values\nalias.append(x)", [1]),
        (list_constant_total, "values.pop()", [1]),
        (list_constant_total, "values.clear()", [1]),
        (list_constant_total, "_append_in_helper(values, x)", [1]),
        (list_constant_total, "values[0] = x", [1]),
        (list_constant_total, "standalone_list_append(values, x)", [1]),
        (list_constant_total, "standalone_list_to_voidptr(values)", [1]),
        (list_constant_total, "standalone_list_free(values)", [1]),
        (dict_constant_lookup, "values[x] = x", {1: 2}),
        (dict_constant_lookup, "values.pop(x)", {1: 2}),
        (dict_constant_lookup, "values.clear()", {1: 2}),
        (dict_constant_lookup, "standalone_dict_insert(values, x, x)", {1: 2}),
        (dict_constant_lookup, "standalone_dict_to_voidptr(values)", {1: 2}),
        (dict_constant_lookup, "standalone_dict_free(values)", {1: 2}),
    ],
)
def test_constant_containers_reject_writes(template, body, values):
    call_globals = {
        "_append_in_helper": _append_in_helper,
        "standalone_list_append": standalone_list_append,
        "standalone_list_to_voidptr": standalone_list_to_voidptr,
        "standalone_list_free": standalone_list_free,
        "standalone_dict_insert": standalone_dict_insert,
        "standalone_dict_to_voidptr": standalone_dict_to_voidptr,
        "standalone_dict_free": standalone_dict_free,
    }
    with pytest.raises(TypingError):
        _compile_body(template, body, Signal(typ=int), values, call_globals=call_globals)


def test_constant_container_lifecycle_and_storage():
    result = compile_function(
        list_constant_with_state,
        values=[2, 3],
        start_body=ast.parse("total = values[0]").body,
        stop_body=ast.parse("total = values[1]").body,
    )
    assert result.state_values == (0, 0)
    assert result.constant_container_indices == (1,)
    left = CompiledNode(result, [int]).start()
    right = CompiledNode(result, [int]).start()
    assert left._state_store[0] == right._state_store[0] == 2
    assert left._state[1] != right._state[1]
    assert left.execute([3]) == (7, True)
    assert left.execute([4]) == (13, True)
    assert right.execute([1]) == (5, True)
    left.stop()
    right.stop()
    assert left._state_store[0] == right._state_store[0] == 3
    assert left._state[1] is right._state[1] is None

    result = compile_function(two_constant_containers, items=[2], mapping={3: 4})
    assert result.state_values == (0, 0)
    assert result.constant_container_indices == (0, 1)
    node = CompiledNode(result, [int]).start()
    assert node._state[0] and node._state[1]
    assert node._state[0] != node._state[1]
    assert node.execute([3]) == (6, True)
    node.stop()
    assert node._state[0] is node._state[1] is None


@pytest.mark.parametrize(("marker", "kind"), [(ListTypeMarker(int), "list"), (DictTypeMarker(int, int), "dict")])
def test_container_input_boundaries_are_readonly(marker, kind):
    with setup_standalone_context():
        binding = HostTypeFactory.resolve(marker)
        source = register_input_source(identity=("tests", f"readonly.{kind}"), fields=())
        source_type = source.value_type(binding)
        assert source_type.payload_type.readonly
        assert source_type.storage_type.readonly
        assert not binding.payload.native_type.readonly
        assert source_type.unify(None, binding.payload.native_type) is None

        bind = bind_input(source_type)

        def read(ptr):
            return len(input_payload(bind(ptr, ())))

        njit(read).compile((types.voidptr,))

        if kind == "list":

            def write(ptr):
                input_payload(bind(ptr, ())).append(1)

        else:

            def write(ptr):
                input_payload(bind(ptr, ()))[1] = 2

        with pytest.raises(TypingError):
            njit(write).compile((types.voidptr,))

    if kind == "list":
        with pytest.raises(TypingError):
            _compile_body(
                list_constant_total,
                "state_values: State[NumbaList] = create_new_list(int)\nselected = state_values if x > 0 else values\nselected.append(x)",
                Signal(typ=int),
                [1],
            )
