"""Execution tests for the standalone NumbaDict / NumbaList runtime.

Unlike ``test_support_units.py`` (which only checks AST lowering / type
parsing) and ``test_compilation.py`` (which needs gcc and only covers scalar
ops), these tests *compile and actually run* container operations through the
JIT -> C FFI boundary using the pure-Python ctypes :class:`CompiledNode`
driver. That makes them platform-agnostic, so they run on Windows CI and would
catch runtime/ABI regressions in the dict/list C runtime (e.g. unexported
symbols, or ``Py_ssize_t``/``Py_hash_t`` width mismatches) *here* rather than
downstream in consumers like csp.
"""

import unittest

from numba.core.errors import TypingError

from numba_cfunc_compiler.node_api import NumbaDict, NumbaList, State, create_new_dict, create_new_list
from numba_cfunc_compiler.tests.harness import CompiledNode, Signal, compile_function, numba_node

# ---- Dict nodes -----------------------------------------------------------


@numba_node
def dict_set_get(key: Signal[int], val: Signal[int]) -> Signal[int]:
    d: State[NumbaDict] = create_new_dict(int, int)
    d[key] = val
    return d[key]


@numba_node
def dict_contains(x: Signal[int]) -> Signal[int]:
    # Mirrors csp's test_dict_contains, the case that segfaulted on Windows.
    seen: State[NumbaDict] = create_new_dict(int, int)
    found = 1 if x in seen else 0
    seen[x] = x * 10
    return found


@numba_node
def dict_len(key: Signal[int], val: Signal[int]) -> Signal[int]:
    d: State[NumbaDict] = create_new_dict(int, int)
    d[key] = val
    return len(d)


@numba_node
def dict_get_default(key: Signal[int], present: Signal[int]) -> Signal[int]:
    d: State[NumbaDict] = create_new_dict(int, int)
    d[present] = present * 100
    return d.get(key, -1)


@numba_node
def dict_float_accumulate(key: Signal[int], val: Signal[float]) -> Signal[float]:
    sums: State[NumbaDict] = create_new_dict(int, float)
    sums[key] = sums.get(key, 0.0) + val
    return sums[key]


@numba_node
def dict_pop_value(key: Signal[int], val: Signal[int]) -> Signal[int]:
    values: State[NumbaDict] = create_new_dict(int, int)
    values[key] = val
    removed = values.pop(key)
    return removed + 1


# ---- List nodes -----------------------------------------------------------


@numba_node
def list_append_len(x: Signal[int]) -> Signal[int]:
    xs: State[NumbaList] = create_new_list(int)
    xs.append(x)
    return len(xs)


@numba_node
def list_append_last(x: Signal[int]) -> Signal[int]:
    xs: State[NumbaList] = create_new_list(int)
    xs.append(x)
    return xs[len(xs) - 1]


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
def dict_float_items(x: Signal[int], value: Signal[float]) -> Signal[float]:
    values: State[NumbaDict] = create_new_dict(int, float)
    values[x] = value
    total = 0.0
    for _, item in values.items():  # noqa: PERF102 - exercise item value loading
        total += item
    return total


@numba_node
def local_containers_are_typed(x: Signal[int]) -> Signal[int]:
    values = create_new_list(int)
    mapping = create_new_dict(int, int)
    values.append(x)
    mapping[x] = x + 1
    total = 0
    for value in values:
        total += value
    for key, value in mapping.items():
        total += key + value
    return total


@numba_node
def invalid_rebound_list_method(x: Signal[int]) -> Signal[int]:
    xs: State[NumbaList] = create_new_list(int)
    alias = xs
    alias = 1
    alias.append(x)
    return x


class TestDictExecution(unittest.TestCase):
    def test_float_item_iterator(self):
        node = CompiledNode(compile_function(dict_float_items), input_types=[int, float]).start()
        self.assertEqual(node.execute([2, 1.25])[0], 1.25)
        self.assertEqual(node.execute([3, 2.5])[0], 3.75)

    def test_local_constructor_results_are_typed_by_numba(self):
        node = CompiledNode(compile_function(local_containers_are_typed), input_types=[int]).start()
        self.assertEqual(node.execute([4])[0], 13)

    def test_alias_iteration_is_typed_by_numba(self):
        node = CompiledNode(compile_function(dict_alias_iteration), input_types=[int]).start()
        self.assertEqual(node.execute([2])[0], 26)
        self.assertEqual(node.execute([3])[0], 65)

    def test_set_get_int(self):
        node = CompiledNode(compile_function(dict_set_get), input_types=[int, int]).start()
        self.assertEqual(node.execute([1, 100])[0], 100)
        self.assertEqual(node.execute([2, 200])[0], 200)
        # overwrite existing key
        self.assertEqual(node.execute([1, 111])[0], 111)

    def test_contains_sequence(self):
        # inputs 1,2,1,3,2 -> contains 0,0,1,0,1 (state persists across ticks)
        node = CompiledNode(compile_function(dict_contains), input_types=[int]).start()
        got = [node.execute([x])[0] for x in (1, 2, 1, 3, 2)]
        self.assertEqual(got, [0, 0, 1, 0, 1])

    def test_len_grows(self):
        node = CompiledNode(compile_function(dict_len), input_types=[int, int]).start()
        self.assertEqual(node.execute([1, 10])[0], 1)
        self.assertEqual(node.execute([2, 20])[0], 2)
        self.assertEqual(node.execute([1, 30])[0], 2)  # existing key, no growth

    def test_get_default(self):
        node = CompiledNode(compile_function(dict_get_default), input_types=[int, int]).start()
        # present key stored, then look up a missing key -> default
        self.assertEqual(node.execute([999, 5])[0], -1)  # 999 missing
        self.assertEqual(node.execute([5, 5])[0], 500)  # 5 was stored (5*100)

    def test_float_values_accumulate(self):
        node = CompiledNode(compile_function(dict_float_accumulate), input_types=[int, float]).start()
        keys = [1, 2, 1, 2, 1]
        vals = [10.0, 20.0, 30.0, 40.0, 50.0]
        got = [node.execute([k, v])[0] for k, v in zip(keys, vals)]
        self.assertEqual(got, [10.0, 20.0, 40.0, 60.0, 90.0])

    def test_pop_result_is_typed_by_numba(self):
        node = CompiledNode(compile_function(dict_pop_value), input_types=[int, int]).start()
        self.assertEqual(node.execute([3, 20]), (21, True))
        self.assertEqual(node.execute([4, 40]), (41, True))

    def test_stop_frees_state(self):
        node = CompiledNode(compile_function(dict_contains), input_types=[int]).start()
        node.execute([1])
        node.stop()
        self.assertIsNone(node._state[0])


class TestListExecution(unittest.TestCase):
    def test_alias_iteration_is_typed_by_numba(self):
        node = CompiledNode(compile_function(list_alias_iteration), input_types=[int]).start()
        self.assertEqual(node.execute([2])[0], 2)
        self.assertEqual(node.execute([3])[0], 5)

    def test_append_len(self):
        node = CompiledNode(compile_function(list_append_len), input_types=[int]).start()
        self.assertEqual(node.execute([10])[0], 1)
        self.assertEqual(node.execute([20])[0], 2)
        self.assertEqual(node.execute([30])[0], 3)

    def test_append_getitem(self):
        node = CompiledNode(compile_function(list_append_last), input_types=[int]).start()
        self.assertEqual(node.execute([10])[0], 10)
        self.assertEqual(node.execute([20])[0], 20)

    def test_pop_result_is_typed_by_numba(self):
        node = CompiledNode(compile_function(list_pop_value), input_types=[int]).start()
        self.assertEqual(node.execute([10]), (11, True))
        self.assertEqual(node.execute([20]), (21, True))

    def test_rebound_alias_method_uses_numba_type(self):
        with self.assertRaises(TypingError):
            compile_function(invalid_rebound_list_method)

    def test_stop_frees_state(self):
        node = CompiledNode(compile_function(list_append_len), input_types=[int]).start()
        node.execute([10])
        node.stop()
        self.assertIsNone(node._state[0])


if __name__ == "__main__":
    unittest.main()
