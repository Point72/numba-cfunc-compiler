"""Execution tests for standalone dictionaries, lists, and fixed arrays.

Unlike ``test_support_units.py`` (which only checks AST lowering / type
parsing) and ``test_compilation.py`` (which needs gcc and only covers scalar
ops), these tests *compile and actually run* container operations through the
JIT -> C FFI boundary using the pure-Python ctypes :class:`CompiledNode`
driver. That makes them platform-agnostic, so they run on Windows CI and would
catch runtime/ABI regressions in the dict/list C runtime (e.g. unexported
symbols, or ``Py_ssize_t``/``Py_hash_t`` width mismatches) *here* rather than
downstream in consumers like csp.
"""

import re
import unittest

from numba_cfunc_compiler.node_api import NumbaArray, NumbaDict, NumbaList, State, create_new_array, create_new_dict, create_new_list
from numba_cfunc_compiler.numba_methods import numba_method
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


@numba_method
def append_and_get_length(values, value):
    values.append(value)
    return len(values)


@numba_node
def list_inline_append_len(x: Signal[int]) -> Signal[int]:
    xs: State[NumbaList] = create_new_list(int)
    return append_and_get_length(xs, x)


@numba_node
def bool_list_append_set(x: Signal[bool]) -> Signal[bool]:
    xs: State[NumbaList] = create_new_list(bool)
    if len(xs) == 0:
        xs.append(False)
    xs[0] = x
    return xs[0]


@numba_node
def list_positive_constant_get_set(x: Signal[int]) -> Signal[int]:
    xs: State[NumbaList] = create_new_list(int)
    xs.append(x)
    xs[0] = x + 1
    return xs[0]


@numba_node
def list_negative_constant_get_set(x: Signal[int]) -> Signal[int]:
    xs: State[NumbaList] = create_new_list(int)
    xs.append(0)
    xs[-1] = x
    return xs[-1]


@numba_node
def list_negative_constant_pop(x: Signal[int]) -> Signal[int]:
    xs: State[NumbaList] = create_new_list(int)
    xs.append(x)
    return xs.pop(-1)


# ---- Fixed array nodes ---------------------------------------------------

ARRAY_LENGTH = 3


@numba_node
def array_state_constant(x: Signal[int]) -> Signal[int]:
    values: State[NumbaArray] = create_new_array(int, 4)
    values[0] = x
    values[-1] = values[0] + len(values)
    return values[-1]


@numba_node
def array_state_dynamic(index: Signal[int], x: Signal[int]) -> Signal[int]:
    values: State[NumbaArray] = create_new_array(int, 4)
    values[index] = x
    return values[index]


@numba_node
def array_state_persists(index: Signal[int], x: Signal[int]) -> Signal[int]:
    values: State[NumbaArray] = create_new_array(int, 4)
    values[index] = values[index] + x
    return values[index]


@numba_node
def array_local_sum(x: Signal[int]) -> Signal[int]:
    values = create_new_array(int, 3)
    values[0] = x
    values[1] = x + 1
    total = 0
    for value in values:
        total += value
    return total


@numba_node
def array_named_length(x: Signal[int]) -> Signal[int]:
    values = create_new_array(int, ARRAY_LENGTH)
    values[2] = x
    return values[2] + len(values)


@numba_node
def array_bool(x: Signal[bool]) -> Signal[bool]:
    values: State[NumbaArray] = create_new_array(bool, 2)
    values[-1] = x
    return values[1]


@numba_node
def array_float(x: Signal[float]) -> Signal[float]:
    values: State[NumbaArray] = create_new_array(float, 2)
    values[0] = x
    return values[0]


@numba_method
def array_increment(values, index, increment):
    values[index] = values[index] + increment
    return values[index]


@numba_node
def array_inline_state(index: Signal[int], x: Signal[int]) -> Signal[int]:
    values: State[NumbaArray] = create_new_array(int, 4)
    return array_increment(values, index, x)


@numba_node
def array_bad_static_index(x: Signal[int]) -> Signal[int]:
    values = create_new_array(int, 2)
    return values[2] + x


@numba_method
def array_bad_helper_index(values):
    return values[2]


@numba_node
def array_bad_helper_index_node(x: Signal[int]) -> Signal[int]:
    values = create_new_array(int, 2)
    return array_bad_helper_index(values) + x


class TestDictExecution(unittest.TestCase):
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

    def test_stop_frees_state(self):
        node = CompiledNode(compile_function(dict_contains), input_types=[int]).start()
        node.execute([1])
        node.stop()
        self.assertIsNone(node._state[0])


class TestListExecution(unittest.TestCase):
    def test_append_len(self):
        node = CompiledNode(compile_function(list_append_len), input_types=[int]).start()
        self.assertEqual(node.execute([10])[0], 1)
        self.assertEqual(node.execute([20])[0], 2)
        self.assertEqual(node.execute([30])[0], 3)

    def test_append_getitem(self):
        node = CompiledNode(compile_function(list_append_last), input_types=[int]).start()
        self.assertEqual(node.execute([10])[0], 10)
        self.assertEqual(node.execute([20])[0], 20)

    def test_inline_helper_mutates_explicit_container_state(self):
        node = CompiledNode(compile_function(list_inline_append_len), input_types=[int]).start()
        self.assertEqual(node.execute([10])[0], 1)
        self.assertEqual(node.execute([20])[0], 2)

    def test_stop_frees_state(self):
        node = CompiledNode(compile_function(list_append_len), input_types=[int]).start()
        node.execute([10])
        node.stop()
        self.assertIsNone(node._state[0])


class TestArrayExecution(unittest.TestCase):
    def test_constant_index_and_length(self):
        result = compile_function(array_state_constant)
        self.assertEqual(result.struct_state_sizes, (32,))
        node = CompiledNode(result, input_types=[int]).start()
        self.assertEqual(node.execute([10])[0], 14)
        self.assertEqual(node.execute([20])[0], 24)

    def test_dynamic_index_and_negative_index(self):
        node = CompiledNode(compile_function(array_state_dynamic), input_types=[int, int]).start()
        self.assertEqual(node.execute([2, 17])[0], 17)
        self.assertEqual(node.execute([-1, 23])[0], 23)

    def test_state_persists(self):
        node = CompiledNode(compile_function(array_state_persists), input_types=[int, int]).start()
        self.assertEqual(node.execute([1, 5])[0], 5)
        self.assertEqual(node.execute([1, 7])[0], 12)
        self.assertEqual(node.execute([2, 4])[0], 4)

    def test_local_array_is_zeroed_on_each_execution_and_iterates(self):
        node = CompiledNode(compile_function(array_local_sum), input_types=[int]).start()
        self.assertEqual(node.execute([10])[0], 21)
        self.assertEqual(node.execute([20])[0], 41)

    def test_length_can_be_a_module_constant(self):
        node = CompiledNode(compile_function(array_named_length), input_types=[int]).start()
        self.assertEqual(node.execute([6])[0], 9)

    def test_bool_and_float(self):
        bool_node = CompiledNode(compile_function(array_bool), input_types=[bool]).start()
        float_node = CompiledNode(compile_function(array_float), input_types=[float]).start()
        self.assertIs(bool_node.execute([True])[0], True)
        self.assertIs(bool_node.execute([False])[0], False)
        self.assertEqual(float_node.execute([1.5])[0], 1.5)

    def test_inline_helper_mutates_array_state(self):
        node = CompiledNode(compile_function(array_inline_state), input_types=[int, int]).start()
        self.assertEqual(node.execute([1, 3])[0], 3)
        self.assertEqual(node.execute([1, 4])[0], 7)

    def test_constant_access_has_no_bounds_branch(self):
        llvm_ir = compile_function(array_state_constant).compiled_func._library.get_llvm_str()
        self.assertNotIn("llvm.trap", llvm_ir)
        self.assertNotIn("numba_list_getitem", llvm_ir)
        self.assertNotIn("numba_list_setitem", llvm_ir)

    def test_dynamic_access_has_one_bounds_branch_per_access(self):
        llvm_ir = compile_function(array_state_dynamic).compiled_func._library.get_llvm_str()
        # This node performs one dynamic store and one dynamic load.
        self.assertEqual(len(re.findall(r"\bcall void @llvm\.trap\(", llvm_ir)), 2)
        self.assertEqual(len(re.findall(r"icmp u(?:lt|le|gt|ge) i64", llvm_ir)), 2)

    def test_invalid_constant_index_is_rejected(self):
        with self.assertRaisesRegex(Exception, "out of range"):
            compile_function(array_bad_static_index)
        with self.assertRaisesRegex(Exception, "out of range"):
            compile_function(array_bad_helper_index_node)


if __name__ == "__main__":
    unittest.main()
