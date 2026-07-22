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

from numba_cfunc_compiler.numba_config import NumbaDict, NumbaList, State, create_new_dict, create_new_list
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


if __name__ == "__main__":
    unittest.main()
