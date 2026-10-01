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
def mutate_constant_containers(x: Signal[int], values: NumbaList[int], mapping: NumbaDict[int, int]) -> Signal[int]:
    result = values[0] * 1_000 + len(values) * 100 + mapping.get(1, -1) * 10 + len(mapping)
    values[0] = x
    values.append(x)
    mapping[1] = x
    mapping[x] = x
    return result


@numba_node
def constant_container_and_state(x: Signal[int], values: NumbaList[int]) -> Signal[int]:
    total: State[int] = 0
    total += x
    return values[0] + total


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

    def test_stop_frees_state(self):
        node = CompiledNode(compile_function(list_append_len), input_types=[int]).start()
        node.execute([10])
        node.stop()
        self.assertIsNone(node._state[0])

    def test_constant_containers_reuse_storage_and_reset_values(self):
        import ctypes

        # Test-only mirrors of the C runtime structs, used to inspect their backing-storage pointers.
        class NBList(ctypes.Structure):
            _fields_ = [
                ("size", ctypes.c_ssize_t),
                ("item_size", ctypes.c_ssize_t),
                ("allocated", ctypes.c_ssize_t),
                ("is_mutable", ctypes.c_int),
                ("item_incref", ctypes.c_void_p),
                ("item_decref", ctypes.c_void_p),
                ("items", ctypes.c_void_p),
            ]

        class NBDict(ctypes.Structure):
            _fields_ = [
                ("used", ctypes.c_ssize_t),
                ("keys", ctypes.c_void_p),
            ]

        result = compile_function(
            mutate_constant_containers,
            values=[7, 8],
            mapping={1: 5},
        )
        self.assertEqual(result.state_values, (0, 0))
        self.assertEqual(result.nrt_state_indices, (0, 1))

        node = CompiledNode(result, input_types=[int]).start()
        container_ptrs = tuple(node._state)
        self.assertTrue(all(container_ptrs))
        list_items = NBList.from_address(container_ptrs[0]).items
        dict_keys = NBDict.from_address(container_ptrs[1]).keys

        for x in range(2, 34):
            self.assertEqual(node.execute([x]), (7_251, True))
            self.assertEqual(tuple(node._state), container_ptrs)
            self.assertEqual(NBList.from_address(container_ptrs[0]).items, list_items)
            self.assertEqual(NBDict.from_address(container_ptrs[1]).keys, dict_keys)

        node.stop()
        self.assertEqual(tuple(node._state), (None, None))

    def test_constant_container_and_declared_state_use_distinct_slots(self):
        result = compile_function(constant_container_and_state, values=[7])
        self.assertEqual(result.state_values, (0, 0))
        self.assertEqual(len(result.nrt_state_indices), 1)

        node = CompiledNode(result, input_types=[int]).start()
        self.assertEqual(node.execute([1])[0], 8)
        self.assertEqual(node.execute([2])[0], 10)
        node.stop()


if __name__ == "__main__":
    unittest.main()
