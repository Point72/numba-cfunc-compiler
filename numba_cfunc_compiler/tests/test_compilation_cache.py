import enum
import math
import os
import subprocess
import sys
import tempfile
import types
import unittest
import warnings
from contextlib import contextmanager
from unittest import mock

import numpy as np

import numba_cfunc_compiler
from numba_cfunc_compiler.compilation_cache import CacheConfig, CompilationCache, get_cache_config, set_cache_config
from numba_cfunc_compiler.node_api import NumbaDict, NumbaList, State, create_new_dict, create_new_list
from numba_cfunc_compiler.numba_core import create_compiled_func
from numba_cfunc_compiler.tests.harness import CompiledNode, Signal, compile_function, numba_node, setup_standalone_context

OFFSET = 1
WEIGHTS = np.array([1.0, 2.0])
params = types.ModuleType("params")
params.SCALE = 2.0


class Side(enum.IntEnum):
    BUY = 1
    SELL = 2


@numba_node
def scale_and_shift(x: Signal[int], factor: int) -> Signal[int]:
    return x * factor + 1


@numba_node
def running_list_sum(x: Signal[int]) -> Signal[int]:
    xs: State[NumbaList] = create_new_list(int)
    xs.append(x)
    total = 0
    for v in xs:
        total += v
    return total


@numba_node
def running_dict_sum(key: Signal[int], val: Signal[int]) -> Signal[int]:
    d: State[NumbaDict] = create_new_dict(int, int)
    d[key] = val
    total = 0
    for k in d:
        total += d[k]
    return total


@numba_node
def reads_global_variable(x: Signal[float]) -> Signal[float]:
    return x + OFFSET


@numba_node
def reads_module_attribute(x: Signal[float]) -> Signal[float]:
    return x * params.SCALE


@numba_node
def reads_enum_member(x: Signal[float]) -> Signal[float]:
    y = 0.0
    if Side.SELL == 2:
        y = 1.0
    return x + y


@numba_node
def reads_global_array(x: Signal[float]) -> Signal[float]:
    return x + WEIGHTS[1]


@numba_node
def reads_module_only(x: Signal[float]) -> Signal[float]:
    y = math.sqrt(x)
    return y


@contextmanager
def cache_config(**kwargs):
    previous = get_cache_config()
    set_cache_config(CacheConfig(enabled=True, **kwargs))
    try:
        yield
    finally:
        set_cache_config(previous)


def compile_with_globals(func):
    with setup_standalone_context():
        return create_compiled_func(func, Signal(typ=float), extract_python_type_fn=lambda s: s.get_type(), call_globals=globals())


class TestCompilationCacheStore(unittest.TestCase):
    def test_memory_only(self):
        cache = CompilationCache()
        self.assertIsNone(cache.get("k"))
        cache.put("k", "ir")
        self.assertEqual(cache.get("k"), "ir")

    def test_disk_shared(self):
        with tempfile.TemporaryDirectory() as tmp:
            CompilationCache(tmp).put("k", "ir")
            self.assertEqual(CompilationCache(tmp).get("k"), "ir")

    def test_disk_versioned(self):
        with tempfile.TemporaryDirectory() as tmp:
            CompilationCache(tmp).put("k", "ir")
            (namespace,) = os.listdir(tmp)
            self.assertTrue(namespace.startswith(f"ncc-{numba_cfunc_compiler.__version__}-numba-"))
            self.assertEqual(os.listdir(os.path.join(tmp, namespace)), ["k.ll"])
            with mock.patch.object(numba_cfunc_compiler, "__version__", "0.0.0"):
                self.assertIsNone(CompilationCache(tmp).get("k"))

    def test_disabled_by_default(self):
        self.assertFalse(CacheConfig().enabled)


class TestCompileWithCache(unittest.TestCase):
    def test_hit_matches_fresh(self):
        with cache_config():
            fresh = compile_function(scale_and_shift, factor=3)
            cached = compile_function(scale_and_shift, factor=3)
        self.assertIsNotNone(fresh.compiled_func)
        self.assertIsNone(cached.compiled_func)
        self.assertEqual(cached.semantic_key, fresh.semantic_key)
        self.assertEqual(cached.native_name, fresh.native_name)
        self.assertEqual(cached.llvm_ir, fresh.llvm_ir)
        self.assertEqual(cached.output_types, fresh.output_types)

    def test_constant_arg_misses(self):
        with cache_config():
            compile_function(scale_and_shift, factor=3)
            other = compile_function(scale_and_shift, factor=4)
        self.assertIsNotNone(other.compiled_func)

    def test_array_global_not_cached(self):
        with tempfile.TemporaryDirectory() as tmp, cache_config(directory=tmp):
            with self.assertWarnsRegex(UserWarning, "Not caching 'reads_global_array'.*WEIGHTS"):
                compile_with_globals(reads_global_array)
                self.assertIsNotNone(compile_with_globals(reads_global_array).compiled_func)
            self.assertEqual(os.listdir(tmp), [])

    def test_constant_globals_keyed(self):
        cases = [
            (reads_global_variable, mock.patch(f"{__name__}.OFFSET", 2)),
            (reads_module_attribute, mock.patch.object(params, "SCALE", 3.0)),
            (reads_enum_member, mock.patch(f"{__name__}.Side", enum.IntEnum("Side", {"BUY": 1, "SELL": 5}))),
        ]
        for func, patch in cases:
            with self.subTest(func.__name__), cache_config(), warnings.catch_warnings():
                warnings.simplefilter("error")
                original = compile_with_globals(func)
                self.assertIsNone(compile_with_globals(func).compiled_func)
                with patch:
                    changed = compile_with_globals(func)
                self.assertNotEqual(changed.semantic_key, original.semantic_key)
                self.assertIsNotNone(changed.compiled_func)

    def test_module_calls_cached(self):
        with cache_config():
            compile_with_globals(reads_module_only)
            self.assertIsNone(compile_with_globals(reads_module_only).compiled_func)

    def test_loop_keys_stable(self):
        for func, inputs, expected in [
            (running_list_sum, [[1], [2], [3]], [1, 3, 6]),
            (running_dict_sum, [[1, 10], [2, 20], [1, 5]], [10, 30, 25]),
        ]:
            with self.subTest(func.__name__):
                first = compile_function(func)
                second = compile_function(func)
                self.assertEqual(first.semantic_key, second.semantic_key)
                node = CompiledNode(second, input_types=[int] * len(inputs[0])).start()
                self.assertEqual([node.execute(x)[0] for x in inputs], expected)
                node.stop()

    def test_cross_process_hit(self):
        script = (
            "import sys\n"
            "from numba_cfunc_compiler.compilation_cache import CacheConfig, set_cache_config\n"
            "from numba_cfunc_compiler.tests.harness import compile_function\n"
            "from numba_cfunc_compiler.tests.test_compilation_cache import running_list_sum\n"
            "set_cache_config(CacheConfig(enabled=True, directory=sys.argv[1]))\n"
            "print(compile_function(running_list_sum).semantic_key)\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = subprocess.run([sys.executable, "-c", script, tmp], capture_output=True, text=True, check=True)
            with cache_config(directory=tmp):
                result = compile_function(running_list_sum)
        self.assertEqual(result.semantic_key, out.stdout.strip().splitlines()[-1])
        self.assertIsNone(result.compiled_func)


if __name__ == "__main__":
    unittest.main()
