"""
numba_cfunc_compiler test suite.

Python compiles the functions, C++ runs and validates them.

Flow:
  1. Python defines @numba_node functions
  2. compile_function() produces Numba cfuncs
  3. Function pointers are passed to the C test runner
  4. C sets up void* arrays, calls the functions, and asserts correctness
"""

import ast
import ctypes
import enum
import os
import shutil
import subprocess
import tempfile
import types
import unittest

from numba_cfunc_compiler.defaults.struct_support import StructFieldInfo, StructType
from numba_cfunc_compiler.node_api import State
from numba_cfunc_compiler.numba_methods import numba_method
from numba_cfunc_compiler.tests.harness import CompiledNode, Signal, compile_function, numba_node, setup_standalone_context


def _build_test_runner():
    """Compile the C test runner into a shared library."""
    src = os.path.join(os.path.dirname(__file__), "cfunc_caller.c")
    lib_path = os.path.join(tempfile.gettempdir(), "cfunc_caller.so")
    result = subprocess.run(
        ["gcc", "-shared", "-fPIC", "-O2", "-lm", "-o", lib_path, src],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gcc failed:\n{result.stderr}")
    lib = ctypes.CDLL(lib_path)
    lib.run_tests.argtypes = [ctypes.c_void_p] * 7
    lib.run_tests.restype = ctypes.c_int
    lib.get_fail_count.restype = ctypes.c_int
    lib.get_last_error.restype = ctypes.c_char_p
    return lib


# ---- Define the functions to compile ----


@numba_node
def add_ints(x: Signal[int], y: Signal[int]) -> Signal[int]:
    return x + y


@numba_node
def add_floats(x: Signal[float], y: Signal[float]) -> Signal[float]:
    return x + y


@numba_node
def multiply(x: Signal[int], factor: int) -> Signal[int]:
    return x * factor


@numba_node
def conditional(x: Signal[int], limit: int) -> Signal[int]:
    if x > limit:
        return x


@numba_node
def negate_if(x: Signal[int], flag: Signal[bool]) -> Signal[int]:
    if flag:
        return 0 - x
    return x


@numba_node
def accumulate(x: Signal[int]) -> Signal[int]:
    total: State[int] = 0
    total = total + x
    return total


@numba_node
def ema(x: Signal[float], alpha: float) -> Signal[float]:
    s: State[float] = 0.0
    s = alpha * x + (1.0 - alpha) * s
    return s


@numba_method
def advance_inline_total(total: int, value: int) -> int:
    # Deliberately reuse a common node-local name: helper scopes are independent.
    value = value + 1
    return total + value


@numba_node
def inline_accumulate(x: Signal[int]) -> Signal[int]:
    total: State[int] = 0
    total = advance_inline_total(total, x)
    return total


class InlineOrder:
    price: float
    count: int


class InlineOrderStorage(ctypes.Structure):
    _fields_ = [("price", ctypes.c_double), ("count", ctypes.c_int64)]


class InlineOrderReversed:
    price: float
    count: int


class InlineOrderReversedStorage(ctypes.Structure):
    _fields_ = [("count", ctypes.c_int64), ("price", ctypes.c_double)]


class InlineOrderType(StructType):
    @classmethod
    def is_type_supported(cls, var_type):
        return var_type is InlineOrder

    @classmethod
    def _get_struct_fields(cls, var_type):
        return {
            "price": StructFieldInfo("price", 0, "float64", 8),
            "count": StructFieldInfo("count", 8, "int64", 8),
        }

    @classmethod
    def _get_struct_size(cls, var_type):
        return 16


class InlineOrderReversedType(InlineOrderType):
    @classmethod
    def is_type_supported(cls, var_type):
        return var_type is InlineOrderReversed

    @classmethod
    def _get_struct_fields(cls, var_type):
        return {
            "count": StructFieldInfo("count", 0, "int64", 8),
            "price": StructFieldInfo("price", 8, "float64", 8),
        }


class InlineOrderAlternateType(InlineOrderType):
    @classmethod
    def _get_struct_fields(cls, var_type):
        return InlineOrderReversedType._get_struct_fields(var_type)


@numba_method
def inline_adjust_order(order, change):
    order.count = order.count + change
    return order.price


@numba_node
def inline_struct_fields(order: Signal[InlineOrder], change: Signal[int]) -> Signal[float]:
    return inline_adjust_order(order, change)


@numba_node
def inline_struct_fields_reversed(order: Signal[InlineOrderReversed], change: Signal[int]) -> Signal[float]:
    return inline_adjust_order(order, change)


@numba_method
def inline_nested_inner(value):
    return value + 2


@numba_method
def method_force_inline(value):
    return value + 1


@numba_method(force_inline=False)
def method_without_forced_inline(value):
    return value + 1


METHOD_MODE_MODULE = types.ModuleType("method_mode_module")
METHOD_MODE_MODULE.apply = method_force_inline


@numba_node
def method_inline_mode(x: Signal[int]) -> Signal[int]:
    return METHOD_MODE_MODULE.apply(x)


@numba_method
def inline_nested_outer(value):
    return inline_nested_inner(value)


INLINE_TEST_MODULE = types.ModuleType("inline_test_module")
INLINE_TEST_MODULE.increment = inline_nested_inner
INLINE_TEST_MODULE.offset = 2


@numba_method
def inline_nested_more(value):
    return value + 3


@numba_node
def inline_nested_local(x: Signal[int]) -> Signal[int]:
    local = x
    return inline_nested_outer(local)


@numba_node
def inline_module_call(x: Signal[int]) -> Signal[int]:
    return INLINE_TEST_MODULE.increment(x)


@numba_method
def inline_nested_module(value):
    return INLINE_TEST_MODULE.increment(value)


@numba_node
def inline_nested_module_call(x: Signal[int]) -> Signal[int]:
    return inline_nested_module(x)


@numba_node
def inline_expression_argument(x: Signal[int]) -> Signal[int]:
    return inline_nested_inner(x + 1)


class InlineAdjustment(enum.Enum):
    BONUS = 3


def lower_inline_adjustment(node, globalns, factory):
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "InlineAdjustment":
        return ast.Constant(globalns["InlineAdjustment"][node.attr].value)
    return None


@numba_method
def inline_with_enum(value):
    return value + InlineAdjustment.BONUS


@numba_node
def inline_enum_node(x: Signal[int]) -> Signal[int]:
    return inline_with_enum(x)


INLINE_GLOBAL_OFFSET = 2


@numba_method
def inline_with_global(value):
    return value + INLINE_GLOBAL_OFFSET


def make_closure_helper():
    offset = 2

    @numba_method
    def helper(value):
        return value + offset

    return helper


inline_with_closure = make_closure_helper()


def plain_helper(value):
    return value + 1


@numba_method
def inline_with_plain_helper(value):
    return plain_helper(value)


@numba_method
def inline_with_module_data(value):
    return value + INLINE_TEST_MODULE.offset


@numba_method
def inline_with_default(value, offset=1):
    return value + offset


@numba_method
def inline_recursive(value):
    return inline_recursive(value)


@numba_method
def inline_with_missing_name(value):
    return value + missing_inline_value  # noqa: F821 - verifies unbound-name diagnostics


@numba_node
def inline_invalid_global(x: Signal[int]) -> Signal[int]:
    return inline_with_global(x)


@numba_node
def inline_invalid_closure(x: Signal[int]) -> Signal[int]:
    return inline_with_closure(x)


@numba_node
def inline_invalid_plain_helper(x: Signal[int]) -> Signal[int]:
    return inline_with_plain_helper(x)


@numba_node
def inline_invalid_module_data(x: Signal[int]) -> Signal[int]:
    return inline_with_module_data(x)


@numba_node
def inline_invalid_default(x: Signal[int]) -> Signal[int]:
    return inline_with_default(x)


@numba_node
def inline_invalid_recursive(x: Signal[int]) -> Signal[int]:
    return inline_recursive(x)


@numba_node
def inline_invalid_missing_name(x: Signal[int]) -> Signal[int]:
    return inline_with_missing_name(x)


class TestCompilation(unittest.TestCase):
    """Compile in Python, test in C."""

    @classmethod
    def setUpClass(cls):
        # This suite drives compiled cfuncs through a gcc-built C runner, so it
        # is limited to platforms with gcc (Linux/macOS). Cross-platform
        # execution coverage (incl. Windows) lives in test_containers.py.
        if shutil.which("gcc") is None:
            raise unittest.SkipTest("gcc not available; see test_containers.py for cross-platform execution tests")
        cls.lib = _build_test_runner()

    def test_all_from_cpp(self):
        """Compile all functions, pass pointers to C, let C run and assert."""
        results = {
            "add_ints": compile_function(add_ints),
            "add_floats": compile_function(add_floats),
            "multiply": compile_function(multiply, factor=3),
            "conditional": compile_function(conditional, limit=10),
            "negate_if": compile_function(negate_if),
            "accumulate": compile_function(accumulate),
            "ema": compile_function(ema, alpha=0.1),
        }

        failures = self.lib.run_tests(
            results["add_ints"].compiled_func.address,
            results["add_floats"].compiled_func.address,
            results["multiply"].compiled_func.address,
            results["conditional"].compiled_func.address,
            results["negate_if"].compiled_func.address,
            results["accumulate"].compiled_func.address,
            results["ema"].compiled_func.address,
        )

        if failures > 0:
            error = self.lib.get_last_error().decode()
            self.fail(f"C test runner reported {failures} failure(s): {error}")

    def test_inline_scalar_state(self):
        result = compile_function(inline_accumulate)
        node = CompiledNode(result, input_types=[int]).start()
        self.assertEqual(node.execute([2])[0], 3)
        self.assertEqual(node.execute([4])[0], 8)

    def test_inline_struct_layouts(self):
        # Both struct layouts reach Numba as voidptr; field offsets come from each caller's StructType.
        context = setup_standalone_context()
        context.type_classes.insert(0, InlineOrderType)
        context.type_classes.insert(0, InlineOrderReversedType)
        try:
            result = compile_function(inline_struct_fields)
            reversed_result = compile_function(inline_struct_fields_reversed)
        finally:
            context.type_classes.remove(InlineOrderType)
            context.type_classes.remove(InlineOrderReversedType)

        node = CompiledNode(result, input_types=[int, int]).start()
        order = InlineOrderStorage(12.5, 3)
        node._inputs[0] = ctypes.addressof(order)
        node._write(node._in_store, 1, int, 4)
        node._call(0)
        self.assertEqual(order.count, 7)
        self.assertEqual(node._read(node._out_store, 0, float), 12.5)

        reversed_node = CompiledNode(reversed_result, input_types=[int, int]).start()
        reversed_order = InlineOrderReversedStorage(5, 21.5)
        reversed_node._inputs[0] = ctypes.addressof(reversed_order)
        reversed_node._write(reversed_node._in_store, 1, int, 2)
        reversed_node._call(0)
        self.assertEqual(reversed_order.count, 7)
        self.assertEqual(reversed_node._read(reversed_node._out_store, 0, float), 21.5)

    def test_inline_struct_layout_changes_key(self):
        context = setup_standalone_context()
        context.type_classes.insert(0, InlineOrderType)
        try:
            original = compile_function(inline_struct_fields)
            context.type_classes.insert(0, InlineOrderAlternateType)
            try:
                alternate = compile_function(inline_struct_fields)
            finally:
                context.type_classes.remove(InlineOrderAlternateType)
        finally:
            context.type_classes.remove(InlineOrderType)
        self.assertNotEqual(original.semantic_key, alternate.semantic_key)

    def test_inline_call_forms(self):
        nested = CompiledNode(compile_function(inline_nested_local), input_types=[int]).start()
        self.assertEqual(nested.execute([5])[0], 7)
        module_result = compile_function(inline_module_call)
        module = CompiledNode(module_result, input_types=[int]).start()
        self.assertEqual(module.execute([5])[0], 7)
        nested_module_result = compile_function(inline_nested_module_call)
        nested_module = CompiledNode(nested_module_result, input_types=[int]).start()
        self.assertEqual(nested_module.execute([5])[0], 7)
        with self.assertRaisesRegex(TypeError, "must be a named variable"):
            compile_function(inline_expression_argument)

        original = INLINE_TEST_MODULE.increment
        INLINE_TEST_MODULE.increment = inline_nested_more
        try:
            rebound_result = compile_function(inline_module_call)
            rebound_nested_result = compile_function(inline_nested_module_call)
        finally:
            INLINE_TEST_MODULE.increment = original
        self.assertEqual(CompiledNode(rebound_result, input_types=[int]).start().execute([5])[0], 8)
        self.assertEqual(CompiledNode(rebound_nested_result, input_types=[int]).start().execute([5])[0], 8)
        self.assertNotEqual(module_result.semantic_key, rebound_result.semantic_key)
        self.assertNotEqual(nested_module_result.semantic_key, rebound_nested_result.semantic_key)

    def test_method_inline_option_changes_key(self):
        forced = compile_function(method_inline_mode)
        self.assertEqual(CompiledNode(forced, input_types=[int]).start().execute([5])[0], 6)
        METHOD_MODE_MODULE.apply = method_without_forced_inline
        try:
            optional = compile_function(method_inline_mode)
        finally:
            METHOD_MODE_MODULE.apply = method_force_inline
        self.assertEqual(CompiledNode(optional, input_types=[int]).start().execute([5])[0], 6)
        self.assertNotEqual(forced.semantic_key, optional.semantic_key)

    def test_inline_enum_lowerer(self):
        context = setup_standalone_context()
        context.attr_lowerers.append(lower_inline_adjustment)
        try:
            result = compile_function(inline_enum_node)
        finally:
            context.attr_lowerers.remove(lower_inline_adjustment)
        node = CompiledNode(result, input_types=[int]).start()
        self.assertEqual(node.execute([4])[0], 7)

    def test_inline_enum_value_changes_key(self):
        context = setup_standalone_context()
        context.attr_lowerers.append(lower_inline_adjustment)
        try:
            original = compile_function(inline_enum_node)
            original_enum = inline_with_enum.__globals__["InlineAdjustment"]
            inline_with_enum.__globals__["InlineAdjustment"] = enum.Enum("AlternateAdjustment", {"BONUS": 5})
            try:
                changed = compile_function(inline_enum_node)
            finally:
                inline_with_enum.__globals__["InlineAdjustment"] = original_enum
        finally:
            context.attr_lowerers.remove(lower_inline_adjustment)
        self.assertNotEqual(original.semantic_key, changed.semantic_key)
        self.assertEqual(CompiledNode(changed, input_types=[int]).start().execute([4])[0], 9)

    def test_inline_rejects_captures_and_defaults(self):
        cases = [
            (inline_invalid_global, "captures 'INLINE_GLOBAL_OFFSET'"),
            (inline_invalid_closure, "captures 'offset'"),
            (inline_invalid_plain_helper, "captures 'plain_helper'"),
            (inline_invalid_module_data, "unresolved global symbols: INLINE_TEST_MODULE"),
            (inline_invalid_default, "cannot declare default arguments"),
            (inline_invalid_recursive, "Recursive @numba_method"),
        ]
        for node, message in cases:
            with self.subTest(node=node.__name__), self.assertRaisesRegex(TypeError, message):
                compile_function(node)
        with self.assertRaisesRegex(NameError, "missing_inline_value"):
            compile_function(inline_invalid_missing_name)


if __name__ == "__main__":
    unittest.main()
