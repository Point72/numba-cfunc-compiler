"""
numba_cfunc_compiler test suite.

Python compiles the functions, C++ runs and validates them.

Flow:
  1. Python defines @numba_node functions
  2. compile_function() produces Numba cfuncs
  3. Function pointers are passed to the C test runner
  4. C sets up void* arrays, calls the functions, and asserts correctness
"""

import ctypes
import os
import subprocess
import tempfile
import unittest

from numba_cfunc_compiler.numba_config import State
from numba_cfunc_compiler.tests.harness import Signal, compile_function, numba_node


def _build_test_runner():
    """Compile the C test runner into a shared library."""
    src = os.path.join(os.path.dirname(__file__), "cfunc_caller.c")
    lib_path = os.path.join(tempfile.gettempdir(), "cfunc_caller.so")
    result = subprocess.run(
        ["gcc", "-shared", "-fPIC", "-O2", "-lm", "-o", lib_path, src],
        capture_output=True,
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


class TestCompilation(unittest.TestCase):
    """Compile in Python, test in C."""

    @classmethod
    def setUpClass(cls):
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


if __name__ == "__main__":
    unittest.main()
