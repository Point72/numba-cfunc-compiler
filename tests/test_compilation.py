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
import shutil
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone

import pytest

from numba_cfunc_compiler.api import NumbaList, State, create_new_list, set_output
from numba_cfunc_compiler.core import compile as compiler_module
from numba_cfunc_compiler.core.analysis import FunctionAnalyzer, OutputAnalysis, OutputTypeHandler
from numba_cfunc_compiler.core.compile import create_compiled_func
from numba_cfunc_compiler.core.context import CompilationContext
from numba_cfunc_compiler.core.defaults import register_all
from numba_cfunc_compiler.extension.callback_components import ComponentRegistry
from tests.harness import CompiledNode, Signal, SignalComponent, SignalInputHandler, compile_function, numba_node


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


@pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc is required for the C callback runner")
def test_c_callback_operations():
    """Compile in Python, then exercise the callbacks through the C runner."""
    lib = _build_test_runner()
    results = (
        compile_function(add_ints),
        compile_function(add_floats),
        compile_function(multiply, factor=3),
        compile_function(conditional, limit=10),
        compile_function(negate_if),
        compile_function(accumulate),
        compile_function(ema, alpha=0.1),
    )
    failures = lib.run_tests(*(result.compiled_func.address for result in results))
    assert failures == 0, f"C test runner reported {failures} failure(s): {lib.get_last_error().decode()}"


@numba_node
def set_int(x: Signal[int]) -> Signal[int]:
    set_output("output_0", x + 1)


@numba_node
def shift_timestamp(stamp: Signal[datetime]) -> Signal[datetime]:
    return stamp + 1_000_000_000


@numba_node
def set_shifted_timestamp(stamp: Signal[datetime]) -> Signal[datetime]:
    set_output("output_0", stamp + 2_000_000_000)


@numba_node
def shift_duration(span: Signal[timedelta]) -> Signal[timedelta]:
    return span + 1_000_000_000


@numba_node
def time_literals(span: Signal[timedelta]) -> Signal[datetime]:
    start: State[datetime] = datetime(2020, 1, 1, tzinfo=timezone.utc)
    offset: State[timedelta] = timedelta(seconds=2, microseconds=3)
    return start + offset + timedelta(seconds=1) + span


@numba_node
def naive_time(span: Signal[timedelta]) -> Signal[datetime]:
    start: State[datetime] = datetime(2020, 1, 1)  # noqa: DTZ001 - verify naive datetime rejection
    return start + span


@numba_node
def private_local(x: Signal[int]) -> Signal[int]:
    __private = x + 1
    return __private


@numba_node
def generated_local(x: Signal[int]) -> Signal[int]:
    __ncc_output_sink_0 = x + 1
    return __ncc_output_sink_0


@numba_node
def related_state_names(x: Signal[int]) -> Signal[int]:
    values: State[NumbaList] = create_new_list(int)
    values_ptr: State[NumbaList] = create_new_list(int)
    values.append(x)
    values_ptr.append(x)
    return len(values) + len(values_ptr)


class NamedPair:
    pass


class NamedSingle:
    pass


class NamedOutputHandler(OutputTypeHandler):
    def try_parse(self, return_annotation, ast_tree):
        if return_annotation is NamedPair:
            return OutputAnalysis(outputs={"second": int, "first": int})
        if return_annotation is NamedSingle:
            return OutputAnalysis(outputs={"output_0": int})
        return None


@numba_node
def named_pair(x: Signal[int]) -> NamedPair:
    set_output("first", x + 5)
    return x + 1, None


@numba_node
def named_single(x: Signal[int]) -> NamedSingle:
    return x * 2


def test_operations():
    result = compile_function(add_ints)
    assert result.outputs == {None: int}
    assert [signal.get_type() for signal in result.ordered_input_signals] == [int, int]
    stamp = 1_700_000_000_000_000_000
    for case, compiled, input_types, values, expected in (
        ("add_ints", result, [int, int], [2, 3], (5, True)),
        ("set_int", compile_function(set_int), [int], [4], (5, True)),
        ("shift_timestamp", compile_function(shift_timestamp), [datetime], [stamp], (stamp + 1_000_000_000, True)),
        ("set_shifted_timestamp", compile_function(set_shifted_timestamp), [datetime], [stamp], (stamp + 2_000_000_000, True)),
        ("shift_duration", compile_function(shift_duration), [timedelta], [5_000_000_000], (6_000_000_000, True)),
    ):
        assert CompiledNode(compiled, input_types).execute(values) == expected, case

    optional = CompiledNode(compile_function(conditional, limit=10), [int])
    assert optional.execute([5]) == (0, False)
    assert optional.execute([12]) == (12, True)

    start = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp() * 1e9)
    time_result = compile_function(time_literals)
    literal_node = CompiledNode(time_result, [timedelta])
    for index, value in enumerate(time_result.state_values):
        literal_node._state_store[index] = value
    literal_node.start()
    assert literal_node.execute([5_000_000_000]) == (start + 8_000_003_000, True)
    literal_node.stop()
    with pytest.raises(TypeError, match="timezone-aware"):
        compile_function(naive_time)

    with CompilationContext():
        register_all()
        ComponentRegistry.register(SignalComponent())
        FunctionAnalyzer.register_input_handler(SignalInputHandler())
        FunctionAnalyzer.register_output_handler(NamedOutputHandler())
        for func, expected_outputs, expected_values in (
            (named_pair, {"second": int, "first": int}, {"second": (4, True), "first": (8, True)}),
            (named_single, {"output_0": int}, {"output_0": (6, True)}),
        ):
            named_result = create_compiled_func(func, Signal(typ=int), extract_python_type_fn=lambda signal: signal.get_type())
            assert list(named_result.outputs.items()) == list(expected_outputs.items())
            assert CompiledNode(named_result, [int]).execute([3]) == expected_values


def test_validation():
    assert CompiledNode(compile_function(private_local), [int]).execute([2]) == (3, True)
    assert compile_function(related_state_names).compiled_func.address
    with pytest.raises(ValueError, match="reserved compiler prefix"):
        compile_function(generated_local)
    with pytest.raises(ValueError, match="At least one output"):
        OutputAnalysis(outputs={})
    with pytest.raises(ValueError, match="anonymous output must be the only output"):
        OutputAnalysis(outputs={None: int, "other": int})
    with pytest.raises(TypeError, match="Output names"):
        OutputAnalysis(outputs={1: int})


def test_semantic_key(monkeypatch):
    first = compile_function(multiply, factor=3)
    same = compile_function(multiply, factor=3)
    changed = compile_function(multiply, factor=4)
    assert first.semantic_key == same.semantic_key
    assert first.semantic_key != changed.semantic_key
    original = compiler_module.build_semantic_key("source", "signature", "options")
    monkeypatch.setattr(
        compiler_module,
        "COMPILER_IMPLEMENTATION_VERSION",
        compiler_module.COMPILER_IMPLEMENTATION_VERSION + 1,
    )
    assert compiler_module.build_semantic_key("source", "signature", "options") != original
