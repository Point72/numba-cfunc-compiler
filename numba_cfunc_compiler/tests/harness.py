"""
Standalone harness for numba_cfunc_compiler tests.

Compiles @numba_node functions and exposes the raw C function pointer
for use by the C test runner.

Usage:
    from tests.harness import Signal, compile_function, numba_node
    from numba_cfunc_compiler.numba_config import State

    @numba_node
    def add(x: Signal[int], y: Signal[int]) -> Signal[int]:
        return x + y

    result = compile_function(add)
    func_ptr = result.compiled_func.address   # raw C function pointer (int)

The compiled function has the C signature:
    void (*)(void** outputs, int8_t* output_ticked,
             void** state, int8_t lifecycle_phase,
             void** inputs, int8_t* input_ticked, int8_t* input_valid)
"""

import ctypes
import inspect
from typing import Any, Generic, Optional, TypeVar, get_args, get_origin

from numba_cfunc_compiler.compilation_context import CompilationContext
from numba_cfunc_compiler.defaults import register_all
from numba_cfunc_compiler.function_analyzer import (
    FunctionAnalyzer,
    InputTypeHandler,
    OutputAnalysis,
    OutputTypeHandler,
)
from numba_cfunc_compiler.models import ParameterInfo
from numba_cfunc_compiler.numba_core import CompilationResult, create_compiled_func
from numba_cfunc_compiler.source_registry import (
    CfuncParam,
    SourceCategory,
    SourceInitFilter,
    SourceRegistry,
)

__all__ = [
    "Signal",
    "numba_node",
    "compile_function",
    "setup_standalone_context",
    "CompiledNode",
]


T = TypeVar("T")

_DEFAULT_VALUES = {int: 0, float: 0.0, bool: False}


class Signal(Generic[T]):
    """Minimal signal type for standalone use. Wraps a typed value."""

    def __init__(self, value: Any = None, typ: type = None):
        self._type = typ or (type(value) if value is not None else int)
        self._value = value if value is not None else _DEFAULT_VALUES.get(self._type, 0)

    def get_type(self) -> type:
        return self._type


class _SignalInputHandler(InputTypeHandler):
    def try_parse(self, param: inspect.Parameter, ann: Any) -> Optional[ParameterInfo]:
        if get_origin(ann) is not Signal:
            return None
        args = get_args(ann)
        if not args:
            return None
        return ParameterInfo(expected_type=args[0], category="signal")

    def validate_value(self, param_name: str, value: Any, expected_type: Any) -> Any:
        if not isinstance(value, Signal):
            raise TypeError(f"Expected Signal, got {type(value)}")
        return value


class _SingleSignalOutputHandler(OutputTypeHandler):
    def try_parse(self, return_annotation: Any, ast_tree) -> Optional[OutputAnalysis]:
        if get_origin(return_annotation) is not Signal:
            return None
        args = get_args(return_annotation)
        if not args:
            return None
        return OutputAnalysis(output_types=[args[0]], named_outputs=None)


def numba_node(f):
    """No-op decorator for standalone use."""
    return f


# Array names for standalone harness
_INPUTS_ARRAY_NAME = "inputs"
_TICKED_ARRAY_NAME = "input_ticked"
_VALID_ARRAY_NAME = "input_valid"


class _SignalCategory(SourceCategory):
    """Signal inputs for standalone harness (inputs, input_ticked, input_valid)."""

    id = "harness.signal"
    order = 0  # mimic a normal user extension appended after built-in categories
    init_filter = SourceInitFilter.ON_EXECUTE

    @property
    def cfunc_params(self):
        return [
            CfuncParam("inputs", "CPointer(voidptr)"),
            CfuncParam("input_ticked", "CPointer(int8)"),
            CfuncParam("input_valid", "CPointer(int8)"),
        ]

    def create_variables(self, info, factory):
        from numba_cfunc_compiler.type_factory import TypeFactory
        from numba_cfunc_compiler.variable_factory import VoidPtrSource

        input_idx = 0
        info.ordered_input_signals = []

        for name, signal_obj in info.input_analysis.get_by_category("signal").items():
            var_type = TypeFactory.get_type(info.extract_python_type_fn(signal_obj))
            var = VoidPtrSource(
                array_idx=input_idx,
                type=var_type,
                name=name,
                storage_location=_INPUTS_ARRAY_NAME,
                force_opaque=True,
            )
            factory.add_variable(var, category=self.id)
            info.ordered_input_signals.append(signal_obj)
            input_idx += 1

    def get_result_metadata(self, info):
        return {"ordered_input_signals": list(info.ordered_input_signals)}


_standalone_ctx: CompilationContext | None = None


def setup_standalone_context() -> CompilationContext:
    """Return the shared standalone CompilationContext (created once)."""
    global _standalone_ctx
    if _standalone_ctx is None:
        ctx = CompilationContext()
        with ctx:
            register_all()
            SourceRegistry.register(_SignalCategory())
            FunctionAnalyzer.register_input_handler(_SignalInputHandler())
            FunctionAnalyzer.register_output_handler(_SingleSignalOutputHandler())
        _standalone_ctx = ctx
    return _standalone_ctx


def compile_function(func, **constants) -> CompilationResult:
    """Compile a @numba_node function and return the result.

    Args:
        func: A function decorated with @numba_node.
        **constants: Values for constant (non-Signal) parameters, baked in at compile time.

    Returns:
        CompilationResult with:
          .compiled_func          - Numba cfunc object
          .compiled_func.address  - raw C function pointer (int)
          .state_values           - initial state values
          .output_types           - list of output Python types
          .named_outputs          - dict of named outputs (or None)

    Example:
        @numba_node
        def scale(x: Signal[int], factor: int) -> Signal[int]:
            return x * factor

        result = compile_function(scale, factor=3)
        func_ptr = result.compiled_func.address
        # Pass func_ptr to C++ runtime
    """
    sig = inspect.signature(func)
    compile_args = []

    for pname, param in sig.parameters.items():
        ann = param.annotation
        if hasattr(ann, "__origin__") and ann.__origin__ is Signal:
            inner_type = get_args(ann)[0]
            compile_args.append(Signal(typ=inner_type))
        else:
            if pname in constants:
                compile_args.append(constants[pname])
            elif param.default is not inspect.Parameter.empty:
                compile_args.append(param.default)
            else:
                raise TypeError(f"Constant parameter '{pname}' has no default value. Provide it as: compile_function(func, {pname}=value)")

    ctx = setup_standalone_context()
    with ctx:
        return create_compiled_func(
            func,
            *compile_args,
            extract_python_type_fn=lambda s: s.get_type(),
            decorator_name="@numba_node",
        )


# ---------------------------------------------------------------------------
# Cross-platform execution driver
#
# Mirrors the ABI set up by cfunc_caller.c, but in pure Python + ctypes so it
# runs on every platform (Windows included) without needing a C compiler.
# This is what actually *executes* compiled cfuncs so runtime/FFI bugs (symbol
# export, ABI/type-width mismatches in the dict/list runtime) surface here.
#
# cfunc signature (see cfunc_caller.c):
#   void (*)(void** outputs, int8_t* output_ticked,
#            void** state, int8_t lifecycle_phase,
#            void** inputs, int8_t* input_ticked, int8_t* input_valid)
# ---------------------------------------------------------------------------

LIFECYCLE_EXECUTE = 0
LIFECYCLE_START = 1
LIFECYCLE_STOP = 2

_CFUNC_T = ctypes.CFUNCTYPE(
    None,  # void
    ctypes.POINTER(ctypes.c_void_p),  # outputs
    ctypes.POINTER(ctypes.c_int8),  # output_ticked
    ctypes.POINTER(ctypes.c_void_p),  # state
    ctypes.c_int8,  # lifecycle_phase
    ctypes.POINTER(ctypes.c_void_p),  # inputs
    ctypes.POINTER(ctypes.c_int8),  # input_ticked
    ctypes.POINTER(ctypes.c_int8),  # input_valid
)


def _slot_arrays(n):
    """Return (int64 storage array, void* array) where ptr[i] points at &storage[i].

    Each cfunc value slot is an 8-byte cell; the void* array is what the cfunc
    receives, matching cfunc_caller.c's `inputs[i] = &input_storage[i]` layout.
    """
    count = max(n, 1)
    storage = (ctypes.c_int64 * count)()
    ptrs = (ctypes.c_void_p * count)()
    base = ctypes.addressof(storage)
    for i in range(count):
        ptrs[i] = base + i * ctypes.sizeof(ctypes.c_int64)
    return storage, ptrs


class CompiledNode:
    """Drive a compiled ``@numba_node`` cfunc from Python via ctypes.

    Example::

        result = compile_function(dict_contains)
        node = CompiledNode(result, input_types=[int]).start()
        val, ticked = node.execute([1])

    ``input_types`` are the plain Python types (``int``/``float``/``bool``) of
    the Signal inputs, in declaration order. Output types are taken from the
    compilation result. Only single-output (``Signal[T]``) nodes are supported,
    which is all these container tests need.
    """

    def __init__(self, result, input_types):
        self._func = _CFUNC_T(result.compiled_func.address)
        self._input_types = list(input_types)
        self._n_inputs = len(self._input_types)

        named = result.named_outputs
        if named is not None:
            self._output_names = list(named.keys())
            self._output_types = list(named.values())
        else:
            self._output_names = None
            self._output_types = list(result.output_types)
        self._n_outputs = len(self._output_types)

        # ``state_values`` is only present when the node declares state.
        self._n_state = len(getattr(result, "state_values", ()) or ())

        self._in_store, self._inputs = _slot_arrays(self._n_inputs)
        self._out_store, self._outputs = _slot_arrays(self._n_outputs)
        # State cells start zeroed (NULL); container state is allocated on START.
        self._state_store, self._state = _slot_arrays(self._n_state)

        n_in = max(self._n_inputs, 1)
        self._in_ticked = (ctypes.c_int8 * n_in)(*([1] * n_in))
        self._in_valid = (ctypes.c_int8 * n_in)(*([1] * n_in))
        self._out_ticked = (ctypes.c_int8 * max(self._n_outputs, 1))()

    @staticmethod
    def _write(store, idx, py_type, value):
        if py_type is float:
            ctypes.c_double.from_buffer(store, idx * ctypes.sizeof(ctypes.c_int64)).value = float(value)
        else:  # int / bool live in the low bytes of the int64 cell
            store[idx] = int(value)

    @staticmethod
    def _read(store, idx, py_type):
        if py_type is float:
            return ctypes.c_double.from_buffer(store, idx * ctypes.sizeof(ctypes.c_int64)).value
        if py_type is bool:
            return bool(store[idx] & 0xFF)
        return int(store[idx])

    def _call(self, phase):
        self._func(
            self._outputs,
            self._out_ticked,
            self._state,
            ctypes.c_int8(phase),
            self._inputs,
            self._in_ticked,
            self._in_valid,
        )

    def start(self):
        """Run the START lifecycle phase (allocates container state). Returns self."""
        self._call(LIFECYCLE_START)
        return self

    def stop(self):
        """Run the STOP lifecycle phase."""
        self._call(LIFECYCLE_STOP)

    def execute(self, values, ticked=None, valid=None):
        """Run one EXECUTE tick with the given input values; return outputs.

        Returns ``(value, ticked)`` for a single-output node.
        """
        for i, value in enumerate(values):
            self._write(self._in_store, i, self._input_types[i], value)
            self._in_ticked[i] = 1 if ticked is None else int(ticked[i])
            self._in_valid[i] = 1 if valid is None else int(valid[i])
        for i in range(self._n_outputs):
            self._out_ticked[i] = 0

        self._call(LIFECYCLE_EXECUTE)

        values_out = [self._read(self._out_store, i, self._output_types[i]) for i in range(self._n_outputs)]
        ticks_out = [bool(self._out_ticked[i]) for i in range(self._n_outputs)]
        if self._output_names is not None:
            return {name: (values_out[i], ticks_out[i]) for i, name in enumerate(self._output_names)}
        if self._n_outputs == 1:
            return values_out[0], ticks_out[0]
        return list(zip(values_out, ticks_out))
