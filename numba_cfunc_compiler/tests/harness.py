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
