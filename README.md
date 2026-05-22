# numba cfunc compiler

Extensible compiler for producing native C-callable functions from Python source with stateful variables, typed containers, and pluggable type systems built on Numba @cfunc

[![Build Status](https://github.com/Point72/numba-cfunc-compiler/actions/workflows/build.yaml/badge.svg?branch=main&event=push)](https://github.com/Point72/numba-cfunc-compiler/actions/workflows/build.yaml)
[![codecov](https://codecov.io/gh/Point72/numba-cfunc-compiler/branch/main/graph/badge.svg)](https://codecov.io/gh/Point72/numba-cfunc-compiler)
[![License](https://img.shields.io/github/license/Point72/numba-cfunc-compiler)](https://github.com/Point72/numba-cfunc-compiler)
[![PyPI](https://img.shields.io/pypi/v/numba-cfunc-compiler.svg)](https://pypi.python.org/pypi/numba-cfunc-compiler)

## Overview

A compilation framework that transforms Python functions into native C-callable functions using Numba `@cfunc` and AST rewriting. Unlike using `@cfunc` directly, it provides:

- **Stateful variables** — declare persistent state with natural syntax that survives across calls with automatic lifecycle management (start/execute/stop)
- **Standalone typed containers** — lists and dicts that work inside compiled functions without Numba's runtime overhead
- **Extensible type resolution** — register custom types that map Python syntax to numba-compatible lowering 
- **Plugin architecture** — all domain-specific behavior (input handlers, output handlers, AST transforms, type inference) is injected through registration APIs

```python
@numba_node
def ema(x: Signal[float], alpha: float) -> Signal[float]:
    s: State[float] = 0.0                    # persistent state, initialized once
    s = alpha * x + (1.0 - alpha) * s        # natural Python syntax
    return s                                  # compiled to native code
```

The distribution is published as `numba-cfunc-compiler` and imported in Python as `numba_cfunc_compiler`.

## Installation

Install from PyPI:

```
pip install numba-cfunc-compiler
```

Install from the conda channel:

```
conda install -c conda-forge numba-cfunc-compiler
```

## Quick Start

```python
from numba_cfunc_compiler.compilation_context import CompilationContext
from numba_cfunc_compiler.defaults import register_all
from numba_cfunc_compiler.numba_core import CompilationOptions, create_compiled_func

with CompilationContext() as ctx:
    register_all()                        # built-in types (int, float, datetime, list, dict, struct)
    my_domain_extension.register()        # your custom types/handlers

    result = create_compiled_func(
        func, *args,
        extract_python_type_fn=get_type,
        options=CompilationOptions(fastmath=True, force_inline=True),
    )
    # result.compiled_func        — the Numba cfunc
    # result.native_name          — exported entry-point name (_gc_numba_<semantic_key>)
    # result.semantic_key         — deterministic hash of transformed code + cfunc signature/options
    # result.llvm_ir              — final LLVM IR after exported-symbol rewrite and optional transforms
    # result.output_types / .named_outputs
    # result.metadata             — category-specific metadata dict
    # result.state_values         — convenience attribute for metadata["state_values"]
    # result.nrt_state_indices / .struct_state_indices / .struct_state_sizes
```

All registrations are scoped to the active `CompilationContext`. When the `with` block exits, the previous context is restored. If no explicit context exists, `CompilationContext.current()` lazily creates a default with all built-in types.

`CompilationResult` stores category-specific metadata in `result.metadata`, and also exposes those keys through attribute access for convenience. Custom source categories can add fields such as `ordered_input_signals` by implementing `get_result_metadata()`.

______________________________________________________________________

## Extension API

Every extension point follows the same pattern: define a handler, register it inside a `register()` function, and call that function within an active `CompilationContext`.

### Registering Custom Types

Subclass `VariableType` to teach the system about a new type. The factory tries registered classes in order — first match wins.

```python
from numba_cfunc_compiler.type_factory import TypeFactory
from numba_cfunc_compiler.models import VariableType, ParameterInfo, StateVariableInfo

@dataclass(frozen=True)
class MyEnumType(VariableType):
    def get_numba_type_name(self) -> str:
        return 'int64'

    @classmethod
    def is_type_supported(cls, var_type):
        return isinstance(var_type, type) and issubclass(var_type, MyEnumBase)

    @classmethod
    def from_type(cls, var_type, value):
        if cls.is_type_supported(var_type):
            return cls(var_type, value)
        return None

    # Optional: parse constant inputs like  def f(mode: MyEnum): ...
    @classmethod
    def try_parse_input(cls, param, ann):
        if cls.is_type_supported(ann):
            return ParameterInfo(expected_type=ann)
        return None

    # Optional: parse state declarations like  x: State[MyEnum] = MyEnum.A
    @classmethod
    def try_parse_state(cls, node, var_name, globalns):
        ...

    # Optional: lower assignments like  x = MyEnum.VALUE → x = <int>
    @classmethod
    def try_lower_assignment(cls, node, rhs, call_globals):
        ...

TypeFactory.register(MyEnumType, priority=0)   # lower priority = tried first
```

### Registering Input Handlers

Input handlers parse function parameter annotations that aren't plain types (e.g., time series signals, baskets).

```python
from numba_cfunc_compiler.function_analyzer import FunctionAnalyzer, InputTypeHandler
from numba_cfunc_compiler.models import ParameterInfo

class TsInputHandler(InputTypeHandler):
    def try_parse(self, param, ann) -> ParameterInfo | None:
        if is_time_series(ann):
            return ParameterInfo(expected_type=extract_type(ann), category="signal")
        return None

    def validate_value(self, param_name, value, expected_type):
        return value

FunctionAnalyzer.register_input_handler(TsInputHandler())
```

### Registering Output Handlers

```python
from numba_cfunc_compiler.function_analyzer import FunctionAnalyzer, OutputTypeHandler
from numba_cfunc_compiler.models import OutputAnalysis

class MyOutputHandler(OutputTypeHandler):
    def try_parse(self, return_annotation, ast_tree) -> OutputAnalysis | None:
        if is_my_output(return_annotation):
            return OutputAnalysis(output_types=[extract_type(return_annotation)])
        return None

FunctionAnalyzer.register_output_handler(MyOutputHandler())
```

### Registering Source Categories

Source categories define two things:

- which variables get materialized into the generated function
- which leading parameters are injected into the generated `@cfunc` signature

Categories are ordered by `order`, and lower values appear earlier in the generated signature. The built-in categories registered by `defaults.register_all()` use negative orders, so extension code can safely start at `order = 0` and append its own parameters after the built-in prefix.

```python
from numba_cfunc_compiler.source_registry import (
    CfuncParam,
    SourceCategory,
    SourceInitFilter,
    SourceRegistry,
)

class MyCategory(SourceCategory):
    id = "my.category"
    order = 0
    init_filter = SourceInitFilter.ON_EXECUTE

    @property
    def cfunc_params(self):
        return [CfuncParam("my_runtime_ctx", "voidptr")]

    def create_variables(self, info, factory):
        ...

SourceRegistry.register(MyCategory())
```

With only the built-in default categories registered, the generated cfunc signature starts with:

```c
void (*)(void** outputs, int8_t* output_ticked, void** state, int8_t lifecycle_phase, ...)
```

Anything after that prefix is determined by any additional source categories you register. Category orders must be unique; duplicate orders are rejected during registration.

### Registering AST Handlers

AST handlers intercept specific node types during AST transformation. Use `@ast_handler` inside your `register()` function:

```python
from numba_cfunc_compiler.ast_handlers import ast_handler, HandlerResult

def register():
    # Pre-handler: runs BEFORE default visit_Call logic.
    # Return non-None to short-circuit; None to pass through.
    @ast_handler('Call', pre=True, priority=5)
    def handle_my_call(converter, node):
        if is_my_special_call(node):
            return transformed_node
        return None

    # Post-handler: runs AFTER default logic, receives the result.
    @ast_handler('Assign', post=True)
    def tweak_assignment(converter, original_node, result):
        return result  # or modify it

    # HandlerResult lets you inject side-effect statements
    @ast_handler('For', pre=True)
    def handle_my_loop(converter, node):
        if is_my_pattern(node):
            setup = generate_setup()
            loop = generate_loop(node)
            return HandlerResult(node=loop, side_effects=[setup])
        return None
```

Supported node types: `Call`, `Expr`, `Return`, `Assign`, `AugAssign`, `Subscript`, `Attribute`, `Compare`, `For`, `Name`.

Priority: lower values run first. Pre-handlers short-circuit (first non-`None` wins). Post-handlers chain.

### Registering Type Inference Handlers

Type inference resolves method call chains (`a.method().field`) during AST transformation.

```python
from numba_cfunc_compiler.numba_type_inference import NumbaTypeInference
from numba_cfunc_compiler.variable_factory import ExpressionSource

# Handle attribute access:  my_var.some_attr
def my_attr_handler(inference, base_var, attr_name, args):
    if isinstance(base_var.type, MyCustomType):
        field_ast = generate_access(base_var, attr_name)
        return ExpressionSource(resolve_type(attr_name), field_ast, inference.variable_factory)
    return None

NumbaTypeInference.register_attr_accessor(my_attr_handler)

# Handle method calls:  my_var.transform(x)
def my_call_handler(inference, base_var, method_name, args):
    if isinstance(base_var.type, MyCustomType) and method_name == 'transform':
        ...
    return None

NumbaTypeInference.register_call_handler(my_call_handler)

# Handle attribute nodes:  MyEnum.VALUE → constant
def my_attribute_transformer(node, globalns, variable_factory):
    ...  # return transformed AST or None

NumbaTypeInference.register_attr_lowerer(my_attribute_transformer)
```

### Registering Signal Processors

Signal processors run on each signal variable during function analysis:

```python
from numba_cfunc_compiler.numba_core import NumbaFunctionInfo

def my_processor(variable, signal_obj, function_info):
    if hasattr(signal_obj, 'custom_attr'):
        function_info.custom_data = signal_obj.custom_attr

NumbaFunctionInfo.register_signal_processor(my_processor)
```

### Putting It All Together

Bundle all registrations into a `register()` function — no side effects on import:

```python
# my_extension.py
def register():
    """Register into the current CompilationContext."""
    TypeFactory.register(MyType)
    FunctionAnalyzer.register_input_handler(MyInputHandler())
    NumbaTypeInference.register_attr_accessor(my_attr_handler)

    @ast_handler('Call', pre=True)
    def handle_my_call(converter, node):
        ...
```

Then use it:

```python
with CompilationContext() as ctx:
    register_all()           # built-in defaults
    my_extension.register()  # your domain types
    # ... compile ...
```

## Built-in Types

Registered by `defaults.register_all()`:

- **Primitives** — `int` (int64), `float` (float64), `bool` (int8). Support `State[int]`, etc.
- **DateTime** — Stored as nanoseconds (int64). Constructor calls like `datetime(2020,1,1,tzinfo=timezone.utc)` are lowered to constants at compile time. Must be timezone-aware.
- **TimeDelta** — Stored as nanoseconds (int64). `timedelta(seconds=5)` lowered to constants.
- **NumbaList** — Standalone typed list (`int`/`float`/`bool` elements). Supports `len`, indexing, `append`, `pop`, `clear`, iteration. Created with `create_new_list(int)`.
- **NumbaDict** — Standalone typed dict (`int` keys, `int`/`float`/`bool` values). Supports `len`, `[]`, `in`/`not in`, `get`, `pop`, `clear`, `items()`, `keys()`. Created with `create_new_dict(int, float)`.
- **Structs** — Opaque void pointers with field metadata. Read/write fields via pointer arithmetic. Base `StructType` must be subclassed with `is_type_supported()`, `_get_struct_fields()`, `_get_struct_size()`.

## Compilation Options & Post-Compilation Transforms

`CompilationOptions` controls compilation flags and post-compilation LLVM IR transforms. Pass it to `create_compiled_func` via the `options` parameter — all flags are opt-in.

```python
from numba_cfunc_compiler.numba_core import CompilationOptions, create_compiled_func

opts = CompilationOptions(
    fastmath=True,          # enable fast-math in @cfunc
    force_inline=True,      # noinline → alwaysinline on cfunc wrapper
)
result = create_compiled_func(func, *args, options=opts, ...)
```

### Flags

**`fastmath`** *(compilation)* — Passes `fastmath=True` to the Numba `@cfunc` decorator, enabling aggressive floating-point optimizations (reassociation, no-NaN, etc.).

**`force_inline`** *(post-compilation)* — Replaces Numba's `noinline` attribute on the cfunc wrapper with `alwaysinline`, letting the LLVM optimizer inline the function body into the wrapper and eliminate the extra call.

Regardless of options, the exported wrapper symbol is renamed to `_gc_numba_<semantic_key>`. `result.native_name`, `result.semantic_key`, and `result.llvm_ir` all reflect that final compiled form.

### FFI Optimization

FFI function declarations are automatically annotated with `readonly` and `nounwind` attributes (in `utils/ffi.py`), enabling LLVM to perform CSE (common subexpression elimination) and LICM (loop-invariant code motion) on FFI calls.

### Module-Level FFI Bitcode Inlining

For applications that compile FFI accessor functions (e.g. order-book price/quantity readers) to LLVM bitcode at build time, `link_ffi_bitcode` can link those bodies into the LLVM module before optimization, allowing the inliner to replace function calls with single-load instructions:

```python
from numba_cfunc_compiler.post_compilation import link_ffi_bitcode

# At module-link time (once, not per-function):
with open('numba_c_interface.bc', 'rb') as f:
    bitcode = f.read()
module = link_ffi_bitcode(module, bitcode)
# Now run the LLVM optimizer — FFI calls will be inlined.
```

`link_ffi_bitcode` handles linking, patching `alwaysinline`, and stripping `target-cpu` / `target-features` to prevent inlining mismatches. Falls back gracefully to external calls on error.

## Implementation Notes

### Compilation Flow

```
Python Function
  → CompilationContext (holds all registries)
  → FunctionAnalyzer (parse params, state, outputs from annotations)
  → VariableFactory (create variable sources for each param/state/output)
  → NumbaASTConverter (rewrite AST via registered handlers)
  → SHA-256 hash of transformed Python source + cfunc signature/options (semantic_key)
  → Numba @cfunc (NRT library lazy-loaded; fastmath applied if enabled)
  → Post-compilation IR updates (optional force_inline, then exported symbol rename)
  → CompilationResult (cfunc + metadata + semantic_key + final native_name/llvm_ir)
```

### CompilationContext

All mutable state lives on a `CompilationContext` instance (backed by `contextvars.ContextVar`). This replaced class-level mutable dicts/lists on `TypeFactory`, `ASTHandlerRegistry`, `NumbaTypeInference`, `FunctionAnalyzer`, `FFIMethodHelper`, and `NumbaFunctionInfo`. Benefits: test isolation, concurrent safety, explicit initialization.

### Variable Sources

The `VariableSource` hierarchy abstracts where a variable lives. The AST converter calls `var.get()` without knowing the storage mechanism:

- `VoidPtrSource` — reads from an external `void*` array such as `inputs` or `state`
- `ConstantSource` — materializes compile-time constants, including container constants
- `OutputSource` — writes to `outputs[i]` and marks `output_ticked[i] = 1`
- `LocalVariableSource` — tracks locals introduced during AST rewriting
- `ExpressionSource` — carries type information for expression-backed values
- `LocalConstantSource` — represents local constant values created during lowering

Signal-style inputs are typically modeled by custom source categories that create `VoidPtrSource` instances with framework-specific metadata.

### AST Transformation

`NumbaASTConverter` is an `ast.NodeTransformer`. Each `visit_*` method is wrapped with `@with_handlers(node_type)` which runs registered pre/post handlers around the default logic. Key transformations:

- `visit_FunctionDef` — replaces args with the fixed cfunc parameter list, injects lifecycle dispatch
- `visit_Return` — converts `return value` to output pointer writes + tick marks
- `visit_Assign` — handles struct fields, managed variables, type-lowered assignments
- `visit_Name` — resolves managed variable names to their AST representation
- `visit_Call` — dispatches method calls through the type inference engine

### Lifecycle Dispatch

Generated code wraps user logic in lifecycle checks:

```python
def compiled_func(outputs, output_ticked, state, lifecycle_phase, ...):
    state_var = cast_voidptr_to_ptr(state[0], 'int64')  # always available
    if lifecycle_phase == 1:   # START — container init, start hooks
        ...
    if lifecycle_phase == 2:   # STOP — cleanup hooks
        ...
    if lifecycle_phase == 0:   # EXECUTE — input loading, user logic, output
        ...
```

### Standalone Containers

`NumbaList` and `NumbaDict` use standalone C implementations (`numba_rt/_py_nrt_init.so`) instead of Numba's reference-counted runtime. Memory is owned by the host framework via state slots. The C library is loaded lazily on first compilation via `CompilationContext.ensure_nrt_loaded()`.
