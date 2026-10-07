# numba cfunc compiler

Extensible compiler for producing native C-callable functions from Python source with stateful variables and Numba typed containers and structs.

[![Build Status](https://github.com/Point72/numba-cfunc-compiler/actions/workflows/build.yaml/badge.svg?branch=main&event=push)](https://github.com/Point72/numba-cfunc-compiler/actions/workflows/build.yaml)
[![codecov](https://codecov.io/gh/Point72/numba-cfunc-compiler/branch/main/graph/badge.svg)](https://codecov.io/gh/Point72/numba-cfunc-compiler)
[![License](https://img.shields.io/github/license/Point72/numba-cfunc-compiler)](https://github.com/Point72/numba-cfunc-compiler)
[![PyPI](https://img.shields.io/pypi/v/numba-cfunc-compiler.svg)](https://pypi.python.org/pypi/numba-cfunc-compiler)

## Overview

A compilation framework that transforms Python functions into native C-callable functions using Numba `@cfunc` and AST rewriting. Unlike using `@cfunc` directly, it provides:

- **Stateful variables** — declare persistent state with natural syntax that survives across calls with automatic lifecycle management (start/execute/stop)
- **Standalone typed containers** — lists and dicts that work inside compiled functions without Numba's runtime overhead
- **Host storage descriptors** — register types that describe annotations and host memory layout
- **Plugin architecture** — domain behavior is registered through input, output, source, AST, and Numba extension APIs

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
from numba_cfunc_compiler.core.context import CompilationContext
from numba_cfunc_compiler.core.defaults import register_all
from numba_cfunc_compiler.core.compile import create_compiled_func
from numba_cfunc_compiler.core.postprocess import CompilationOptions

with CompilationContext() as ctx:
    register_all()                        # built-in types (int, float, datetime, list, dict, struct)
    my_domain_extension.register()        # your custom types/handlers

    result = create_compiled_func(
        func, *args,
        extract_python_type_fn=get_type,
        options=CompilationOptions(fastmath=True, force_inline=True),
    )
    # result.compiled_func        — the Numba cfunc
    # result.native_name          — exported entry-point name (_ncc_numba_<semantic_key>)
    # result.semantic_key         — deterministic hash of transformed code + cfunc signature/options
    # result.llvm_ir              — final LLVM IR after exported-symbol rewrite and optional transforms
    # result.outputs           — ordered mapping: {None: type} or {name: type, ...}
    # result.metadata             — callback component metadata dict
    # result.state_values         — convenience attribute for metadata["state_values"]
    # result.nrt_state_indices / .struct_state_indices / .struct_state_sizes
    # result.constant_container_indices — hidden state slots for constant lists/dicts
```

All registrations are scoped to the active `CompilationContext`. When the `with` block exits, the previous context is restored. If no explicit context exists, `CompilationContext.current()` lazily creates a default with all built-in types.

Enum and enumset ABI layouts come from the registered enum family and host type definitions. The compiler has no process-wide enum width setting.

`CompilationResult` stores callback component metadata in `result.metadata`, and also exposes those keys through attribute access for convenience. Components receive a `ComponentInputs` record containing the input, state, and output analyses plus `extract_python_type_fn`; they return any metadata from `create_variables()` after registering their variables.

`NumbaList` and `NumbaDict` constant parameters are read-only in compiled callbacks. Mutating methods and item assignment fail during compilation, including through aliases and helper calls. The compiler allocates each constant container once per node instance during START, reuses it during EXECUTE, and frees it during STOP. Hosts must allocate the full `state_values` array, including the hidden slots in `constant_container_indices`, and call START once before EXECUTE and STOP once when the instance ends. `State[NumbaList]` and `State[NumbaDict]` remain mutable.

______________________________________________________________________

## Extension API

Every extension point follows the same pattern: define a handler, register it inside a `register()` function, and call that function within an active `CompilationContext`.

### Registering Custom Types

Subclass `HostType` to describe a host annotation and its storage. It is the
single extension class for a value family; the compiler resolves it into an
internal binding once per declared variable. The factory tries registered
classes in order — first match wins. Numba owns the types of user locals,
aliases, and expressions.

```python
from dataclasses import dataclass

from numba_cfunc_compiler.types.factory import HostTypeFactory
from numba_cfunc_compiler.types.base import HostType
from numba_cfunc_compiler.core.analysis import ParameterInfo

@dataclass(frozen=True)
class MyEnumHostType(HostType):
    def get_numba_type_name(self) -> str:
        return 'int64'

    @classmethod
    def is_type_supported(cls, var_type):
        return isinstance(var_type, type) and issubclass(var_type, MyEnumBase)

    # Optional: parse constant inputs like  def f(mode: MyEnum): ...
    @classmethod
    def try_parse_input(cls, param, ann):
        if cls.is_type_supported(ann):
            return ParameterInfo(expected_type=ann)
        return None

    # Optional: parse state declarations like  x: State[MyEnum] = MyEnum.A
    # Return (initial_value, declared_type), or None if not handled.
    @classmethod
    def try_parse_state(cls, node, var_name, globalns):
        ...

    # Required for State[MyEnum]: register its native payload, host layout,
    # copy/borrow semantics, and typed store policy.
    def get_state_payload(self):
        ...

HostTypeFactory.register(MyEnumHostType, priority=0)   # lower priority = tried first
```

`HostType` provides default constant and pointer-read behavior. A family can
override `constant_plan()`, `slot_read()`, or `state_plan()` when it needs
different materialization or lifecycle code. For a statement-level constructor,
implement `lower_local_assignment()` on the same class. The compiler tries
registered type classes in priority order for simple call assignments after
handling state assignments; each class returns `None` for calls it does not
handle. Unsupported declared types fail during resolution; the runtime value
does not silently select another registered type.

### Registering Input Handlers

Input handlers parse function parameter annotations that aren't plain types (e.g., time series signals, baskets).

```python
from enum import Enum, auto

from numba_cfunc_compiler.core.analysis import FunctionAnalyzer, InputTypeHandler
from numba_cfunc_compiler.core.analysis import ParameterInfo

class MyInputCategory(Enum):
    SIGNAL = auto()

class TsInputHandler(InputTypeHandler):
    def try_parse(self, param, ann) -> ParameterInfo | None:
        if is_time_series(ann):
            return ParameterInfo(expected_type=extract_type(ann), category=MyInputCategory.SIGNAL)
        return None

    def validate_value(self, param_name, value, expected_type):
        return value

FunctionAnalyzer.register_input_handler(TsInputHandler())
```

Input categories are enum members. The built-in `InputCategory.CONSTANT` is
used when `ParameterInfo.category` is omitted; host components can define
their own enum members for other input groups.

### Registering Output Handlers

```python
from numba_cfunc_compiler.core.analysis import FunctionAnalyzer, OutputTypeHandler
from numba_cfunc_compiler.core.analysis import OutputAnalysis

class MyOutputHandler(OutputTypeHandler):
    def try_parse(self, return_annotation, ast_tree) -> OutputAnalysis | None:
        if is_my_output(return_annotation):
            return OutputAnalysis(outputs={None: extract_type(return_annotation)})
        return None

FunctionAnalyzer.register_output_handler(MyOutputHandler())
```

For named outputs, return `OutputAnalysis(outputs={"bid": float, "ask": float})`.
Mapping order determines output slot order. `CompilationResult.outputs` exposes
the same mapping so hosts can allocate slots and build named results.

### Registering Callback Components

Callback components define two things:

- which variables get materialized into the generated function
- which leading parameters are injected into the generated `@cfunc` signature

Components are ordered by `order`, and lower values appear earlier in the generated signature. The built-in components registered by `defaults.register_all()` use negative orders, so extension code can safely start at `order = 0` and append its own parameters after the built-in prefix.

```python
from numba_cfunc_compiler.extension.callback_components import (
    CallbackComponent,
    CfuncParam,
    ComponentRegistry,
    MaterializationPhase,
)

class MyComponent(CallbackComponent):
    id = "my.component"
    order = 0
    materialization_phase = MaterializationPhase.EXECUTE

    @property
    def cfunc_params(self):
        return [CfuncParam("my_runtime_ctx", "voidptr")]

    def create_variables(self, inputs, factory):
        ...
        return {}

ComponentRegistry.register(MyComponent())
```

With only the built-in components registered, the generated cfunc signature starts with:

```c
void (*)(void** outputs, int8_t* output_ticked, void** state, int8_t lifecycle_phase, ...)
```

Anything after that prefix is determined by additional components you register. Component orders must be unique; duplicate orders are rejected during registration.

### Registering AST Handlers

AST handlers intercept specific node types during AST transformation. Use `@ast_handler` inside your `register()` function:

```python
from numba_cfunc_compiler.extension.ast import ast_handler, HandlerResult

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

### Extending Value Behavior

Use Numba extension types and `@overload_method` for value attributes and methods. An input has a typed payload and host-defined source metadata. An alias retains its source type, so source methods work on aliases without an AST method dispatcher. Payload methods can be registered with `@binding.method(name)` or the argument-sensitive `@binding.overload_method(name)`. A client value binding declares `semantics=ValueSemantics.COPY` or `semantics=ValueSemantics.BORROWED_VIEW` and the host layout before its descriptor uses `binding.state_payload()`.

The host registers each input source once. The compiler creates a distinct Numba subclass for that descriptor, and the host defines its methods. For example:

```python
from numba import types

from numba_cfunc_compiler import SourceField, register_input_source
from numba_cfunc_compiler.core.variable_access import SourceAccess

active = register_input_source(
    identity=("my_host", "signal.active"),
    fields=(
        SourceField("valid", types.CPointer(types.int8)),
        SourceField("ticked", types.CPointer(types.int8)),
        SourceField("index", types.intp),
    ),
)

@active.method("valid")
def active_valid(signal):
    return signal.source.valid[signal.source.index] != 0

@active.method("ticked")
def active_ticked(signal):
    return signal.source.ticked[signal.source.index] != 0

# Inside a host CallbackComponent.create_variables():
factory.add_variable(
    SourceAccess(
        array_idx=slot,
        type=payload_binding,
        name=name,
        storage_location="inputs",
        source=active,
        metadata={"valid": "input_valid", "ticked": "input_ticked", "index": slot},
    ),
    component=self.id,
)
```

The component declares the corresponding callback parameters with `CfuncParam`; the host runtime passes their arrays for each invocation. A passive source can register only `valid` and `index`, with no ticked pointer or method. The `source` attribute is reserved for metadata: if a struct payload also has a field named `source`, the input wrapper exposes the metadata there. Register each source and its methods before compilation; registrations remain fixed for the lifetime of the process. If a host changes only a method's implementation between runs, it must invalidate its persistent compiled-library cache; the registration key records method names and signatures, not their bodies.

Keyed inputs use `register_keyed_input()` with a separate basket identity, an element source, typed fields including `slots`, `start`, and `length`, and an `element_projection` mapping basket fields to element metadata. `SLOT_INDEX` projects the selected absolute host slot. `basket.iter_method("validkeys", filter=KeyFilter.nonzero("valid"))` registers a filtered iterator. Basket iterators yield zero-based integer positions. Dynamic integer lookup requires homogeneous elements in contiguous host slots; indexes outside the basket length are a caller precondition and are not checked. Literal keys use the host's declared key-to-slot mapping.

The AST handles only syntax that identifies host sources, enum member literals, `EnumSet[E]([...])` constructors, and named outputs. Numba types ordinary expressions, local aliases, struct fields, list and dict operations, and order book methods. Pointer-backed struct fields need explicit typed layout metadata.

### Registering Signal Processors

Signal processors run on each signal variable during function analysis:

```python
from numba_cfunc_compiler.core.function import NumbaFunctionInfo

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
    HostTypeFactory.register(MyType)
    FunctionAnalyzer.register_input_handler(MyInputHandler())
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
- **NumbaList** — Standalone typed list (`int`/`float`/`bool` elements). Supports `len`, indexing, `append`, `pop`, `clear`, iteration. Initialize state with `values: State[NumbaList] = create_new_list(int)`.
- **NumbaArray** — Zero-initialized fixed array of `int`/`float`/`bool`. Supports `len`, indexing, assignment, and iteration. Initialize state with `values: State[NumbaArray] = create_new_array(int, 8)`, or pass a read-only constant input annotated `NumbaArray[int, 8]`. Length must be a positive compile-time integer. Invalid dynamic indices terminate the callback process.
- **NumbaDict** — Standalone typed dict (`int` keys, `int`/`float`/`bool` values). Supports `len`, `[]`, `in`/`not in`, `get`, `pop`, `clear`, `items()`, `keys()`. Initialize state with `values: State[NumbaDict] = create_new_dict(int, float)`.
- **Structs** — Opaque void pointers with field metadata. Read/write fields via pointer arithmetic. Base `StructHostType` must be subclassed with `is_type_supported()`, `get_struct_fields()`, `get_struct_size()`.

The list, dict, struct, and enum families live under `types/builtin/<family>/`. `host.py` describes annotations and host storage, `native.py` defines Numba types and lowering, and `register.py` holds the family's default registrations where needed. Shared input, output, and state value machinery remains under `types/native/`.

## State Assignments

The AST converter lowers assignments to a declared `State[T]` name into typed
store calls. Each store writes its host cell immediately and rebinds the name
to the registered native value type. `State[int]`, `State[float]`, `State[bool]`,
`State[datetime]`, and `State[timedelta]` have copied values: an alias retains
the snapshot it took before a later state assignment. Registered struct states
have borrowed views: field writes and whole-object replacement update the
original host cell, so earlier aliases see the new fields. Standalone list and
dict states have borrowed views and support item mutation, but reject
whole-object replacement at compile time. Borrowed host storage must remain
valid throughout the callback.

Direct and augmented assignments to a state name are supported, including
inside branches and loops. Chained assignment, unpacking, and `for` loop
targets involving state names are rejected. Named expressions are unsupported
throughout node code. `del state_name` is rejected. An alias is an ordinary
local and never becomes a state store target; Numba types ordinary local
reassignments. Invalid state stores report the declared name and RHS Numba
type.

For the generated code, supported syntax, and copy versus borrowed alias
behavior, see [How state stores work](docs/current-architecture.md#5-how-state-stores-work).

## Compilation Options & Post-Compilation Transforms

`CompilationOptions` controls compilation flags and post-compilation LLVM IR transforms. Pass it to `create_compiled_func` via the `options` parameter — all flags are opt-in.

```python
from numba_cfunc_compiler.core.compile import create_compiled_func
from numba_cfunc_compiler.core.postprocess import CompilationOptions

opts = CompilationOptions(
    fastmath=True,          # enable fast-math in @cfunc
    force_inline=True,      # noinline → alwaysinline on cfunc wrapper
)
result = create_compiled_func(func, *args, options=opts, ...)
```

### Flags

**`fastmath`** *(compilation)* — Passes `fastmath=True` to the Numba `@cfunc` decorator, enabling aggressive floating-point optimizations (reassociation, no-NaN, etc.).

**`force_inline`** *(post-compilation)* — Replaces Numba's `noinline` attribute on the cfunc wrapper with `alwaysinline`, letting the LLVM optimizer inline the function body into the wrapper and eliminate the extra call.

Regardless of options, the exported wrapper symbol is renamed to `_ncc_numba_<semantic_key>`. `result.native_name`, `result.semantic_key`, and `result.llvm_ir` all reflect that final compiled form.

### FFI Optimization

Native FFI references use nominal Numba pointer types. Each registered method declares an exported symbol and exact signature; Numba emits a direct call with `nounwind`. Native iterator types support order book level and order iteration. `link_ffi_bitcode` can inline accessor bodies during module linking.

### Module-Level FFI Bitcode Inlining

For applications that compile FFI accessor functions (e.g. order-book price/quantity readers) to LLVM bitcode at build time, `link_ffi_bitcode` can link those bodies into the LLVM module before optimization, allowing the inliner to replace function calls with single-load instructions:

```python
from numba_cfunc_compiler.core.postprocess import link_ffi_bitcode

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
  → VariableFactory (create variable access objects for each param/state/output)
  → NumbaASTConverter (rewrite AST via registered handlers)
  → SHA-256 hash of generated source, signature, and native bindings (semantic_key)
  → Numba @cfunc with typed state store calls in the generated source
  → Post-compilation IR updates (optional force_inline, then exported symbol rename)
  → CompilationResult (cfunc + metadata + semantic_key + final native_name/llvm_ir)
```

### CompilationContext

Host type, callback component, AST syntax adapter, and function analysis registrations live on a `CompilationContext` instance (backed by `contextvars.ContextVar`). Numba extension registrations are process-wide. Re-registering the same method object is idempotent; replacing a method within the process is rejected. The compiled semantic key includes native layouts, input semantics, method names and signatures, and FFI symbol signatures. It does not hash method bodies, so artifacts must be invalidated if those bodies change across processes.

### Variable Access

The `VariableAccess` hierarchy abstracts where a variable lives. The AST converter calls `var.get()` without knowing the storage mechanism:

- `PointerSlotAccess` — reads from an external `void*` array such as `inputs` or `state`
- `SourceAccess` — binds an input slot and host-declared source metadata
- `KeyedInputAccess` — binds a keyed basket and its element source
- `ConstantAccess` — materializes compile-time constants, including container constants
- `OutputAccess` — creates a typed output sink that writes `outputs[i]` and sets `output_ticked[i]`

Signal-style inputs are modeled by custom callback components that create `SourceAccess` instances with framework-specific metadata.

### AST Transformation

`NumbaASTConverter` is an `ast.NodeTransformer`. Each `visit_*` method is wrapped with `@with_handlers(node_type)` which runs registered pre/post handlers around the default logic. Key transformations:

- `visit_FunctionDef` — replaces args with the fixed cfunc parameter list, injects lifecycle dispatch
- `visit_Return` — converts `return value` to typed output sink writes and ticks
- `visit_Assign` — handles non-state source stores and syntax-specific constructors
- `visit_Name` — reads non-state host sources; state and local names pass through to Numba
- enum AST handlers in `types/builtin/enum/register.py` — lower registered enum member literals and `EnumSet[E]([...])` constructors

### Lifecycle Dispatch

Generated code wraps user logic in lifecycle checks:

```python
def compiled_func(outputs, output_ticked, state, lifecycle_phase, ...):
    state_slot = state[0]
    state_var = bind_state(state_slot)
    if lifecycle_phase == 1:   # START — container init, start hooks
        ...
    if lifecycle_phase == 2:   # STOP — cleanup hooks
        ...
    if lifecycle_phase == 0:   # EXECUTE — input loading, user logic, output
        ...
```

### Standalone Containers

`NumbaList` and `NumbaDict` use standalone C implementations (`runtime/_py_nrt_init.so`) instead of Numba's reference-counted runtime. State container memory is owned by the host framework via state slots. The C library is loaded lazily on first compilation via `CompilationContext.ensure_nrt_loaded()`.

`create_new_list()` and `create_new_dict()` are supported only as direct initializers of matching `State[...]` declarations. State containers are freed on STOP. Assigning a state or input container to a local alias is supported; creating a new container in a local expression is rejected at compile time.
