import ast
import hashlib
import inspect
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numba
from numba import cfunc, float64, int8, int64
from numba.types import CPointer

from numba_cfunc_compiler.core.context import CompilationContext
from numba_cfunc_compiler.core.converter import NumbaASTConverter
from numba_cfunc_compiler.core.function import NumbaFunctionInfo, validate_user_identifiers
from numba_cfunc_compiler.core.names import (
    UNWRAP_SOURCE_NAME,
    bind_input_name,
    bind_output_name,
    bind_state_name,
    state_store_name,
)
from numba_cfunc_compiler.core.postprocess import (
    CompilationOptions,
    apply_post_compilation,
)
from numba_cfunc_compiler.extension.callback_components import ComponentRegistry
from numba_cfunc_compiler.types.builtin.array.native import standalone_array_new, standalone_array_readonly
from numba_cfunc_compiler.types.builtin.dict.native import (
    standalone_dict_free,
    standalone_dict_from_voidptr,
    standalone_dict_length,
    standalone_dict_new,
    standalone_dict_to_voidptr,
)
from numba_cfunc_compiler.types.builtin.list.native import (
    standalone_list_free,
    standalone_list_from_voidptr,
    standalone_list_new,
    standalone_list_to_voidptr,
)
from numba_cfunc_compiler.types.native.pointers import cast_voidptr_to_int, cast_voidptr_to_ptr, voidptr_null

# Bump when a compiler implementation change can alter generated native code
# without changing the generated source or the binding/layout fingerprints.
COMPILER_IMPLEMENTATION_VERSION = 2

__all__ = [
    "CompilationResult",
    "create_compiled_func",
]


@dataclass
class CompilationResult:
    """Result of create_compiled_func — contains the compiled cfunc and all metadata
    needed by the host framework to wire the node.

    Core fields are always present. Component-specific metadata (e.g.
    ``ordered_input_signals``, ``nrt_state_indices``) lives in the
    ``metadata`` dict, keyed by the strings documented in each built-in
    :class:`CallbackComponent`.
    """

    compiled_func: Any
    outputs: dict[str | None, Any]
    native_name: str = ""
    semantic_key: str = ""
    llvm_ir: str = ""
    metadata: dict = field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        """Allow attribute access to metadata keys (e.g. result.state_values)."""
        metadata = self.__dict__.get("metadata")
        if metadata and name in metadata:
            return metadata[name]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")


def build_semantic_key(
    new_func_code: str,
    cfunc_sig: str,
    cfunc_kwargs: str,
    typed_struct_layouts: dict | None = None,
    state_bindings_key: tuple = (),
    input_bindings_key: tuple = (),
    output_bindings_key: tuple = (),
    keyed_input_bindings_key: tuple = (),
    method_bindings_key: tuple = (),
) -> str:
    payload = {
        "compiler_implementation_version": COMPILER_IMPLEMENTATION_VERSION,
        "new_func_code": new_func_code,
        "cfunc_sig": cfunc_sig,
        "cfunc_kwargs": cfunc_kwargs,
    }
    if typed_struct_layouts:
        payload["typed_structs"] = [layout.key for _, layout in sorted(typed_struct_layouts.items())]
    payload["state_bindings"] = state_bindings_key
    if input_bindings_key:
        payload["input_bindings"] = input_bindings_key
    if output_bindings_key:
        payload["output_bindings"] = output_bindings_key
    if keyed_input_bindings_key:
        payload["keyed_input_bindings"] = keyed_input_bindings_key
    if method_bindings_key:
        payload["method_bindings"] = method_bindings_key
    payload["compiler_pipeline"] = {
        "numba": numba.__version__,
        "python": sys.version_info[:2],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def create_compiled_func(
    func,
    *args,
    extract_python_type_fn: Callable[[Any], type],
    decorator_name: str = "@numba_node",
    func_globals: dict | None = None,
    signature: inspect.Signature | None = None,
    call_globals: dict | None = None,
    start_body: list[ast.AST] | None = None,
    stop_body: list[ast.AST] | None = None,
    options: CompilationOptions | None = None,
    **kwargs,
) -> CompilationResult:
    """
    Analyze and compile a function to a Numba cfunc.

    This is the main entry point for compilation.  It analyzes the function
    signature (inputs, outputs, state), builds a VariableFactory, transforms
    the AST, and compiles to a native cfunc.

    Args:
        func: The function to compile — either a callable or an ast.FunctionDef.
        *args, **kwargs: Runtime argument values (edges, constants) bound to the signature.
        extract_python_type_fn: Extracts a Python type from an input object.
        decorator_name: Decorator name for AST extraction (default '@numba_node').
        func_globals: Globals dict (required when func is an AST node).
        signature: Pre-computed inspect.Signature (required when func is an AST node).
        call_globals: Extra globals available during Numba compilation (e.g., enum classes).
        start_body: AST statements for the START lifecycle phase.
        stop_body: AST statements for the STOP lifecycle phase.
        options: Optional CompilationOptions controlling compilation flags
            (e.g. fastmath) and post-compilation IR transforms
            (e.g. force_inline).

    Returns:
        CompilationResult with the compiled cfunc and all wiring metadata.
    """
    opts = options or CompilationOptions()
    info = NumbaFunctionInfo(
        func,
        *args,
        extract_python_type_fn=extract_python_type_fn,
        decorator_name=decorator_name,
        func_globals=func_globals,
        call_globals=call_globals,
        signature=signature,
        **kwargs,
    )

    tree = info.tree
    name = info.name
    variable_factory = info.variable_factory

    from numba_cfunc_compiler.core.variable_access import OutputAccess, SourceAccess
    from numba_cfunc_compiler.extension.callback_components import CallbackComponentId

    validate_user_identifiers(tree, info.sig, variable_factory.variable_by_name, start_body or [], stop_body or [])
    state_vars = variable_factory.get_by_component(CallbackComponentId.STATE)
    state_payloads = {var.name: var.type.payload for var in state_vars}
    input_bindings = tuple((var.name, var.input_value_type()) for var in variable_factory.get_accesses(SourceAccess))
    output_bindings = tuple((var.array_idx, var.output_sink_type()) for var in variable_factory.get_accesses(OutputAccess))
    state_bindings_key = tuple((var.name, var.array_idx, state_payloads[var.name].key) for var in state_vars)

    # Lazy-load the NRT C library on first compilation
    CompilationContext.current().ensure_nrt_loaded()

    transformer = NumbaASTConverter(
        tree,
        variable_factory,
        start_body=start_body,
        stop_body=stop_body,
        call_globals=info.call_globals,
        host_globals=info.func_globals,
        state_names=frozenset(state_payloads),
    )
    new_tree = transformer.visit(tree)
    new_func_code = ast.unparse(new_tree)
    keyed_input_bindings = tuple(sorted(variable_factory.typed_input_bindings.items()))
    keyed_basket_bindings = tuple(sorted(variable_factory.typed_keyed_bindings.items()))
    source_descriptors = {value_type.source_descriptor for _, value_type in (*input_bindings, *keyed_input_bindings)}
    basket_descriptors = {basket_type.descriptor for _, basket_type in keyed_basket_bindings}
    for basket in basket_descriptors:
        source_descriptors.add(basket.element_source)
    source_registrations = tuple(sorted((source.registration_key(freeze=True) for source in source_descriptors), key=lambda item: item[0][0]))
    basket_registrations = tuple(sorted((basket.registration_key(freeze=True) for basket in basket_descriptors), key=lambda item: item[0][0]))
    from numba_cfunc_compiler.extension.ffi import ffi_binding_fingerprint
    from numba_cfunc_compiler.extension.methods import method_registration_key

    method_types = set()
    for payload in state_payloads.values():
        native = payload.native_type
        method_types.add(getattr(native, "payload_type", native))
    for _, value_type in (*input_bindings, *keyed_input_bindings):
        method_types.add(value_type.payload_type)
    for _, basket_type in keyed_basket_bindings:
        method_types.update(item_type.payload_type for _, _, item_type in basket_type.layout_key)
    for _, sink_type in output_bindings:
        method_types.add(sink_type.payload_type)
    method_bindings_key = tuple(
        sorted((native.name, method_registration_key(native, freeze=True), ffi_binding_fingerprint(native)) for native in method_types)
    )

    cfunc_sig = ComponentRegistry.build_cfunc_signature()
    cfunc_kwargs = "nopython=True, nogil=True, _nrt=False, error_model='numpy'"
    if opts.fastmath:
        cfunc_kwargs += ", fastmath=True"
    semantic_key = build_semantic_key(
        new_func_code,
        cfunc_sig,
        cfunc_kwargs,
        variable_factory.typed_struct_layouts,
        state_bindings_key,
        (tuple((name, value_type.name) for name, value_type in input_bindings), source_registrations),
        tuple((idx, sink_type.name) for idx, sink_type in output_bindings),
        (tuple((name, value_type.name) for name, value_type in (*keyed_input_bindings, *keyed_basket_bindings)), basket_registrations),
        method_bindings_key,
    )
    cfunc_code = f"""
@cfunc({cfunc_sig}, {cfunc_kwargs})
{new_func_code}
"""

    exec_globals = {}
    exec_globals.update(globals())
    exec_globals.update(transformer.call_globals)

    exec_globals.update(
        {
            "cfunc": cfunc,
            "CPointer": CPointer,
            "int64": int64,
            "int8": int8,
            "float64": float64,
            "voidptr": numba.types.voidptr,
            "cast_voidptr_to_ptr": cast_voidptr_to_ptr,
            "voidptr_null": voidptr_null,
            "cast_voidptr_to_int": cast_voidptr_to_int,
            "standalone_array_new": standalone_array_new,
            "standalone_array_readonly": standalone_array_readonly,
            # standalone list (NRT-free)
            "standalone_list_new": standalone_list_new,
            "standalone_list_from_voidptr": standalone_list_from_voidptr,
            "standalone_list_free": standalone_list_free,
            "standalone_list_to_voidptr": standalone_list_to_voidptr,
            # standalone dict (NRT-free)
            "standalone_dict_new": standalone_dict_new,
            "standalone_dict_from_voidptr": standalone_dict_from_voidptr,
            "standalone_dict_free": standalone_dict_free,
            "standalone_dict_to_voidptr": standalone_dict_to_voidptr,
            "standalone_dict_length": standalone_dict_length,
        }
    )
    exec_globals.update(variable_factory.typed_struct_bindings)
    from numba_cfunc_compiler.types.native.input import bind_input, unwrap_source

    exec_globals.update({bind_input_name(name): bind_input(value_type) for name, value_type in input_bindings})
    exec_globals.update({name: bind_input(value_type) for name, value_type in keyed_input_bindings})
    from numba_cfunc_compiler.types.native.keyed import bind_keyed

    exec_globals.update({name: bind_keyed(value_type) for name, value_type in keyed_basket_bindings})
    exec_globals[UNWRAP_SOURCE_NAME] = unwrap_source
    from numba_cfunc_compiler.types.native.output import bind_output

    exec_globals.update({bind_output_name(idx): bind_output(sink_type) for idx, sink_type in output_bindings})
    exec_globals.update({bind_state_name(name): payload.bind_function for name, payload in state_payloads.items()})
    exec_globals.update({state_store_name(name): payload.store_function(name) for name, payload in state_payloads.items()})
    exec(cfunc_code, exec_globals)  # noqa: S102 - generated function source

    compiled_func = exec_globals[name]

    # --- post-compilation IR transforms ---
    result_ir, exported_entry_point = apply_post_compilation(compiled_func, semantic_key, opts)

    return CompilationResult(
        compiled_func=compiled_func,
        outputs=dict(info.output_analysis.outputs),
        native_name=exported_entry_point,
        semantic_key=semantic_key,
        llvm_ir=result_ir,
        metadata=info.component_metadata,
    )
