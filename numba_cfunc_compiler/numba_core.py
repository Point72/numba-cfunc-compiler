import ast
import hashlib
import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numba
from numba import cfunc, float64, int8, int64
from numba.types import CPointer

from numba_cfunc_compiler.compilation_context import CompilationContext
from numba_cfunc_compiler.config import get_numba_config
from numba_cfunc_compiler.defaults.struct_support import StructType
from numba_cfunc_compiler.function_analyzer import FunctionAnalyzer
from numba_cfunc_compiler.models import ContainerType, VariableType
from numba_cfunc_compiler.numba_ast_converter import NumbaASTConverter
from numba_cfunc_compiler.post_compilation import (
    CompilationOptions,
    apply_post_compilation,
)
from numba_cfunc_compiler.source_registry import SourceRegistry
from numba_cfunc_compiler.standalone.dict import (
    _standalone_dict_iter_begin,
    _standalone_dict_iter_next_item,
    _standalone_dict_iter_next_key,
    standalone_dict_clear,
    standalone_dict_free,
    standalone_dict_from_voidptr,
    standalone_dict_length,
    standalone_dict_new,
    standalone_dict_to_voidptr,
)
from numba_cfunc_compiler.standalone.list import (
    standalone_list_free,
    standalone_list_from_voidptr,
    standalone_list_new,
    standalone_list_reset_size,
    standalone_list_to_voidptr,
)
from numba_cfunc_compiler.utils.ast import AST
from numba_cfunc_compiler.utils.enum import make_enum
from numba_cfunc_compiler.utils.enumset import make_enumset
from numba_cfunc_compiler.utils.ffi import FFIMethodHelper
from numba_cfunc_compiler.utils.struct import StructHelper
from numba_cfunc_compiler.variable_factory import (
    VariableFactory,
)

__all__ = [
    "CompilationOptions",
    "CompilationResult",
    "NumbaFunctionInfo",  # kept for advanced use cases
    "create_compiled_func",
]


@dataclass
class CompilationResult:
    """Result of create_compiled_func — contains the compiled cfunc and all metadata
    needed by the host framework to wire the node.

    Core fields are always present. State storage and category-specific metadata
    (e.g. ``ordered_input_signals``) live in the ``metadata`` dict.
    """

    compiled_func: Any
    output_types: list[type]
    named_outputs: dict | None  # None for single output
    native_name: str = ""
    semantic_key: str = ""
    llvm_ir: str = ""
    metadata: dict = None  # populated at build time

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def __getattr__(self, name: str) -> Any:
        """Allow attribute access to metadata keys (e.g. result.state_values)."""
        metadata = self.__dict__.get("metadata")
        if metadata and name in metadata:
            return metadata[name]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")


def _build_semantic_key(
    new_func_code: str,
    cfunc_sig: str,
    cfunc_kwargs: str,
) -> str:
    payload = {
        "new_func_code": new_func_code,
        "cfunc_sig": cfunc_sig,
        "cfunc_kwargs": cfunc_kwargs,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


class NumbaFunctionInfo:
    """
    Holds analysis information for a decorated function.

    This class analyzes function signatures, inputs, outputs, and state variables,
    and creates the variable factory for AST transformation.
    """

    def __init__(
        self,
        func,
        *args,
        extract_python_type_fn: Callable[[Any], type],
        decorator_name: str,
        func_globals: dict | None = None,
        signature: inspect.Signature | None = None,
        **kwargs,
    ):
        # Handle both callable and ast.FunctionDef
        if isinstance(func, ast.FunctionDef):
            self.tree = func
            self.name = func.name
            if signature is None:
                raise ValueError("signature is required when func is an ast.FunctionDef")
            self.sig = signature
            self._func_globals = func_globals or {}
        else:
            self.tree = FunctionAnalyzer.get_function_ast(func, decorator_name)
            self.name = func.__name__
            self.sig = signature if signature is not None else inspect.signature(func)
            self._func_globals = getattr(func, "__globals__", {})

        self.func = func
        self.analyzer = FunctionAnalyzer()
        self.extract_python_type_fn = extract_python_type_fn

        # Use try/except to re-throw with additional info (numba_node function name)
        try:
            self._initialize_and_validate(*args, **kwargs)
        except Exception as e:
            error_msg = f"Error in numba_node function '{self.name}': {e!s}"
            raise type(e)(error_msg) from e

    def _initialize_and_validate(self, *args, **kwargs):
        import inspect as _inspect

        try:
            bound = self.sig.bind_partial(*args, **kwargs)
        except TypeError as e:
            raise ValueError(f"expects {self.sig.parameters} arguments, got {args} positional and {kwargs} keyword arguments. {e}")

        # Fill in defaults for any missing values
        for pname, param in self.sig.parameters.items():
            if pname not in bound.arguments and param.default is not _inspect._empty:
                bound.arguments[pname] = param.default

        # Ensure all required params provided
        missing = [pname for pname, param in self.sig.parameters.items() if pname not in bound.arguments and param.default is _inspect._empty]
        if missing:
            raise ValueError(f"expects {len(self.sig.parameters)} arguments, got {len(args) + len(kwargs)}")

        self.input_analysis = self.analyzer.parse_input_annotation(self.sig, dict(bound.arguments))
        self.output_analysis = self.analyzer.parse_output_annotation(self.tree, self.sig)
        self.state_analysis = self.analyzer.parse_state_annotation(self.tree, self._func_globals)
        self.state_slots: list[tuple[Any, VariableType]] = []
        self.variable_factory = self._create_variable_factory()

    def _create_variable_factory(self) -> VariableFactory:
        variable_factory = VariableFactory()
        for category in SourceRegistry.get_ordered():
            category.create_variables(self, variable_factory)
        return variable_factory


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
    get_numba_config()
    opts = options or CompilationOptions()
    info = NumbaFunctionInfo(
        func,
        *args,
        extract_python_type_fn=extract_python_type_fn,
        decorator_name=decorator_name,
        func_globals=func_globals,
        signature=signature,
        **kwargs,
    )

    tree = info.tree
    name = info.name
    variable_factory = info.variable_factory

    # Lazy-load the NRT C library on first compilation
    CompilationContext.current().ensure_nrt_loaded()

    transformer = NumbaASTConverter(
        tree,
        variable_factory,
        start_body=start_body,
        stop_body=stop_body,
        call_globals=call_globals,
    )
    new_tree = transformer.visit(tree)
    new_func_code = ast.unparse(new_tree)

    cfunc_sig = SourceRegistry.build_cfunc_signature()
    cfunc_kwargs = "nopython=True, nogil=True, _nrt=False, error_model='numpy'"
    if opts.fastmath:
        cfunc_kwargs += ", fastmath=True"
    semantic_key = _build_semantic_key(new_func_code, cfunc_sig, cfunc_kwargs)
    cfunc_code = f"""
@cfunc({cfunc_sig}, {cfunc_kwargs})
{new_func_code}
"""

    exec_globals = {}
    exec_globals.update(globals())
    if call_globals:
        exec_globals.update(call_globals)

    exec_globals.update(
        {
            "cfunc": cfunc,
            "CPointer": CPointer,
            "int64": int64,
            "int8": int8,
            "float64": float64,
            "voidptr": numba.types.voidptr,
            "cast_voidptr_to_ptr": AST.cast_voidptr_to_ptr,
            "struct_field_access": StructHelper.struct_field_access,
            "struct_field_ptr": StructHelper.struct_field_ptr,
            "struct_field_store": StructHelper.struct_field_store,
            "struct_memcpy": StructHelper.struct_memcpy,
            "voidptr_null": AST.voidptr_null,
            "ffi_tuple_args": AST.ffi_tuple_args,
            "cast_voidptr_to_int": AST.cast_voidptr_to_int,
            "make_int8": AST.make_int8,
            "make_enum": make_enum,
            "make_enumset": make_enumset,
            "ffi_call": FFIMethodHelper.ffi_call,
            "voidptr_to_intp": AST.voidptr_to_intp,
            # standalone list (NRT-free)
            "standalone_list_new": standalone_list_new,
            "standalone_list_from_voidptr": standalone_list_from_voidptr,
            "standalone_list_free": standalone_list_free,
            "standalone_list_reset_size": standalone_list_reset_size,
            "standalone_list_to_voidptr": standalone_list_to_voidptr,
            # standalone dict (NRT-free)
            "standalone_dict_new": standalone_dict_new,
            "standalone_dict_from_voidptr": standalone_dict_from_voidptr,
            "standalone_dict_free": standalone_dict_free,
            "standalone_dict_clear": standalone_dict_clear,
            "standalone_dict_to_voidptr": standalone_dict_to_voidptr,
            "standalone_dict_length": standalone_dict_length,
            "_standalone_dict_iter_begin": _standalone_dict_iter_begin,
            "_standalone_dict_iter_next_item": _standalone_dict_iter_next_item,
            "_standalone_dict_iter_next_key": _standalone_dict_iter_next_key,
        }
    )
    exec(cfunc_code, exec_globals)  # noqa: S102 - generated function source

    compiled_func = exec_globals[name]

    # --- post-compilation IR transforms ---
    result_ir, exported_entry_point = apply_post_compilation(compiled_func, semantic_key, opts)

    # --- build result ---
    if info.output_analysis.named_outputs is not None:
        output_types = list(info.output_analysis.named_outputs.values())
        named_outputs = dict(info.output_analysis.named_outputs)
    else:
        output_types = list(info.output_analysis.output_types)
        named_outputs = None

    # Collect storage metadata and metadata from registered source categories.
    nrt_indices = []
    struct_indices = []
    struct_sizes = []
    for index, (_, var_type) in enumerate(info.state_slots):
        if isinstance(var_type, ContainerType):
            nrt_indices.append(index)
        elif isinstance(var_type, StructType):
            struct_indices.append(index)
            struct_sizes.append(var_type.get_size())
    result_metadata = {
        "state_values": tuple(value for value, _ in info.state_slots),
        "nrt_state_indices": tuple(nrt_indices),
        "struct_state_indices": tuple(struct_indices),
        "struct_state_sizes": tuple(struct_sizes),
    }
    for category in SourceRegistry.get_ordered():
        result_metadata.update(category.get_result_metadata(info))

    return CompilationResult(
        compiled_func=compiled_func,
        output_types=output_types,
        named_outputs=named_outputs,
        native_name=exported_entry_point,
        semantic_key=semantic_key,
        llvm_ir=result_ir,
        metadata=result_metadata,
    )
