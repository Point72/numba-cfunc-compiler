import logging
from dataclasses import dataclass
from typing import Any

from llvmlite import binding as llvm

from numba_cfunc_compiler.utils.llvm import LLVMIRHelper

__all__ = [
    "CompilationOptions",
    "apply_post_compilation",
    "link_ffi_bitcode",
]

log = logging.getLogger("numba_cfunc_compiler")


@dataclass(frozen=True)
class CompilationOptions:
    """Opt-in compilation flags and post-compilation IR transforms.

    fastmath:        enable fast-math in @cfunc.
    force_inline:    replace noinline → alwaysinline on the cfunc wrapper.
    """

    fastmath: bool = False
    force_inline: bool = False


def apply_post_compilation(
    compiled_func: Any,
    semantic_key: str,
    opts: CompilationOptions,
) -> tuple[str, str]:
    ir_text = compiled_func._library.get_llvm_str()
    raw_native_name = compiled_func.native_name
    exported_entry_point = f"_gc_numba_{semantic_key}"

    if opts.force_inline:
        ir_text = LLVMIRHelper.force_inline(ir_text)

    module = llvm.parse_assembly(ir_text)
    LLVMIRHelper.rename_exported_symbol(module, raw_native_name, exported_entry_point)
    # The graph runtime discovers this entry point by name. All other function
    # bodies belong to this compilation unit and can be inlined and removed by
    # the graph module's optimization pipeline.
    LLVMIRHelper.internalize_defined_functions(module, preserve_symbols={exported_entry_point})
    module.verify()

    return str(module), exported_entry_point


def link_ffi_bitcode(module: Any, bitcode: bytes, *, internalize: bool = False) -> Any:
    """Link FFI function bodies into an LLVM module for inlining.

    Args:
        module: An llvmlite.binding.ModuleRef (the linked LLVM module).
        bitcode: Raw bytes of a .bc file containing the FFI function
            implementations (e.g. compiled from the C interface source).
        internalize: Give definitions supplied by the FFI module internal
            linkage so they can be removed after their callers are inlined.
            Disabled by default for callers that use the definitions as an ABI.

    Returns: The (possibly re-parsed) ModuleRef with FFI bodies linked in and
        patched for inlining.  Falls back to the original *module* on error.
    """
    try:
        ffi_module = llvm.parse_bitcode(bitcode)

        ffi_definition_names = {func.name for func in ffi_module.functions if not func.is_declaration}
        # Collect names of functions that are currently just declarations
        # (i.e. FFI stubs).  After linking these become definitions that
        # we want the inliner to pull in.
        ffi_decl_names = {func.name for func in module.functions if func.is_declaration and not func.name.startswith("llvm.")}

        module.link_in(ffi_module, preserve=False)

        ir_text = LLVMIRHelper.mark_always_inline(module, ffi_decl_names & ffi_definition_names)

        # Strip target-cpu and target-features from FFI attribute groups so
        # they match the numba function's (empty) target attrs. llvmlite's
        # binding does not expose removing these attributes.
        ir_text = LLVMIRHelper.strip_target_attributes(ir_text)
        module = llvm.parse_assembly(ir_text)

        if internalize:
            LLVMIRHelper.internalize_defined_functions(module, only_symbols=ffi_definition_names)

        module.verify()

    except Exception as e:  # noqa: BLE001 - optimization failure must fall back
        log.warning(f"Failed to link FFI bitcode for inlining (falling back to external calls): {e}")

    return module
