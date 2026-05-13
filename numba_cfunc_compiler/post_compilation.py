import logging
import re
from dataclasses import dataclass
from typing import Any

__all__ = [
    "CompilationOptions",
    "apply_post_compilation",
    "link_ffi_bitcode",
]

logger = logging.getLogger("graph_compute")


@dataclass(frozen=True)
class CompilationOptions:
    """Opt-in compilation flags and post-compilation IR transforms.

    fastmath:        enable fast-math in @cfunc.
    force_inline:    replace noinline → alwaysinline on the cfunc wrapper.
    """

    fastmath: bool = False
    force_inline: bool = False


def _force_inline(llvm_ir_text: str) -> str:
    """Replace noinline with alwaysinline in LLVM attribute groups."""
    return re.sub(
        r"attributes\s+#(\d+)\s*=\s*\{\s*noinline\s*\}",
        r"attributes #\1 = { alwaysinline }",
        llvm_ir_text,
    )


def _rename_exported_symbol(llvm_ir_text: str, old_symbol: str, new_symbol: str) -> str:
    if old_symbol == new_symbol:
        return llvm_ir_text

    renamed_ir, substitutions = re.subn(
        rf"@{re.escape(old_symbol)}(?=[^\w.$])",
        f"@{new_symbol}",
        llvm_ir_text,
    )
    if substitutions == 0:
        raise ValueError(f"Failed to find compiled symbol {old_symbol!r} in generated LLVM IR")
    return renamed_ir


def apply_post_compilation(
    compiled_func: Any,
    semantic_key: str,
    opts: CompilationOptions,
) -> tuple[str, str]:
    ir_text = compiled_func._library.get_llvm_str()
    raw_native_name = compiled_func.native_name
    exported_entry_point = f"_gc_numba_{semantic_key}"

    if opts.force_inline:
        ir_text = _force_inline(ir_text)

    ir_text = _rename_exported_symbol(ir_text, raw_native_name, exported_entry_point)

    return ir_text, exported_entry_point


def link_ffi_bitcode(module: Any, bitcode: bytes) -> Any:
    """Link FFI function bodies into an LLVM module for inlining.

    Args:
        module: An llvmlite.binding.ModuleRef (the linked LLVM module).
        bitcode: Raw bytes of a .bc file containing the FFI function
            implementations (e.g. compiled from the C interface source).

    Returns: The (possibly re-parsed) ModuleRef with FFI bodies linked in and
        patched for inlining.  Falls back to the original *module* on error.
    """
    import llvmlite.binding as llvm

    try:
        ffi_module = llvm.parse_bitcode(bitcode)

        # Collect names of functions that are currently just declarations
        # (i.e. FFI stubs).  After linking these become definitions that
        # we want the inliner to pull in.
        ffi_decl_names = {func.name for func in module.functions if func.is_declaration and not func.name.startswith("llvm.")}

        module.link_in(ffi_module, preserve=False)

        # Patch the linked-in FFI definitions to ``alwaysinline`` and strip
        # target attributes that would cause inlining mismatches.
        ir_text = str(module)

        patched_lines = []
        for line in ir_text.split("\n"):
            if line.startswith("define ") and any(f"@{n}(" in line for n in ffi_decl_names):
                # Insert 'alwaysinline' before the #N attribute group ref
                line = re.sub(r"(#\d+)", r"alwaysinline \1", line, count=1)
            patched_lines.append(line)
        ir_text = "\n".join(patched_lines)

        # Strip target-cpu and target-features from FFI attribute groups so
        # they match the numba function's (empty) target attrs.
        ir_text = re.sub(r'"target-cpu"="[^"]*"', "", ir_text)
        ir_text = re.sub(r'"target-features"="[^"]*"', "", ir_text)

        module = llvm.parse_assembly(ir_text)
        module.verify()

    except Exception as e:
        logger.warning(f"Failed to link FFI bitcode for inlining (falling back to external calls): {e}")

    return module
