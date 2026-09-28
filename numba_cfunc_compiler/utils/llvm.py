"""LLVM IR utilities used after Numba compilation."""

import re

from llvmlite import binding as llvm

__all__ = ["LLVMIRHelper"]


class LLVMIRHelper:
    """Use parsed LLVM values for symbols and linkage, with scoped attribute edits."""

    _ATTRIBUTE_REF_FOLLOW = r"(?=[ \t]*(?:\{|$|!|,|section\b|comdat\b|align\b|gc\b|prefix\b|prologue\b|personality\b|to\b))"

    @staticmethod
    def force_inline(llvm_ir_text: str) -> str:
        """Replace noinline with alwaysinline in LLVM attribute groups."""
        return re.sub(
            r"(?m)^attributes[ \t]+#(\d+)[ \t]*=[ \t]*\{[ \t]*noinline[ \t]*\}",
            r"attributes #\1 = { alwaysinline }",
            llvm_ir_text,
        )

    @staticmethod
    def strip_target_attributes(llvm_ir_text: str) -> str:
        """Remove FFI target attributes that prevent inlining into Numba code."""
        lines = []
        for line in llvm_ir_text.splitlines(keepends=True):
            if line.startswith("attributes #"):
                line = re.sub(r'"target-cpu"="[^"]*"', "", line)
                line = re.sub(r'"target-features"="[^"]*"', "", line)
            lines.append(line)
        llvm_ir_text = "".join(lines)

        # LLVM rejects an attribute group once stripping leaves it empty. Drop
        # both the group and its references from the canonical printed IR.
        empty_group = re.compile(r"(?m)^attributes #(\d+) = \{[ \t]*\}[ \t]*(?:\n|$)")
        while match := empty_group.search(llvm_ir_text):
            llvm_ir_text = empty_group.sub("", llvm_ir_text, count=1)
            llvm_ir_text = re.sub(rf" #{match.group(1)}\b{LLVMIRHelper._ATTRIBUTE_REF_FOLLOW}", "", llvm_ir_text, flags=re.MULTILINE)
        return llvm_ir_text

    @staticmethod
    def rename_exported_symbol(module: llvm.ModuleRef, old_symbol: str, new_symbol: str) -> None:
        if old_symbol == new_symbol:
            return

        try:
            function = module.get_function(old_symbol)
        except NameError as exc:
            raise ValueError(f"Failed to find compiled symbol {old_symbol!r} in generated LLVM IR") from exc

        function.name = new_symbol
        if function.name != new_symbol:
            raise ValueError(f"Failed to rename compiled symbol {old_symbol!r} to {new_symbol!r}")

    @staticmethod
    def internalize_defined_functions(
        module: llvm.ModuleRef,
        *,
        only_symbols: set[str] | None = None,
        preserve_symbols: set[str] | None = None,
    ) -> None:
        """Give selected definitions internal linkage, leaving declarations external."""
        preserved = preserve_symbols or set()
        for function in module.functions:
            if function.is_declaration or function.name in preserved:
                continue
            if only_symbols is None or function.name in only_symbols:
                function.linkage = llvm.Linkage.internal

    @staticmethod
    def mark_always_inline(module: llvm.ModuleRef, names: set[str]) -> str:
        """Mark linked FFI definitions that replaced declarations for inlining.

        llvmlite.binding's add_function_attribute emits invalid IR for this
        attribute on some supported versions, so edit each selected header and
        immediately reparse the result.
        """
        ir_text = str(module)
        for function in module.functions:
            if function.name in names and not function.is_declaration:
                header = str(function).split("\n", 1)[0]
                group = re.search(rf" #\d+\b{LLVMIRHelper._ATTRIBUTE_REF_FOLLOW}", header)
                insertion_point = group.start() if group else header.rfind(" {")
                if insertion_point < 0 or header not in ir_text:
                    raise ValueError(f"Cannot mark {function.name!r} alwaysinline in LLVM IR")
                patched_header = header[:insertion_point] + " alwaysinline" + header[insertion_point:]
                ir_text = ir_text.replace(header, patched_header, 1)
        return ir_text
