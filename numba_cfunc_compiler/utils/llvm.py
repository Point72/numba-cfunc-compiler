"""LLVM IR utilities used after Numba compilation."""

import operator
import re

from llvmlite import binding as llvm

__all__ = ["LLVMIRHelper"]


class LLVMIRHelper:
    """Use parsed LLVM values for symbols and linkage, with scoped attribute edits."""

    _ATTRIBUTE_REF_FOLLOW = r"(?=[ \t]*(?:\{|$|!|,|section\b|comdat\b|align\b|gc\b|prefix\b|prologue\b|personality\b|to\b))"

    @staticmethod
    def return_constants(function) -> set[int] | None:
        """Conservatively bound an optimized function's integer return values.

        Unknown instructions and cyclic data flow fail the proof, rather than
        accepting an exception path that we cannot analyze. This is independent
        of Numba's mangled names and exception serialization format.
        """
        instructions = {instruction: instruction for block in function.blocks for instruction in block.instructions}
        active = set()
        binary = {"add": operator.add, "sub": operator.sub, "mul": operator.mul, "and": operator.and_, "or": operator.or_, "xor": operator.xor}

        def union(values):
            values = list(values)
            if any(value is None for value in values):
                return None
            result = set().union(*values)
            return result if len(result) <= 32 else None

        def resolve(value):
            if value.is_constant:
                try:
                    return {int(value.get_constant_value())}
                except (TypeError, ValueError):
                    return None
            instruction = instructions.get(value)
            if instruction is None or value in active:
                return None
            active.add(value)
            try:
                operands = list(instruction.operands)
                operation = instruction.opcode
                if operation in ("icmp", "fcmp"):
                    return {0, 1}
                if operation == "phi":
                    return union(resolve(operand) for operand in operands)
                if operation == "select":
                    return union(resolve(operand) for operand in operands[1:])
                if operation in ("zext", "sext", "trunc"):
                    values = resolve(operands[0])
                    # Status values are small nonnegative integers. Reject casts
                    # whose truncation/sign extension would change those values.
                    bits = min(operands[0].type.type_width, instruction.type.type_width)
                    if values is not None and all(0 <= item < (1 << (bits if operation == "zext" else bits - 1)) for item in values):
                        return values
                if operation in binary:
                    left, right = (resolve(operand) for operand in operands)
                    if left is not None and right is not None:
                        bits = instruction.type.type_width
                        return union({binary[operation](a, b) % (1 << bits)} for a in left for b in right)
                return None
            finally:
                active.remove(value)

        return union(
            resolve(next(iter(instruction.operands)))
            for block in function.blocks
            for instruction in block.instructions
            if instruction.opcode == "ret"
        )

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
                # Enum function attributes are printed as a "; Function Attrs:"
                # comment line ahead of the definition, so skip to the define.
                header = next((line for line in str(function).splitlines() if line.startswith("define")), "")
                group = re.search(rf" #\d+\b{LLVMIRHelper._ATTRIBUTE_REF_FOLLOW}", header)
                insertion_point = group.start() if group else header.rfind(" {")
                if insertion_point < 0 or header not in ir_text:
                    raise ValueError(f"Cannot mark {function.name!r} alwaysinline in LLVM IR")
                patched_header = header[:insertion_point] + " alwaysinline" + header[insertion_point:]
                ir_text = ir_text.replace(header, patched_header, 1)
        return ir_text
