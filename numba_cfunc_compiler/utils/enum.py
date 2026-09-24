"""Configured native representation for scalar enum values."""

from typing import ClassVar

from llvmlite import ir
from numba import types
from numba.extending import intrinsic

from numba_cfunc_compiler.numba_config import ENUM_BIT_WIDTH

__all__ = [
    "EnumRepresentation",
    "enum_representation",
    "make_enum",
]


class EnumRepresentation:
    """Numba and LLVM types derived from the configured enum ABI width."""

    _NUMBA_TYPES: ClassVar[dict[int, types.Integer]] = {
        8: types.int8,
        16: types.int16,
        32: types.int32,
        64: types.int64,
    }

    def __init__(self, bit_width: int) -> None:
        try:
            self.numba_type = self._NUMBA_TYPES[bit_width]
        except KeyError:
            supported = ", ".join(str(width) for width in self._NUMBA_TYPES)
            raise ValueError(f"ENUM_BIT_WIDTH must be one of {supported}; got {bit_width}") from None

        self.bit_width = bit_width
        self.byte_width = bit_width // 8
        self.llvm_type = ir.IntType(bit_width)


enum_representation = EnumRepresentation(ENUM_BIT_WIDTH)


@intrinsic
def make_enum(typingctx):
    """Return a zero value carrying the configured enum representation."""

    sig = enum_representation.numba_type()

    def codegen(context, builder, signature, args):
        return ir.Constant(enum_representation.llvm_type, 0)

    return sig, codegen
