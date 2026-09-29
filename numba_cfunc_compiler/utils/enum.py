"""Configured native representation for scalar enum values."""

from dataclasses import dataclass
from typing import ClassVar

from llvmlite import ir
from numba import types
from numba.extending import intrinsic

from numba_cfunc_compiler.config import get_numba_config

__all__ = [
    "EnumRepresentation",
    "enum_representation",
    "make_enum",
]


@dataclass(frozen=True, init=False)
class EnumRepresentation:
    """Numba and LLVM types derived from the configured enum ABI width."""

    _bit_width: int | None = None

    _NUMBA_TYPES: ClassVar[dict[int, types.Integer]] = {
        8: types.int8,
        16: types.int16,
        32: types.int32,
        64: types.int64,
    }

    def __init__(self, bit_width: int | None = None) -> None:
        if bit_width is not None and bit_width not in self._NUMBA_TYPES:
            supported = ", ".join(str(width) for width in self._NUMBA_TYPES)
            raise ValueError(f"ENUM_BIT_WIDTH must be one of {supported}; got {bit_width}")
        object.__setattr__(self, "_bit_width", bit_width)

    @property
    def bit_width(self) -> int:
        return self._bit_width if self._bit_width is not None else get_numba_config().enum_bit_width

    @property
    def numba_type(self) -> types.Integer:
        return self._NUMBA_TYPES[self.bit_width]

    @property
    def byte_width(self) -> int:
        return self.bit_width // 8

    @property
    def llvm_type(self) -> ir.IntType:
        return ir.IntType(self.bit_width)


enum_representation = EnumRepresentation()


@intrinsic
def make_enum(typingctx):
    """Return a zero value carrying the configured enum representation."""

    sig = enum_representation.numba_type()

    def codegen(context, builder, signature, args):
        return ir.Constant(enum_representation.llvm_type, 0)

    return sig, codegen
