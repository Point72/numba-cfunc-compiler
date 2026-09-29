"""Numba value type used for fixed-width enumsets."""

from llvmlite import ir
from numba import types
from numba.extending import intrinsic, models, register_model

from numba_cfunc_compiler.config import get_numba_config

__all__ = [
    "EnumSetNumbaType",
    "enumset_type",
    "make_enumset",
]


class EnumSetNumbaType(types.Type):
    """Opaque-at-Python-level scalar with a configurable integer representation."""

    def __init__(self) -> None:
        super().__init__(name="enumset")

    @property
    def bit_width(self) -> int:
        return get_numba_config().enumset_bit_width

    @property
    def byte_width(self) -> int:
        return self.bit_width // 8

    @property
    def llvm_type(self) -> ir.IntType:
        return ir.IntType(self.bit_width)


enumset_type = EnumSetNumbaType()


@register_model(EnumSetNumbaType)
class EnumSetModel(models.PrimitiveModel):
    def __init__(self, dmm, fe_type):
        super().__init__(dmm, fe_type, fe_type.llvm_type)


@intrinsic
def make_enumset(typingctx):
    """Return a zero enumset.

    Besides being useful directly, this gives the generic FFI intrinsic a typed
    placeholder from which it can infer an enumset return type.
    """

    sig = enumset_type()

    def codegen(context, builder, signature, args):
        return ir.Constant(enumset_type.llvm_type, 0)

    return sig, codegen
