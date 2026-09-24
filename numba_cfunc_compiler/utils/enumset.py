"""Numba value type used for fixed-width 128-bit enumsets."""

from llvmlite import ir
from numba import types
from numba.extending import intrinsic, models, register_model

__all__ = [
    "EnumSetNumbaType",
    "enumset_type",
    "make_enumset",
]


class EnumSetNumbaType(types.Type):
    """Opaque-at-Python-level scalar whose native representation is LLVM ``i128``."""

    def __init__(self) -> None:
        super().__init__(name="enumset")


enumset_type = EnumSetNumbaType()


@register_model(EnumSetNumbaType)
class EnumSetModel(models.PrimitiveModel):
    def __init__(self, dmm, fe_type):
        super().__init__(dmm, fe_type, ir.IntType(128))


@intrinsic
def make_enumset(typingctx):
    """Return a zero enumset.

    Besides being useful directly, this gives the generic FFI intrinsic a typed
    placeholder from which it can infer an enumset return type.
    """

    sig = enumset_type()

    def codegen(context, builder, signature, args):
        return ir.Constant(ir.IntType(128), 0)

    return sig, codegen
