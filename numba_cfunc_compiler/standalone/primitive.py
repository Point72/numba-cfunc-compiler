"""Numba-typed writes into host-owned primitive output cells."""

from numba import types
from numba.core.errors import TypingError
from numba.extending import intrinsic

from numba_cfunc_compiler.state_values import CopyStateValueType

__all__ = ["primitive_output_store"]

OUTPUT_STORE_VERSION = 1


_OUTPUT_POINTER_TYPES = {"int": types.int64, "float": types.float64, "bool": types.int8}


def _same_primitive_family(kind, value_type) -> bool:
    if isinstance(value_type, CopyStateValueType):
        value_type = value_type.payload_type
    if kind == "bool":
        return isinstance(value_type, types.Boolean)
    if kind == "int":
        return isinstance(value_type, types.Integer)
    if kind == "float":
        return isinstance(value_type, types.Float)
    return False


@intrinsic
def primitive_output_store(typingctx, destination, value, kind):
    """Store a value only when Numba assigns it the output's primitive kind."""
    if not isinstance(kind, types.Literal) or kind.literal_value not in _OUTPUT_POINTER_TYPES:
        return None
    if destination != types.CPointer(_OUTPUT_POINTER_TYPES[kind.literal_value]):
        return None
    output_type = destination.dtype
    if not _same_primitive_family(kind.literal_value, value):
        raise TypingError(f"Primitive {kind.literal_value} output cannot store value of type {value}")

    signature = types.void(destination, value, kind)

    def codegen(context, builder, sig, args):
        if isinstance(sig.args[1], CopyStateValueType):
            stored_value = context.make_helper(builder, sig.args[1], value=args[1]).value
            stored_value = context.cast(builder, stored_value, sig.args[1].payload_type, output_type)
        else:
            stored_value = context.cast(builder, args[1], sig.args[1], output_type)
        builder.store(stored_value, args[0])
        return context.get_dummy_value()

    return signature, codegen
