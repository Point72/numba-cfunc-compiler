"""Typed writes and tick flags for host-owned output cells."""

from functools import cache

from llvmlite import ir
from numba import types
from numba.core import cgutils
from numba.core.errors import TypingError
from numba.extending import intrinsic, models, overload_method, register_model

from numba_cfunc_compiler.types.builtin.struct.native import StructPtrType
from numba_cfunc_compiler.types.native.input import InputValueType
from numba_cfunc_compiler.types.native.state import CopyStateValueType
from numba_cfunc_compiler.types.policy import ValueSemantics


class OutputSinkType(types.Type):
    def __init__(self, payload_type, storage_type, semantics: ValueSemantics, host_size, layout_key):
        if not isinstance(semantics, ValueSemantics):
            raise TypeError(f"Output semantics must be a ValueSemantics member, got {semantics!r}")
        self.payload_type = payload_type
        self.storage_type = storage_type
        self.semantics = semantics
        self.host_size = host_size
        self.layout_key = layout_key
        super().__init__(f"OutputSink[{payload_type}:{semantics.value}:{layout_key}]")

    @property
    def key(self):
        return self.payload_type, self.storage_type, self.semantics, self.host_size, self.layout_key


@register_model(OutputSinkType)
class OutputSinkModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        super().__init__(dmm, fe_type, [("slot", types.voidptr), ("ticks", types.CPointer(types.int8)), ("index", types.intp)])


@cache
def output_sink_type(payload_type, storage_type, semantics: ValueSemantics, host_size, layout_key):
    return OutputSinkType(payload_type, storage_type, semantics, host_size, layout_key)


@cache
def bind_output(sink_type: OutputSinkType):
    @intrinsic
    def bind(typingctx, slot, ticks, index):
        if slot != types.voidptr or ticks != types.CPointer(types.int8) or not isinstance(index, types.Integer):
            return None
        sig = sink_type(slot, ticks, index)

        def codegen(context, builder, signature, args):
            result = context.make_helper(builder, sink_type)
            result.slot = args[0]
            result.ticks = args[1]
            result.index = context.cast(builder, args[2], signature.args[2], types.intp)
            return result._getvalue()

        return sig, codegen

    return bind


def _unwrapped_type(typ):
    if isinstance(typ, (InputValueType, CopyStateValueType)):
        return typ.payload_type
    return typ


def _compatible(sink, rhs):
    rhs = _unwrapped_type(rhs)
    payload = sink.payload_type
    if isinstance(payload, types.Boolean):
        return isinstance(rhs, types.Boolean) or rhs == types.int8
    if isinstance(payload, types.Integer):
        return isinstance(rhs, types.Integer) and rhs.bitwidth <= payload.bitwidth
    if isinstance(payload, types.Float):
        return isinstance(rhs, types.Float) and rhs.bitwidth <= payload.bitwidth
    return rhs == payload


@intrinsic
def output_write(typingctx, sink, rhs):
    if not isinstance(sink, OutputSinkType):
        return None
    if not _compatible(sink, rhs):
        raise TypingError(f"Output {sink.payload_type} cannot store value of type {rhs}")
    sig = types.void(sink, rhs)

    def codegen(context, builder, signature, args):
        output = context.make_helper(builder, sink, value=args[0])
        rhs_type = signature.args[1]
        value = args[1]
        if isinstance(rhs_type, (InputValueType, CopyStateValueType)):
            value = context.make_helper(builder, rhs_type, value=value).value
            rhs_type = rhs_type.payload_type
        if isinstance(sink.payload_type, StructPtrType):
            cgutils.memcpy(builder, output.slot, value, context.get_constant(types.intp, sink.host_size))
        else:
            stored = context.cast(builder, value, rhs_type, sink.storage_type)
            pointer = builder.bitcast(output.slot, context.get_value_type(sink.storage_type).as_pointer())
            builder.store(stored, pointer)
        return context.get_dummy_value()

    return sig, codegen


@intrinsic
def output_tick(typingctx, sink):
    if not isinstance(sink, OutputSinkType):
        return None
    sig = types.void(sink)

    def codegen(context, builder, signature, args):
        output = context.make_helper(builder, sink, value=args[0])
        pointer = builder.gep(output.ticks, [output.index])
        builder.store(ir.Constant(ir.IntType(8), 1), pointer)
        return context.get_dummy_value()

    return sig, codegen


@overload_method(OutputSinkType, "write")
def output_sink_write(sink, value):
    def impl(sink, value):
        output_write(sink, value)

    return impl


@overload_method(OutputSinkType, "tick")
def output_sink_tick(sink):
    def impl(sink):
        output_tick(sink)

    return impl
