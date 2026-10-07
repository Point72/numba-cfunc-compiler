"""Typed input values that retain source capabilities across local aliases."""

import hashlib
import math
import operator
from functools import cache

from llvmlite import ir
from numba import types
from numba.core import cgutils
from numba.core.typing.templates import AttributeTemplate
from numba.extending import (
    infer_getattr,
    intrinsic,
    lower_builtin,
    lower_cast,
    lower_getattr_generic,
    lower_setattr_generic,
    models,
    overload,
    register_model,
)

from numba_cfunc_compiler.types.policy import ValueSemantics


class SourceMetadataType(types.Type):
    """One source descriptor's typed callback metadata."""

    def __init__(self, identity, fields):
        self.identity = identity
        self.fields = fields
        super().__init__(f"SourceMetadata[{identity!r}:{tuple((field.name, field.numba_type.name) for field in fields)}]")

    @property
    def key(self):
        return self.identity, tuple((field.name, field.numba_type) for field in self.fields)


@register_model(SourceMetadataType)
class SourceMetadataModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        super().__init__(dmm, fe_type, [(field.name, field.numba_type) for field in fe_type.fields])


@infer_getattr
class SourceMetadataAttributes(AttributeTemplate):
    key = SourceMetadataType

    def generic_resolve(self, typ, attr):
        for field in typ.fields:
            if field.name == attr:
                return field.numba_type
        return None


@lower_getattr_generic(SourceMetadataType)
def source_metadata_getattr(context, builder, typ, value, attr):
    if attr not in {field.name for field in typ.fields}:
        raise AttributeError(attr)
    return getattr(context.make_helper(builder, typ, value=value), attr)


@cache
def source_metadata_type(identity, fields):
    return SourceMetadataType(identity, fields)


class InputValueType(types.Type):
    """A payload snapshot or borrowed view with nominal source metadata."""

    def __init__(self, payload_type, storage_type, semantics: ValueSemantics, layout_key, mutable):
        if not isinstance(semantics, ValueSemantics):
            raise TypeError(f"Input semantics must be a ValueSemantics member, got {semantics!r}")
        descriptor = type(self).source_descriptor
        self.payload_type = payload_type
        self.storage_type = storage_type
        self.semantics = semantics
        self.source_descriptor = descriptor
        self.metadata_type = source_metadata_type(descriptor.identity, descriptor.fields)
        self.layout_key = layout_key
        self.mutable = mutable
        super().__init__(f"InputValue[{descriptor.key!r}:{payload_type}:{storage_type}:{semantics.value}:{layout_key}:{mutable}]")

    @property
    def key(self):
        return (
            self.source_descriptor.key,
            self.payload_type,
            self.storage_type,
            self.semantics,
            self.layout_key,
            self.mutable,
        )

    def unify(self, typingctx, other):
        from numba_cfunc_compiler.types.native.state import CopyStateValueType

        if other == self.payload_type or (isinstance(other, (InputValueType, CopyStateValueType)) and self.payload_type == other.payload_type):
            return self.payload_type
        return None


@register_model(InputValueType)
class InputValueModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        super().__init__(
            dmm,
            fe_type,
            [
                ("value", fe_type.payload_type),
                ("source", fe_type.metadata_type),
            ],
        )


@cache
def source_value_type(type_class, payload_type, storage_type, semantics: ValueSemantics, layout_key=(), mutable=False):
    return type_class(payload_type, storage_type, semantics, layout_key, mutable)


def make_source_type_class(identity, fields):
    token = hashlib.sha256(repr(identity).encode()).hexdigest()[:12]
    descriptor_proxy = None
    source_class = type(f"InputSource_{token}", (InputValueType,), {"source_descriptor": descriptor_proxy})
    register_model(source_class)(InputValueModel)
    return source_class


def construct_source_value(context, builder, value_type, slot, metadata_values):
    """The common payload and metadata construction path for scalar and keyed reads."""
    result = context.make_helper(builder, value_type)
    if value_type.semantics is ValueSemantics.COPY:
        storage_llvm_type = context.get_value_type(value_type.storage_type)
        stored_slot = cgutils.alloca_once(builder, storage_llvm_type)
        builder.store(ir.Constant(storage_llvm_type, None), stored_slot)
        with builder.if_then(builder.icmp_unsigned("!=", slot, ir.Constant(slot.type, None))):
            pointer = builder.bitcast(slot, storage_llvm_type.as_pointer())
            builder.store(builder.load(pointer), stored_slot)
        stored = builder.load(stored_slot)
        result.value = context.cast(builder, stored, value_type.storage_type, value_type.payload_type)
    else:
        result.value = builder.bitcast(slot, context.get_value_type(value_type.payload_type))
    source = context.make_helper(builder, value_type.metadata_type)
    for field, field_value in zip(value_type.source_descriptor.fields, metadata_values):
        setattr(source, field.name, field_value)
    result.source = source._getvalue()
    return result._getvalue()


@cache
def bind_input(value_type: InputValueType):
    """Load one host slot and attach the descriptor's metadata fields."""

    @intrinsic
    def bind(typingctx, slot, metadata_values):
        fields = value_type.source_descriptor.fields
        if slot != types.voidptr or not isinstance(metadata_values, types.BaseTuple) or len(metadata_values) != len(fields):
            return None
        for field, actual in zip(fields, metadata_values):
            if actual != field.numba_type and not (field.numba_type == types.intp and isinstance(actual, types.Integer)):
                return None
        sig = value_type(slot, metadata_values)

        def codegen(context, builder, signature, args):
            values = cgutils.unpack_tuple(builder, args[1], len(fields))
            converted = [
                context.cast(builder, values[position], metadata_values[position], field.numba_type) for position, field in enumerate(fields)
            ]
            return construct_source_value(context, builder, value_type, args[0], converted)

        return sig, codegen

    return bind


@lower_cast(InputValueType, types.Boolean)
@lower_cast(InputValueType, types.Type)
def input_value_cast(context, builder, from_type, to_type, value):
    if to_type != from_type.payload_type:
        raise TypeError(f"Cannot convert {from_type} to {to_type}")
    return context.make_helper(builder, from_type, value=value).value


@lower_builtin(bool, InputValueType)
def input_value_bool(context, builder, signature, args):
    source = signature.args[0]
    value = context.make_helper(builder, source, value=args[0]).value
    return context.cast(builder, value, source.payload_type, types.boolean)


@intrinsic
def input_payload(typingctx, value):
    from numba_cfunc_compiler.types.native.state import CopyStateValueType

    if not isinstance(value, (InputValueType, CopyStateValueType)):
        return None
    sig = value.payload_type(value)

    def codegen(context, builder, signature, args):
        return context.make_helper(builder, value, value=args[0]).value

    return sig, codegen


def unwrap_source(value):
    """Expose the payload when a native operation requires its exact type."""
    return value


@overload(unwrap_source)
def overload_unwrap_source(value):
    from numba_cfunc_compiler.types.native.state import CopyStateValueType

    if isinstance(value, (InputValueType, CopyStateValueType)):

        def impl(value):
            return input_payload(value)

        return impl

    def impl(value):
        return value

    return impl


def payload_type_of(value_type):
    from numba_cfunc_compiler.types.native.state import CopyStateValueType

    return value_type.payload_type if isinstance(value_type, (InputValueType, CopyStateValueType)) else value_type


@intrinsic
def unwrap_source_args(typingctx, args):
    """Unwrap each source in a statically typed argument tuple."""
    from numba_cfunc_compiler.types.native.state import CopyStateValueType

    if not isinstance(args, types.BaseTuple):
        return None
    result_type = types.Tuple(tuple(payload_type_of(arg) for arg in args))
    sig = result_type(args)

    def codegen(context, builder, signature, values):
        items = cgutils.unpack_tuple(builder, values[0], len(args))
        unwrapped = [
            context.make_helper(builder, arg_type, value=item).value if isinstance(arg_type, (InputValueType, CopyStateValueType)) else item
            for arg_type, item in zip(args, items)
        ]
        return context.make_tuple(builder, result_type, unwrapped)

    return sig, codegen


def _install_binary_operator(operation):
    @overload(operation)
    def source_binary(left, right):
        from numba_cfunc_compiler.types.native.state import CopyStateValueType

        if not isinstance(left, (InputValueType, CopyStateValueType)) and not isinstance(right, (InputValueType, CopyStateValueType)):
            return None

        def impl(left, right):
            return operation(unwrap_source(left), unwrap_source(right))

        return impl


for _operation in (
    operator.add,
    operator.sub,
    operator.mul,
    operator.truediv,
    operator.floordiv,
    operator.mod,
    operator.pow,
    operator.and_,
    operator.or_,
    operator.xor,
    operator.lshift,
    operator.rshift,
    operator.eq,
    operator.ne,
    operator.lt,
    operator.le,
    operator.gt,
    operator.ge,
    operator.iadd,
    operator.isub,
    operator.imul,
    operator.itruediv,
    operator.ifloordiv,
    operator.imod,
    operator.iand,
    operator.ior,
    operator.ixor,
):
    _install_binary_operator(_operation)


def _install_unary_operator(operation):
    @overload(operation)
    def source_unary(value):
        from numba_cfunc_compiler.types.native.state import CopyStateValueType

        if not isinstance(value, (InputValueType, CopyStateValueType)):
            return None

        def impl(value):
            return operation(unwrap_source(value))

        return impl


for _operation in (abs, operator.neg, operator.pos, operator.invert, operator.not_):
    _install_unary_operator(_operation)


@overload(math.fabs)
def input_math_fabs(value):
    if not isinstance(value, InputValueType):
        return None

    def impl(value):
        return math.fabs(input_payload(value))

    return impl


@overload(range)
def input_range(start, stop=None, step=None):
    if not any(isinstance(arg, InputValueType) for arg in (start, stop, step)):
        return None
    if stop is None or isinstance(stop, (types.NoneType, types.Omitted)):

        def impl(start, stop=None, step=None):
            return range(unwrap_source(start))

        return impl
    if step is None or isinstance(step, (types.NoneType, types.Omitted)):

        def impl(start, stop=None, step=None):
            return range(unwrap_source(start), unwrap_source(stop))

        return impl

    def impl(start, stop=None, step=None):
        return range(unwrap_source(start), unwrap_source(stop), unwrap_source(step))

    return impl


@overload(operator.truth)
def input_truth(value):
    if not isinstance(value, InputValueType):
        return None

    def impl(value):
        return bool(input_payload(value))

    return impl


@infer_getattr
class InputValueAttributes(AttributeTemplate):
    key = InputValueType

    def generic_resolve(self, typ, attr):
        from numba_cfunc_compiler.types.builtin.struct.native import StructPtrType

        if attr == "source":
            return typ.metadata_type
        if isinstance(typ.payload_type, StructPtrType):
            field = typ.payload_type.layout.get_field(attr)
            if field is not None:
                return field.numba_type
        return None


@lower_getattr_generic(InputValueType)
def input_value_getattr(context, builder, typ, value, attr):
    from numba_cfunc_compiler.types.builtin.struct.native import StructPtrType

    wrapped = context.make_helper(builder, typ, value=value)
    if attr == "source":
        return wrapped.source
    if isinstance(typ.payload_type, StructPtrType) and typ.payload_type.layout.get_field(attr) is not None:
        return context.get_getattr(typ.payload_type, attr)(context, builder, typ.payload_type, wrapped.value, attr)
    raise AttributeError(attr)


@lower_setattr_generic(InputValueType)
def input_value_setattr(context, builder, sig, args, attr):
    from numba_cfunc_compiler.types.builtin.struct.native import StructPtrType

    typ, value_type = sig.args
    if not typ.mutable or not isinstance(typ.payload_type, StructPtrType):
        raise TypeError(f"{typ} does not allow field mutation")
    field = typ.payload_type.layout.get_field(attr)
    if field is None:
        raise AttributeError(attr)
    wrapped = context.make_helper(builder, typ, value=args[0])
    pointer = builder.bitcast(wrapped.value, ir.IntType(8).as_pointer())
    pointer = builder.gep(pointer, [context.get_constant(types.intp, field.offset)])
    pointer = builder.bitcast(pointer, context.get_value_type(field.numba_type).as_pointer())
    builder.store(context.cast(builder, args[1], value_type, field.numba_type), pointer, align=1)
