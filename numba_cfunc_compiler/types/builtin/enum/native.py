"""Nominal enum and copied enum-set values for compiled nodes."""

import hashlib
import operator
from functools import cache

from llvmlite import ir
from numba import types
from numba.extending import intrinsic, lower_cast, models, overload, overload_method, register_model


class EnumValueType(types.Type):
    def __init__(self, family_key, storage_type):
        self.family_key = family_key
        self.storage_type = storage_type
        fingerprint = hashlib.sha256(repr(family_key).encode()).hexdigest()[:16]
        super().__init__(f"EnumValue[{family_key[1]}:{fingerprint}]")

    @property
    def key(self):
        return self.family_key, self.storage_type


class EnumSetValueType(types.Type):
    def __init__(self, family_key, bit_width):
        self.family_key = family_key
        self.bit_width = bit_width
        self.bitwidth = bit_width
        fingerprint = hashlib.sha256(repr(family_key).encode()).hexdigest()[:16]
        super().__init__(f"EnumSetValue[{family_key[1]}:{fingerprint}:{bit_width}]")

    @property
    def key(self):
        return self.family_key, self.bit_width


@register_model(EnumValueType)
class EnumValueModel(models.PrimitiveModel):
    def __init__(self, dmm, fe_type):
        super().__init__(dmm, fe_type, ir.IntType(fe_type.storage_type.bitwidth))


@register_model(EnumSetValueType)
class EnumSetValueModel(models.PrimitiveModel):
    def __init__(self, dmm, fe_type):
        super().__init__(dmm, fe_type, ir.IntType(fe_type.bit_width))


@lower_cast(EnumValueType, types.Integer)
def enum_to_storage(context, builder, from_type, to_type, value):
    if to_type != from_type.storage_type:
        raise TypeError(f"Cannot cast {from_type} to {to_type}")
    return value


@lower_cast(types.Integer, EnumValueType)
def storage_to_enum(context, builder, from_type, to_type, value):
    if from_type != to_type.storage_type:
        raise TypeError(f"Cannot cast {from_type} to {to_type}")
    return value


@cache
def family_types(family_key, storage_type, bit_width):
    return EnumValueType(family_key, storage_type), EnumSetValueType(family_key, bit_width)


@intrinsic
def enum_code(typingctx, value):
    if not isinstance(value, EnumValueType):
        return None
    sig = value.storage_type(value)

    def codegen(context, builder, signature, args):
        return args[0]

    return sig, codegen


@intrinsic
def enum_set_has(typingctx, value, candidate_set):
    if not isinstance(value, EnumValueType) or not isinstance(candidate_set, EnumSetValueType):
        return None
    if value.family_key != candidate_set.family_key:
        return None
    sig = types.boolean(value, candidate_set)

    def codegen(context, builder, signature, args):
        width = candidate_set.bit_width
        bit_type = ir.IntType(width)
        check_type = ir.IntType(max(width, value.storage_type.bitwidth))
        code = args[0]
        if value.storage_type.bitwidth < check_type.width:
            code = builder.sext(code, check_type) if value.storage_type.signed else builder.zext(code, check_type)
        in_range = builder.icmp_unsigned("<", code, ir.Constant(check_type, width))
        safe_code = builder.select(in_range, code, ir.Constant(check_type, 0))
        if check_type.width > width:
            safe_code = builder.trunc(safe_code, bit_type)
        shifted = builder.shl(ir.Constant(bit_type, 1), safe_code)
        present = builder.icmp_unsigned("!=", builder.and_(args[1], shifted), ir.Constant(bit_type, 0))
        return builder.and_(in_range, present)

    return sig, codegen


@intrinsic
def enum_set_any(typingctx, left, right):
    if not isinstance(left, EnumSetValueType) or left != right:
        return None
    sig = types.boolean(left, right)

    def codegen(context, builder, signature, args):
        return builder.icmp_unsigned("!=", builder.and_(args[0], args[1]), ir.Constant(ir.IntType(left.bit_width), 0))

    return sig, codegen


@intrinsic
def enum_set_all(typingctx, left, right):
    if not isinstance(left, EnumSetValueType) or left != right:
        return None
    sig = types.boolean(left, right)

    def codegen(context, builder, signature, args):
        return builder.icmp_unsigned("==", builder.and_(args[0], args[1]), args[1])

    return sig, codegen


@overload_method(EnumValueType, "isin")
def enum_isin(value, candidate_set):
    if isinstance(candidate_set, EnumSetValueType) and value.family_key == candidate_set.family_key:

        def impl(value, candidate_set):
            return enum_set_has(value, candidate_set)

        return impl


@overload_method(EnumSetValueType, "contains_any")
def enum_contains_any(value, other):
    if isinstance(other, EnumSetValueType) and value == other:

        def impl(value, other):
            return enum_set_any(value, other)

        return impl


@overload_method(EnumSetValueType, "contains_all")
def enum_contains_all(value, other):
    if isinstance(other, EnumSetValueType) and value == other:

        def impl(value, other):
            return enum_set_all(value, other)

        return impl


@overload(operator.eq)
def enum_equal(left, right):
    if isinstance(left, EnumValueType) and left == right:

        def impl(left, right):
            return enum_code(left) == enum_code(right)

        return impl


@overload(operator.ne)
def enum_not_equal(left, right):
    if isinstance(left, EnumValueType) and left == right:

        def impl(left, right):
            return enum_code(left) != enum_code(right)

        return impl


@cache
def enum_literal(value_type, value):
    @intrinsic
    def literal(typingctx):
        sig = value_type()

        def codegen(context, builder, signature, args):
            return ir.Constant(ir.IntType(value_type.storage_type.bitwidth), value)

        return sig, codegen

    return literal


@cache
def set_builder(set_type, value_type):
    @intrinsic
    def build(typingctx, values):
        if not isinstance(values, types.BaseTuple):
            return None
        from numba_cfunc_compiler.types.native.input import InputValueType

        for entry in values:
            native = entry.payload_type if isinstance(entry, InputValueType) else entry
            if native != value_type:
                return None
        sig = set_type(values)

        def codegen(context, builder, signature, args):
            bit_type = ir.IntType(set_type.bit_width)
            check_type = ir.IntType(max(set_type.bit_width, value_type.storage_type.bitwidth))
            result = ir.Constant(bit_type, 0)
            for idx, entry in enumerate(values):
                code = builder.extract_value(args[0], idx)
                if isinstance(entry, InputValueType):
                    code = context.make_helper(builder, entry, value=code).value
                if value_type.storage_type.bitwidth < check_type.width:
                    code = builder.sext(code, check_type) if value_type.storage_type.signed else builder.zext(code, check_type)
                in_range = builder.icmp_unsigned("<", code, ir.Constant(check_type, set_type.bit_width))
                safe_code = builder.select(in_range, code, ir.Constant(check_type, 0))
                if check_type.width > set_type.bit_width:
                    safe_code = builder.trunc(safe_code, bit_type)
                bit = builder.shl(ir.Constant(bit_type, 1), safe_code)
                result = builder.or_(result, builder.select(in_range, bit, ir.Constant(bit_type, 0)))
            return result

        return sig, codegen

    return build
