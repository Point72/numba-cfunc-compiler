"""Nominal host baskets and typed filtered position iterators."""

import hashlib
import operator
from functools import cache

from llvmlite import ir
from numba import types
from numba.core import cgutils
from numba.core.datamodel import models
from numba.core.imputils import RefType, impl_ret_borrowed, iternext_impl
from numba.core.typing.templates import AttributeTemplate
from numba.extending import infer_getattr, intrinsic, lower_builtin, lower_getattr_generic, overload, overload_method, register_model

from numba_cfunc_compiler.extension.input_sources import SLOT_INDEX, _require_dense_slots
from numba_cfunc_compiler.types.native.input import InputValueType, construct_source_value, unwrap_source


class KeyedValueType(types.IterableType):
    def __init__(self, layout_key):
        layout_key = tuple(layout_key)
        _require_dense_slots(index for _, index, _ in layout_key)
        self.descriptor = type(self).basket_descriptor
        self.layout_key = layout_key
        super().__init__(f"KeyedValue[{self.descriptor.key!r}:{layout_key}]")

    @property
    def key(self):
        return self.descriptor.key, self.layout_key

    @property
    def iterator_type(self):
        from numba_cfunc_compiler.extension.input_sources import KeyFilter

        return KeyedIteratorType(self, KeyFilter.all())


class KeyedIterableType(types.IterableType):
    def __init__(self, parent, filter):
        self.parent = parent
        self.filter = filter
        super().__init__(f"KeyedIterable[{filter.key}:{parent}]")

    @property
    def key(self):
        return self.parent, self.filter.key

    @property
    def iterator_type(self):
        return KeyedIteratorType(self.parent, self.filter)


class KeyedIteratorType(types.SimpleIteratorType):
    def __init__(self, parent, filter):
        self.parent = parent
        self.filter = filter
        super().__init__(f"KeyedIterator[{filter.key}:{parent}]", types.intp)

    @property
    def key(self):
        return self.parent, self.filter.key


@register_model(KeyedValueType)
@register_model(KeyedIterableType)
class KeyedValueModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        descriptor = fe_type.parent.descriptor if isinstance(fe_type, KeyedIterableType) else fe_type.descriptor
        super().__init__(dmm, fe_type, [(field.name, field.numba_type) for field in descriptor.fields])


@register_model(KeyedIteratorType)
class KeyedIteratorModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        super().__init__(dmm, fe_type, [("parent", fe_type.parent), ("index", types.EphemeralPointer(types.intp))])


@infer_getattr
class KeyedValueAttributes(AttributeTemplate):
    key = KeyedValueType

    def generic_resolve(self, typ, attr):
        for field in typ.descriptor.fields:
            if field.name == attr:
                return field.numba_type
        return None


@lower_getattr_generic(KeyedValueType)
def keyed_value_getattr(context, builder, typ, value, attr):
    if attr not in {field.name for field in typ.descriptor.fields}:
        raise AttributeError(attr)
    return getattr(context.make_helper(builder, typ, value=value), attr)


def make_keyed_type_class(identity):
    token = hashlib.sha256(repr(identity).encode()).hexdigest()[:12]
    basket_class = type(f"KeyedInput_{token}", (KeyedValueType,), {"basket_descriptor": None})
    register_model(basket_class)(KeyedValueModel)
    return basket_class


@cache
def keyed_value_type(type_class, layout_key):
    return type_class(layout_key)


@cache
def bind_keyed(value_type):
    @intrinsic
    def bind(typingctx, field_values):
        fields = value_type.descriptor.fields
        if not isinstance(field_values, types.BaseTuple) or len(field_values) != len(fields):
            return None
        for field, actual in zip(fields, field_values):
            if actual != field.numba_type and not (field.numba_type == types.intp and isinstance(actual, types.Integer)):
                return None
        sig = value_type(field_values)

        def codegen(context, builder, signature, args):
            result = context.make_helper(builder, value_type)
            values = cgutils.unpack_tuple(builder, args[0], len(fields))
            for position, field in enumerate(fields):
                setattr(result, field.name, context.cast(builder, values[position], field_values[position], field.numba_type))
            return result._getvalue()

        return sig, codegen

    return bind


@cache
def _item_at(item_type):
    @intrinsic
    def at(typingctx, basket, index):
        if not isinstance(basket, KeyedValueType) or not isinstance(index, types.Integer):
            return None
        sig = item_type(basket, index)

        def codegen(context, builder, signature, args):
            parent = context.make_helper(builder, basket, value=args[0])
            if basket.descriptor.element_source is not item_type.source_descriptor:
                raise TypeError("Basket element source does not match its declared source")
            idx = context.cast(builder, args[1], signature.args[1], types.intp)
            slot = builder.load(builder.gep(parent.slots, [idx]))
            metadata = []
            for field in item_type.source_descriptor.fields:
                projection = basket.descriptor.element_projection[field.name]
                metadata.append(idx if projection is SLOT_INDEX else getattr(parent, projection))
            return construct_source_value(context, builder, item_type, slot, metadata)

        return sig, codegen

    return at


def _static_item_impl(item_type, index):
    at = _item_at(item_type)

    def impl(basket, key):
        return at(basket, index)

    return impl


@overload(operator.getitem, prefer_literal=True)
def keyed_getitem(basket, key):
    if not isinstance(basket, KeyedValueType):
        return None
    if isinstance(key, types.Literal):
        for candidate, index, item_type in basket.layout_key:
            if type(candidate) is type(key.literal_value) and candidate == key.literal_value:
                return _static_item_impl(item_type, index)
    return None


@overload_method(KeyedValueType, "at")
def keyed_at(basket, position):
    if not isinstance(basket, KeyedValueType):
        return None
    index_type = position.payload_type if isinstance(position, InputValueType) else position
    if not isinstance(index_type, types.Integer):
        return None
    item_types = {item_type for _, _, item_type in basket.layout_key}
    if len(item_types) != 1:
        return None
    at = _item_at(next(iter(item_types)))
    start = min(index for _, index, _ in basket.layout_key)

    def impl(basket, position):
        return at(basket, start + unwrap_source(position))

    return impl


@intrinsic
def keyed_length(typingctx, value):
    if not isinstance(value, KeyedValueType):
        return None
    sig = types.intp(value)

    def codegen(context, builder, signature, args):
        return context.make_helper(builder, value, value=args[0]).length

    return sig, codegen


@overload(len)
def overload_keyed_len(value):
    if isinstance(value, KeyedValueType):

        def impl(value):
            return keyed_length(value)

        return impl


def register_iter_method(owner, name):
    @overload_method(owner, name)
    def overload_iter_method(value):
        if not isinstance(value, owner):
            return None
        filter = value.descriptor._iter_methods.get(name)
        if filter is None:
            return None
        iterable_type = KeyedIterableType(value, filter)

        @intrinsic
        def cast(typingctx, source):
            if source != value:
                return None
            sig = iterable_type(source)

            def codegen(context, builder, signature, args):
                return args[0]

            return sig, codegen

        def impl(value):
            return cast(value)

        return impl


@lower_builtin("getiter", KeyedValueType)
@lower_builtin("getiter", KeyedIterableType)
def lower_keyed_getiter(context, builder, sig, args):
    iterator = context.make_helper(builder, sig.return_type)
    iterator.parent = args[0]
    index = cgutils.alloca_once(builder, context.get_value_type(types.intp))
    builder.store(context.get_constant(types.intp, 0), index)
    iterator.index = index
    return impl_ret_borrowed(context, builder, sig.return_type, iterator._getvalue())


@lower_builtin("iternext", KeyedIteratorType)
@iternext_impl(RefType.BORROWED)
def lower_keyed_iternext(context, builder, sig, args, result):
    iterator_type = sig.args[0]
    iterator = context.make_helper(builder, iterator_type, args[0])
    parent = context.make_helper(builder, iterator_type.parent, iterator.parent)
    scan = builder.append_basic_block("keyed.scan")
    candidate = builder.append_basic_block("keyed.candidate")
    found = builder.append_basic_block("keyed.found")
    skip = builder.append_basic_block("keyed.skip")
    exhausted = builder.append_basic_block("keyed.exhausted")
    done = builder.append_basic_block("keyed.done")
    builder.branch(scan)

    builder.position_at_end(scan)
    index = builder.load(iterator.index)
    builder.cbranch(builder.icmp_signed("<", index, parent.length), candidate, exhausted)

    builder.position_at_end(candidate)
    filter = iterator_type.filter
    if filter.operation == "all":
        builder.branch(found)
    else:
        absolute = builder.add(parent.start, index)
        flags = getattr(parent, filter.field)
        flag_value = builder.load(builder.gep(flags, [absolute]))
        expected = ir.Constant(flag_value.type, filter.expected if filter.operation == "equals" else 0)
        predicate = builder.icmp_unsigned("==" if filter.operation == "equals" else "!=", flag_value, expected)
        builder.cbranch(predicate, found, skip)

    builder.position_at_end(found)
    result.set_valid(ir.Constant(ir.IntType(1), 1))
    result.yield_(index)
    builder.store(builder.add(index, context.get_constant(types.intp, 1)), iterator.index)
    builder.branch(done)

    builder.position_at_end(skip)
    builder.store(builder.add(index, context.get_constant(types.intp, 1)), iterator.index)
    builder.branch(scan)

    builder.position_at_end(exhausted)
    result.set_valid(ir.Constant(ir.IntType(1), 0))
    builder.branch(done)
    builder.position_at_end(done)
