"""Host registrations for nominal input sources and keyed baskets."""

import inspect
from dataclasses import dataclass

from numba import types
from numba.extending import overload_method, register_jitable


@dataclass(frozen=True)
class SourceField:
    name: str
    numba_type: types.Type

    def __post_init__(self):
        if not self.name.isidentifier() or not isinstance(self.numba_type, types.Type):
            raise TypeError("Source fields need an identifier and a Numba type")


@dataclass(frozen=True)
class KeyFilter:
    """A typed predicate over a basket's per-slot metadata."""

    operation: str
    field: str | None = None
    expected: int = 0

    def __post_init__(self):
        if self.operation == "all":
            if self.field is not None or self.expected != 0:
                raise ValueError("An unfiltered iterator cannot name a field or expected value")
        elif self.operation in {"nonzero", "equals"}:
            if not isinstance(self.field, str) or not self.field.isidentifier() or not isinstance(self.expected, int):
                raise TypeError("Filtered iterators need a field name and integer expected value")
            if self.operation == "nonzero" and self.expected != 0:
                raise ValueError("A nonzero filter cannot set an expected value")
        else:
            raise ValueError(f"Unknown basket filter {self.operation!r}")

    @classmethod
    def all(cls):
        return cls("all")

    @classmethod
    def nonzero(cls, field: str):
        return cls("nonzero", field)

    @classmethod
    def equals(cls, field: str, expected: int):
        return cls("equals", field, expected)

    @property
    def key(self):
        return self.operation, self.field, self.expected


class _SlotIndex:
    pass


SLOT_INDEX = _SlotIndex()


def _require_dense_slots(indices):
    indices = tuple(indices)
    if any(type(index) is not int or index < 0 for index in indices):
        raise TypeError("Basket host slots must be non-negative integers")
    if indices and sorted(indices) != list(range(min(indices), min(indices) + len(indices))):
        raise TypeError("Basket host slots must be contiguous and unique")


def _identity(identity):
    if not isinstance(identity, tuple) or len(identity) != 2 or not all(isinstance(part, str) and part for part in identity):
        raise ValueError("Source identity must be (nonempty namespace, nonempty name)")
    return identity


def _fields(fields):
    fields = tuple(fields)
    if any(not isinstance(field, SourceField) for field in fields):
        raise TypeError("Fields must be SourceField instances")
    names = [field.name for field in fields]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate metadata field")
    return fields


def _method_arguments(function):
    parameters = tuple(inspect.signature(function).parameters.values())
    if not parameters or any(param.kind not in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD) for param in parameters):
        raise TypeError("Source methods require a fixed positional signature")
    return tuple(param.name for param in parameters[1:])


def _install_method(owner, name, implementation, arg_names, *, factory=False):
    if not name.isidentifier():
        raise ValueError(f"Invalid method name {name!r}")
    args = ", ".join(("value", *arg_names))
    body = (
        f"def overload({args}):\n"
        "    if isinstance(value, owner):\n"
        + (
            f"        return implementation({args})\n"
            if factory
            else f"        def impl({args}):\n            return implementation({args})\n        return impl\n"
        )
        + "    return None\n"
    )
    namespace = {"owner": owner, "implementation": implementation}
    exec(body, namespace)  # noqa: S102 - identifiers and fixed template are validated
    overload_method(owner, name)(namespace["overload"])


class InputSourceDescriptor:
    def __init__(self, identity, fields):
        from numba_cfunc_compiler.types.native.input import make_source_type_class

        self.identity = identity
        self.fields = fields
        self.type_class = make_source_type_class(identity, fields)
        self.type_class.source_descriptor = self
        self._methods = {}
        self._frozen = False

    def type_for(self, payload_type, storage_type, semantics, layout_key=(), mutable=False):
        from numba_cfunc_compiler.types.native.input import source_value_type

        return source_value_type(self.type_class, payload_type, storage_type, semantics, layout_key, mutable)

    def value_type(self, binding):
        payload_type, storage_type = binding.input_boundary_types()
        return self.type_for(payload_type, storage_type, binding.payload.semantics, binding.payload.key, binding.payload.mutable)

    def method(self, name, *, overload=False):
        def decorate(function):
            if name == "source":
                raise ValueError("Method name 'source' is reserved for source metadata")
            arg_names = _method_arguments(function)
            existing = self._methods.get(name)
            if existing is not None:
                if existing[0] is not function or existing[1:] != (arg_names, overload):
                    raise ValueError(f"Method {name!r} is already registered on {self.identity}")
                return function
            if self._frozen:
                raise RuntimeError(f"Source methods for {self.identity} are frozen after compilation")
            implementation = function if overload else register_jitable(inline="always")(function)
            _install_method(self.type_class, name, implementation, arg_names, factory=overload)
            self._methods[name] = (function, arg_names, overload)
            return function

        return decorate

    @property
    def key(self):
        return self.identity, tuple((field.name, field.numba_type.name) for field in self.fields)

    def registration_key(self, *, freeze=False):
        if freeze:
            self._frozen = True
        return self.key, tuple(sorted((name, arg_names, overload) for name, (_, arg_names, overload) in self._methods.items()))


class KeyedInputDescriptor:
    def __init__(self, identity, element_source, fields, element_projection):
        from numba_cfunc_compiler.types.native.keyed import make_keyed_type_class

        self.identity = identity
        self.element_source = element_source
        self.fields = fields
        self.element_projection = dict(element_projection)
        self.type_class = make_keyed_type_class(identity)
        self.type_class.basket_descriptor = self
        self._methods = {}
        self._iter_methods = {}
        self._frozen = False

    def value_type(self, layout_key):
        from numba_cfunc_compiler.types.native.keyed import keyed_value_type

        return keyed_value_type(self.type_class, tuple(layout_key))

    def method(self, name, *, overload=False):
        def decorate(function):
            if name == "at" or name in {field.name for field in self.fields}:
                raise ValueError(f"Method name {name!r} conflicts with basket metadata")
            arg_names = _method_arguments(function)
            existing = self._methods.get(name)
            if existing is not None:
                if existing[0] is not function or existing[1:] != (arg_names, overload):
                    raise ValueError(f"Method {name!r} is already registered on {self.identity}")
                return function
            if self._frozen:
                raise RuntimeError(f"Basket methods for {self.identity} are frozen after compilation")
            if name in self._iter_methods:
                raise ValueError(f"Method {name!r} is already registered on {self.identity}")
            implementation = function if overload else register_jitable(inline="always")(function)
            _install_method(self.type_class, name, implementation, arg_names, factory=overload)
            self._methods[name] = (function, arg_names, overload)
            return function

        return decorate

    def iter_method(self, name, *, filter: KeyFilter):
        from numba_cfunc_compiler.types.native.keyed import register_iter_method

        if not name.isidentifier() or not isinstance(filter, KeyFilter):
            raise TypeError("Iterator method needs an identifier and a KeyFilter")
        if name == "at" or name in {field.name for field in self.fields}:
            raise ValueError(f"Iterator method name {name!r} conflicts with basket metadata")
        if filter.operation != "all":
            field = next((field for field in self.fields if field.name == filter.field), None)
            if field is None or not isinstance(field.numba_type, types.CPointer) or not isinstance(field.numba_type.dtype, types.Integer):
                raise TypeError("Basket filter field must be a pointer to integers")
        if name in self._methods:
            raise ValueError(f"Method {name!r} is already registered on {self.identity}")
        existing = self._iter_methods.get(name)
        if existing is not None:
            if existing != filter:
                raise ValueError(f"Iterator method {name!r} is already registered on {self.identity}")
            return
        if self._frozen:
            raise RuntimeError(f"Basket methods for {self.identity} are frozen after compilation")
        self._iter_methods[name] = filter
        register_iter_method(self.type_class, name)

    @property
    def key(self):
        projection = tuple(sorted((name, "slot_index" if value is SLOT_INDEX else value) for name, value in self.element_projection.items()))
        return self.identity, self.element_source.key, tuple((field.name, field.numba_type.name) for field in self.fields), projection

    def registration_key(self, *, freeze=False):
        if freeze:
            self._frozen = True
            self.element_source.registration_key(freeze=True)
        return (
            self.key,
            tuple(sorted((name, arg_names, overload) for name, (_, arg_names, overload) in self._methods.items())),
            tuple(sorted((name, filter.key) for name, filter in self._iter_methods.items())),
        )


_SOURCES = {}
_BASKETS = {}


def register_input_source(*, identity, fields):
    identity = _identity(identity)
    fields = _fields(fields)
    existing = _SOURCES.get(identity)
    if existing is not None:
        if existing.fields != fields:
            raise ValueError(f"Source identity {identity!r} already has a different layout")
        return existing
    source = InputSourceDescriptor(identity, fields)
    _SOURCES[identity] = source
    return source


def register_keyed_input(*, identity, element_source, fields, element_projection):
    identity = _identity(identity)
    if not isinstance(element_source, InputSourceDescriptor):
        raise TypeError("Basket element_source must be an input source descriptor")
    fields = _fields(fields)
    field_types = {field.name: field.numba_type for field in fields}
    projection = dict(element_projection)
    if set(projection) != {field.name for field in element_source.fields}:
        raise ValueError("Basket projection must provide every element metadata field")
    for source_field in element_source.fields:
        projected = projection[source_field.name]
        if projected is SLOT_INDEX:
            if source_field.numba_type != types.intp:
                raise TypeError("Projected slot index must have intp type")
        elif not isinstance(projected, str) or field_types.get(projected) != source_field.numba_type:
            raise TypeError(f"Invalid basket projection for {source_field.name!r}")
    required = {"slots": types.CPointer(types.voidptr), "start": types.intp, "length": types.intp}
    if any(field_types.get(name) != typ for name, typ in required.items()):
        raise TypeError("Basket fields must include slots:void**, start:intp, and length:intp")
    existing = _BASKETS.get(identity)
    if existing is not None:
        if existing.element_source is not element_source or existing.fields != fields or existing.element_projection != projection:
            raise ValueError(f"Basket identity {identity!r} already has a different layout")
        return existing
    basket = KeyedInputDescriptor(identity, element_source, fields, projection)
    _BASKETS[identity] = basket
    return basket
