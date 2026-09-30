"""Explicit host storage bindings for client-defined Numba value types."""

from dataclasses import dataclass, replace

from numba import types

from numba_cfunc_compiler.extension.methods import register_value_method, register_value_overload
from numba_cfunc_compiler.types.builtin.struct.native import StructLayout, struct_ptr_type
from numba_cfunc_compiler.types.native.ffi import FFIRefType
from numba_cfunc_compiler.types.native.state import borrowed_ref_payload, borrowed_struct_payload, nominal_copy_state_payload
from numba_cfunc_compiler.types.policy import ValueSemantics


@dataclass(frozen=True)
class ValueBinding:
    """Native representation, host storage contract, and Numba methods."""

    python_type: type
    numba_type: types.Type
    storage: object
    semantics: ValueSemantics
    mutable: bool
    owner: str | None
    payload: object

    def state_payload(self):
        return self.payload

    def method(self, name: str):
        def decorate(implementation):
            register_value_method(self.numba_type, name, implementation)
            return implementation

        return decorate

    def overload_method(self, name: str):
        def decorate(factory):
            register_value_overload(self.numba_type, name, factory)
            return factory

        return decorate


_BINDINGS = {}


def register_value_type(python_type, *, numba_type, storage, semantics: ValueSemantics, mutable=False, owner=None) -> ValueBinding:
    """Bind a client value to explicit copy or host-owned view semantics.

    Copied scalar storage is a Numba type, or ``(type, size, alignment)``.
    Borrowed struct storage is a :class:`StructLayout`; borrowed FFI storage
    is a stable layout key tuple. Borrowed views require an owner contract.
    The host descriptor uses ``binding.state_payload()`` for source storage.
    """
    if not isinstance(python_type, type) or not isinstance(numba_type, types.Type):
        raise TypeError("A Python class and a native Numba type are required")
    if not isinstance(semantics, ValueSemantics):
        raise TypeError("Value semantics must be a ValueSemantics member")
    if not isinstance(mutable, bool):
        raise TypeError("Mutation policy must be a boolean")
    nominal_key = (python_type.__module__, python_type.__qualname__)
    if semantics is ValueSemantics.COPY:
        if owner is not None or mutable:
            raise ValueError("Copied values cannot declare a borrowed owner or in-place mutation")
        if isinstance(storage, tuple) and len(storage) == 3:
            storage_type, size, alignment = storage
        elif isinstance(storage, types.Type):
            storage_type = storage
            width = getattr(storage, "bitwidth", getattr(storage, "bit_width", 0))
            if width <= 0 or width % 8:
                raise ValueError("Copied storage needs an explicit byte size and alignment")
            size = alignment = width // 8
        else:
            raise TypeError("Copied storage must be a Numba type or (type, size, alignment)")
        payload = nominal_copy_state_payload(nominal_key, numba_type, storage_type, size, alignment)
    else:
        if not isinstance(owner, str) or not owner:
            raise ValueError("Borrowed views require a named host owner/lifetime contract")
        if isinstance(storage, StructLayout):
            if numba_type != struct_ptr_type(storage):
                raise TypeError("Borrowed struct Numba type must match its layout")
            payload = borrowed_struct_payload(storage)
        elif isinstance(numba_type, FFIRefType) and isinstance(storage, tuple):
            if mutable:
                raise ValueError("Borrowed FFI references do not support in-place mutation")
            payload = borrowed_ref_payload(python_type.__qualname__, numba_type, storage)
        else:
            raise TypeError("Borrowed views require a StructLayout or an FFI reference layout key")
        payload = replace(payload, layout_key=(payload.layout_key, owner, mutable), mutable=mutable)
    binding = ValueBinding(python_type, numba_type, storage, semantics, mutable, owner, payload)
    existing = _BINDINGS.get(python_type)
    if existing is not None:
        if (existing.numba_type, existing.storage, existing.semantics, existing.mutable, existing.owner) != (
            numba_type,
            storage,
            semantics,
            mutable,
            owner,
        ):
            raise ValueError(f"{python_type} already has a different value binding")
        return existing
    _BINDINGS[python_type] = binding
    return binding


def value_binding(python_type) -> ValueBinding:
    return _BINDINGS[python_type]
