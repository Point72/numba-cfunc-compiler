"""Callback ABI names and reserved names emitted into generated cfunc code."""

import hashlib

__all__ = [
    "INTERNAL_PREFIX",
    "LIFECYCLE_EXECUTE",
    "LIFECYCLE_PARAM_NAME",
    "LIFECYCLE_START",
    "LIFECYCLE_STOP",
    "OUTPUTS_ARRAY_NAME",
    "STATE_ARRAY_NAME",
    "TICKED_OUTPUTS_ARRAY_NAME",
    "UNWRAP_SOURCE_NAME",
    "bind_input_name",
    "bind_keyed_basket_name",
    "bind_keyed_input_name",
    "bind_output_name",
    "bind_state_name",
    "container_ptr_name",
    "enum_literal_name",
    "enumset_builder_name",
    "output_sink_name",
    "state_aug_name",
    "state_init_name",
    "state_loaded_name",
    "state_slot_name",
    "state_store_name",
    "state_value_name",
    "typed_struct_name",
]

INTERNAL_PREFIX = "__ncc_"

# Callback ABI names and phase values. Hosts use the same parameter order and
# integer phase values when calling the compiled function.
STATE_ARRAY_NAME = "state"
OUTPUTS_ARRAY_NAME = "outputs"
TICKED_OUTPUTS_ARRAY_NAME = "output_ticked"
LIFECYCLE_PARAM_NAME = "lifecycle_phase"
LIFECYCLE_EXECUTE = 0
LIFECYCLE_START = 1
LIFECYCLE_STOP = 2


def _internal_name(*parts: str | int) -> str:
    return INTERNAL_PREFIX + "_".join(str(part) for part in parts)


def _fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:16]


UNWRAP_SOURCE_NAME = _internal_name("unwrap_source")


def bind_input_name(name: str) -> str:
    return _internal_name("bind_input", name)


def bind_keyed_basket_name(name: str) -> str:
    return _internal_name("bind_keyed_basket", name)


def bind_keyed_input_name(type_name: str) -> str:
    return _internal_name("bind_keyed_input", _fingerprint(type_name))


def bind_output_name(index: int) -> str:
    return _internal_name("bind_output", index)


def bind_state_name(name: str) -> str:
    return _internal_name("bind_state", name)


def container_ptr_name(name: str) -> str:
    return _internal_name("container_ptr", name)


def enum_literal_name(family_key: tuple, member_name: str) -> str:
    return _internal_name("enum_literal", _fingerprint(repr(family_key)), member_name)


def enumset_builder_name(family_key: tuple) -> str:
    return _internal_name("enumset_builder", _fingerprint(repr(family_key)))


def output_sink_name(index: int) -> str:
    return _internal_name("output_sink", index)


def state_init_name(name: str) -> str:
    return _internal_name("state_init", name)


def state_aug_name(name: str, index: int) -> str:
    return _internal_name("state_aug", name, index)


def state_loaded_name(name: str) -> str:
    return _internal_name("state_loaded", name)


def state_slot_name(name: str) -> str:
    return _internal_name("state_slot", name)


def state_store_name(name: str) -> str:
    return _internal_name("state_store", name)


def state_value_name(name: str) -> str:
    return _internal_name("state_value", name)


def typed_struct_name(operation: str, fingerprint: str) -> str:
    return _internal_name("typed_struct", operation, fingerprint)
