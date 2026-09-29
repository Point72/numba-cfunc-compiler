"""Tests for the process-wide ABI configuration."""

from dataclasses import FrozenInstanceError

import pytest

from numba_cfunc_compiler import config, get_numba_config, numba_config, set_numba_config
from numba_cfunc_compiler.compilation_context import CompilationContext
from numba_cfunc_compiler.config import NumbaConfig
from numba_cfunc_compiler.node_api import State
from numba_cfunc_compiler.type_registry import NumbaTypeRegistry
from numba_cfunc_compiler.utils.enum import enum_representation
from numba_cfunc_compiler.utils.enumset import enumset_type


def test_legacy_exports():
    assert numba_config.State is State
    assert numba_config.NumbaTypeRegistry is NumbaTypeRegistry
    assert numba_config.get_numba_config is get_numba_config
    assert numba_config.set_numba_config is set_numba_config


@pytest.mark.parametrize(
    ("enum_width", "enumset_width"),
    [(7, 64), (True, 64), (32, 0), (32, 7), (32, True)],
)
def test_invalid_widths(enum_width, enumset_width):
    with pytest.raises(ValueError):
        NumbaConfig(enum_width, enumset_width)


def test_config_is_frozen():
    with pytest.raises(FrozenInstanceError):
        get_numba_config().enum_bit_width = 32


def test_setter_is_one_time():
    get_numba_config()
    with pytest.raises(RuntimeError, match="already set"):
        set_numba_config(enum_bit_width=32, enumset_bit_width=64)


def test_lazy_defaults(monkeypatch):
    monkeypatch.setattr(config, "_numba_config", None)
    with CompilationContext():
        assert get_numba_config() == NumbaConfig()
        assert enum_representation.bit_width == 16
        assert enumset_type.byte_width == 16
    with pytest.raises(RuntimeError, match="already set"):
        set_numba_config(enum_bit_width=32)


def test_custom_widths(monkeypatch):
    monkeypatch.setattr(config, "_numba_config", None)
    set_numba_config(enum_bit_width=32, enumset_bit_width=64)
    assert enum_representation.bit_width == 32
    assert enumset_type.byte_width == 8
