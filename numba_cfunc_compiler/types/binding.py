"""Resolved host type contracts used throughout one compilation."""

import ast
from dataclasses import dataclass, field
from typing import Any

from numba_cfunc_compiler.types.native.state import CopyStateValueType, StatePayload


@dataclass(frozen=True)
class ConstantPlan:
    setup: tuple[ast.stmt, ...]
    value: ast.expr


@dataclass(frozen=True)
class StatePlan:
    before: tuple[ast.stmt, ...] = ()
    start: tuple[ast.stmt, ...] = ()
    execute: tuple[ast.stmt, ...] = ()
    stop_before: tuple[ast.stmt, ...] = ()
    stop_after: tuple[ast.stmt, ...] = ()
    metadata: dict[str, tuple] = field(default_factory=dict)


@dataclass(frozen=True)
class TypeBinding:
    """One resolved representation; HostType remains the public extension API."""

    host_type: Any
    payload: StatePayload

    @property
    def host_size(self) -> int:
        return self.payload.host_size

    @property
    def alignment(self) -> int:
        return self.payload.alignment

    def _boundary_types(self):
        native = self.payload.native_type
        if isinstance(native, CopyStateValueType):
            return native.payload_type, native.storage_type
        return native, native

    def output_sink_type(self):
        from numba_cfunc_compiler.types.native.output import output_sink_type

        native, storage = self._boundary_types()
        return output_sink_type(native, storage, self.payload.semantics, self.host_size, self.payload.key)

    def constant_plan(self, name: str, value: Any, call_globals: dict) -> ConstantPlan:
        return self.host_type.constant_plan(name, value, call_globals)

    def slot_read(self, name: str, slot: ast.expr, variable_factory) -> ast.stmt | list[ast.stmt]:
        return self.host_type.slot_read(name, slot, variable_factory)

    def state_plan(self, variable) -> StatePlan:
        return self.host_type.state_plan(variable)

    def is_opaque_pointer(self) -> bool:
        return self.host_type.is_opaque_pointer()

    def lower_loaded_value(self, value: ast.expr) -> ast.expr:
        return self.host_type.lower_loaded_value(value)
