"""Numba IR passes for declared state assignments and user local types."""

import dis
import re
from collections import Counter
from dataclasses import dataclass
from functools import cache

from numba import types
from numba.core import ir
from numba.core.compiler import Compiler, DefaultPassBuilder, LiteralUnroll
from numba.core.compiler_machinery import FunctionPass, register_pass
from numba.core.errors import NotDefinedError, RedefinedError, TypingError
from numba.core.ir_utils import build_definitions
from numba.core.typed_passes import NopythonTypeInference

from numba_cfunc_compiler.state_values import StatePayload

PIPELINE_VERSION = 1
STABLE_LOCAL_POLICY_VERSION = 1


@dataclass(frozen=True)
class StateBinding:
    """Immutable compiler metadata for a declared host state cell."""

    name: str
    slot_name: str
    payload: StatePayload

    def __post_init__(self):
        if not self.name.isidentifier() or not self.slot_name.isidentifier() or self.name == self.slot_name:
            raise ValueError("State and slot names must be distinct Python identifiers")

    @property
    def bind_function(self):
        return self.payload.bind_function

    @property
    def store_function(self):
        return self.payload.store_function(self.name)


@register_pass(mutates_CFG=False, analysis_only=False)
class StateAssignmentPass(FunctionPass):
    _name = "state_assignment"

    def __init__(self):
        super().__init__()

    def run_pass(self, state):
        bindings = state.state_bindings
        func_ir = state.func_ir
        by_name = {binding.name: binding for binding in bindings}
        if len(by_name) != len(bindings):
            raise ValueError("Duplicate state binding name")
        slot_names = {binding.slot_name for binding in bindings}
        if len(slot_names) != len(bindings) or slot_names & by_name.keys():
            raise ValueError("Generated state slots must have unique names that do not shadow state names")

        self._validate_provenance(func_ir, by_name)
        marker_counts = {name: 0 for name in by_name}
        slot_counts = {name: 0 for name in slot_names}
        frontend_copies = self._frontend_copies(func_ir)
        chained_sources = self._chained_bindings(func_ir, by_name)
        raw_values = {}
        changed = False
        for block in func_ir.blocks.values():
            body = []
            for stmt in block.body:
                if not isinstance(stmt, ir.Assign):
                    body.append(stmt)
                    continue
                name = stmt.target.unversioned_name
                if name in slot_counts:
                    slot_counts[name] += 1
                copied_result = frontend_copies.get(id(stmt))
                source = chained_sources.get(id(stmt))
                if source is not None and id(source) not in raw_values:
                    raise TypingError(f"Cannot identify chained assignment value for '{name}'", loc=stmt.loc)
                value = raw_values[id(source)] if source is not None else stmt.value
                binding = by_name.get(name)
                if binding is None:
                    if copied_result is None and source is None:
                        body.append(stmt)
                    else:
                        rhs = stmt.target.scope.make_temp(stmt.loc)
                        body.extend(
                            [
                                ir.Assign(value, rhs, stmt.loc),
                                ir.Assign(rhs, stmt.target, stmt.loc),
                                *([ir.Assign(rhs, copied_result, stmt.loc)] if copied_result is not None else []),
                            ]
                        )
                        changed = True
                    continue
                if self._is_marker(func_ir, stmt, binding):
                    marker_counts[name] += 1
                    body.append(stmt)
                    continue
                if isinstance(stmt.value, ir.Arg):
                    raise TypingError(f"State name '{name}' cannot be a callback parameter", loc=stmt.loc)
                try:
                    slot = stmt.target.scope.get_exact(binding.slot_name)
                except NotDefinedError as exc:
                    raise TypingError(f"Missing generated slot '{binding.slot_name}' for State '{name}'", loc=stmt.loc) from exc
                scope = stmt.target.scope
                rhs = scope.make_temp(stmt.loc)
                store_global = scope.make_temp(stmt.loc)
                body.extend(
                    [
                        ir.Assign(value, rhs, stmt.loc),
                        *([ir.Assign(rhs, copied_result, stmt.loc)] if copied_result is not None else []),
                        ir.Assign(ir.Global(f"_state_store_{name}", binding.store_function, stmt.loc), store_global, stmt.loc),
                        ir.Assign(ir.Expr.call(store_global, [slot, rhs], (), stmt.loc), stmt.target, stmt.loc),
                    ]
                )
                raw_values[id(stmt)] = rhs
                changed = True
            block.body = body
        for name, count in marker_counts.items():
            if not count:
                raise TypingError(f"State '{name}' has no generated bind marker")
        for name, count in slot_counts.items():
            if count != 1:
                raise TypingError(f"Generated slot '{name}' must be bound exactly once; found {count} assignments")
        if changed:
            func_ir._definitions = build_definitions(func_ir.blocks)
        return changed

    @staticmethod
    def _validate_provenance(func_ir, by_name):
        """Require every state-name IR assignment to match a local store."""
        stores = Counter(
            (line, instruction.argval)
            for instruction, line in StateAssignmentPass._bytecode_with_lines(func_ir.func_id.func)
            if instruction.opname == "STORE_FAST" and instruction.argval in by_name
        )
        assignments = Counter(
            (stmt.loc.line, stmt.target.unversioned_name)
            for block in func_ir.blocks.values()
            for stmt in block.body
            if isinstance(stmt, ir.Assign) and stmt.target.unversioned_name in by_name
        )
        if stores != assignments:
            line, name = next(key for key in stores.keys() | assignments.keys() if stores[key] != assignments[key])
            raise TypingError(f"Cannot identify State '{name}' assignment provenance at line {line}")

    @staticmethod
    def _bytecode_with_lines(func):
        """Read instruction lines on both Python 3.10 and 3.11+."""
        line = func.__code__.co_firstlineno
        for instruction in dis.get_instructions(func):
            if instruction.starts_line is not None:
                line = instruction.starts_line
            position = getattr(instruction, "positions", None)
            yield instruction, position.lineno if position is not None and position.lineno is not None else line

    @staticmethod
    def _chained_bindings(func_ir, by_name):
        """Find later targets of a duplicated RHS whose first target is state."""
        bytecode = list(StateAssignmentPass._bytecode_with_lines(func_ir.func_id.func))
        instructions = [instruction for instruction, _ in bytecode]
        line_by_offset = {instruction.offset: line for instruction, line in bytecode}
        stores = {}
        for instruction in instructions:
            if instruction.opname == "STORE_FAST":
                key = (line_by_offset[instruction.offset], instruction.argval)
                stores.setdefault(key, []).append(instruction.offset)
        assignments = {}
        for block in func_ir.blocks.values():
            for stmt in block.body:
                if isinstance(stmt, ir.Assign):
                    assignments.setdefault((stmt.loc.line, stmt.target.unversioned_name), []).append(stmt)

        def assignment_for(instruction):
            key = (line_by_offset[instruction.offset], instruction.argval)
            candidates = assignments.get(key, ())
            offsets = stores[key]
            if len(candidates) != len(offsets):
                raise TypingError(f"Cannot identify chained assignment binding for '{key[1]}' at line {key[0]}")
            return candidates[offsets.index(instruction.offset)]

        chained = {}
        for i, instruction in enumerate(instructions):
            if instruction.opname != "STORE_FAST" or instruction.argval not in by_name or i == 0:
                continue
            if instructions[i - 1].opname not in ("COPY", "DUP_TOP"):
                continue
            source = None
            j = i + 1
            while j < len(instructions):
                if instructions[j].opname in ("COPY", "DUP_TOP"):
                    j += 1
                if j >= len(instructions) or instructions[j].opname != "STORE_FAST":
                    break
                if source is None:
                    source = assignment_for(instruction)
                follower = assignment_for(instructions[j])
                if not isinstance(follower.value, ir.Var):
                    raise TypingError(f"Cannot identify chained assignment value for '{follower.target.unversioned_name}'", loc=follower.loc)
                chained[id(follower)] = source
                j += 1
        return chained

    @staticmethod
    def _frontend_copies(func_ir):
        """Restore expression values lost by Numba 0.67's named-expression frontend.

        CPython duplicates a walrus RHS immediately before STORE_FAST. Numba
        assigns the RHS to the local but can leave its duplicated stack value
        as an undefined temporary. Its bytecode offset identifies the
        instruction boundary after the RHS producer and before COPY (DUP_TOP
        on Python 3.10). Match stores on a source line in bytecode and IR
        order, failing closed if either representation is ambiguous.
        """
        bytecode = list(StateAssignmentPass._bytecode_with_lines(func_ir.func_id.func))
        instructions = [instruction for instruction, _ in bytecode]
        line_by_offset = {instruction.offset: line for instruction, line in bytecode}
        copy_stores = {}
        stores = {}
        for i, instruction in enumerate(instructions):
            if instruction.opname != "STORE_FAST":
                continue
            key = (line_by_offset[instruction.offset], instruction.argval)
            stores.setdefault(key, []).append(instruction.offset)
            if i >= 2 and instructions[i - 1].opname in ("COPY", "DUP_TOP"):
                copy_stores[instructions[i - 2].offset + 2] = (key, instruction.offset)

        definitions = {stmt.target.name for block in func_ir.blocks.values() for stmt in block.body if isinstance(stmt, ir.Assign)}
        undefined = {var.name for block in func_ir.blocks.values() for stmt in block.body for var in stmt.list_vars() if var.name not in definitions}
        needed = {}
        for name in undefined:
            if not name.startswith("$"):
                continue
            match = re.match(r"\$(\d+)", name) or re.search(r"(\d+)\.\d+(?:\.\d+)?$", name)
            if match is None:
                continue
            offset = int(match.group(1))
            if offset in copy_stores:
                needed[offset] = name
        if not needed:
            return {}

        assignments = {}
        for block in func_ir.blocks.values():
            for stmt in block.body:
                if isinstance(stmt, ir.Assign):
                    assignments.setdefault((stmt.loc.line, stmt.target.unversioned_name), []).append(stmt)
        repaired = {}
        for offset, name in needed.items():
            key, store_offset = copy_stores[offset]
            bytecode_stores = stores[key]
            ir_assignments = assignments.get(key, ())
            if len(bytecode_stores) != len(ir_assignments):
                raise TypingError(f"Cannot identify named-expression binding for '{key[1]}' at line {key[0]}")
            stmt = ir_assignments[bytecode_stores.index(store_offset)]
            repaired[id(stmt)] = stmt.target.scope.get_exact(name)
        return repaired

    @staticmethod
    def _is_marker(func_ir, stmt, binding):
        value = stmt.value
        if not isinstance(value, ir.Expr) or value.op != "call":
            return False
        try:
            definition = func_ir.get_definition(value.func)
        except (KeyError, RedefinedError):
            return False
        return isinstance(definition, (ir.Global, ir.FreeVar)) and definition.value is binding.bind_function


@register_pass(mutates_CFG=False, analysis_only=True)
class StableLocalTypesPass(FunctionPass):
    _name = "stable_local_types"

    def __init__(self):
        super().__init__()

    def run_pass(self, state):
        user_names = state.stable_local_names
        if not user_names:
            return False
        first_types = {}
        for block in state.func_ir.blocks.values():
            for stmt in block.body:
                if not isinstance(stmt, ir.Assign):
                    continue
                name = stmt.target.unversioned_name
                if name not in user_names:
                    continue
                rhs = stmt.value
                if isinstance(rhs, ir.Expr) and rhs.op == "phi":
                    continue
                target_type = state.typemap.get(stmt.target.name)
                if target_type is None:
                    continue
                target_type = types.unliteral(target_type)
                if isinstance(rhs, ir.Var):
                    rhs_type = state.typemap.get(rhs.name)
                elif isinstance(rhs, ir.Expr):
                    signature = state.calltypes.get(rhs)
                    rhs_type = signature.return_type if signature is not None else None
                else:
                    rhs_type = None
                if rhs_type is not None and types.unliteral(rhs_type) != target_type:
                    raise TypingError(f"Local '{name}' assignment changes type from {rhs_type} to {target_type}", loc=stmt.loc)
                previous = first_types.setdefault(name, target_type)
                if target_type != previous:
                    raise TypingError(f"Local '{name}' changes type from {previous} to {target_type}", loc=stmt.loc)
        return False


@cache
def make_state_pipeline(bindings: tuple[StateBinding, ...], stable_local_names: tuple[str, ...] = ()):
    """Build a cfunc pipeline with immutable state-binding metadata."""
    if not isinstance(bindings, tuple) or not all(isinstance(binding, StateBinding) for binding in bindings):
        raise TypeError("bindings must be a tuple of StateBinding")

    class StateCompiler(Compiler):
        def define_pipelines(self):
            self.state.state_bindings = bindings
            self.state.stable_local_names = frozenset(stable_local_names)
            pipeline = DefaultPassBuilder.define_nopython_pipeline(self.state)
            pipeline.add_pass_after(StateAssignmentPass, LiteralUnroll)
            pipeline.add_pass_after(StableLocalTypesPass, NopythonTypeInference)
            pipeline.finalize()
            return [pipeline]

    return StateCompiler
