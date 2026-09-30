"""Shared lifecycle plan for standalone list and dictionary state."""

import ast
import copy

from numba_cfunc_compiler.core.names import (
    bind_state_name,
    state_init_name,
    state_loaded_name,
    state_slot_name,
    state_value_name,
)
from numba_cfunc_compiler.types.binding import StatePlan
from numba_cfunc_compiler.utils.ast import AST


def container_state_plan(host_type, variable) -> StatePlan:
    name = variable.name
    slot = AST.array_access(variable.get_storage_location(), variable.array_idx)
    init_name = state_init_name(name)
    loaded_name = state_loaded_name(name)
    value_name = state_value_name(name)

    start = list(host_type.create_new_container(init_name))
    pointer = AST.function_call(host_type.to_voidptr_func_name(), ast.Name(id=init_name, ctx=ast.Load()))
    start.extend(
        [
            AST.assignment(loaded_name, pointer),
            AST.assignment(copy.deepcopy(slot), ast.Name(id=loaded_name, ctx=ast.Load())),
            AST.assignment(name, AST.function_call(bind_state_name(name), ast.Name(id=init_name, ctx=ast.Load()))),
        ]
    )

    load = [
        AST.assignment(loaded_name, copy.deepcopy(slot)),
        host_type.from_voidptr(value_name, name, ast.Name(id=loaded_name, ctx=ast.Load())),
        AST.assignment(name, AST.function_call(bind_state_name(name), ast.Name(id=value_name, ctx=ast.Load()))),
    ]
    stop_after = [
        ast.Expr(value=AST.function_call(host_type.free_func_name(), ast.Name(id=name, ctx=ast.Load()))),
        AST.assignment(copy.deepcopy(slot), AST.function_call("voidptr_null")),
    ]
    return StatePlan(
        before=(AST.assignment(state_slot_name(name), copy.deepcopy(slot)),),
        start=tuple(start),
        execute=tuple(load),
        stop_before=tuple(copy.deepcopy(load)),
        stop_after=tuple(stop_after),
        metadata={"nrt_state_indices": (variable.array_idx,)},
    )
