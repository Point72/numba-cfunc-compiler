"""Shared lifecycle plan for standalone list and dictionary state."""

import ast
import copy

from numba_cfunc_compiler.core.names import (
    STATE_ARRAY_NAME,
    bind_state_name,
    container_ptr_name,
    state_init_name,
    state_loaded_name,
    state_slot_name,
    state_value_name,
)
from numba_cfunc_compiler.extension.ast import LifecycleBody
from numba_cfunc_compiler.types.binding import ConstantPlan, StatePlan
from numba_cfunc_compiler.utils.ast import AST


def require_container_state_initializer(node: ast.AnnAssign, expected_type: str, *, body: LifecycleBody | None) -> None:
    """A constructor declaration must match its persistent state type."""
    if body is LifecycleBody.EXECUTE and isinstance(node.target, ast.Name):
        annotation = node.annotation
        if isinstance(annotation, ast.Subscript) and isinstance(annotation.value, ast.Name) and annotation.value.id == "State":
            declared_type = annotation.slice
            if isinstance(declared_type, ast.Subscript):
                declared_type = declared_type.value
            if isinstance(declared_type, ast.Name) and declared_type.id == expected_type:
                return
    reject_container_constructor(node.value, expected_type)


def reject_container_constructor(call: ast.Call, expected_type: str) -> None:
    """Report a constructor call that cannot be owned by persistent state."""
    raise TypeError(
        f"{call.func.id}() is only supported as a State[{expected_type}] initializer "
        f"at line {getattr(call, 'lineno', '?')}; standalone container locals have no cleanup"
    )


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


def container_constant_plan(host_type, name: str, slot_index: int, populate: list[ast.stmt]) -> ConstantPlan:
    """Keep one compiled constant container in an instance-owned state slot."""
    slot = AST.array_access(STATE_ARRAY_NAME, slot_index)
    init_name = state_init_name(name)
    loaded_name = state_loaded_name(name)
    cleanup_name = state_value_name(name)
    create = host_type.create_new_container(init_name)
    pointer_name = container_ptr_name(init_name)
    start = [
        create[0],
        AST.assignment(copy.deepcopy(slot), ast.Name(id=pointer_name, ctx=ast.Load())),
        *create[1:],
        *populate,
        host_type.from_voidptr(name, name, ast.Name(id=pointer_name, ctx=ast.Load()), readonly=True),
    ]
    load = [
        AST.assignment(loaded_name, copy.deepcopy(slot)),
        host_type.from_voidptr(name, name, ast.Name(id=loaded_name, ctx=ast.Load()), readonly=True),
    ]
    cleanup = [
        host_type.from_voidptr(cleanup_name, cleanup_name, ast.Name(id=loaded_name, ctx=ast.Load())),
        ast.Expr(value=AST.function_call(host_type.free_func_name(), ast.Name(id=cleanup_name, ctx=ast.Load()))),
        AST.assignment(copy.deepcopy(slot), AST.function_call("voidptr_null")),
    ]
    return ConstantPlan((), ast.Name(id=name, ctx=ast.Load()), tuple(start), tuple(copy.deepcopy(load)), tuple(load), tuple(cleanup), state_slots=1)
