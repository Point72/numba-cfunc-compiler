import ast
from typing import Iterable

__all__ = [
    "STATE_ANNOTATION_NAME",
    "is_state_annotation",
    "state_annotation_target",
    "inject_state_params",
    "append_state_values_to_return",
]


STATE_ANNOTATION_NAME = "State"


def is_state_annotation(node: ast.AST) -> bool:
    """True if ``node`` is a ``State[...]: var = ...`` annotated assignment."""
    return (
        isinstance(node, ast.AnnAssign)
        and isinstance(node.annotation, ast.Subscript)
        and isinstance(node.annotation.value, ast.Name)
        and node.annotation.value.id == STATE_ANNOTATION_NAME
        and isinstance(node.target, ast.Name)
    )


def state_annotation_target(node: ast.AnnAssign) -> str:
    """Return the variable name bound by a ``State[...]: var = ...`` annotation."""
    if not is_state_annotation(node):
        raise ValueError("node is not a State[...] annotation")
    assert isinstance(node.target, ast.Name)  # narrowed by is_state_annotation
    return node.target.id


def inject_state_params(
    func_def: ast.FunctionDef,
    param_names: Iterable[str],
) -> None:
    """Append state parameters to a function's signature in place."""
    for name in param_names:
        func_def.args.args.append(ast.arg(arg=name, annotation=None))


def append_state_values_to_return(
    return_node: ast.Return,
    state_var_names: Iterable[str],
) -> ast.Return:
    """
    Return a new ast.Return whose value is a tuple with state_var_names
    appended to the original returned value.

    Used by backends that thread state through the return value
    rather than through pointer side-effects.
    """
    extras = [ast.Name(id=name, ctx=ast.Load()) for name in state_var_names]
    original = return_node.value
    if original is None:
        new_value: ast.expr = ast.Tuple(elts=extras, ctx=ast.Load())
    elif isinstance(original, ast.Tuple):
        new_value = ast.Tuple(elts=list(original.elts) + extras, ctx=ast.Load())
    else:
        new_value = ast.Tuple(elts=[original] + extras, ctx=ast.Load())
    return ast.Return(value=new_value)
