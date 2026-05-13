import ast
from typing import Annotated, Any, Dict, List, Optional, get_args, get_origin

__all__ = [
    "collect_return_nodes",
    "validate_single_return",
    "validate_return_arity",
    "parse_annotated_metadata_dict",
]


def collect_return_nodes(ast_tree: ast.AST) -> List[ast.Return]:
    """Return every `return <expr>` node in the tree (skips bare `return`)."""
    return [node for node in ast.walk(ast_tree) if isinstance(node, ast.Return) and node.value is not None]


def validate_single_return(ast_tree: ast.AST) -> None:
    """Ensure the function has at least one return and none of them are tuples."""
    return_nodes = collect_return_nodes(ast_tree)
    if not return_nodes:
        raise ValueError("expects 1 output but has no return statements")
    for return_node in return_nodes:
        if isinstance(return_node.value, ast.Tuple):
            raise ValueError(f"returns {len(return_node.value.elts)} values but annotation expects 1.")


def validate_return_arity(ast_tree: ast.AST, expected: int) -> None:
    """Ensure every return statement returns exactly `expected` values."""
    for return_node in collect_return_nodes(ast_tree):
        actual = len(return_node.value.elts) if isinstance(return_node.value, ast.Tuple) else 1
        if actual != expected:
            raise ValueError(f"returns {actual} values but annotation expects {expected}.")


def parse_annotated_metadata_dict(
    annotation: Any,
    expected_base: type,
    base_label: str,
    example: str,
) -> Optional[Dict[str, Any]]:
    """Validate an `Annotated[expected_base, {name: spec, ...}]` annotation.

    Returns the metadata dict on success, or None if `annotation` is not an
    `Annotated[...]` at all (so callers can fall through to the next handler).
    Raises TypeError when the annotation is `Annotated` but malformed.

    `base_label` and `example` are only used to format error messages, e.g.
    base_label="SignalSet", example="Annotated[SignalSet, {'x': int}]".
    """
    if get_origin(annotation) is not Annotated:
        return None

    ann_args = get_args(annotation)
    if len(ann_args) < 2:
        raise TypeError(f"Annotated outputs must be Annotated[{base_label}, {{name: type, ...}}] with a metadata dict.")

    base = ann_args[0]
    meta_list = ann_args[1:]

    if base is not expected_base:
        raise TypeError(f"Annotated outputs must use {base_label} as the base type, e.g., {example}.")

    if len(meta_list) != 1 or not isinstance(meta_list[0], dict):
        raise TypeError(f"Annotated outputs must provide a single dict metadata mapping names to types, e.g., {example}.")

    return meta_list[0]
