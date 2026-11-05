import ast
import enum
import inspect
from dataclasses import dataclass
from typing import Any

import numba
import numpy as np

__all__ = ["FrozenGlobals", "find_frozen_globals"]


@dataclass(frozen=True)
class FrozenGlobals:
    """Global values read by a function.

    constants:  path -> text for plain constants, e.g. {"params.WINDOW": "int64:20"}.
    unkeyable:  paths of other values (arrays, objects), which can't be keyed.
    """

    constants: dict[str, str]
    unkeyable: tuple[str, ...]


def find_frozen_globals(tree: ast.AST, call_globals: dict | None) -> FrozenGlobals:
    """Find the global values Numba freezes into tree's IR.

    Follows attribute access (params.WINDOW, Side.BUY) past modules, types and
    callables, which are code and not covered.
    """
    if not call_globals:
        return FrozenGlobals({}, ())
    # One walk: locals can be assigned after use, so chains are resolved afterwards.
    # Parents are visited first, so each chain's inner nodes are marked before they are reached.
    local, chains, inner = set(), [], set()
    for node in ast.walk(tree):
        if isinstance(node, ast.arg):
            local.add(node.arg)
        elif isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            local.add(node.id)
        elif isinstance(node, (ast.Name, ast.Attribute)) and node not in inner:
            chains.append(node)
        if isinstance(node, ast.Attribute):
            inner.add(node.value)

    constants: dict[str, str] = {}
    unkeyable: set[str] = set()
    for expr in chains:
        attrs = []
        while isinstance(expr, ast.Attribute):
            attrs.append(expr.attr)
            expr = expr.value
        if not isinstance(expr, ast.Name) or expr.id in local or expr.id not in call_globals:
            continue
        path, value = expr.id, call_globals[expr.id]
        for attr in reversed(attrs):
            if not _is_code(value) or not hasattr(value, attr):
                break
            path, value = f"{path}.{attr}", getattr(value, attr)
        if _is_code(value):
            continue
        if (text := _constant_text(value)) is None:
            unkeyable.add(path)
        else:
            constants[path] = text
    return FrozenGlobals(constants, tuple(sorted(unkeyable)))


def _is_code(value: Any) -> bool:
    return inspect.ismodule(value) or isinstance(value, type) or callable(value)


_CONSTANT_TYPES = (type(None), bool, int, float, complex, str, bytes, np.generic)


def _is_constant(value: Any) -> bool:
    if isinstance(value, enum.Enum):
        return _is_constant(value.value)
    if isinstance(value, tuple):
        return all(_is_constant(item) for item in value)
    return isinstance(value, _CONSTANT_TYPES)


def _constant_text(value: Any) -> str | None:
    """Return the Numba type and repr of a plain constant, else None."""
    if not _is_constant(value):
        return None
    try:
        return f"{numba.typeof(value)}:{value!r}"
    except ValueError:  # e.g. an int too large for Numba, which then fails to compile it
        return None
