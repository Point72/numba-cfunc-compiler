"""Shared parsing and lowering for datetime and timedelta literals."""

import ast
from datetime import datetime, timedelta, timezone
from typing import Any

__all__ = ["evaluate_time_constructor", "lower_time_constructor", "time_constructor_name"]

_CONSTRUCTORS = {"datetime": datetime, "timedelta": timedelta, "timezone": timezone}
_TIMEZONES = {"utc": timezone.utc, "min": timezone.min, "max": timezone.max}


def time_constructor_name(node: ast.AST) -> str | None:
    """Return the datetime or timedelta constructor named by an AST call."""
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Name) and func.id in ("datetime", "timedelta"):
        return func.id
    if (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "datetime"
        and func.attr in ("datetime", "timedelta")
    ):
        return func.attr
    return None


def _evaluate_literal(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return ast.literal_eval(node)
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "timezone" and node.attr in _TIMEZONES:
        return _TIMEZONES[node.attr]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _CONSTRUCTORS:
        return _call_constructor(node.func.id, node)
    raise TypeError(f"Unsupported node in time literal: {ast.dump(node)}")


def _call_constructor(name: str, node: ast.Call) -> Any:
    args = [_evaluate_literal(arg) for arg in node.args]
    kwargs = {}
    for keyword in node.keywords:
        if keyword.arg is None or keyword.arg in kwargs:
            raise TypeError("Time constructors require distinct, explicit keyword arguments")
        kwargs[keyword.arg] = _evaluate_literal(keyword.value)
    return _CONSTRUCTORS[name](*args, **kwargs)


def evaluate_time_constructor(node: ast.AST) -> datetime | timedelta | None:
    """Evaluate a supported time constructor without executing generated code."""
    name = time_constructor_name(node)
    if name is None:
        return None
    try:
        return _call_constructor(name, node)
    except Exception as exc:
        raise TypeError(f"Failed to evaluate {name}() constructor: {exc}") from exc


def lower_time_constructor(node: ast.AST) -> ast.Constant | None:
    """Convert a datetime or timedelta constructor to nanoseconds."""
    value = evaluate_time_constructor(node)
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise TypeError(f"{value} is not a timezone-aware datetime")
        nanos = int(value.timestamp() * 1e9)
    else:
        nanos = int(value.total_seconds() * 1e9)
    return ast.copy_location(ast.Constant(value=nanos), node)
