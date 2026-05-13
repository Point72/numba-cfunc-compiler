import ast
from datetime import (
    datetime as _PyDatetime,
    timedelta as _PyTimedelta,
    timezone as _PyTimezone,
)
from typing import Any, Optional

__all__ = [
    "TypeHelper",
]


class TypeHelper:
    """Helper utilities for datetime/timedelta parsing and conversion."""

    _TIME_EVAL_GLOBALS = {
        "datetime": _PyDatetime,
        "timedelta": _PyTimedelta,
        "timezone": _PyTimezone,
    }
    _TIME_ALLOWED_NAMES = frozenset(_TIME_EVAL_GLOBALS.keys())

    @staticmethod
    def get_time_func_name(node: ast.AST) -> Optional[str]:
        """Extract function name if node is a datetime/timedelta call, else None."""
        if not isinstance(node, ast.Call):
            return None
        func = node.func
        # datetime(...), timedelta(...)
        if isinstance(func, ast.Name) and func.id in ("datetime", "timedelta"):
            return func.id
        # datetime.datetime(...), datetime.timedelta(...)
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "datetime":
            if func.attr in ("datetime", "timedelta"):
                return func.attr
        return None

    @staticmethod
    def _validate_time_call(node: ast.Call) -> None:
        """Ensure the AST for a datetime/timedelta literal is simple and safe to eval."""
        for sub in ast.walk(node):
            if isinstance(sub, (ast.Call, ast.Constant, ast.keyword, ast.Load)):
                continue
            if isinstance(sub, ast.Name) and sub.id in TypeHelper._TIME_ALLOWED_NAMES:
                continue
            if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name) and sub.value.id == "timezone":
                continue
            raise TypeError(f"Unsupported node in time literal: {ast.dump(sub)}")

    @staticmethod
    def eval_time_constructor(node: ast.Call) -> tuple[Any, Optional[str]]:
        """
        Evaluate a datetime/timedelta AST call and return (value, func_name).
        Returns (None, None) if not a time constructor.
        """
        func_name = TypeHelper.get_time_func_name(node)
        if func_name is None:
            return None, None

        try:
            eval_call = ast.Call(
                func=ast.Name(id=func_name, ctx=ast.Load()),
                args=node.args,
                keywords=node.keywords,
            )
            TypeHelper._validate_time_call(eval_call)

            expr = ast.Expression(body=eval_call)
            ast.fix_missing_locations(expr)
            code = compile(expr, "<time-literal>", "eval")
            val = eval(code, {"__builtins__": {}}, TypeHelper._TIME_EVAL_GLOBALS)
            return val, func_name
        except Exception as e:
            raise TypeError(f"Failed to evaluate {func_name}() constructor: {e}") from e

    @staticmethod
    def lower_time_constructor(node: ast.AST) -> Optional[ast.Constant]:
        """Convert a datetime/timedelta AST call to a nanoseconds constant node."""
        func_name = TypeHelper.get_time_func_name(node)
        if func_name is None:
            return None

        val, _ = TypeHelper.eval_time_constructor(node)
        if val is None:
            return None

        if isinstance(val, _PyDatetime):
            if val.tzinfo is None or val.utcoffset() is None:
                raise TypeError(f"{val} is not a timezone-aware datetime")
            nanos = int(val.timestamp() * 1e9)
        elif isinstance(val, _PyTimedelta):
            nanos = int(val.total_seconds() * 1e9)
        else:
            raise TypeError(f"Expected datetime or timedelta, got {type(val).__name__}")

        return ast.copy_location(ast.Constant(value=nanos), node)
