"""Markers and helpers written in functions compiled by the host."""

from typing import Any, Generic, TypeVar

__all__ = [
    "NumbaDict",
    "NumbaList",
    "State",
    "create_new_dict",
    "create_new_list",
    "set_output",
]

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


def set_output(name: str, value: Any):
    """
    Set output at index `name` to `value` and mark it ticked.
    """
    from numba_cfunc_compiler.utils.ast import AST

    return AST.set_output(name, value)


class State(Generic[T]):
    """Marks a variable as stateful (persistent between function calls)"""


class NumbaList(Generic[T]):
    """
    Type annotation for list input parameters in @numba_node functions.

    Supported element types: int, float, bool

    Usage:
        @numba_node
        def my_node(data: NumbaList[int]) -> Signal[int]:
            ...

    For local variables, use create_new_list():
        l = create_new_list(int)

    For state variables, use State[NumbaList] with create_new_list():
        my_list: State[NumbaList] = create_new_list(int)
    """


class NumbaDict(Generic[K, V]):
    """
    Type annotation for dict input parameters in @numba_node functions.

    Supported key types: int
    Supported value types: int, float, bool

    Usage:
        @numba_node
        def my_node(data: NumbaDict[int, float]) -> Signal[float]:
            ...

    For local variables, use create_new_dict():
        d = create_new_dict(int, float)

    For state variables, use State[NumbaDict] with create_new_dict():
        my_dict: State[NumbaDict] = create_new_dict(int, int)
    """


def create_new_list(element_type: type) -> NumbaList:
    """
    Create a new empty list with the specified element type.

    Args:
        element_type: The type of elements (int, float, or bool)

    Returns:
        A new empty NumbaList
    """
    raise NotImplementedError("create_new_list is transformed at compile time by numba_node")


def create_new_dict(key_type: type, value_type: type) -> NumbaDict:
    """
    Create a new empty dict with the specified key and value types.

    Args:
        key_type: The type of keys (only int is supported)
        value_type: The type of values (int, float, or bool)

    Returns:
        A new empty NumbaDict
    """
    raise NotImplementedError("create_new_dict is transformed at compile time by numba_node")
