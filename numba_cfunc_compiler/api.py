"""Markers and helpers written in functions compiled by the host."""

from typing import Any, Generic, TypeVar

__all__ = [
    "NumbaArray",
    "NumbaDict",
    "NumbaList",
    "State",
    "create_new_array",
    "create_new_dict",
    "create_new_list",
    "set_output",
]

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")
N = TypeVar("N")


def set_output(name: str, value: Any):
    """Marker lowered to an output write and tick by the node compiler."""
    raise NotImplementedError("set_output is transformed at compile time by numba_node")


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

    For persistent state, use State[NumbaList] with create_new_list():
        my_list: State[NumbaList] = create_new_list(int)

    Local aliases of state and input lists are supported.
    """


class NumbaArray(Generic[T, N]):
    """Fixed array state or read-only constant input annotated ``NumbaArray[type, length]``."""


def create_new_array(element_type: type, length: int) -> NumbaArray:
    """Initialize a zeroed ``State[NumbaArray]`` with a fixed positive length."""
    raise NotImplementedError("create_new_array is transformed at compile time by numba_node")


class NumbaDict(Generic[K, V]):
    """
    Type annotation for dict input parameters in @numba_node functions.

    Supported key types: int
    Supported value types: int, float, bool

    Usage:
        @numba_node
        def my_node(data: NumbaDict[int, float]) -> Signal[float]:
            ...

    For persistent state, use State[NumbaDict] with create_new_dict():
        my_dict: State[NumbaDict] = create_new_dict(int, int)

    Local aliases of state and input dicts are supported.
    """


def create_new_list(element_type: type) -> NumbaList:
    """
    Initialize an empty State[NumbaList] with the specified element type.

    This constructor is only supported directly in a state declaration.

    Args:
        element_type: The type of elements (int, float, or bool)

    Returns:
        A new empty NumbaList
    """
    raise NotImplementedError("create_new_list is transformed at compile time by numba_node")


def create_new_dict(key_type: type, value_type: type) -> NumbaDict:
    """
    Initialize an empty State[NumbaDict] with the specified key and value types.

    This constructor is only supported directly in a state declaration.

    Args:
        key_type: The type of keys (only int is supported)
        value_type: The type of values (int, float, or bool)

    Returns:
        A new empty NumbaDict
    """
    raise NotImplementedError("create_new_dict is transformed at compile time by numba_node")
