from dataclasses import dataclass
from typing import Any, Dict, Generic, TypeVar

from numba_cfunc_compiler.compilation_context import CompilationContext

### Public API ###

__all__ = (
    "set_output",
    "State",
    "create_new_list",
    "create_new_dict",
    "NumbaList",
    "NumbaDict",
    "NumbaTypeInfo",
    "NumbaTypeRegistry",
)


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

    pass


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

    pass


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

    pass


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


@dataclass(frozen=True)
class NumbaTypeInfo:
    """Complete info about a type's numba representation."""

    python_type: type | None  # None for voidptr (no Python equivalent)
    numba_name: str
    numba_type: Any  # types.int64, types.voidptr, etc.
    size: int
    is_numeric: bool
    is_primitive: bool  # True for int/float/bool, False for voidptr and custom types
    type_name: str | None = None  # Optional name for State[type] annotations (e.g., 'int', 'datetime')


class NumbaTypeRegistry:
    """Registry for Numba type information.

    All mutable state lives in the active CompilationContext.
    """

    @classmethod
    def register_type(cls, info: NumbaTypeInfo):
        """Register a new type with the registry."""
        CompilationContext.current().numba_types.append(info)

    @classmethod
    def get_by_python_type(cls, py_type: type) -> NumbaTypeInfo | None:
        """Look up type info by Python type."""
        for info in CompilationContext.current().numba_types:
            if info.python_type is py_type:
                return info
        return None

    @classmethod
    def get_by_numba_name(cls, numba_name: str) -> NumbaTypeInfo | None:
        """Look up type info by numba type name string."""
        for info in CompilationContext.current().numba_types:
            if info.numba_name == numba_name:
                return info
        return None

    @classmethod
    def get_by_numba_type(cls, numba_type: Any) -> NumbaTypeInfo | None:
        """Look up type info by numba type object."""
        for info in CompilationContext.current().numba_types:
            if info.numba_type is numba_type:
                return info
        return None

    @classmethod
    def resolve_numba_name(cls, py_type: Any) -> str:
        """Get the numba type name for a Python type, defaulting to 'voidptr'."""
        info = cls.get_by_python_type(py_type)
        return info.numba_name if info else "voidptr"

    @classmethod
    def get_numba_type(cls, numba_name: str) -> Any:
        """Get the numba type object for a numba type name. Raises KeyError if not found."""
        info = cls.get_by_numba_name(numba_name)
        if info is None:
            raise KeyError(numba_name)
        return info.numba_type

    @classmethod
    def resolve_to_numba_type(cls, py_type: Any) -> Any:
        """Get the numba type object for a Python type."""
        return cls.get_numba_type(cls.resolve_numba_name(py_type))

    @classmethod
    def get_size(cls, py_type: type) -> int:
        """Get the size in bytes for a Python type. Raises KeyError if not found."""
        info = cls.get_by_python_type(py_type)
        if info is None:
            raise KeyError(py_type)
        return info.size

    @classmethod
    def get_size_for_numba_type(cls, numba_type: Any) -> int:
        """Get the size in bytes for a numba type object. Raises KeyError if not found."""
        info = cls.get_by_numba_type(numba_type)
        if info is None:
            raise KeyError(numba_type)
        return info.size

    @classmethod
    def get_size_for_numba_name(cls, numba_name: str) -> int:
        """Get the size in bytes for a numba type name string. Raises KeyError if not found."""
        info = cls.get_by_numba_name(numba_name)
        if info is None:
            raise KeyError(numba_name)
        return info.size

    @classmethod
    def has_numba_name(cls, numba_name: str) -> bool:
        """Check if a numba type name is registered."""
        return cls.get_by_numba_name(numba_name) is not None

    @classmethod
    def is_numeric(cls, numba_name: str) -> bool:
        """Check if a numba type name is numeric."""
        info = cls.get_by_numba_name(numba_name)
        return info is not None and info.is_numeric

    @classmethod
    def has_python_type(cls, py_type: type) -> bool:
        """Check if a Python type is registered."""
        return cls.get_by_python_type(py_type) is not None

    @classmethod
    def get_supported_type_names(cls) -> Dict[str, type]:
        """Get all registered type names for State[type] annotations."""
        return {
            info.type_name: info.python_type
            for info in CompilationContext.current().numba_types
            if info.type_name is not None and info.python_type is not None
        }

    @classmethod
    def is_supported_type(cls, t: Any) -> bool:
        """Check if a type is supported for numba compilation."""
        if cls.get_by_python_type(t) is not None:
            return True

        # Query registered type classes via the context
        for type_class in CompilationContext.current().type_classes:
            if type_class.is_type_supported(t):
                return True

        return False

    @classmethod
    def get_list_element_types(cls) -> tuple[type, ...]:
        """Get supported element types for NumbaList."""
        return CompilationContext.current().list_element_types

    @classmethod
    def get_dict_key_types(cls) -> tuple[type, ...]:
        """Get supported key types for NumbaDict."""
        return CompilationContext.current().dict_key_types

    @classmethod
    def get_dict_value_types(cls) -> tuple[type, ...]:
        """Get supported value types for NumbaDict."""
        return CompilationContext.current().dict_value_types

    @classmethod
    def get_numba_type_map(cls, python_types: tuple[type, ...]) -> dict[str, Any]:
        """Create a map from numba type names to numba type objects for given Python types."""
        result = {}
        for py_type in python_types:
            info = cls.get_by_python_type(py_type)
            if info:
                result[info.numba_name] = info.numba_type
        return result

    # Mapping from C++ type names (as returned by struct metadata) to numba type names
    # This is a static constant — not part of CompilationContext.
    _CPP_TYPE_TO_NUMBA: dict[str, str] = {
        "BOOL": "int8",
        "INT8": "int8",
        "UINT8": "int8",
        "INT16": "int64",
        "UINT16": "int64",
        "INT32": "int64",
        "UINT32": "int64",
        "INT64": "int64",
        "UINT64": "int64",
        "DOUBLE": "float64",
        "FLOAT": "float64",
        "DATETIME": "int64",
        "TIMEDELTA": "int64",
        "STRUCT": "voidptr",
        "STRING": "voidptr",
        "ENUM": "int64",
    }

    @classmethod
    def cpp_type_to_numba_name(cls, cpp_type: str) -> str:
        """Convert a C++ type name (e.g., 'INT64', 'DOUBLE') to a numba type name."""
        return cls._CPP_TYPE_TO_NUMBA.get(cpp_type, "voidptr")


def numba_type_to_python(ty):
    try:
        from numba.core import types as numba_types

        if ty is numba_types.int64 or ty is numba_types.intp:
            return int
        elif ty is numba_types.float64:
            return float
        elif ty is numba_types.boolean or ty is numba_types.int8:
            return bool
    except ImportError:
        pass
    return ty


# Primitive types supported in Signal/Output annotations (base set)
SUPPORTED_SIGNAL_PRIMITIVES = (int, float, bool)


# Array name constants used in generated code
STATE_ARRAY_NAME = "state"

TICKED_OUTPUTS_ARRAY_NAME = "output_ticked"
OUTPUTS_ARRAY_NAME = "outputs"

# Lifecycle phase constants
LIFECYCLE_PARAM_NAME = "lifecycle_phase"
LIFECYCLE_EXECUTE = 0  # Normal execution
LIFECYCLE_START = 1  # Called once at node start
LIFECYCLE_STOP = 2  # Called once at node stop
