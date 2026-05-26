"""
Generic struct support for numba_cfunc_compiler.

Provides:
- StructFieldInfo: Metadata for struct fields
- StructType: Struct type represented as void pointer with field metadata
- struct_attribute_transformer: Transforms struct field access to intrinsic calls (registered as attr lowerer)
- struct_attr_handler: Type inference for struct field access (registered as attr accessor)
- register(): Registers struct support
"""

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional

from numba_cfunc_compiler.models import UnknownNumbaValue, VariableType
from numba_cfunc_compiler.type_factory import TypeFactory
from numba_cfunc_compiler.utils.ast import AST

if TYPE_CHECKING:
    from numba_cfunc_compiler.numba_type_inference import NumbaTypeInference
    from numba_cfunc_compiler.variable_factory import ExpressionSource, VariableSource


@dataclass(frozen=True)
class StructFieldInfo:
    """
    Metadata for a single struct field.

    Attributes:
        name: Field name
        offset: Byte offset from struct start
        numba_type_name: Numba type name (e.g., 'int64', 'float64', 'voidptr')
        size: Size of the field in bytes
    """

    name: str
    offset: int
    numba_type_name: str
    size: int


@dataclass(frozen=True)
class StructType(VariableType):
    """
    Struct type represented as a void pointer with field metadata.

    Fields are accessed via pointer arithmetic and intrinsics:
    - get_field(): Read a field value
    - set_field(): Write a field value
    """

    fields: Dict[str, StructFieldInfo] = None
    size: int = 0

    def get_numba_type_name(self) -> str:
        return "voidptr"

    def is_opaque_pointer(self) -> bool:
        return True

    def get_methods(self):
        return []

    def _get_field_info(self, field_name: str) -> StructFieldInfo:
        """Get and validate field metadata."""
        if self.fields is None:
            raise TypeError("StructType has no field metadata")
        if field_name not in self.fields:
            struct_name = self.value.__name__ if self.value else "unknown"
            raise KeyError(f"Field '{field_name}' not found in struct {struct_name}")

        field_info = self.fields[field_name]
        if field_info.numba_type_name == "voidptr":
            raise TypeError(f"Field '{field_name}' has type 'voidptr' (nested struct/string/etc.) which is not supported for direct access")
        return field_info

    def get_size(self) -> int:
        return self.size

    def get_field(self, struct_expr, field_name: str) -> ast.AST:
        """Generate AST for reading a struct field.

        Args:
            struct_expr: Either a string (variable name) or an AST expression for the struct pointer
            field_name: Name of the field to access
        """
        field_info = self._get_field_info(field_name)

        # Accept either a string name or an AST expression
        if isinstance(struct_expr, str):
            struct_ptr = ast.Name(id=struct_expr, ctx=ast.Load())
        else:
            struct_ptr = struct_expr

        offset = ast.Constant(value=field_info.offset)

        if field_info.numba_type_name == "voidptr":
            return AST.function_call("struct_field_ptr", struct_ptr, offset)

        type_name = ast.Constant(value=field_info.numba_type_name)
        return AST.function_call("struct_field_access", struct_ptr, offset, type_name)

    def set_field(self, struct_name: str, field_name: str, value_expr: ast.AST) -> ast.AST:
        """Generate AST for writing to a struct field."""
        field_info = self._get_field_info(field_name)
        struct_ptr = ast.Name(id=struct_name, ctx=ast.Load())
        offset = ast.Constant(value=field_info.offset)
        type_name = ast.Constant(value=field_info.numba_type_name)
        return AST.function_call("struct_field_store", struct_ptr, offset, type_name, value_expr)

    @classmethod
    def _get_struct_fields(cls, var_type: type) -> Dict[str, StructFieldInfo]:
        """Override in subclasses to provide field metadata."""
        return {}

    @classmethod
    def _get_struct_size(cls, var_type: type) -> int:
        """Override in subclasses to provide struct size."""
        return 0

    @classmethod
    def from_type(cls, var_type: Any, value: Any) -> Optional["StructType"]:
        """Create StructType from a Python type or struct instance."""
        if cls.is_type_supported(var_type):
            return cls(
                value=var_type,
                runtime_value=value,
                fields=cls._get_struct_fields(var_type),
                size=cls._get_struct_size(var_type),
            )
        # Check runtime value type
        if not isinstance(value, UnknownNumbaValue) and value is not None:
            runtime_type = type(value)
            if cls.is_type_supported(runtime_type):
                return cls(
                    value=runtime_type,
                    runtime_value=value,
                    fields=cls._get_struct_fields(runtime_type),
                    size=cls._get_struct_size(runtime_type),
                )
        return None


def is_struct_type(var_type) -> bool:
    return isinstance(var_type, StructType)


def struct_attribute_transformer(node: ast.AST, globalns: dict, variable_factory) -> Optional[ast.AST]:
    """Transform struct field access to struct_field_access call.

    Handles both simple access (struct_var.field) and dynamic access (basket[i].field).
    """
    if not isinstance(node, ast.Attribute):
        return None
    if variable_factory is None:
        return None

    # Case 1: Simple variable access - struct_var.field
    if isinstance(node.value, ast.Name):
        base_name = node.value.id
        base_var = variable_factory.from_name(base_name)

        if base_var is None or not is_struct_type(base_var.type):
            return None

        try:
            return base_var.type.get_field(base_name, node.attr)
        except TypeError:
            return None

    # Case 2: Dynamic signal access - basket[i].field
    if isinstance(node.value, ast.Subscript) and isinstance(node.value.value, ast.Name):
        container_name = node.value.value.id
        container_var = variable_factory.from_name(container_name)

        # Check if this is a keyed container (e.g., SignalSetSource) with struct elements
        # Use duck typing: containers with key_to_child_name support dynamic access
        if not hasattr(container_var, "key_to_child_name"):
            return None

        element_type = getattr(container_var, "element_type", None)
        if element_type is None or not is_struct_type(element_type):
            return None

        # Create a dynamic access for the indexed element
        # The container must implement create_dynamic_access()
        if not hasattr(container_var, "create_dynamic_access"):
            return None

        dynamic_access = container_var.create_dynamic_access(node.value.slice, variable_factory)

        # Get the struct pointer expression (without [0] dereference for opaque types)
        struct_ptr_expr = dynamic_access.get()

        try:
            return element_type.get_field(struct_ptr_expr, node.attr)
        except TypeError:
            return None

    return None


def struct_attr_handler(
    inference: "NumbaTypeInference",
    base_var: "VariableSource",
    attr_name: str,
    args: list,
) -> Optional["ExpressionSource"]:
    """Type inference handler for struct field access."""
    from numba_cfunc_compiler.variable_factory import ExpressionSource

    if not is_struct_type(base_var.type):
        return None

    struct_class = base_var.type.value
    if struct_class is None:
        return None

    annotations = getattr(struct_class, "__annotations__", {})
    if attr_name not in annotations:
        return None

    try:
        # Use base_var.get() to get the actual AST expression for the struct pointer
        # This is important for dynamic signal access (e.g., basket[i].field)
        struct_expr = base_var.get() if hasattr(base_var, "get") else base_var.name
        field_ast = base_var.type.get_field(struct_expr, attr_name)
    except (TypeError, KeyError):
        return None

    field_annotation = annotations[attr_name]
    var_type = TypeFactory.get_type(field_annotation)

    return ExpressionSource(var_type, field_ast, variable_factory=inference.variable_factory)


def register():
    """
    Register struct attribute handler for type inference.

    Note: Base StructType returns False for is_type_supported(), so it won't
    match any types by default. Client code should register their own StructType
    subclasses for struct support.
    """
    from numba_cfunc_compiler.numba_type_inference import NumbaTypeInference

    NumbaTypeInference.register_attr_accessor(struct_attr_handler)
