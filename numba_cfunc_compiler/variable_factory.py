import ast
from collections import defaultdict
from typing import Any

from numba_cfunc_compiler.compiler_constants import OUTPUTS_ARRAY_NAME
from numba_cfunc_compiler.defaults.primitive_support import PrimitiveType
from numba_cfunc_compiler.models import (
    ContainerType,
    VariableType,
)
from numba_cfunc_compiler.utils.ast import AST

__all__ = [
    "ConstantSource",
    "ExpressionSource",
    "LocalVariableSource",
    "OutputSource",
    "VariableFactory",
    "VariableSource",
    "VoidPtrSource",
]


class VariableSource:
    """
    VariableSource represents variables that we manage, and provides an API to interface it with the Python AST to create valid numba code.

    source: sources inherit from this base class and implement the methods below. The source represents 'where the variable comes from / where its value is stored' which changes how we read and write to it.
    type: Represents the type of the variable, different types expose different methods. For example, we can call contains_any on an EnumSet, but not an Enum.
    name: the name of the variable
    supported_methods: this is a list of methods that the variable supports. Its a combination of methods supported by its source (e.g SignalSource supports Valid and Ticked) and methods supported by its type (e.g EnumSetType supports ContainsAll and ContainsAny)
    handler: the dispatcher that deals with method calls for this variable.
    """

    def __init__(
        self,
        type: VariableType,
        name: str,
        supported_methods=None,
        variable_factory=None,
    ):
        from numba_cfunc_compiler.method_factory import method_handler_factory

        self.type = type
        self.name = name
        self.variable_factory = variable_factory
        self.category: Any = None  # Set by VariableFactory.add_variable
        supported_methods = supported_methods or []
        supported_methods = supported_methods + type.get_methods()
        self.handler = method_handler_factory(self.__class__.__name__, supported_methods)

    def local_variable_name(self):
        raise NotImplementedError(f"local_variable_name method not implemented for {self.name}")

    def get_storage_location(self):
        raise NotImplementedError(f"get_storage_location method not implemented for {self.name}")

    def read(self):
        raise NotImplementedError(f"read method not implemented for {self.name}")

    def write(self):
        raise NotImplementedError(f"write method not implemented for {self.name}")

    def get(self):
        raise NotImplementedError(f"get method not implemented for {self.name}")

    def read_value(self):
        return self.get()

    def call(self, method, *args):
        return self.handler.handle(method, self, *args)

    def is_opaque_pointer(self) -> bool:
        if self.type is not None:
            return self.type.is_opaque_pointer()
        return False


class VoidPtrSource(VariableSource):
    """Configurable void-pointer source for external array variables.

    Args:
        array_idx (int): Index into the storage array.
        type (VariableType): The variable's type descriptor.
        name (str): Variable name used in generated code.
        storage_location (str): Name of the array this variable is read from (e.g. "inputs").
        supported_methods (list, optional): Method handler classes.
        force_opaque (bool): If True, always treat as opaque pointer regardless of type.
        skip_pre_read (bool): If True, read() returns None.
    """

    def __init__(
        self,
        array_idx: int,
        type: VariableType,
        name: str,
        storage_location: str,
        supported_methods=None,
        force_opaque: bool = False,
        skip_pre_read: bool = False,
    ):
        self.array_idx = array_idx
        self._storage_location = storage_location
        self._force_opaque = force_opaque
        self.skip_pre_read = skip_pre_read
        super().__init__(type, name, supported_methods)

    def local_variable_name(self):
        return self.name

    def get_storage_location(self):
        return self._storage_location

    def is_opaque_pointer(self) -> bool:
        return self._force_opaque or super().is_opaque_pointer()

    def read(self):
        if self.skip_pre_read:
            return None
        loaded_value = AST.array_access(self.get_storage_location(), self.array_idx)
        # ContainerType handles its own read logic (e.g. list/dict from voidptr)
        if isinstance(self.type, ContainerType):
            return self.type.read(self.local_variable_name(), self.name, loaded_value)
        # Allow types to prepare themselves before reading
        self.type = self.type.prepare_voidptr_read(self)
        from numba_cfunc_compiler.defaults.struct_support import StructType

        if isinstance(self.type, StructType) and not isinstance(self, OutputSource):
            view_name = self.variable_factory.bind_struct_view(self.type)
            return AST.assignment(self.local_variable_name(), AST.function_call(view_name, loaded_value))
        value = AST.cast_from_voidptr(loaded_value, self.type.get_numba_type_name())
        return AST.assignment(self.local_variable_name(), value)

    def get(self):
        if self.type.is_opaque_pointer():
            # opaque pointer-like types (structs, order books, containers) are represented as void pointers
            return ast.Name(id=self.local_variable_name(), ctx=ast.Load())
        # for other types we dereference the pointer
        return AST.deref_pointer(self.local_variable_name())

    def read_value(self):
        value = self.get()
        if isinstance(self.type, PrimitiveType) and self.type.value is bool:
            return ast.Compare(left=value, ops=[ast.NotEq()], comparators=[ast.Constant(value=0)])
        return value


class OutputSource(VoidPtrSource):
    def __init__(self, array_idx: int, type: VariableType, name: str):
        from numba_cfunc_compiler.method_factory import Output

        super().__init__(
            array_idx=array_idx,
            type=type,
            name=name,
            storage_location=OUTPUTS_ARRAY_NAME,
            supported_methods=[Output],
        )

    def local_variable_name(self):
        return f"output_{self.array_idx}_ptr"

    def write(self, value: Any):
        """
        Write a value to the host-owned output cell.

        Primitive outputs use a Numba-typed store. Structs use a
        layout-checked copy.
        """
        from numba_cfunc_compiler.defaults.struct_support import StructType

        if isinstance(self.type, StructType):
            output_size = self.type.get_size()
            if output_size <= 0:
                raise TypeError(f"Struct output {self.type.value} has invalid size {output_size}")
            copy_name = self.variable_factory.bind_struct_copy(self.type)
            copy_call = AST.function_call(copy_name, ast.Name(id=self.local_variable_name(), ctx=ast.Load()), value)
            return ast.Expr(value=copy_call)

        if isinstance(self.type, PrimitiveType):
            return ast.Expr(
                value=AST.function_call(
                    "primitive_output_store",
                    ast.Name(id=self.local_variable_name(), ctx=ast.Load()),
                    value,
                    ast.Constant(value=self.type.value.__name__),
                )
            )
        output_ptr_value = AST.deref_pointer(self.local_variable_name())
        return AST.assignment(output_ptr_value, value)


class LocalVariableSource(VariableSource):
    def __init__(self, type: VariableType, name: str):
        super().__init__(type, name)

    def local_variable_name(self):
        return self.name

    def get(self):
        return ast.Name(id=self.local_variable_name(), ctx=ast.Load())


class ExpressionSource(VariableSource):
    """Host source whose value is produced by a keyed access expression."""

    def __init__(self, type: VariableType, expr: ast.AST, variable_factory, name: str = "_expr"):
        super().__init__(type, name, variable_factory=variable_factory)
        self.expr = expr

    def local_variable_name(self):
        # No local variable exists; this is expression-backed.
        return self.name

    def get(self):
        return self.expr


class ConstantSource(VariableSource):
    """Constants that are passed in as arguments to the node"""

    def __init__(self, type: VariableType, name: str):
        super().__init__(type, name)

    def local_variable_name(self):
        return self.name

    def get_storage_location(self):
        raise ValueError("Constants are not stored in a storage location")

    def read(self):
        # Delegate to the type's read_constant method for polymorphic initialization
        return self.type.read_constant(self.local_variable_name())

    def get(self):
        # Opaque pointer types (structs, containers, etc.) must return the local variable name
        # since their value cannot be inlined as a constant
        if self.is_opaque_pointer():
            return ast.Name(id=self.local_variable_name(), ctx=ast.Load())

        # For value types, return the actual constant value directly.
        # This ensures constants work in all lifecycle phases (start, execute, stop)
        # without relying on a local variable that may not exist yet.
        if hasattr(self.type, "get_value_node"):
            return self.type.get_value_node()
        return ast.Constant(value=self.type.runtime_value)


class VariableFactory:
    def __init__(self):
        self.variable_sources = defaultdict(list)
        self.category_variables = defaultdict(list)
        self.source_name_map = {}
        self.ast_converter = None
        self.typed_struct_bindings = {}
        self.typed_struct_layouts = {}

    def _bind_struct(self, struct_type, operation: str) -> str:
        from numba_cfunc_compiler.standalone.struct import struct_copy, struct_view

        layout = struct_type.get_typed_layout()
        name = f"_typed_struct_{operation}_{layout.fingerprint}"
        self.typed_struct_bindings[name] = struct_view(layout) if operation == "view" else struct_copy(layout)
        self.typed_struct_layouts[layout.fingerprint] = layout
        return name

    def bind_struct_view(self, struct_type) -> str:
        return self._bind_struct(struct_type, "view")

    def bind_struct_copy(self, struct_type) -> str:
        return self._bind_struct(struct_type, "copy")

    def add_variable(self, variable: VariableSource, category: Any = None):
        if variable.name in self.source_name_map:
            raise ValueError(f"variable {variable.name} already exists")
        variable.variable_factory = self
        self.variable_sources[type(variable)].append(variable)
        if category is not None:
            variable.category = category
            self.category_variables[category].append(variable)
        self.source_name_map[variable.name] = variable

    def get_source(self, source_type: type):
        if not issubclass(source_type, VariableSource):
            raise TypeError(f"source_type must be a subclass of VariableSource, got {source_type}")
        return self.variable_sources[source_type]

    def get_by_category(self, category_id: Any) -> list:
        """Get all variables registered under *category_id*."""
        return self.category_variables.get(category_id, [])

    def from_source_name(self, name: str):
        """Look up a declared host source."""
        return self.source_name_map.get(name)

    def get_output_by_idx(self, idx: int):
        output = self.variable_sources[OutputSource][idx]
        if output.array_idx != idx:
            raise RuntimeError(f"output {output.name} has array index {output.array_idx} but expected {idx}")
        return output

    def lower_value_expression(self, visitor, ast_node: ast.AST, statements: list[ast.stmt]) -> ast.AST:
        """Lower an expression without recording an inferred Python-side type.

        Keyed source access still needs source lookup. Ordinary locals and
        expressions are left for Numba to type in the generated function.
        """
        keyed_source = self.resolve_keyed_source(visitor, ast_node)
        if keyed_source is not None:
            return keyed_source.get()

        value = visitor.visit(ast_node)
        if isinstance(value, list):
            statements.extend(value[:-1])
            return value[-1]
        return value

    def _get_static_key(self, key_node) -> Any:
        """Extract a static key value from an AST node, or return None if dynamic."""
        if isinstance(key_node, ast.Constant):
            return key_node.value
        if hasattr(key_node, "value") and isinstance(key_node.value, ast.Constant):
            return key_node.value.value
        return None

    def _handle_container_subscript(self, visitor, container, key_node):
        """Handle subscripting into a container that supports keyed access.

        The container must implement:
        - key_to_child_name: dict mapping keys to child variable names
        - _idx_to_key: dict mapping integer indices to keys (optional)
        - get_key_index(key): return index for a key
        - create_dynamic_access(index_expr, variable_factory): create dynamic accessor
        """
        key = self._get_static_key(key_node)

        if key is not None:
            # Static access: a['key'] or a[0]
            child_var_name = container.key_to_child_name.get(key)

            # Integer index access for dict baskets with string keys
            if child_var_name is None and isinstance(key, int):
                idx_to_key = getattr(container, "_idx_to_key", {})
                original_key = idx_to_key.get(key)
                if original_key is not None:
                    child_var_name = container.key_to_child_name.get(original_key)

            if child_var_name is None:
                raise KeyError(f"Container has no key '{key}'")
            child_var = self.from_source_name(child_var_name)
            if child_var is None:
                raise RuntimeError(f"Internal error: child variable '{child_var_name}' not found for key '{key}'")
            # If child has skip_pre_read, it was never loaded into a local variable.
            # Use dynamic access with a constant index to generate inline access.
            if getattr(child_var, "skip_pre_read", False):
                key_index = container.get_key_index(key)
                return container.create_dynamic_access(ast.Constant(value=key_index), variable_factory=self)
            return child_var

        if isinstance(key_node, ast.Name):
            key_var = self.from_source_name(key_node.id)
            if key_var is not None and hasattr(key_var, "resolve_index_expr"):
                return container.create_dynamic_access(key_var.resolve_index_expr(container), variable_factory=self)

        # Dynamic access: a[some_variable]
        index_expr = visitor.visit(key_node) if visitor else key_node
        return container.create_dynamic_access(index_expr, variable_factory=self)

    def resolve_keyed_source(self, visitor, ast_node: ast.AST):
        """Resolve host-backed keyed access without inferring an expression type."""
        if not isinstance(ast_node, ast.Subscript) or not isinstance(ast_node.value, ast.Name):
            return None
        container = self.from_source_name(ast_node.value.id)
        if not hasattr(container, "key_to_child_name") or not hasattr(container, "create_dynamic_access"):
            return None
        return self._handle_container_subscript(visitor, container, ast_node.slice)
