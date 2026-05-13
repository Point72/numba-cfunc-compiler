import ast
import copy
from collections import defaultdict
from typing import Any, List

from numba_cfunc_compiler.defaults.primitive_support import PrimitiveType
from numba_cfunc_compiler.models import (
    ContainerType,
    UnknownNumbaType,
    UnknownType,
    VariableType,
)
from numba_cfunc_compiler.numba_config import (
    OUTPUTS_ARRAY_NAME,
)
from numba_cfunc_compiler.type_factory import TypeFactory
from numba_cfunc_compiler.utils.ast import AST

__all__ = [
    "VariableSource",
    "VoidPtrSource",
    "OutputSource",
    "LocalVariableSource",
    "ExpressionSource",
    "LocalConstantSource",
    "ConstantSource",
    "VariableFactory",
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

    def call(self, method, *args):
        return self.handler.handle(method, self, *args)

    def is_opaque_pointer(self) -> bool:
        if self.type is not None:
            return self.type.is_opaque_pointer()
        return False

    def clone_with_name(self, new_name: str):
        new_source = copy.copy(self)
        new_source.name = new_name
        return new_source


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
        # Allow types to prepare themselves before reading (e.g., OrderBookType loads from book_ptrs)
        self.type = self.type.prepare_voidptr_read(self)
        value = AST.cast_from_voidptr(loaded_value, self.type.get_numba_type_name())
        return AST.assignment(self.local_variable_name(), value)

    def get(self):
        if self.type.is_opaque_pointer():
            # opaque pointer-like types (structs, order books, containers) are represented as void pointers
            return ast.Name(id=self.local_variable_name(), ctx=ast.Load())
        # for other types we dereference the pointer
        return AST.deref_pointer(self.local_variable_name())


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

    def write(self, value: any):
        """
        Create: output_(output_idx)_ptr[0] = value
        """
        if isinstance(self.type, PrimitiveType):
            value_type = None
            if isinstance(value, ast.Constant):
                value_type = type(value.value)
            elif isinstance(value, ast.Name):
                var = self.variable_factory.from_name(value.id)
                if var is not None:
                    value_type = var.type.value
            if value_type is not None and not isinstance(value_type, UnknownNumbaType) and not self.type.accepts_value_type(value_type):
                raise TypeError(f"Return value at position {self.array_idx} has type {value_type}, expected {self.type.value}")
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
    """Variable backed by an AST expression, used to keep track of type information."""

    def __init__(self, type: VariableType, expr: ast.AST, variable_factory, name: str = "_expr"):
        super().__init__(type, name, variable_factory=variable_factory)
        self.expr = expr

    def local_variable_name(self):
        # No local variable exists; this is expression-backed.
        return self.name

    def get(self):
        return self.expr


class LocalConstantSource(VariableSource):
    """Local constants created inside the node (mainly used when returning a constant value)"""

    def __init__(self, type: VariableType, name: str, var_value: any):
        super().__init__(type, name)
        self.var_value = var_value

    def local_variable_name(self):
        return self.name

    def get(self):
        if isinstance(self.type, PrimitiveType):
            return ast.Constant(value=self.var_value)
        raise ValueError(f"LocalConstantSource cannot be used for type {self.type}")


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
        self.variable_name_map = dict()
        self.temporary_variable_counter = 0
        self.ast_converter = None

    def add_variable(self, variable: VariableSource, category: Any = None):
        variable.variable_factory = self
        self.variable_sources[type(variable)].append(variable)
        if category is not None:
            variable.category = category
            self.category_variables[category].append(variable)
        if variable.name in self.variable_name_map:
            raise ValueError(f"variable {variable.name} already exists")
        self.variable_name_map[variable.name] = variable

    def get_source(self, source_type: type):
        if not issubclass(source_type, VariableSource):
            raise TypeError(f"source_type must be a subclass of VariableSource, got {source_type}")
        return self.variable_sources[source_type]

    def get_by_category(self, category_id: Any) -> list:
        """Get all variables registered under *category_id*."""
        return self.category_variables.get(category_id, [])

    def from_name(self, name: str):
        if name not in self.variable_name_map:
            return None
        return self.variable_name_map[name]

    def get_output_by_idx(self, idx: int):
        output = self.variable_sources[OutputSource][idx]
        if output.array_idx != idx:
            raise RuntimeError(f"output {output.name} has array index {output.array_idx} but expected {idx}")
        return output

    def create_temporary_variable_name(self):
        name = f"tmp_{self.temporary_variable_counter}"
        self.temporary_variable_counter += 1
        return name

    def add_local_variable(self, type, var_name: str, value):
        # If caller provides a VariableType instance use it directly; otherwise derive it
        var_type = type if isinstance(type, VariableType) else TypeFactory.get_type(type)
        var = LocalVariableSource(var_type, var_name)
        self.add_variable(var)
        assign = AST.assignment(var_name, value)
        return var, assign

    def create_temporary_variable(self, type, value, statements: List[ast.stmt]):
        name = self.create_temporary_variable_name()
        var, assign = self.add_local_variable(type, name, value)
        statements.append(assign)
        return var

    def _visit_and_create_temp_var(self, visitor, ast_node, statements: List[ast.stmt]):
        """Visit an AST node and create a temporary variable for its result."""
        values = visitor.visit(ast_node)
        # If visiting returns a list, preceding items are statements; last is the value
        if isinstance(values, list):
            statements.extend(values[:-1])
            value = values[-1]
        else:
            value = values
        name = self.create_temporary_variable_name()
        statements.append(AST.assignment(name, value))
        var_type = TypeFactory.get_type_from_ast(value)
        var = LocalVariableSource(var_type, name)
        self.add_variable(var)
        return var

    def _get_static_key(self, key_node) -> any:
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
            child_var = self.from_name(child_var_name)
            if child_var is None:
                raise RuntimeError(f"Internal error: child variable '{child_var_name}' not found for key '{key}'")
            # If child has skip_pre_read, it was never loaded into a local variable.
            # Use dynamic access with a constant index to generate inline access.
            if getattr(child_var, "skip_pre_read", False):
                key_index = container.get_key_index(key)
                return container.create_dynamic_access(ast.Constant(value=key_index), variable_factory=self)
            return child_var

        if isinstance(key_node, ast.Name):
            key_var = self.from_name(key_node.id)
            if key_var is not None and hasattr(key_var, "resolve_index_expr"):
                return container.create_dynamic_access(key_var.resolve_index_expr(container), variable_factory=self)

        # Dynamic access: a[some_variable]
        index_expr = visitor.visit(key_node) if visitor else key_node
        return container.create_dynamic_access(index_expr, variable_factory=self)

    def from_ast(self, visitor, ast_node: ast.AST, statements: List[ast.stmt]):
        if isinstance(ast_node, ast.Name):
            existing = self.from_name(ast_node.id)
            if existing is not None:
                return existing
            # Unknown variable - create with unknown type
            var = LocalVariableSource(UnknownType(UnknownNumbaType(), ast_node), ast_node.id)
            self.add_variable(var)
            return var

        if isinstance(ast_node, ast.Constant):
            name = self.create_temporary_variable_name()
            var_type = TypeFactory.get_type(type(ast_node.value))
            var = LocalConstantSource(var_type, name, ast_node.value)
            self.add_variable(var)
            return var

        if isinstance(ast_node, ast.Subscript) and isinstance(ast_node.value, ast.Name):
            container = self.from_name(ast_node.value.id)
            # Duck-typed: any container with key_to_child_name and create_dynamic_access
            if hasattr(container, "key_to_child_name") and hasattr(container, "create_dynamic_access"):
                return self._handle_container_subscript(visitor, container, ast_node.slice)

        # Default: visit expression and create a temporary variable
        return self._visit_and_create_temp_var(visitor, ast_node, statements)

    def add_alias(self, alias_name: str, variable: VariableSource):
        """
        Register an alias for an existing variable so future lookups by alias_name
        return the same VariableSource (preserving source-specific methods).
        """
        if alias_name in self.variable_name_map:
            raise ValueError(f"variable {alias_name} already exists")
        self.variable_name_map[alias_name] = variable

    def copy_source(self, src: VariableSource, new_name: str):
        """
        Create a new VariableSource of the same subclass as `src`, pointing to the same underlying storage
        (array index, pointer semantics, etc.) but with a distinct name for type tracking.
        """
        if new_name in self.variable_name_map:
            raise ValueError(f"variable {new_name} already exists")
        return src.clone_with_name(new_name)
