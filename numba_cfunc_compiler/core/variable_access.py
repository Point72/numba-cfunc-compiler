import ast
import copy
from typing import Any

from numba_cfunc_compiler.core.names import (
    OUTPUTS_ARRAY_NAME,
    TICKED_OUTPUTS_ARRAY_NAME,
    bind_input_name,
    bind_keyed_basket_name,
    bind_output_name,
    output_sink_name,
)
from numba_cfunc_compiler.extension.input_sources import SLOT_INDEX, InputSourceDescriptor, KeyedInputDescriptor, _require_dense_slots
from numba_cfunc_compiler.types.binding import TypeBinding
from numba_cfunc_compiler.utils.ast import AST


class VariableAccess:
    """Describe how generated AST reads or writes one declared variable."""

    def __init__(
        self,
        type: TypeBinding | None,
        name: str,
        variable_factory=None,
    ):
        self.type = type
        self.name = name
        self.variable_factory = variable_factory
        self.component: Any = None  # Set by VariableFactory.add_variable

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


class PointerSlotAccess(VariableAccess):
    """Access a value in a host-owned array of void pointers.

    Args:
        array_idx (int): Index into the storage array.
        type (TypeBinding): The resolved value representation.
        name (str): Variable name used in generated code.
        storage_location (str): Name of the array this variable is read from (e.g. "inputs").
    """

    def __init__(
        self,
        array_idx: int,
        type: TypeBinding,
        name: str,
        storage_location: str,
    ):
        self.array_idx = array_idx
        self._storage_location = storage_location
        super().__init__(type, name)

    def local_variable_name(self):
        return self.name

    def get_storage_location(self):
        return self._storage_location

    def read(self):
        loaded_value = AST.array_access(self.get_storage_location(), self.array_idx)
        return self.type.slot_read(self.local_variable_name(), loaded_value, self.variable_factory)

    def get(self):
        if self.type.is_opaque_pointer():
            # opaque pointer-like types (structs, order books, containers) are represented as void pointers
            return ast.Name(id=self.local_variable_name(), ctx=ast.Load())
        # for other types we dereference the pointer
        return AST.deref_pointer(self.local_variable_name())

    def read_value(self):
        value = self.get()
        return self.type.lower_loaded_value(value)


def _field_expr(value):
    if isinstance(value, ast.expr):
        return copy.deepcopy(value)
    if isinstance(value, str):
        return ast.Name(id=value, ctx=ast.Load())
    if isinstance(value, int):
        return ast.Constant(value=value)
    raise TypeError(f"Unsupported source field expression {value!r}")


class SourceAccess(VariableAccess):
    """One input slot bound with host-declared source metadata."""

    def __init__(self, array_idx: int, type: TypeBinding, name: str, storage_location: str, source: InputSourceDescriptor, metadata: dict):
        if not isinstance(source, InputSourceDescriptor):
            raise TypeError("Source access needs an input source descriptor")
        if set(metadata) != {field.name for field in source.fields}:
            raise ValueError("Source access must bind every declared metadata field")
        super().__init__(type, name)
        self.array_idx = array_idx
        self.storage_location = storage_location
        self.source = source
        self.metadata = dict(metadata)

    def input_value_type(self):
        return self.source.value_type(self.type)

    def local_variable_name(self):
        return self.name

    def get_storage_location(self):
        return self.storage_location

    def read(self):
        return AST.assignment(
            self.name,
            AST.function_call(
                bind_input_name(self.name),
                AST.array_access(self.storage_location, self.array_idx),
                ast.Tuple(elts=[_field_expr(self.metadata[field.name]) for field in self.source.fields], ctx=ast.Load()),
            ),
        )

    def get(self):
        return ast.Name(id=self.name, ctx=ast.Load())

    def read_value(self):
        return self.get()


class OutputAccess(PointerSlotAccess):
    def __init__(self, array_idx: int, type: TypeBinding, name: str):
        super().__init__(
            array_idx=array_idx,
            type=type,
            name=name,
            storage_location=OUTPUTS_ARRAY_NAME,
        )

    def local_variable_name(self):
        return output_sink_name(self.array_idx)

    def output_sink_type(self):
        return self.type.output_sink_type()

    def read(self):
        return AST.assignment(
            self.local_variable_name(),
            AST.function_call(
                bind_output_name(self.array_idx),
                AST.array_access(self.get_storage_location(), self.array_idx),
                ast.Name(id=TICKED_OUTPUTS_ARRAY_NAME, ctx=ast.Load()),
                ast.Constant(value=self.array_idx),
            ),
        )

    def write(self, value: Any):
        return ast.Expr(
            value=ast.Call(
                func=ast.Attribute(value=ast.Name(id=self.local_variable_name(), ctx=ast.Load()), attr="write", ctx=ast.Load()),
                args=[value],
                keywords=[],
            )
        )

    def tick(self):
        return ast.Expr(
            value=ast.Call(
                func=ast.Attribute(value=ast.Name(id=self.local_variable_name(), ctx=ast.Load()), attr="tick", ctx=ast.Load()),
                args=[],
                keywords=[],
            )
        )


class KeyedInputAccess(VariableAccess):
    """Host basket whose indexed elements become generated Numba input values."""

    def __init__(self, name, children, basket: KeyedInputDescriptor, fields: dict):
        if not isinstance(basket, KeyedInputDescriptor):
            raise TypeError("Keyed input needs a basket descriptor")
        _require_dense_slots(index for index, _ in children.values())
        field_values = dict(fields)
        field_values.setdefault("start", min((index for index, _ in children.values()), default=0))
        field_values.setdefault("length", len(children))
        if set(field_values) != {field.name for field in basket.fields}:
            raise ValueError("Keyed input must bind every basket metadata field")
        if not isinstance(field_values["slots"], str):
            raise TypeError("Keyed slot array must be a callback parameter name")
        super().__init__(None, name)
        self.children = dict(children)  # key -> (host array index, HostType)
        self.basket = basket
        self.fields = field_values
        self.storage_location = field_values["slots"]

    def local_variable_name(self):
        return self.name

    def get_storage_location(self):
        return self.storage_location

    def read(self):
        layout_key = tuple(
            (key, index, self.basket.element_source.value_type(var_type))
            for key, (index, var_type) in sorted(self.children.items(), key=lambda item: repr(item[0]))
        )
        value_type = self.basket.value_type(layout_key)
        bind_name = bind_keyed_basket_name(self.name)
        self.variable_factory.typed_keyed_bindings[bind_name] = value_type
        return AST.assignment(
            self.name,
            AST.function_call(
                bind_name,
                ast.Tuple(elts=[_field_expr(self.fields[field.name]) for field in self.basket.fields], ctx=ast.Load()),
            ),
        )

    def get(self):
        raise TypeError(f"Keyed input '{self.name}' must be indexed")

    def read_value(self):
        return ast.Name(id=self.name, ctx=ast.Load())

    def resolve_access(self, visitor, key_node):
        if isinstance(key_node, ast.Constant):
            match = next(
                (
                    (index, var_type)
                    for key, (index, var_type) in self.children.items()
                    if type(key) is type(key_node.value) and key == key_node.value
                ),
                None,
            )
            if match is None:
                raise KeyError(f"Keyed input '{self.name}' has no key {key_node.value!r}")
            index, var_type = match
            index_expr = ast.Constant(value=index)
        else:
            raise TypeError(f"Dynamic indexing of '{self.name}' uses {self.name}.at(position)")
        value_type = self.basket.element_source.value_type(var_type)
        bind_name = self.variable_factory.bind_input_type(value_type)
        metadata = []
        for field in self.basket.element_source.fields:
            projection = self.basket.element_projection[field.name]
            metadata.append(copy.deepcopy(index_expr) if projection is SLOT_INDEX else _field_expr(self.fields[projection]))
        return AST.function_call(
            bind_name,
            AST.array_access(self.storage_location, copy.deepcopy(index_expr)),
            ast.Tuple(elts=metadata, ctx=ast.Load()),
        )


class ConstantAccess(VariableAccess):
    """Materialize a validated argument in code rather than a callback slot."""

    def __init__(self, type: TypeBinding, name: str, value: Any):
        super().__init__(type, name)
        self.value = value
        self._constant_plan = None

    def _plan(self):
        if self._constant_plan is None:
            self._constant_plan = self.type.constant_plan(self.name, self.value, self.variable_factory.ast_converter.call_globals)
        return self._constant_plan

    def local_variable_name(self):
        return self.name

    def get_storage_location(self):
        raise ValueError("Constants are not stored in a storage location")

    def read(self):
        return list(self._plan().setup)

    def get(self):
        return copy.deepcopy(self._plan().value)
