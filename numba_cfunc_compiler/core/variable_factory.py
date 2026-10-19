import ast
from collections import defaultdict
from typing import Any

from numba_cfunc_compiler.core.names import bind_keyed_input_name, typed_struct_name
from numba_cfunc_compiler.core.variable_access import KeyedInputAccess, OutputAccess, VariableAccess
from numba_cfunc_compiler.extension.callback_components import ComponentRegistry, MaterializationPhase


class CallbackBody:
    """Statements contributed by variables to each callback phase."""

    def __init__(self):
        self.setup = []
        self.execute_init = []
        self._lifecycle = {phase: [] for phase in ("start", "execute", "stop_before", "stop_after")}

    def add_lifecycle(self, order, plan):
        for phase, entries in self._lifecycle.items():
            entries.append((order, getattr(plan, phase)))

    def phase(self, name):
        # Later components bind first; cleanup reverses that order.
        entries = sorted(self._lifecycle[name], key=lambda entry: entry[0], reverse=name != "stop_after")
        return [statement for _, statements in entries for statement in statements]


class VariableFactory:
    def __init__(self):
        self.variables_by_access_type = defaultdict(list)
        self.component_variables = defaultdict(list)
        self.variable_by_name = {}
        self.typed_struct_bindings = {}
        self.typed_struct_layouts = {}
        self.typed_input_bindings = {}
        self.typed_keyed_bindings = {}

    def bind_input_type(self, value_type):
        name = bind_keyed_input_name(value_type.name)
        self.typed_input_bindings[name] = value_type
        return name

    def _bind_struct(self, struct_type, operation: str) -> str:
        from numba_cfunc_compiler.types.builtin.struct.native import struct_copy, struct_view

        layout = struct_type.get_typed_layout()
        name = typed_struct_name(operation, layout.fingerprint)
        self.typed_struct_bindings[name] = struct_view(layout) if operation == "view" else struct_copy(layout)
        self.typed_struct_layouts[layout.fingerprint] = layout
        return name

    def bind_struct_view(self, struct_type) -> str:
        return self._bind_struct(struct_type, "view")

    def bind_struct_copy(self, struct_type) -> str:
        return self._bind_struct(struct_type, "copy")

    def add_variable(self, variable: VariableAccess, component: Any = None):
        if variable.name in self.variable_by_name:
            raise ValueError(f"variable {variable.name} already exists")
        variable.variable_factory = self
        self.variables_by_access_type[type(variable)].append(variable)
        if component is not None:
            variable.component = component
            self.component_variables[component].append(variable)
        self.variable_by_name[variable.name] = variable

    def get_accesses(self, access_type: type):
        if not issubclass(access_type, VariableAccess):
            raise TypeError(f"access_type must be a subclass of VariableAccess, got {access_type}")
        return self.variables_by_access_type[access_type]

    def get_by_component(self, component_id: Any) -> list:
        """Get variables registered under *component_id*."""
        return self.component_variables.get(component_id, [])

    def build_func_args(self) -> list[ast.arg]:
        return ComponentRegistry.build_func_args()

    def materialize(self) -> CallbackBody:
        body = CallbackBody()
        for component in ComponentRegistry.get_ordered():
            if component.materialization_phase is MaterializationPhase.NONE:
                continue
            for variable in self.get_by_component(component.id):
                variable.contribute(body, component.materialization_phase, component.order)
        return body

    def from_variable_name(self, name: str):
        """Look up a declared variable access by name."""
        return self.variable_by_name.get(name)

    def get_output_by_idx(self, idx: int):
        output = self.variables_by_access_type[OutputAccess][idx]
        if output.array_idx != idx:
            raise RuntimeError(f"output {output.name} has array index {output.array_idx} but expected {idx}")
        return output

    def lower_value_expression(self, visitor, ast_node: ast.AST, statements: list[ast.stmt]) -> ast.AST:
        """Lower an expression without recording an inferred Python-side type.

        Keyed input access still needs variable lookup. Ordinary locals and
        expressions are left for Numba to type in the generated function.
        """
        keyed_access = self.resolve_keyed_access(visitor, ast_node)
        if keyed_access is not None:
            return keyed_access

        value = visitor.visit(ast_node)
        if isinstance(value, list):
            statements.extend(value[:-1])
            return value[-1]
        return value

    def resolve_keyed_access(self, visitor, ast_node: ast.AST):
        """Resolve host-backed keyed access without inferring an expression type."""
        if not isinstance(ast_node, ast.Subscript) or not isinstance(ast_node.value, ast.Name):
            return None
        container = self.from_variable_name(ast_node.value.id)
        if isinstance(container, KeyedInputAccess):
            return container.resolve_access(visitor, ast_node.slice)
        return None
