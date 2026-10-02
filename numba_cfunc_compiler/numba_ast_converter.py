import ast
import copy

from numba_cfunc_compiler.models import (
    ContainerType,
)
from numba_cfunc_compiler.source_registry import (
    SourceCategoryId,
    SourceInitFilter,
    SourceRegistry,
)
from numba_cfunc_compiler.state_ast import (
    is_state_annotation,
    state_annotation_target,
)
from numba_cfunc_compiler.type_factory import TypeFactory
from numba_cfunc_compiler.variable_factory import (
    VariableFactory,
)

__all__ = [
    "NumbaASTConverter",
]

from numba_cfunc_compiler.ast_handlers import (
    with_handlers,
)
from numba_cfunc_compiler.utils.ast import (
    AST,
    add_statement_to_list,
)


class NumbaASTConverter(ast.NodeTransformer):
    """
    AST transformer to convert decorated function to @cfunc format.

    Handlers can be registered using the @ast_handler decorator from ast_handlers.py.
    """

    def __init__(
        self,
        tree: ast.AST,
        variable_factory: VariableFactory,
        start_body: list[ast.AST] | None = None,
        stop_body: list[ast.AST] | None = None,
        call_globals: dict | None = None,
        state_bindings: dict | None = None,
    ):
        self.tree = tree
        self.variable_factory = variable_factory
        self.variable_factory.ast_converter = self
        self.call_globals = call_globals or {}
        self.start_body = start_body or []
        self.stop_body = stop_body or []
        self.state_bindings = state_bindings or {}
        for body in (tree, *self.start_body, *self.stop_body):
            self._validate_state_binding_forms(body)

    def _validate_state_binding_forms(self, tree):
        """Reject bindings whose lexical scope is not a node-body local."""
        state_names = self.state_bindings.keys()
        for node in ast.walk(tree):
            if isinstance(node, ast.Lambda):
                raise TypeError(f"Lambda expressions in nodes are unsupported at line {node.lineno}")
            if isinstance(node, ast.comprehension):
                targets = {name.id for name in ast.walk(node.target) if isinstance(name, ast.Name)}
                for name in targets & state_names:
                    raise TypeError(f"State '{name}' cannot be bound in a comprehension at line {node.target.lineno}")
            if isinstance(node, ast.ExceptHandler) and node.name in state_names:
                raise TypeError(f"State '{node.name}' cannot be an exception target at line {node.lineno}")
            if isinstance(node, (ast.MatchAs, ast.MatchStar, ast.MatchMapping)):
                name = node.rest if isinstance(node, ast.MatchMapping) else node.name
                if name in state_names:
                    raise TypeError(f"State '{name}' cannot be a pattern capture at line {node.lineno}")
            if isinstance(node, ast.withitem) and node.optional_vars is not None:
                targets = {name.id for name in ast.walk(node.optional_vars) if isinstance(name, ast.Name)}
                for name in targets & state_names:
                    raise TypeError(f"State '{name}' cannot be a with target at line {node.optional_vars.lineno}")

    def _state_bind(self, var):
        binding = self.state_bindings[var.name]
        return [
            AST.assignment(binding.slot_name, AST.array_access(var.get_storage_location(), var.array_idx)),
            AST.assignment(var.name, AST.function_call(f"__ncc_bind_state_{var.name}", ast.Name(id=binding.slot_name, ctx=ast.Load()))),
        ]

    def _container_init(self, variables):
        statements = []
        for var in variables:
            value_name = f"__ncc_state_init_{var.name}"
            loaded_name = f"__ncc_state_loaded_{var.name}"
            state_slot = AST.array_access(var.get_storage_location(), var.array_idx)
            statements.extend(var.type.init_statements(value_name, loaded_name, state_slot))
            statements.append(AST.assignment(var.name, AST.function_call(f"__ncc_bind_state_{var.name}", ast.Name(id=value_name, ctx=ast.Load()))))
        return statements

    def _container_load(self, variables):
        statements = []
        for var in variables:
            value_name = f"__ncc_state_value_{var.name}"
            loaded_name = f"__ncc_state_loaded_{var.name}"
            state_slot = AST.array_access(var.get_storage_location(), var.array_idx)
            statements.append(AST.assignment(loaded_name, state_slot))
            statements.extend(var.type.post_load_statements(value_name, var.name, ast.Name(id=loaded_name, ctx=ast.Load())))
            statements.append(AST.assignment(var.name, AST.function_call(f"__ncc_bind_state_{var.name}", ast.Name(id=value_name, ctx=ast.Load()))))
        return statements

    def visit_FunctionDef(self, node):
        from numba_cfunc_compiler.compiler_constants import (
            LIFECYCLE_EXECUTE,
            LIFECYCLE_PARAM_NAME,
            LIFECYCLE_START,
            LIFECYCLE_STOP,
        )

        # Build function arguments dynamically from registered source categories
        node.args.args = SourceRegistry.build_func_args()

        # Group variables by their category's init_filter
        top_body = []
        execute_init_body = []
        container_state_vars = []

        for category in SourceRegistry.get_ordered():
            if category.init_filter == SourceInitFilter.NEVER:
                continue
            for var in self.variable_factory.get_by_category(category.id):
                if category.init_filter == SourceInitFilter.EVERYTIME:
                    if category.id == SourceCategoryId.STATE and isinstance(var.type, ContainerType):
                        container_state_vars.append(var)
                        add_statement_to_list(
                            top_body,
                            AST.assignment(self.state_bindings[var.name].slot_name, AST.array_access(var.get_storage_location(), var.array_idx)),
                        )
                        continue
                    init = self._state_bind(var) if category.id == SourceCategoryId.STATE else var.read()
                    if init is None:
                        continue
                    add_statement_to_list(top_body, init)
                elif category.init_filter == SourceInitFilter.ON_EXECUTE:
                    init = var.read()
                    if init is None:
                        continue
                    add_statement_to_list(execute_init_body, init)

        # Transform user's start_body statements
        transformed_start_body = []
        for stmt in self.start_body:
            transformed_stmt = self.visit(stmt)
            add_statement_to_list(transformed_start_body, transformed_stmt)

        # Transform user's stop_body statements
        transformed_stop_body = []
        for stmt in self.stop_body:
            transformed_stmt = self.visit(stmt)
            add_statement_to_list(transformed_stop_body, transformed_stmt)

        # Transform user's execution body
        execution_body = []
        for stmt in node.body:
            transformed_stmt = self.visit(stmt)
            add_statement_to_list(execution_body, transformed_stmt)

        # Container state: init in start phase, load in execute phase
        if container_state_vars:
            # Prepend container initialization to start_body
            container_init = self._container_init(container_state_vars)
            transformed_start_body = container_init + transformed_start_body

            # Prepend container loading to execution_body
            container_load = self._container_load(container_state_vars)
            execution_body = copy.deepcopy(container_load) + execution_body

            # STOP needs typed container values for user cleanup code, then must
            # release the native allocations before the host drops its slots.
            container_free = ContainerType.emit_container_state_free(container_state_vars)
            transformed_stop_body = container_load + transformed_stop_body + container_free

        # Build the lifecycle-aware body
        lifecycle_body = []

        # Add start phase check: if lifecycle_phase == LIFECYCLE_START: ...
        if transformed_start_body:
            start_if = ast.If(
                test=ast.Compare(
                    left=ast.Name(id=LIFECYCLE_PARAM_NAME, ctx=ast.Load()),
                    ops=[ast.Eq()],
                    comparators=[ast.Constant(value=LIFECYCLE_START)],
                ),
                body=transformed_start_body,
                orelse=[],
            )
            lifecycle_body.append(start_if)

        # Add stop phase check: if lifecycle_phase == LIFECYCLE_STOP: ...
        if transformed_stop_body:
            stop_if = ast.If(
                test=ast.Compare(
                    left=ast.Name(id=LIFECYCLE_PARAM_NAME, ctx=ast.Load()),
                    ops=[ast.Eq()],
                    comparators=[ast.Constant(value=LIFECYCLE_STOP)],
                ),
                body=transformed_stop_body,
                orelse=[],
            )
            lifecycle_body.append(stop_if)

        exec_if = ast.If(
            test=ast.Compare(
                left=ast.Name(id=LIFECYCLE_PARAM_NAME, ctx=ast.Load()),
                ops=[ast.Eq()],
                comparators=[ast.Constant(value=LIFECYCLE_EXECUTE)],
            ),
            body=(execute_init_body + execution_body) if (execute_init_body or execution_body) else [ast.Pass()],
            orelse=[],
        )
        lifecycle_body.append(exec_if)

        node.body = top_body + lifecycle_body
        node.returns = None

        ast.fix_missing_locations(node)
        return node

    @with_handlers("Return")
    def visit_Return(self, node):
        if not node.value:
            return node

        statements = []

        if isinstance(node.value, ast.Tuple):
            elements = node.value.elts
        else:
            elements = [node.value]

        for i, elt in enumerate(elements):
            output_var = self.variable_factory.get_output_by_idx(i)

            # if we return None, do nothing
            if (isinstance(elt, ast.Name) and elt.id == "None") or (isinstance(elt, ast.Constant) and elt.value is None):
                continue

            value = self.variable_factory.lower_value_expression(self, elt, statements)
            statements.append(output_var.write(value))
            statements.append(output_var.call("output", None))

        # Add a void return at the end
        statements.append(ast.Return(value=None))
        return statements

    @with_handlers("Call")
    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute):
            base = node.func.value
            source = None
            if isinstance(base, ast.Name) and base.id in self.state_bindings:
                return self.generic_visit(node)
            if isinstance(base, ast.Name):
                source = self.variable_factory.from_source_name(base.id)
            elif isinstance(base, ast.Subscript):
                source = self.variable_factory.resolve_keyed_source(self, base)
            if source is not None:
                method = getattr(source.handler, "METHODS", {}).get(node.func.attr)
                if method is not None:
                    args = [self.visit(arg) for arg in node.args]
                    result = source.call(node.func.attr, args)
                    if result is not None:
                        return result

        return self.generic_visit(node)

    @with_handlers("Expr")
    def visit_Expr(self, node):
        # Lower helper set_output(name, value) when used as a standalone statement
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "set_output":
            if len(node.value.args) != 2:
                invalid_call_str = ast.unparse(node.value)
                raise ValueError(f"set_output expects exactly 2 arguments: (name, value) got {invalid_call_str}")
            return AST.set_output(self.variable_factory, self, node.value.args[0], node.value.args[1])
        value = self.visit(node.value)
        if isinstance(value, (ast.stmt, list)):
            return value
        node.value = value
        return node

    @with_handlers("Subscript")
    def visit_Subscript(self, node):
        source = self.variable_factory.resolve_keyed_source(self, node)
        if source is not None and isinstance(node.ctx, ast.Load):
            return source.get()
        return self.generic_visit(node)

    @with_handlers("Attribute")
    def visit_Attribute(self, node):
        return self.generic_visit(node)

    @with_handlers("Assign")
    def visit_Assign(self, node):
        if len(node.targets) != 1:
            return self.generic_visit(node)

        target = node.targets[0]
        if isinstance(target, ast.Name):
            source = self.variable_factory.from_source_name(target.id)
            if source is not None:
                if target.id in self.state_bindings:
                    return self.generic_visit(node)
                return AST.assignment(source.get(), self.visit(node.value))
            lowered = TypeFactory.try_lower_assignment(node, node.value, self.call_globals)
            if lowered is not None:
                return [self.visit(stmt) for stmt in lowered]

        return self.generic_visit(node)

    @with_handlers("AugAssign")
    def visit_AugAssign(self, node):
        return self.generic_visit(node)

    @with_handlers("Compare")
    def visit_Compare(self, node):
        return self.generic_visit(node)

    @with_handlers("For")
    def visit_For(self, node):
        return self.generic_visit(node)

    @with_handlers("Name")
    def visit_Name(self, node):
        if node.id in self.state_bindings:
            return node
        var = self.variable_factory.from_source_name(node.id)
        # Only declared sources need host storage rewrites. Numba owns locals.
        if var is not None:
            return var.read_value() if isinstance(node.ctx, ast.Load) else var.get()
        return node

    def visit_Delete(self, node):
        for target in node.targets:
            for name in ast.walk(target):
                if isinstance(name, ast.Name) and name.id in self.state_bindings:
                    raise TypeError(f"Cannot delete State '{name.id}' at line {name.lineno}")
        return self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if is_state_annotation(node):
            var_name = state_annotation_target(node)
            var = self.variable_factory.from_source_name(var_name)
            if getattr(var, "category", None) != SourceCategoryId.STATE:
                raise TypeError(f"{var_name} is not a state variable")
            return None

        return self.generic_visit(node)
