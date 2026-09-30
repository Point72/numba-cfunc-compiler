import ast

from numba_cfunc_compiler.core.names import (
    state_aug_name,
    state_slot_name,
    state_store_name,
)
from numba_cfunc_compiler.core.variable_access import OutputAccess, SourceAccess
from numba_cfunc_compiler.core.variable_factory import VariableFactory
from numba_cfunc_compiler.extension.callback_components import (
    CallbackComponentId,
    ComponentRegistry,
    MaterializationPhase,
)
from numba_cfunc_compiler.types.factory import HostTypeFactory

__all__ = [
    "NumbaASTConverter",
]

from numba_cfunc_compiler.extension.ast import (
    with_handlers,
)
from numba_cfunc_compiler.utils.ast import (
    AST,
    add_statement_to_list,
)


class NumbaASTConverter(ast.NodeTransformer):
    """Rewrite a decorated function into a lifecycle-aware ``@cfunc`` callback.

    Handlers registered through ``extension.ast`` can intercept selected AST
    visitors.
    """

    def __init__(
        self,
        tree: ast.AST,
        variable_factory: VariableFactory,
        start_body: list[ast.AST] | None = None,
        stop_body: list[ast.AST] | None = None,
        call_globals: dict | None = None,
        host_globals: dict | None = None,
        state_names: frozenset[str] = frozenset(),
    ):
        self.tree = tree
        self.variable_factory = variable_factory
        self.variable_factory.ast_converter = self
        self.call_globals = dict(call_globals or {})
        self.host_globals = host_globals or {}
        self.start_body = start_body or []
        self.stop_body = stop_body or []
        self.state_names = state_names
        self._state_aug_index = 0
        for body in (tree, *self.start_body, *self.stop_body):
            self._validate_state_binding_forms(body)

    def _validate_state_binding_forms(self, tree):
        """Accept only direct state assignments and augmented assignments."""
        allowed_targets = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.NamedExpr):
                raise TypeError(f"Named expressions in nodes are unsupported at line {node.lineno}")
            if isinstance(node, ast.Lambda):
                raise TypeError(f"Lambda expressions in nodes are unsupported at line {node.lineno}")
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                allowed_targets.add(id(node.targets[0]))
            if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                allowed_targets.add(id(node.target))
            if (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and isinstance(node.annotation, ast.Subscript)
                and isinstance(node.annotation.value, ast.Name)
                and node.annotation.value.id == "State"
            ):
                allowed_targets.add(id(node.target))
            if isinstance(node, ast.ExceptHandler) and node.name in self.state_names:
                raise TypeError(f"State '{node.name}' cannot be an exception target at line {node.lineno}")
            if isinstance(node, (ast.MatchAs, ast.MatchStar, ast.MatchMapping)):
                name = node.rest if isinstance(node, ast.MatchMapping) else node.name
                if name in self.state_names:
                    raise TypeError(f"State '{name}' cannot be a pattern capture at line {node.lineno}")
            if isinstance(node, (ast.Global, ast.Nonlocal)):
                for name in self.state_names.intersection(node.names):
                    raise TypeError(f"State '{name}' cannot be declared global or nonlocal at line {node.lineno}")
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id in self.state_names and id(node) not in allowed_targets:
                raise TypeError(f"State '{node.id}' requires a single-name assignment at line {node.lineno}")

    @staticmethod
    def _state_store(name: str, value: ast.AST) -> ast.Call:
        return AST.function_call(state_store_name(name), ast.Name(id=state_slot_name(name), ctx=ast.Load()), value)

    def visit_FunctionDef(self, node):
        from numba_cfunc_compiler.core.names import (
            LIFECYCLE_EXECUTE,
            LIFECYCLE_PARAM_NAME,
            LIFECYCLE_START,
            LIFECYCLE_STOP,
        )

        # Build function arguments dynamically from registered callback components
        node.args.args = ComponentRegistry.build_func_args()

        # Group variables by their component's materialization phase
        top_body = []
        execute_init_body = []
        state_start = []
        state_execute = []
        state_stop_before = []
        state_stop_after = []

        for component in ComponentRegistry.get_ordered():
            if component.materialization_phase == MaterializationPhase.NONE:
                continue
            for var in self.variable_factory.get_by_component(component.id):
                if component.materialization_phase == MaterializationPhase.ALWAYS:
                    if component.id == CallbackComponentId.STATE:
                        plan = var.state_plan
                        top_body.extend(plan.before)
                        state_start.extend(plan.start)
                        state_execute.extend(plan.execute)
                        state_stop_before.extend(plan.stop_before)
                        state_stop_after.extend(plan.stop_after)
                        continue
                    init = var.read()
                    if init is None:
                        continue
                    add_statement_to_list(top_body, init)
                elif component.materialization_phase == MaterializationPhase.EXECUTE:
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

        transformed_start_body = state_start + transformed_start_body
        execution_body = state_execute + execution_body
        transformed_stop_body = state_stop_before + transformed_stop_body + state_stop_after

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
            statements.append(output_var.tick())

        # Add a void return at the end
        statements.append(ast.Return(value=None))
        return statements

    @with_handlers("Call")
    def visit_Call(self, node):
        return self.generic_visit(node)

    @with_handlers("Expr")
    def visit_Expr(self, node):
        # Lower helper set_output(name, value) when used as a standalone statement
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "set_output":
            if len(node.value.args) != 2:
                invalid_call_str = ast.unparse(node.value)
                raise ValueError(f"set_output expects exactly 2 arguments: (name, value) got {invalid_call_str}")
            return self._lower_set_output(node.value.args[0], node.value.args[1])
        value = self.visit(node.value)
        if isinstance(value, (ast.stmt, list)):
            return value
        node.value = value
        return node

    def _lower_set_output(self, name_node, value_node):
        if not isinstance(name_node, ast.Constant) or not isinstance(name_node.value, str):
            raise TypeError(f"set_output name must be a string constant matching an output name: {ast.unparse(name_node)}")
        output_var = self.variable_factory.from_variable_name(name_node.value)
        if output_var is None:
            raise KeyError(f"set_output called with unknown output name '{name_node.value}'")
        if not isinstance(output_var, OutputAccess):
            raise TypeError(f"{name_node.value} is not a declared output")
        statements = []
        value_expr = self.variable_factory.lower_value_expression(self, value_node, statements)
        statements.append(output_var.write(value_expr))
        statements.append(output_var.tick())
        return statements

    @with_handlers("Subscript")
    def visit_Subscript(self, node):
        access = self.variable_factory.resolve_keyed_access(self, node)
        if access is not None and isinstance(node.ctx, ast.Load):
            return access
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
            if target.id in self.state_names:
                statements = []
                value = self.variable_factory.lower_value_expression(self, node.value, statements)
                statements.append(ast.copy_location(AST.assignment(target.id, self._state_store(target.id, value)), node))
                return statements if len(statements) > 1 else statements[0]
            access = self.variable_factory.from_variable_name(target.id)
            if access is not None and not isinstance(access, OutputAccess):
                if isinstance(access, SourceAccess):
                    return self.generic_visit(node)
                return AST.assignment(access.get(), self.visit(node.value))
            lowered = HostTypeFactory.lower_local_assignment(node, self.call_globals)
            if lowered is not None:
                return [self.visit(stmt) for stmt in lowered]

        return self.generic_visit(node)

    @with_handlers("AugAssign")
    def visit_AugAssign(self, node):
        if isinstance(node.target, ast.Name) and node.target.id in self.state_names:
            name = node.target.id
            self._state_aug_index += 1
            temp_name = state_aug_name(name, self._state_aug_index)
            statements = [AST.assignment(temp_name, ast.Name(id=name, ctx=ast.Load()))]
            value = self.variable_factory.lower_value_expression(self, node.value, statements)
            statements.extend(
                [
                    ast.AugAssign(target=ast.Name(id=temp_name, ctx=ast.Store()), op=node.op, value=value),
                    AST.assignment(name, self._state_store(name, ast.Name(id=temp_name, ctx=ast.Load()))),
                ]
            )
            return [ast.copy_location(stmt, node) for stmt in statements]
        return self.generic_visit(node)

    @with_handlers("Compare")
    def visit_Compare(self, node):
        return self.generic_visit(node)

    @with_handlers("For")
    def visit_For(self, node):
        return self.generic_visit(node)

    @with_handlers("Name")
    def visit_Name(self, node):
        if node.id in self.state_names:
            return node
        var = self.variable_factory.from_variable_name(node.id)
        if isinstance(var, OutputAccess):
            return node
        if isinstance(var, SourceAccess):
            return node
        # Only declared variables need host access rewrites. Numba owns locals.
        if var is not None:
            return var.read_value() if isinstance(node.ctx, ast.Load) else var.get()
        return node

    def visit_Delete(self, node):
        for target in node.targets:
            for name in ast.walk(target):
                if isinstance(name, ast.Name) and name.id in self.state_names:
                    raise TypeError(f"Cannot delete State '{name.id}' at line {name.lineno}")
        return self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if (
            isinstance(node.annotation, ast.Subscript)
            and isinstance(node.annotation.value, ast.Name)
            and node.annotation.value.id == "State"
            and isinstance(node.target, ast.Name)
        ):
            var_name = node.target.id
            var = self.variable_factory.from_variable_name(var_name)
            if getattr(var, "component", None) != CallbackComponentId.STATE:
                raise TypeError(f"{var_name} is not a state variable")
            return None

        return self.generic_visit(node)
