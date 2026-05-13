import ast
from typing import List

from numba_cfunc_compiler.defaults.struct_support import StructType
from numba_cfunc_compiler.models import (
    ContainerType,
    NoneType,
    UnknownType,
)
from numba_cfunc_compiler.numba_type_inference import (
    NumbaTypeInference,
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
        start_body: List[ast.AST] = None,
        stop_body: List[ast.AST] = None,
        call_globals: dict = None,
    ):
        self.tree = tree
        self.variable_factory = variable_factory
        self.variable_factory.ast_converter = self
        self.call_globals = call_globals or {}
        self.numba_type_inference = NumbaTypeInference(variable_factory, self.call_globals)
        self.start_body = start_body or []
        self.stop_body = stop_body or []

    def visit_FunctionDef(self, node):
        from numba_cfunc_compiler.numba_config import (
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
                        continue
                    init = var.read()
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
            container_init = ContainerType.emit_container_state_init(container_state_vars)
            transformed_start_body = container_init + transformed_start_body

            # Prepend container loading to execution_body
            container_load = ContainerType.emit_container_state_load(container_state_vars)
            execution_body = container_load + execution_body

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

            if isinstance(elt, ast.Name):
                # if we return None, do nothing
                if elt.id == "None":
                    continue

            # get or create a local variable for the return value
            var = self.variable_factory.from_ast(visitor=self, ast_node=elt, statements=statements)

            if isinstance(var.type, UnknownType) and isinstance(var.type.runtime_value, NoneType):
                continue

            value = var.get()
            statements.append(output_var.write(value))
            statements.append(output_var.call("output", None))

        # Add a void return at the end
        statements.append(ast.Return(value=None))
        return statements

    @with_handlers("Call")
    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute):
            final_var = self.numba_type_inference.handle_call_chain(node)
            if final_var is not None:
                return final_var.get()

        return self.generic_visit(node)

    @with_handlers("Expr")
    def visit_Expr(self, node):
        # Lower helper set_output(name, value) when used as a standalone statement
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            if node.value.func.id == "set_output":
                if len(node.value.args) != 2:
                    invalid_call_str = ast.unparse(node.value)
                    raise ValueError(f"set_output expects exactly 2 arguments: (name, value) got {invalid_call_str}")
                return AST.set_output(self.variable_factory, self, node.value.args[0], node.value.args[1])
        return self.generic_visit(node)

    @with_handlers("Subscript")
    def visit_Subscript(self, node):
        # included here so user types that require subscripting work
        return self.generic_visit(node)

    @with_handlers("Attribute")
    def visit_Attribute(self, node):
        result = self.numba_type_inference.try_attr_lowerers(node)
        if result is not None:
            return result
        return self.generic_visit(node)

    @with_handlers("Assign")
    def visit_Assign(self, node):
        # Single target only
        if len(node.targets) != 1:
            return self.generic_visit(node)

        target = node.targets[0]

        # Case 1: Struct field assignment (my_struct.field = expr)
        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
            base_name = target.value.id
            field_name = target.attr
            base_var = self.variable_factory.from_name(base_name)

            if base_var is not None and isinstance(base_var.type, StructType):
                # Visit RHS first to transform any nested expressions
                value_expr = self.visit(node.value)
                # Generate struct_field_store call
                store_call = base_var.type.set_field(base_name, field_name, value_expr)
                return ast.Expr(value=store_call)

        # Case 2: Existing managed variables
        if isinstance(target, ast.Name):
            assignment = self.numba_type_inference.create_assignment_variable(node, node.value)
            if assignment is not None:
                return assignment

        # Case 3: Try lowering assignment via registered type classes
        rhs = node.value
        assign = self.numba_type_inference.try_lower_assignment(node, rhs)
        if assign is not None:
            # assign may be a list of statements or a single AST node
            if isinstance(assign, list):
                return [self.visit(stmt) if isinstance(stmt, ast.AST) else stmt for stmt in assign]
            return self.generic_visit(assign)

        return self.generic_visit(node)

    @with_handlers("AugAssign")
    def visit_AugAssign(self, node):
        return self.generic_visit(node)

    @with_handlers("Compare")
    def visit_Compare(self, node):
        node = self.generic_visit(node)

        transformed_left = self.numba_type_inference.try_attr_lowerers(node.left)
        if transformed_left is not None:
            node.left = transformed_left
        elif isinstance(node.left, ast.Name):
            var = self.variable_factory.from_name(node.left.id)
            if var is not None:
                node.left = var.get()

        # Transform comparators
        new_comparators = []
        for comp in node.comparators:
            transformed_comp = self.numba_type_inference.try_attr_lowerers(comp)
            if transformed_comp is not None:
                comp = transformed_comp
            elif isinstance(comp, ast.Name):
                var = self.variable_factory.from_name(comp.id)
                if var is not None:
                    comp = var.get()
            new_comparators.append(comp)
        node.comparators = new_comparators

        return node

    @with_handlers("For")
    def visit_For(self, node):
        return self.generic_visit(node)

    @with_handlers("Name")
    def visit_Name(self, node):
        var = self.variable_factory.from_name(node.id)
        # Let managed variables decide how to represent themselves
        if var is not None:
            return var.get()
        return node

    def visit_AnnAssign(self, node):
        if is_state_annotation(node):
            var_name = state_annotation_target(node)
            var = self.variable_factory.from_name(var_name)
            if getattr(var, "category", None) != SourceCategoryId.STATE:
                raise TypeError(f"{var_name} is not a state variable")
            return None

        return self.generic_visit(node)
