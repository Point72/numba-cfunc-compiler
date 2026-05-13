import ast
from enum import Enum
from typing import Callable

from numba_cfunc_compiler.compilation_context import CompilationContext
from numba_cfunc_compiler.models import VariableType
from numba_cfunc_compiler.type_factory import TypeFactory
from numba_cfunc_compiler.utils.ast import AST
from numba_cfunc_compiler.variable_factory import (
    ExpressionSource,
    LocalVariableSource,
    VariableFactory,
    VariableSource,
)

__all__ = [
    "OpKind",
    "CallHandler",
    "AttrAccessor",
    "AttrLowerer",
    "AssignmentHandler",
    "NumbaTypeInference",
]


class OpKind(Enum):
    ATTR = "attr"
    CALL = "call"


# Type aliases for handlers
CallHandler = Callable[["NumbaTypeInference", VariableSource, str, list], ExpressionSource | None]
AttrAccessor = Callable[["NumbaTypeInference", VariableSource, str, list], ExpressionSource | None]
AttrLowerer = Callable[[ast.AST, dict, "VariableFactory"], ast.AST | None]
AssignmentHandler = Callable[["NumbaTypeInference", ast.Assign, ast.AST], ast.AST | None]


class NumbaTypeInference:
    """
    Type inference engine for numba_node functions.

    Supports registration of custom handlers for:
    - Assignment handlers: Handle entire assignment statements (called first)
    - Call handlers: Handle method calls on specific variable types
    - Attr accessors: Handle attribute access on specific variable types
    - Variable creators: Handle creation of typed variables from assignments
    - Attr lowerers: Transform attribute nodes (e.g., enum.VALUE -> constant)

    All handler registries live in the active CompilationContext.
    """

    @classmethod
    def register_assignment_handler(cls, handler: AssignmentHandler):
        CompilationContext.current().assignment_handlers.append(handler)

    @classmethod
    def register_call_handler(cls, handler: CallHandler):
        CompilationContext.current().call_handlers.append(handler)

    @classmethod
    def register_attr_accessor(cls, handler: AttrAccessor):
        CompilationContext.current().attr_accessors.append(handler)

    @classmethod
    def register_attr_lowerer(cls, transformer: AttrLowerer):
        CompilationContext.current().attr_lowerers.append(transformer)

    @classmethod
    def clear_handlers(cls):
        ctx = CompilationContext.current()
        ctx.assignment_handlers.clear()
        ctx.call_handlers.clear()
        ctx.attr_accessors.clear()
        ctx.attr_lowerers.clear()

    def __init__(self, variable_factory: VariableFactory, call_globals: dict = None):
        self.variable_factory = variable_factory
        self.call_globals = call_globals or {}

    def _dispatch_method_call(
        self,
        base_var: VariableSource,
        method_name: str,
        args: list[ast.AST],
    ) -> ExpressionSource:
        ctx = CompilationContext.current()
        # Try registered call handlers first
        for handler in ctx.call_handlers:
            result = handler(self, base_var, method_name, args)
            if result is not None:
                return result

        # Fall back to the variable's call() method for VariableType
        if isinstance(base_var.type, VariableType):
            try:
                lowered_expr = base_var.call(method_name, args)
            except ValueError as e:
                raise ValueError(f"{base_var.name} {e}")
            if lowered_expr is None:
                type_name = type(base_var.type).__name__
                raise ValueError(f"Method '{method_name}' is not supported on variable '{base_var.name}' of type {type_name}")
            var_type = TypeFactory.get_type_from_ast(lowered_expr)
            return ExpressionSource(var_type, lowered_expr, variable_factory=self.variable_factory)

        raise ValueError(f"Expected {base_var.name} to be a valid type, got {type(base_var)}")

    def _dispatch_attr_access(
        self,
        base_var: VariableSource,
        attr_name: str,
        args: list[ast.AST],
    ) -> ExpressionSource | None:
        ctx = CompilationContext.current()
        for handler in ctx.attr_accessors:
            result = handler(self, base_var, attr_name, args)
            if result is not None:
                return result
        return None

    def handle_call_chain(self, node: ast.AST) -> ExpressionSource | None:
        """
        Unwind attribute/call chain starting at `node` and return the result.

        Handles chains like: a.method(), a['key'].field, a.method().field
        Returns None if the node is not a supported chain.
        """
        # Build chain from outside-in
        chain = []
        cur = node
        while True:
            if isinstance(cur, ast.Call) and isinstance(cur.func, ast.Attribute):
                chain.append((OpKind.CALL, cur.func.attr, cur.args))
                cur = cur.func.value
            elif isinstance(cur, ast.Attribute):
                chain.append((OpKind.ATTR, cur.attr, []))
                cur = cur.value
            else:
                break

        if not chain:
            return None

        chain.reverse()

        # Resolve base of the chain
        if isinstance(cur, ast.Name):
            current_var = self.variable_factory.from_name(cur.id)
        elif isinstance(cur, ast.Subscript):
            current_var = self.variable_factory.from_ast(visitor=self.variable_factory.ast_converter, ast_node=cur, statements=[])
        else:
            return None

        # Walk forward through the chain
        for op_kind, name, args in chain:
            if current_var is None:
                return None
            if op_kind == OpKind.ATTR:
                current_var = self._dispatch_attr_access(current_var, name, args)
            else:  # OpKind.CALL
                current_var = self._dispatch_method_call(current_var, name, args)

        return current_var

    def try_lower_assignment(self, node: ast.Assign, rhs: ast.AST) -> ast.AST | None:
        """Try to lower assignment via registered type classes."""
        result = TypeFactory.try_lower_assignment(node, rhs, self.call_globals)
        if result is not None:
            stmts, var_type = result
            var_name = node.targets[0].id
            var = LocalVariableSource(var_type, var_name)
            self.variable_factory.add_variable(var)
            return stmts
        return None

    def try_attr_lowerers(self, node: ast.AST) -> ast.AST | None:
        ctx = CompilationContext.current()
        for transformer in ctx.attr_lowerers:
            try:
                result = transformer(node, self.call_globals, self.variable_factory)
                if result is not None:
                    return result
            except ValueError:
                pass  # Transformer couldn't handle this node
        return None

    def _is_keyed_container(self, var) -> bool:
        """Check if variable is a keyed container (e.g., SignalSet)."""
        return hasattr(var, "key_to_child_name") and hasattr(var, "create_dynamic_access")

    def _resolve_source_variable(self, node: ast.Assign) -> VariableSource | None:
        rhs = node.value

        # Case: x = container['key'] - get element from keyed container
        if isinstance(rhs, ast.Subscript) and isinstance(rhs.value, ast.Name):
            container = self.variable_factory.from_name(rhs.value.id)
            if self._is_keyed_container(container):
                return self.variable_factory.from_ast(visitor=self, ast_node=rhs, statements=[])
            return None

        # Case: x = some_var - simple variable reference
        if isinstance(rhs, ast.Name):
            return self.variable_factory.from_name(rhs.id)

        return None

    def create_assignment_variable(self, node: ast.Assign, rhs: ast.AST):
        """
        Handle assignment to a managed variable.

        Returns transformed AST or None if not a managed assignment.
        """
        target_name = node.targets[0].id
        ctx = CompilationContext.current()

        # 1. Try registered assignment handlers
        for handler in ctx.assignment_handlers:
            result = handler(self, node, rhs)
            if result is not None:
                return result

        # 2. Try resolving RHS as a call/attr chain (a.method(), a['key'].field, etc.)
        chain_result = self.handle_call_chain(rhs)
        if chain_result is not None:
            existing_var = self.variable_factory.from_name(target_name)
            if existing_var is not None:
                return AST.assignment(existing_var.get(), chain_result.get())
            self.variable_factory.add_variable(LocalVariableSource(chain_result.type, target_name))
            return AST.assignment(target_name, chain_result.get())

        # 3. Try resolving source variable (simple name or SignalSet subscript)
        src_var = self._resolve_source_variable(node)

        if src_var is None:
            # 4. If target exists and RHS is transformable (e.g., enum attr), handle it
            existing_var = self.variable_factory.from_name(target_name)
            if existing_var is not None:
                transformed_rhs = self.try_attr_lowerers(node.value)
                if transformed_rhs is not None:
                    return AST.assignment(existing_var.get(), transformed_rhs)
            return None

        # 5. Generate assignment for resolved source variable
        is_keyed_container = self._is_keyed_container(src_var)
        src_value = src_var if is_keyed_container else src_var.get()

        existing_var = self.variable_factory.from_name(target_name)
        if existing_var is not None:
            # Assigning to existing variable
            return AST.assignment(existing_var.get(), src_value)

        # Creating new variable
        if src_var.is_opaque_pointer():
            # Opaque pointers: create alias (both names point to same data)
            self.variable_factory.add_alias(target_name, src_var)
            if is_keyed_container:
                return ast.Pass()
            return AST.assignment(target_name, src_value)

        # Value types: create new local variable (copy semantics)
        self.variable_factory.add_variable(LocalVariableSource(src_var.type, target_name))
        return AST.assignment(target_name, src_value)
