import ast
import inspect
from collections.abc import Callable
from typing import Any

from numba_cfunc_compiler.core.analysis import FunctionAnalyzer
from numba_cfunc_compiler.core.names import INTERNAL_PREFIX
from numba_cfunc_compiler.core.variable_factory import VariableFactory
from numba_cfunc_compiler.extension.callback_components import ComponentInputs, ComponentRegistry


class NumbaFunctionInfo:
    """
    Holds analysis information for a decorated function.

    This class analyzes function signatures, inputs, outputs, and state variables,
    and creates the variable factory for AST transformation.
    """

    def __init__(
        self,
        func,
        *args,
        extract_python_type_fn: Callable[[Any], type],
        decorator_name: str,
        func_globals: dict | None = None,
        call_globals: dict | None = None,
        signature: inspect.Signature | None = None,
        **kwargs,
    ):
        # Handle both callable and ast.FunctionDef
        if isinstance(func, ast.FunctionDef):
            self.tree = func
            self.name = func.name
            if signature is None:
                raise ValueError("signature is required when func is an ast.FunctionDef")
            FunctionAnalyzer.validate_no_nested_scopes(func, decorator_name)
            self.sig = signature
            self.func_globals = func_globals or {}
        else:
            self.tree = FunctionAnalyzer.get_function_ast(func, decorator_name)
            self.name = func.__name__
            self.sig = signature if signature is not None else inspect.signature(func)
            self.func_globals = func_globals if func_globals is not None else getattr(func, "__globals__", {})

        self.func = func
        self.analyzer = FunctionAnalyzer()
        self.extract_python_type_fn = extract_python_type_fn
        self.call_globals = dict(call_globals or {})

        # Use try/except to re-throw with additional info (numba_node function name)
        try:
            self._initialize_and_validate(*args, **kwargs)
        except Exception as e:
            error_msg = f"Error in numba_node function '{self.name}': {e!s}"
            raise type(e)(error_msg) from e

    def _initialize_and_validate(self, *args, **kwargs):
        import inspect as _inspect

        try:
            bound = self.sig.bind_partial(*args, **kwargs)
        except TypeError as e:
            raise ValueError(f"expects {self.sig.parameters} arguments, got {args} positional and {kwargs} keyword arguments. {e}")

        # Fill in defaults for any missing values
        for pname, param in self.sig.parameters.items():
            if pname not in bound.arguments and param.default is not _inspect._empty:
                bound.arguments[pname] = param.default

        # Ensure all required params provided
        missing = [pname for pname, param in self.sig.parameters.items() if pname not in bound.arguments and param.default is _inspect._empty]
        if missing:
            raise ValueError(f"expects {len(self.sig.parameters)} arguments, got {len(args) + len(kwargs)}")

        self.input_analysis = self.analyzer.parse_input_annotation(self.sig, dict(bound.arguments))
        self.output_analysis = self.analyzer.parse_output_annotation(self.tree, self.sig)
        self.state_analysis = self.analyzer.parse_state_annotation(self.tree, self.func_globals)
        self.variable_factory, self.component_metadata = self._create_variable_factory()

    def _create_variable_factory(self) -> tuple[VariableFactory, dict[str, Any]]:
        from numba_cfunc_compiler.core.variable_access import ConstantAccess

        variable_factory = VariableFactory()
        component_inputs = ComponentInputs(
            input_analysis=self.input_analysis,
            state_analysis=self.state_analysis,
            output_analysis=self.output_analysis,
            extract_python_type_fn=self.extract_python_type_fn,
        )
        metadata = {}
        for component in ComponentRegistry.get_ordered():
            metadata.update(component.create_variables(component_inputs, variable_factory))
        # Constant containers live from START to STOP, so give each instance a
        # hidden pointer slot after its declared state slots.
        state_values = list(metadata.get("state_values", ()))
        constant_indices = []
        for variable in variable_factory.get_accesses(ConstantAccess):
            plan = variable.prepare_plan(self.call_globals, len(state_values))
            constant_indices.extend(range(len(state_values), len(state_values) + plan.state_slots))
            state_values.extend((0,) * plan.state_slots)
        metadata["state_values"] = tuple(state_values)
        metadata["constant_container_indices"] = tuple(constant_indices)
        return variable_factory, metadata


def validate_user_identifiers(tree, signature, declared_names, start_body, stop_body):
    """Keep the compiler's generated Python names out of callback source."""
    names = set(signature.parameters) | set(declared_names)
    names.update(param.name for param in ComponentRegistry.build_cfunc_params())
    for root in (tree, *start_body, *stop_body):
        for node in ast.walk(root):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.arg):
                names.add(node.arg)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, (ast.Global, ast.Nonlocal)):
                names.update(node.names)
            elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and node.name:
                names.add(node.name)
            elif isinstance(node, ast.MatchMapping) and node.rest:
                names.add(node.rest)
    reserved = sorted(name for name in names if name.startswith(INTERNAL_PREFIX))
    if reserved:
        raise ValueError(f"Identifier '{reserved[0]}' uses the reserved compiler prefix '{INTERNAL_PREFIX}'")
