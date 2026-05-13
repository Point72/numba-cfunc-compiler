import ast
import inspect
from typing import (
    Any,
    Dict,
    Optional,
    Protocol,
    runtime_checkable,
)

from numba_cfunc_compiler.compilation_context import CompilationContext
from numba_cfunc_compiler.models import (
    InputAnalysis,
    OutputAnalysis,
    ParameterInfo,
    StateAnalysis,
    StateVariableInfo,
)
from numba_cfunc_compiler.numba_config import NumbaTypeRegistry
from numba_cfunc_compiler.type_factory import TypeFactory

__all__ = [
    "InputTypeHandler",
    "OutputTypeHandler",
    "FunctionAnalyzer",
]


@runtime_checkable
class InputTypeHandler(Protocol):
    """Protocol for input type handlers that parse signal-related input patterns."""

    def try_parse(self, param: inspect.Parameter, ann: Any) -> Optional[ParameterInfo]:
        """Try to parse the annotation. Returns ParameterInfo if handled, None otherwise."""
        ...

    def validate_value(self, param_name: str, value: Any, expected_type: Any) -> Any:
        """Validate and potentially transform the input value."""
        ...


@runtime_checkable
class OutputTypeHandler(Protocol):
    """Protocol for output type handlers that parse return type annotations."""

    def try_parse(self, return_annotation: Any, ast_tree: ast.AST) -> Optional[OutputAnalysis]:
        """Try to parse the return annotation. Returns OutputAnalysis if handled, None otherwise."""
        ...


class FunctionAnalyzer:
    """
    Analyzes numba_node function signatures and bodies.

    Responsible for:
    - Parsing input parameter annotations and validating values
    - Parsing state variable declarations from the function body
    - Parsing output type annotations

    All handler registries live in the active CompilationContext.
    """

    @classmethod
    def register_input_handler(cls, handler: InputTypeHandler) -> None:
        CompilationContext.current().input_handlers.append(handler)

    @classmethod
    def register_output_handler(cls, handler: OutputTypeHandler) -> None:
        CompilationContext.current().output_handlers.append(handler)

    def __init__(self):
        ctx = CompilationContext.current()
        self.input_handlers = ctx.input_handlers
        self.output_handlers = ctx.output_handlers

    @staticmethod
    def get_function_ast(func, decorator_name: str) -> ast.AST:
        import textwrap

        source = inspect.getsource(func)
        func_source = textwrap.dedent(source)

        lines = func_source.split("\n")
        if decorator_name not in lines[0].strip():
            raise ValueError(f"Expected {decorator_name}, got {lines[0].strip()}")
        func_source = "\n".join(lines[1:])

        tree = ast.parse(func_source)
        FunctionAnalyzer.validate_no_nested_scopes(tree, decorator_name)
        return tree

    @staticmethod
    def validate_no_nested_scopes(ast_tree: ast.AST, decorator_name: str) -> None:
        """
        Reject nested def / async def / class inside the decorated
        function body.
        """
        outer_fn: Optional[ast.AST] = None
        for node in ast.iter_child_nodes(ast_tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                outer_fn = node
                break
        if outer_fn is None:
            return

        kind_labels = {
            ast.FunctionDef: "def",
            ast.AsyncFunctionDef: "async def",
            ast.ClassDef: "class",
        }
        outer_name = getattr(outer_fn, "name", "<function>")
        for node in ast.walk(outer_fn):
            if node is outer_fn:
                continue
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                kind = kind_labels[type(node)]
                inner_name = getattr(node, "name", "<anonymous>")
                lineno = getattr(node, "lineno", "?")
                raise TypeError(
                    f"{decorator_name} function {outer_name!r} contains a nested "
                    f"'{kind} {inner_name}' at line {lineno}. "
                    f"{decorator_name} does not support nested function or class "
                    f"definitions — move {inner_name!r} to module scope."
                )

    def parse_input_annotation(self, sig: inspect.Signature, args_by_name: Dict[str, Any]) -> InputAnalysis:
        """Parse and validate input parameters.

        Returns an InputAnalysis with parameters stored by name.
        Use input_analysis.get_by_category(category) to filter by category.
        """
        result = InputAnalysis()

        if len(args_by_name) == 0:
            raise ValueError("has no input parameters. Numba nodes must have at least one input parameter.")

        for param_name, param in sig.parameters.items():
            if param_name not in args_by_name:
                continue

            value = args_by_name[param_name]
            ann = getattr(param, "annotation", None)

            if ann is None or ann == inspect.Parameter.empty:
                raise TypeError(f"Parameter '{param_name}' must have a type annotation.")

            # First try type classes for constant inputs
            type_result = TypeFactory.try_parse_input(param, ann)

            if type_result is not None:
                type_class, param_info = type_result
                validated_value = type_class.validate_input(param_name, value, param_info.expected_type)
            else:
                # Fall back to registered input handlers
                matched_handler: Optional[InputTypeHandler] = None
                param_info: Optional[ParameterInfo] = None

                for handler in self.input_handlers:
                    param_info = handler.try_parse(param, ann)
                    if param_info is not None:
                        matched_handler = handler
                        break

                if param_info is None or matched_handler is None:
                    raise TypeError(f"Unable to parse type annotation for parameter '{param_name}'")

                validated_value = matched_handler.validate_value(param_name, value, param_info.expected_type)

            # Store parameter with its info (category is set by the handler)
            result.parameters[param_name] = (validated_value, param_info)

        return result

    def parse_state_annotation(self, ast_tree: ast.AST, globalns: dict) -> StateAnalysis:
        state_vars: Dict[str, StateVariableInfo] = {}

        for node in ast.walk(ast_tree):
            if not isinstance(node, ast.AnnAssign) or not node.annotation:
                continue

            ann = node.annotation

            # Check for State[...] annotation
            if not (isinstance(ann, ast.Subscript) and isinstance(ann.value, ast.Name) and ann.value.id == "State"):
                if isinstance(ann, ast.Name) and ann.id == "State":
                    var_name = node.target.id if isinstance(node.target, ast.Name) else "unknown"
                    supported_names = NumbaTypeRegistry.get_supported_type_names()
                    raise TypeError(f"State variable '{var_name}' is missing type argument. Use State[{', '.join(supported_names.keys())}].")
                continue

            if not isinstance(node.target, ast.Name):
                raise TypeError("State annotations can only be applied to simple variable names.")

            var_name = node.target.id

            if var_name in state_vars:
                raise ValueError(f"State variable '{var_name}' is declared multiple times. Each state variable can only be declared once.")

            if not node.value:
                raise TypeError(f"State variable '{var_name}' must have an explicit initial value")

            # Try type classes in order until one parses successfully
            state_var_info = TypeFactory.try_parse_state(node, var_name, globalns)

            if state_var_info is None:
                raise TypeError(f"Unsupported State type for '{var_name}'. ")

            state_vars[var_name] = state_var_info

        return StateAnalysis(state_vars)

    def parse_output_annotation(self, ast_tree: ast.AST, sig: inspect.Signature) -> OutputAnalysis:
        if not sig.return_annotation or sig.return_annotation == inspect.Signature.empty:
            raise TypeError("Missing output annotation: function must specify a return type annotation.")

        return_annotation = sig.return_annotation

        # Try each handler in order until one matches
        for handler in self.output_handlers:
            result = handler.try_parse(return_annotation, ast_tree)
            if result is not None:
                return result

        raise TypeError("Output has unsupported type. No output handler could parse the return annotation.")
