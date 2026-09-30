import ast
import inspect
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Protocol,
    runtime_checkable,
)

from numba_cfunc_compiler.core.context import CompilationContext
from numba_cfunc_compiler.types.binding import TypeBinding
from numba_cfunc_compiler.types.registry import NumbaTypeRegistry

__all__ = [
    "FunctionAnalyzer",
    "InputAnalysis",
    "InputCategory",
    "InputTypeHandler",
    "OutputAnalysis",
    "OutputTypeHandler",
    "ParameterInfo",
    "StateAnalysis",
    "StateVariableInfo",
]


class InputCategory(Enum):
    """Input categories owned by the compiler; hosts may define their own enums."""

    CONSTANT = auto()


@dataclass(frozen=True)
class ParameterInfo:
    """Info about a parsed input parameter.

    Args:
        expected_type: The expected Python type for this parameter.
        category: An :class:`Enum` member selecting a host input component.
            Defaults to :attr:`InputCategory.CONSTANT`. Hosts may define their
            own enum members for signal, basket, or other input categories.
    """

    expected_type: Any
    category: Enum = InputCategory.CONSTANT

    def __post_init__(self):
        if not isinstance(self.category, Enum):
            raise TypeError("Input category must be an Enum member")


@dataclass(frozen=True)
class StateVariableInfo:
    """State initializer and resolved host binding for one declaration."""

    name: str
    initial_value: Any
    binding: TypeBinding


@dataclass
class InputAnalysis:
    """Parsed input parameters grouped by category.

    Args:
        parameters: Dict mapping param name to (validated_value, ParameterInfo).
    """

    parameters: dict[str, tuple] = field(default_factory=dict)  # name -> (value, ParameterInfo)

    def get_by_category(self, category: Enum) -> dict[str, Any]:
        """Get parameters by category. Returns {name: value}."""
        if not isinstance(category, Enum):
            raise TypeError("Input category must be an Enum member")
        return {name: value for name, (value, info) in self.parameters.items() if info.category == category}

    def get_params_by_category(self, category: Enum) -> dict[str, tuple]:
        """Get full parameter info by category. Returns {name: (value, ParameterInfo)}."""
        if not isinstance(category, Enum):
            raise TypeError("Input category must be an Enum member")
        return {name: (value, info) for name, (value, info) in self.parameters.items() if info.category == category}


@dataclass
class StateAnalysis:
    """Resolved state declarations for one compiled function."""

    state_vars: dict[str, StateVariableInfo] = field(default_factory=dict)

    def sorted_by_size(self) -> list[StateVariableInfo]:
        """Sort state variables by size (largest first), then by name."""
        return sorted(
            self.state_vars.values(),
            key=lambda info: (-info.binding.host_size, info.name),
        )


@dataclass(frozen=True)
class OutputAnalysis:
    """Outputs in native slot order; ``None`` names the single anonymous output."""

    outputs: dict[str | None, Any]

    def __post_init__(self):
        if not self.outputs:
            raise ValueError("At least one output is required")
        if None in self.outputs and len(self.outputs) != 1:
            raise ValueError("An anonymous output must be the only output")
        if any(name is not None and not isinstance(name, str) for name in self.outputs):
            raise TypeError("Output names must be strings or None")


@runtime_checkable
class InputTypeHandler(Protocol):
    """Protocol for input type handlers that parse signal-related input patterns."""

    def try_parse(self, param: inspect.Parameter, ann: Any) -> ParameterInfo | None:
        """Try to parse the annotation. Returns ParameterInfo if handled, None otherwise."""
        ...

    def validate_value(self, param_name: str, value: Any, expected_type: Any) -> Any:
        """Validate and potentially transform the input value."""
        ...


@runtime_checkable
class OutputTypeHandler(Protocol):
    """Protocol for output type handlers that parse return type annotations."""

    def try_parse(self, return_annotation: Any, ast_tree: ast.AST) -> OutputAnalysis | None:
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
        outer_fn: ast.AST | None = ast_tree if isinstance(ast_tree, (ast.FunctionDef, ast.AsyncFunctionDef)) else None
        if outer_fn is None:
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

    def parse_input_annotation(self, sig: inspect.Signature, args_by_name: dict[str, Any]) -> InputAnalysis:
        """Parse and validate input parameters.

        Returns an InputAnalysis with parameters stored by name.
        Use input_analysis.get_by_category(category) to filter by category.
        """
        from numba_cfunc_compiler.types.factory import HostTypeFactory

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
            type_result = HostTypeFactory.try_parse_input(param, ann)

            if type_result is not None:
                type_class, param_info = type_result
                validated_value = type_class.validate_input(param_name, value, param_info.expected_type)
            else:
                # Fall back to registered input handlers
                matched_handler: InputTypeHandler | None = None
                param_info: ParameterInfo | None = None

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
        """Parse ``State[T]`` declarations and resolve their host type bindings."""

        from numba_cfunc_compiler.types.factory import HostTypeFactory

        state_vars: dict[str, StateVariableInfo] = {}

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
            parsed_state = HostTypeFactory.try_parse_state(node, var_name, globalns)

            if parsed_state is None:
                raise TypeError(f"Unsupported State type for '{var_name}'. ")

            initial_value, state_type = parsed_state
            state_vars[var_name] = StateVariableInfo(var_name, initial_value, HostTypeFactory.resolve(state_type))

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
