"""Numba-callable methods used by compiled node functions."""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import types
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from numba.extending import register_jitable

from numba_cfunc_compiler.function_analyzer import FunctionAnalyzer
from numba_cfunc_compiler.state_ast import is_state_type_annotation

__all__ = ["numba_method"]


_NUMBA_METHOD_OPTIONS_ATTRIBUTE = "__numba_cfunc_method_options__"
_F = TypeVar("_F", bound=Callable)


@dataclass(frozen=True)
class _NumbaMethodOptions:
    force_inline: bool


def _method_options(value: Any) -> _NumbaMethodOptions | None:
    options = getattr(value, _NUMBA_METHOD_OPTIONS_ATTRIBUTE, None)
    return options if inspect.isfunction(value) and isinstance(options, _NumbaMethodOptions) else None


def numba_method(func: _F | None = None, *, force_inline: bool = True) -> _F | Callable[[_F], _F]:
    """Compile a Python helper in its calling node's context.

    Helpers are compiled in the calling ``numba_node`` context. They do not
    own node state or outputs: pass values explicitly, return scalar state
    updates explicitly, and mutate only explicitly passed mutable containers.
    Compiled helpers cannot capture runtime globals or use default arguments.
    Numba inlining is forced by default; ``force_inline=False`` disables that
    request and leaves later LLVM optimization free to inline the call.
    """
    if not isinstance(force_inline, bool):
        raise TypeError("@numba_method force_inline must be a bool")

    def decorate(method: _F) -> _F:
        if not inspect.isfunction(method):
            raise TypeError("@numba_method can only decorate a Python function")
        existing = _method_options(method)
        if existing is not None:
            if existing.force_inline != force_inline:
                raise TypeError("@numba_method cannot change force_inline on an already decorated function")
            return method

        options = _NumbaMethodOptions(force_inline=force_inline)
        setattr(method, _NUMBA_METHOD_OPTIONS_ATTRIBUTE, options)
        registered = register_jitable(inline="always" if force_inline else "never")(method)
        setattr(registered, _NUMBA_METHOD_OPTIONS_ATTRIBUTE, options)
        return registered

    return decorate if func is None else decorate(func)


def _is_numba_method(value: Any) -> bool:
    return _method_options(value) is not None


@dataclass(frozen=True)
class _LoweredHelperDefinition:
    helper: Callable
    function_node: ast.FunctionDef
    namespace: dict[str, Any]
    options: _NumbaMethodOptions


def _validate_helper_ast(func: Callable, function_node: ast.FunctionDef) -> None:
    for node in ast.walk(function_node):
        if node is not function_node and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            raise TypeError(f"@numba_method helper '{func.__qualname__}' cannot contain nested functions or classes")
        if isinstance(node, (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            raise TypeError(f"@numba_method helper '{func.__qualname__}' cannot contain nested scopes")
        if isinstance(node, ast.AnnAssign) and is_state_type_annotation(node.annotation):
            raise TypeError(
                f"@numba_method helper '{func.__qualname__}' cannot declare State; declare state in the numba_node and pass it explicitly"
            )
        if isinstance(node, (ast.Yield, ast.YieldFrom, ast.Await)):
            raise TypeError(f"@numba_method helper '{func.__qualname__}' cannot be a generator or coroutine")


def _allowed_helper_symbols(func: Callable, function_node: ast.FunctionDef, closure: inspect.ClosureVars) -> set[str]:
    """Reject captured data and return symbolic globals that lowering must consume."""
    # Python 3.11 may report attribute names as unbound; only AST names can be missing globals.
    loaded_names = {node.id for node in ast.walk(function_node) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
    missing_names = closure.unbound & loaded_names
    if missing_names:
        missing = ", ".join(sorted(missing_names))
        raise NameError(f"@numba_method helper '{func.__qualname__}' has unbound names: {missing}")

    symbols = set()
    for name, value in {**closure.globals, **closure.nonlocals}.items():
        if _is_numba_method(value) or isinstance(value, (types.ModuleType, type)):
            symbols.add(name)
        else:
            raise TypeError(f"@numba_method helper '{func.__qualname__}' captures '{name}'; pass it as a helper argument")
    return symbols


def _check_symbols_lowered(func: Callable, lowered: ast.FunctionDef, symbols: set[str]) -> None:
    remaining = symbols & {node.id for node in ast.walk(lowered) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
    if remaining:
        names = ", ".join(sorted(remaining))
        raise TypeError(f"@numba_method helper '{func.__qualname__}' has unresolved global symbols: {names}")


class NumbaMethodManager:
    """Track method calls and compile copies using types from each call site."""

    def __init__(self, tree: ast.AST, globalns: dict[str, Any]) -> None:
        self._namespaces: dict[ast.AST, dict[str, Any]] = {tree: dict(globalns)}
        self.definitions: list[_LoweredHelperDefinition] = []
        self.lowered_helper_hashes: dict[str, str] = {}
        self.active_helpers: set[Callable] = set()
        self.next_id = 0

    def rewrite_call(self, node: ast.Call, converter: Any) -> ast.Call | None:
        namespace = self._namespaces[converter.tree]
        function = node.func
        attributes = []
        while isinstance(function, ast.Attribute):
            attributes.append(function.attr)
            function = function.value
        if not isinstance(function, ast.Name):
            return None
        if converter.variable_factory.from_name(function.id) is not None:
            return None
        helper = namespace.get(function.id)
        if attributes and not isinstance(helper, types.ModuleType):
            return None
        try:
            for attribute in reversed(attributes):
                helper = getattr(helper, attribute)
        except AttributeError:
            return None
        if not _is_numba_method(helper):
            return None

        if helper in self.active_helpers:
            raise TypeError(f"Recursive @numba_method helper call involving '{helper.__qualname__}' is not supported")
        self.active_helpers.add(helper)
        try:
            return self._lower_call(node, converter, helper)
        finally:
            self.active_helpers.remove(helper)

    def _lower_call(self, node: ast.Call, converter: Any, helper: Callable) -> ast.Call:

        from numba_cfunc_compiler.models import UnknownType
        from numba_cfunc_compiler.numba_ast_converter import NumbaMethodASTConverter
        from numba_cfunc_compiler.variable_factory import LocalVariableSource, VariableFactory

        signature = inspect.signature(helper)
        if any(param.default is not inspect.Parameter.empty for param in signature.parameters.values()):
            raise TypeError(f"@numba_method helper '{helper.__qualname__}' cannot declare default arguments; pass every argument explicitly")
        if any(isinstance(arg, ast.Starred) for arg in node.args) or any(keyword.arg is None for keyword in node.keywords):
            raise TypeError(f"@numba_method helper '{helper.__qualname__}' requires named variable arguments")
        try:
            bound = signature.bind(*node.args, **{keyword.arg: keyword.value for keyword in node.keywords})
        except TypeError as exc:
            raise TypeError(f"Invalid call to @numba_method helper '{helper.__qualname__}': {exc}") from exc

        factory = VariableFactory()
        for name, argument in bound.arguments.items():
            if not isinstance(argument, ast.Name):
                raise TypeError(f"@numba_method helper '{helper.__qualname__}' argument '{name}' must be a named variable")
            source = converter.variable_factory.from_name(argument.id)
            if source is None:
                raise TypeError(f"@numba_method helper '{helper.__qualname__}' argument '{name}' must be a tracked variable")
            variable_type = source.type
            if isinstance(variable_type, UnknownType):
                raise TypeError(f"@numba_method helper '{helper.__qualname__}' argument '{name}' has no known compiler type")
            factory.add_variable(LocalVariableSource(variable_type, name))

        name = f"__ncc_method_{self.next_id}"
        self.next_id += 1
        function_node = FunctionAnalyzer.parse_function_source(helper)
        _validate_helper_ast(helper, function_node)
        closure = inspect.getclosurevars(helper)
        symbols = _allowed_helper_symbols(helper, function_node, closure)
        function_node.name = name
        function_node.decorator_list = []
        function_node.returns = None
        for arg in (*function_node.args.posonlyargs, *function_node.args.args, *function_node.args.kwonlyargs):
            arg.annotation = None

        namespace = dict(helper.__globals__)
        namespace.update(closure.nonlocals)
        FunctionAnalyzer.validate_no_global_shadowing(function_node, namespace)
        self._namespaces[function_node] = namespace
        lowered = NumbaMethodASTConverter(
            function_node,
            factory,
            call_globals=namespace,
            method_manager=self,
        ).visit(function_node)
        _check_symbols_lowered(helper, lowered, symbols)
        options = _method_options(helper)
        assert options is not None  # resolved calls only target decorated methods
        self.definitions.append(_LoweredHelperDefinition(helper=helper, function_node=lowered, namespace=namespace, options=options))

        from numba_cfunc_compiler.config import get_numba_config

        config = get_numba_config()
        payload = {
            "lowered": ast.dump(lowered, include_attributes=False),
            "enum_width": config.enum_bit_width,
            "enumset_width": config.enumset_bit_width,
            "force_inline": options.force_inline,
        }
        self.lowered_helper_hashes[name] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        result = copy.deepcopy(node)
        result.func = ast.Name(id=name, ctx=ast.Load())
        return result

    def build_bindings(self, compiler_globals: dict[str, Any]) -> dict[str, Callable]:
        bindings = {}
        namespaces = []
        for definition in self.definitions:
            namespace = dict(definition.namespace)
            namespace.update(compiler_globals)
            tree = ast.fix_missing_locations(ast.Module(body=[definition.function_node], type_ignores=[]))
            exec(compile(tree, definition.helper.__code__.co_filename, "exec"), namespace)  # noqa: S102 - lowered helper source
            name = definition.function_node.name
            inline = "always" if definition.options.force_inline else "never"
            bindings[name] = register_jitable(inline=inline)(namespace[name])
            namespaces.append(namespace)
        for namespace in namespaces:
            namespace.update(bindings)
        return bindings
