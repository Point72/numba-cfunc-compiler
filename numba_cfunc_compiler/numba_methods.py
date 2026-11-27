"""Numba-callable helpers for compiled nodes."""

from __future__ import annotations

import ast
import builtins
import hashlib
import inspect
import json
import textwrap
import types
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from numba import types as numba_types
from numba.core.compiler_lock import global_compiler_lock
from numba.core.registry import cpu_target
from numba.extending import register_jitable

from numba_cfunc_compiler.core.converter import NumbaMethodASTConverter
from numba_cfunc_compiler.core.variable_factory import VariableFactory
from numba_cfunc_compiler.frozen_globals import _constant_text, find_frozen_globals

__all__ = ["numba_method"]

_F = TypeVar("_F", bound=Callable)
_OPTIONS = "__numba_cfunc_method_options__"


@dataclass(frozen=True)
class _MethodOptions:
    force_inline: bool


def _method_options(value: Any) -> _MethodOptions | None:
    options = getattr(value, _OPTIONS, None)
    return options if inspect.isfunction(value) and isinstance(options, _MethodOptions) else None


def _is_numba_callable(value: Any) -> bool:
    """Preserve functions with existing Numba typing, including overloads."""
    if not callable(value):
        return False
    try:
        return isinstance(cpu_target.typing_context.resolve_value_type(value), numba_types.Callable)
    except ValueError:
        return False


def numba_method(func: _F | None = None, *, force_inline: bool = True) -> _F | Callable[[_F], _F]:
    """Optionally configure a Python helper compiled in the calling node's context."""
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
        setattr(method, _OPTIONS, _MethodOptions(force_inline))
        return method

    return decorate if func is None else decorate(func)


def _function_ast(func: Callable) -> ast.FunctionDef:
    try:
        lines, first_line = inspect.getsourcelines(func)
        source = textwrap.dedent("".join(lines))
    except (OSError, TypeError) as exc:
        raise TypeError(f"Unable to inspect function '{func.__qualname__}'") from exc
    functions = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1:
        raise TypeError(f"Unable to identify function '{func.__qualname__}' in its source")
    return ast.increment_lineno(functions[0], first_line - 1)


def _validate_helper(func: Callable, node: ast.FunctionDef) -> dict[str, Any]:
    for child in ast.walk(node):
        if child is not node and isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            raise TypeError(f"Compiled helper '{func.__qualname__}' cannot contain nested functions or classes")
        if isinstance(child, (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            raise TypeError(f"Compiled helper '{func.__qualname__}' cannot contain nested scopes")
        if isinstance(child, ast.AnnAssign) and isinstance(child.annotation, ast.Subscript):
            annotation = child.annotation.value
            if isinstance(annotation, ast.Name) and annotation.id == "State":
                raise TypeError(f"Compiled helper '{func.__qualname__}' cannot declare State")
        if isinstance(child, (ast.Yield, ast.YieldFrom, ast.Await)):
            raise TypeError(f"Compiled helper '{func.__qualname__}' cannot be a generator or coroutine")
        if isinstance(child, (ast.Global, ast.Nonlocal, ast.Import, ast.ImportFrom)):
            raise TypeError(f"Compiled helper '{func.__qualname__}' cannot contain {type(child).__name__} at line {child.lineno}")

    namespace = dict(func.__globals__)
    namespace.update((name, cell.cell_contents) for name, cell in zip(func.__code__.co_freevars, func.__closure__ or ()))
    local = _local_names(node)
    body = ast.Module(body=node.body, type_ignores=[])
    loads = {child.id for child in ast.walk(body) if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)}
    builtin_namespace = getattr(func, "__builtins__", vars(builtins))
    for name in sorted(loads - local):
        if name not in namespace:
            if name in builtin_namespace:
                continue
            raise NameError(f"Compiled helper '{func.__qualname__}' has unbound name '{name}'")
        value = namespace[name]
        if (
            not inspect.isfunction(value)
            and not isinstance(value, (types.ModuleType, type))
            and _constant_text(value) is None
            and not _is_numba_callable(value)
        ):
            raise TypeError(f"Compiled helper '{func.__qualname__}' captures '{name}' ({type(value).__name__}); pass it as a helper argument")
    return namespace


def _local_names(tree: ast.AST) -> set[str]:
    """Names Python binds in this function, which cannot denote global helpers."""
    function = tree if isinstance(tree, ast.FunctionDef) else next((node for node in tree.body if isinstance(node, ast.FunctionDef)), None)
    if function is None:
        return set()
    names = {arg.arg for arg in (*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs)}
    names.update(node.id for node in ast.walk(function) if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load))
    names.update(node.name for node in ast.walk(function) if isinstance(node, ast.ExceptHandler) and node.name)
    return names


@dataclass
class _Definition:
    helper: Callable
    node: ast.FunctionDef
    namespace: dict[str, Any]
    options: _MethodOptions


class NumbaMethodManager:
    """Lower each helper call with the active node's AST handlers."""

    def __init__(self, variable_factory: VariableFactory):
        # Numba installs some library overloads lazily. Load them before deciding
        # which Python functions need our AST lowering, even on the first compile.
        with global_compiler_lock:
            cpu_target.target_context.refresh()
            cpu_target.typing_context.refresh()
        self.variable_factory = variable_factory
        self.definitions: list[_Definition] = []
        self.hashes: dict[str, str] = {}
        self.lowered_names: dict[Callable, str] = {}
        self.active: set[Callable] = set()
        self.next_id = 0
        self.unkeyable: set[str] = set()

    def rewrite_call(self, call: ast.Call, converter) -> ast.Call | None:
        function = call.func
        attributes = []
        while isinstance(function, ast.Attribute):
            attributes.append(function.attr)
            function = function.value
        if (
            not isinstance(function, ast.Name)
            or function.id in _local_names(converter.tree)
            or converter.variable_factory.from_variable_name(function.id) is not None
        ):
            return None
        namespace = converter.host_globals
        helper = namespace.get(function.id)
        if attributes and not isinstance(helper, types.ModuleType):
            return None
        try:
            for attribute in reversed(attributes):
                helper = getattr(helper, attribute)
        except AttributeError:
            return None
        if not inspect.isfunction(helper):
            return None
        if _method_options(helper) is None and _is_numba_callable(helper):
            return None
        if helper in self.active:
            raise TypeError(f"Recursive compiled helper call involving '{helper.__qualname__}' is not supported")
        self._validate_call(call, helper)
        name = self.lowered_names.get(helper)
        if name is None:
            self.active.add(helper)
            try:
                name = self._lower(helper)
            finally:
                self.active.remove(helper)
            self.lowered_names[helper] = name
        call.func = ast.Name(id=name, ctx=ast.Load())
        return call

    @staticmethod
    def _validate_call(call: ast.Call, helper: Callable) -> None:
        signature = inspect.signature(helper)
        keyword_only = [param.name for param in signature.parameters.values() if param.kind is inspect.Parameter.KEYWORD_ONLY]
        if keyword_only:
            names = ", ".join(keyword_only)
            raise TypeError(
                f"Compiled helper '{helper.__qualname__}' cannot declare keyword-only parameters ({names}); "
                "Numba cannot call them from compiled code; use regular parameters instead"
            )
        if any(param.default is not inspect.Parameter.empty for param in signature.parameters.values()):
            raise TypeError(f"Compiled helper '{helper.__qualname__}' cannot declare default arguments; pass every argument explicitly")
        if any(isinstance(arg, ast.Starred) for arg in call.args) or any(keyword.arg is None for keyword in call.keywords):
            raise TypeError(f"Compiled helper '{helper.__qualname__}' requires named variable arguments")
        try:
            signature.bind(*call.args, **{keyword.arg: keyword.value for keyword in call.keywords})
        except TypeError as exc:
            raise TypeError(f"Invalid call to compiled helper '{helper.__qualname__}': {exc}") from exc

    def _lower(self, helper: Callable) -> str:
        node = _function_ast(helper)
        namespace = _validate_helper(helper, node)
        name = f"__ncc_method_{self.next_id}"
        self.next_id += 1
        node.name = name
        node.decorator_list = []
        node.returns = None
        for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
            arg.annotation = None
        factory = VariableFactory()
        converter = NumbaMethodASTConverter(node, factory, call_globals=namespace, host_globals=namespace, method_manager=self)
        lowered = converter.visit(node)
        self.variable_factory.typed_struct_bindings.update(factory.typed_struct_bindings)
        self.variable_factory.typed_struct_layouts.update(factory.typed_struct_layouts)
        self.variable_factory.typed_input_bindings.update(factory.typed_input_bindings)
        self.variable_factory.typed_keyed_bindings.update(factory.typed_keyed_bindings)
        options = _method_options(helper) or _MethodOptions(force_inline=True)
        self.definitions.append(_Definition(helper, lowered, converter.call_globals, options))
        frozen = find_frozen_globals(lowered, converter.call_globals)
        self.unkeyable.update(f"{helper.__qualname__}.{path}" for path in frozen.unkeyable)
        payload = {
            "lowered": ast.dump(lowered, include_attributes=False),
            "force_inline": options.force_inline,
            "lowering_version": 3,
            "frozen_globals": frozen.constants,
        }
        self.hashes[name] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return name

    def build_bindings(self, compiler_globals: dict[str, Any]) -> dict[str, Callable]:
        bindings = {}
        namespaces = []
        for definition in self.definitions:
            namespace = dict(compiler_globals)
            namespace.update(definition.namespace)
            tree = ast.fix_missing_locations(ast.Module(body=[definition.node], type_ignores=[]))
            exec(compile(tree, definition.helper.__code__.co_filename, "exec"), namespace)  # noqa: S102 - lowered helper source
            name = definition.node.name
            inline = "always" if definition.options.force_inline else "never"
            bindings[name] = register_jitable(inline=inline, error_model="numpy")(namespace[name])
            namespaces.append(namespace)
        for namespace in namespaces:
            namespace.update(bindings)
        return bindings
