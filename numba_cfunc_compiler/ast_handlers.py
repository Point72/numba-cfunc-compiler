import ast
import functools
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

from numba_cfunc_compiler.compilation_context import CompilationContext

__all__ = [
    "ASTHandlerRegistry",
    "HandlerPhase",
    "HandlerResult",
    "with_handlers",
    "ast_handler",
]


class HandlerPhase(Enum):
    """When a handler runs relative to default processing."""

    PRE = auto()  # Before default processing
    POST = auto()  # After default processing


@dataclass
class HandlerResult:
    """
    Result from a handler that may include side effects.

    Side effects are AST statements inserted before the current statement.
    """

    node: ast.AST | None
    side_effects: list[ast.AST] = field(default_factory=list)


@dataclass
class _HandlerInfo:
    """Internal: information about a registered handler."""

    func: Callable
    phase: HandlerPhase
    priority: int = 0


def _normalize_result(result: Any) -> tuple[Any, list[ast.AST]]:
    """Normalize handler return to (node, side_effects) tuple."""
    if result is None:
        return None, []
    if isinstance(result, HandlerResult):
        return result.node, result.side_effects
    return result, []


def _combine_with_side_effects(result: Any, side_effects: list[ast.AST]) -> Any:
    if not side_effects:
        return result
    if isinstance(result, list):
        return side_effects + result
    return side_effects + [result]


class ASTHandlerRegistry:
    """Central registry for AST transformation handlers.

    All mutable state lives in the active CompilationContext.
    """

    @classmethod
    def register(
        cls,
        node_type: str,
        handler: Callable,
        phase: HandlerPhase = HandlerPhase.PRE,
        priority: int = 0,
    ):
        handlers = CompilationContext.current().ast_handlers
        if node_type not in handlers:
            handlers[node_type] = {HandlerPhase.PRE: [], HandlerPhase.POST: []}

        info = _HandlerInfo(func=handler, phase=phase, priority=priority)
        handlers[node_type][phase].append(info)
        handlers[node_type][phase].sort(key=lambda h: h.priority)

    @classmethod
    def get_handlers(cls, node_type: str, phase: HandlerPhase) -> list[Callable]:
        handlers = CompilationContext.current().ast_handlers
        if node_type not in handlers:
            return []
        return [h.func for h in handlers[node_type].get(phase, [])]

    @classmethod
    def run_pre_handlers(
        cls,
        node_type: str,
        converter: Any,
        node: ast.AST,
    ) -> tuple[Any, list[ast.AST]]:
        """Run pre-handlers. Returns first non-None result (short-circuits)."""
        for handler in cls.get_handlers(node_type, HandlerPhase.PRE):
            result, side_effects = _normalize_result(handler(converter, node))
            if result is not None:
                return result, side_effects
        return None, []

    @classmethod
    def run_post_handlers(
        cls,
        node_type: str,
        converter: Any,
        node: ast.AST,
        result: Any,
        accumulated_side_effects: list[ast.AST],
    ) -> tuple[Any, list[ast.AST]]:
        """Run post-handlers. Each receives output of the previous."""
        side_effects = list(accumulated_side_effects)
        for handler in cls.get_handlers(node_type, HandlerPhase.POST):
            result, new_side_effects = _normalize_result(handler(converter, node, result))
            side_effects.extend(new_side_effects)
        return result, side_effects

    @classmethod
    def clear(cls, node_type: str | None = None):
        handlers = CompilationContext.current().ast_handlers
        if node_type is None:
            handlers.clear()
        elif node_type in handlers:
            handlers[node_type] = {HandlerPhase.PRE: [], HandlerPhase.POST: []}


def with_handlers(node_type: str):
    """Decorator for visit_* methods that adds pre/post handler support."""

    def decorator(method: Callable) -> Callable:
        @functools.wraps(method)
        def wrapper(self, node: ast.AST) -> Any:
            # Pre-handlers (can short-circuit)
            pre_result, pre_side_effects = ASTHandlerRegistry.run_pre_handlers(node_type, self, node)
            if pre_result is not None:
                return _combine_with_side_effects(pre_result, pre_side_effects)

            # Default processing
            result, method_side_effects = _normalize_result(method(self, node))
            accumulated = pre_side_effects + method_side_effects

            # Post-handlers
            final_result, final_side_effects = ASTHandlerRegistry.run_post_handlers(node_type, self, node, result, accumulated)
            return _combine_with_side_effects(final_result, final_side_effects)

        return wrapper

    return decorator


def ast_handler(
    node_type: str,
    pre: bool = False,
    post: bool = False,
    priority: int = 0,
):
    """
    Decorator to register a handler for an AST node type.

    Args:
        node_type: AST node type name (must match @with_handlers)
        pre: Register as pre-handler (runs before default logic)
        post: Register as post-handler (runs after default logic)
        priority: Lower values run first

    Pre-handler: (converter, node) -> ast.AST | HandlerResult | None
        Return non-None to short-circuit, None to continue

    Post-handler: (converter, node, result) -> Any | HandlerResult
        Receives and can transform the result

    Example:
        @ast_handler('Call', pre=True)
        def handle_special_call(converter, node):
            if is_special(node):
                return transformed_node
            return None
    """
    if not pre and not post:
        raise ValueError("Must specify pre=True or post=True")
    if pre and post:
        raise ValueError("Cannot specify both pre=True and post=True")

    phase = HandlerPhase.PRE if pre else HandlerPhase.POST

    def decorator(func: Callable) -> Callable:
        ASTHandlerRegistry.register(node_type, func, phase, priority)
        return func

    return decorator
