from __future__ import annotations

import contextvars
from typing import Any, Dict, List

__all__ = ["CompilationContext"]

_current_context: contextvars.ContextVar[CompilationContext] = contextvars.ContextVar(
    "compilation_context",
)


class CompilationContext:
    """Encapsulates all mutable compilation state.

    Each attribute group corresponds to a registry class that previously
    stored its state in class-level mutable collections.
    """

    def __init__(self) -> None:
        # -- NumbaTypeRegistry state --
        self.numba_types: List[Any] = []  # List[NumbaTypeInfo]
        self.list_element_types: tuple[type, ...] = (int, float, bool)
        self.dict_key_types: tuple[type, ...] = (int,)
        self.dict_value_types: tuple[type, ...] = (int, float, bool)

        # -- TypeFactory state --
        self.type_classes: List[Any] = []  # List[Type[VariableType]]

        # -- FunctionAnalyzer state --
        self.input_handlers: List[Any] = []  # List[InputTypeHandler]
        self.output_handlers: List[Any] = []  # List[OutputTypeHandler]

        # -- ASTHandlerRegistry state --
        self.ast_handlers: Dict[str, Dict[Any, list]] = {}

        # -- NumbaTypeInference state --
        self.assignment_handlers: List[Any] = []
        self.call_handlers: List[Any] = []
        self.attr_accessors: List[Any] = []
        self.attr_lowerers: List[Any] = []

        # -- FFIMethodHelper state --
        self.ffi_opcode_cache: Dict[str, int] = {}
        self.ffi_next_opcode: int = 1

        # -- SourceCategory state --
        self.source_categories: List[Any] = []  # List[SourceCategory]

        # -- NRT library --
        self._nrt_loaded: bool = False

    def __enter__(self) -> CompilationContext:
        self._token = _current_context.set(self)
        return self

    def __exit__(self, *exc: Any) -> None:
        _current_context.reset(self._token)

    @staticmethod
    def current() -> CompilationContext:
        """Return the active context, lazily creating a default if needed."""
        try:
            return _current_context.get()
        except LookupError:
            ctx = CompilationContext()
            # Set BEFORE registering defaults to prevent recursion —
            # register_defaults() will call back into current().
            _current_context.set(ctx)
            ctx.register_defaults()
            return ctx

    def ensure_nrt_loaded(self) -> None:
        """Load the NRT C library on first call (no-op afterwards)."""
        if not self._nrt_loaded:
            import llvmlite.binding as llvm

            from numba_cfunc_compiler.numba_rt import _py_nrt_init

            llvm.load_library_permanently(_py_nrt_init.__file__)
            self._nrt_loaded = True

    def register_defaults(self) -> None:
        from numba_cfunc_compiler.defaults import register_all

        register_all()
