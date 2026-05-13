from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, List

if TYPE_CHECKING:
    from numba_cfunc_compiler.variable_factory import VariableFactory

__all__ = [
    "SourceCategoryId",
    "SourceInitFilter",
    "CfuncParam",
    "SourceCategory",
    "SourceRegistry",
    "register_default_categories",
]


class SourceCategoryId(Enum):
    """Built-in source category identifiers."""

    CONSTANT = auto()
    OUTPUT = auto()
    STATE = auto()
    LIFECYCLE = auto()


class SourceInitFilter(Enum):
    """Controls when a source category materializes its variables."""

    EVERYTIME = auto()  # before lifecycle checks (always runs)
    ON_EXECUTE = auto()  # inside the execute-phase block.
    NEVER = auto()  # no automatic initialisation.


@dataclass(frozen=True)
class CfuncParam:
    """Declares one parameter in the generated cfunc signature."""

    name: str  # argument name, e.g. "inputs"
    numba_type: str  # e.g. "CPointer(voidptr)", "CPointer(int8)", "int8"


class SourceCategory(ABC):
    """Describes how a family of variables integrates with the compiled cfunc."""

    @property
    @abstractmethod
    def id(self) -> Any:
        """Unique identifier (typically a :class:`SourceCategoryId` member)."""
        ...

    @property
    @abstractmethod
    def order(self) -> int:
        """Position in cfunc signature. Lower = earlier.

        Built-in categories reserve negative orders so extensions can safely
        start at ``order = 0`` and append user-defined parameters after the
        framework-owned prefix.
        """
        ...

    @property
    def cfunc_params(self) -> List[CfuncParam]:
        """Cfunc parameters this category contributes. Default: none."""
        return []

    @property
    def init_filter(self) -> SourceInitFilter:
        """When variables are initialised.

        ``SourceInitFilter.EVERYTIME``  — before lifecycle checks (always runs).
        ``SourceInitFilter.ON_EXECUTE`` — inside the execute-phase block.
        ``SourceInitFilter.NEVER``      — no automatic initialisation.
        """
        return SourceInitFilter.ON_EXECUTE

    @abstractmethod
    def create_variables(self, info: Any, factory: "VariableFactory") -> None:
        """Create :class:`VariableSource` instances and add them to *factory*."""
        ...

    def get_result_metadata(self, info: Any) -> dict:
        """Optional: contribute fields to ``CompilationResult.metadata``."""
        return {}


class SourceRegistry:
    @classmethod
    def register(cls, category: SourceCategory) -> None:
        from numba_cfunc_compiler.compilation_context import CompilationContext

        ctx = CompilationContext.current()
        for existing in ctx.source_categories:
            if existing.id == category.id:
                raise ValueError(f"Source category '{category.id}' is already registered")
            if existing.order == category.order:
                raise ValueError(f"Source category order {category.order} is already used by '{existing.id}'")
        ctx.source_categories.append(category)

    @classmethod
    def get_ordered(cls) -> List[SourceCategory]:
        from numba_cfunc_compiler.compilation_context import CompilationContext

        return sorted(
            CompilationContext.current().source_categories,
            key=lambda c: c.order,
        )

    @classmethod
    def build_cfunc_params(cls) -> List[CfuncParam]:
        """Flat list of all cfunc parameters, in category order."""
        params: List[CfuncParam] = []
        for cat in cls.get_ordered():
            params.extend(cat.cfunc_params)
        return params

    @classmethod
    def build_cfunc_signature(cls) -> str:
        params = cls.build_cfunc_params()
        param_types = ", ".join(p.numba_type for p in params)
        return f'"void({param_types})"'

    @classmethod
    def build_func_args(cls) -> List[ast.arg]:
        return [ast.arg(arg=p.name, annotation=None) for p in cls.build_cfunc_params()]


def register_default_categories() -> None:
    from numba_cfunc_compiler.defaults.source_categories import (
        register_default_categories as _r,
    )

    _r()
