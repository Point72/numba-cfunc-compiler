from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numba_cfunc_compiler.core.analysis import InputAnalysis, OutputAnalysis, StateAnalysis
    from numba_cfunc_compiler.core.variable_factory import VariableFactory

__all__ = [
    "CallbackComponent",
    "CallbackComponentId",
    "CfuncParam",
    "ComponentInputs",
    "ComponentRegistry",
    "MaterializationPhase",
]


class CallbackComponentId(Enum):
    """Built-in callback component identifiers."""

    CONSTANT = auto()
    OUTPUT = auto()
    STATE = auto()
    LIFECYCLE = auto()


class MaterializationPhase(Enum):
    """Controls when a callback component materializes its variables."""

    ALWAYS = auto()  # before lifecycle checks on every callback
    EXECUTE = auto()  # inside the execute-phase block
    NONE = auto()  # no automatic initialization


@dataclass(frozen=True)
class CfuncParam:
    """Declares one parameter in the generated cfunc signature."""

    name: str  # argument name, e.g. "inputs"
    numba_type: str  # e.g. "CPointer(voidptr)", "CPointer(int8)", "int8"


@dataclass(frozen=True)
class ComponentInputs:
    """Analysis data available to callback components while creating variables."""

    input_analysis: InputAnalysis
    state_analysis: StateAnalysis
    output_analysis: OutputAnalysis
    extract_python_type_fn: Callable[[Any], type]


class CallbackComponent(ABC):
    """Contribute callback parameters, variable access, and result metadata."""

    @property
    @abstractmethod
    def id(self) -> Any:
        """Unique identifier (typically a :class:`CallbackComponentId` member)."""
        ...

    @property
    @abstractmethod
    def order(self) -> int:
        """Position in cfunc signature. Lower = earlier.

        Built-in components reserve negative orders so extensions can safely
        start at ``order = 0`` and append user-defined parameters after the
        framework-owned prefix.
        """
        ...

    @property
    def cfunc_params(self) -> list[CfuncParam]:
        """Cfunc parameters this component contributes. Default: none."""
        return []

    @property
    def materialization_phase(self) -> MaterializationPhase:
        """When variables are initialised.

        ``MaterializationPhase.ALWAYS``  — before lifecycle checks on every callback.
        ``MaterializationPhase.EXECUTE`` — inside the execute-phase block.
        ``MaterializationPhase.NONE``    — no automatic initialization.
        """
        return MaterializationPhase.EXECUTE

    @abstractmethod
    def create_variables(self, inputs: ComponentInputs, factory: VariableFactory) -> dict[str, Any]:
        """Add variable accesses to *factory* and return result metadata."""
        ...


class ComponentRegistry:
    @classmethod
    def register(cls, component: CallbackComponent) -> None:
        from numba_cfunc_compiler.core.context import CompilationContext

        ctx = CompilationContext.current()
        for existing in ctx.callback_components:
            if existing.id == component.id:
                raise ValueError(f"Callback component '{component.id}' is already registered")
            if existing.order == component.order:
                raise ValueError(f"Callback component order {component.order} is already used by '{existing.id}'")
        ctx.callback_components.append(component)

    @classmethod
    def get_ordered(cls) -> list[CallbackComponent]:
        from numba_cfunc_compiler.core.context import CompilationContext

        return sorted(
            CompilationContext.current().callback_components,
            key=lambda c: c.order,
        )

    @classmethod
    def build_cfunc_params(cls) -> list[CfuncParam]:
        """Flat list of all cfunc parameters, in component order."""
        params: list[CfuncParam] = []
        for cat in cls.get_ordered():
            params.extend(cat.cfunc_params)
        return params

    @classmethod
    def build_cfunc_signature(cls) -> str:
        params = cls.build_cfunc_params()
        param_types = ", ".join(p.numba_type for p in params)
        return f'"void({param_types})"'

    @classmethod
    def build_func_args(cls) -> list[ast.arg]:
        return [ast.arg(arg=p.name, annotation=None) for p in cls.build_cfunc_params()]
