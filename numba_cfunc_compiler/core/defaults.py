"""Assemble the default type and callback registrations."""

from numba_cfunc_compiler.core.default_components import register_default_components
from numba_cfunc_compiler.types.builtin import register_types
from numba_cfunc_compiler.types.builtin.enum.host import register_ast_handlers


def register_all() -> None:
    register_types()
    register_default_components()
    register_ast_handlers()
