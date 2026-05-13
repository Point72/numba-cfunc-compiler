import ast
from abc import ABC, abstractmethod
from typing import Dict, List, Type

from numba_cfunc_compiler.numba_config import (
    TICKED_OUTPUTS_ARRAY_NAME,
    NumbaTypeRegistry,
)
from numba_cfunc_compiler.utils.ast import AST
from numba_cfunc_compiler.utils.ffi import FFIMethodHelper

__all__ = [
    "MethodBase",
    "NativeMethod",
    "Output",
    "ffi_method_factory",
    "method_handler_factory",
]


class MethodBase(ABC):
    """Base class for all method handlers.

    Subclasses must implement:
        get_name(): Returns the method name as called in user code.
        handle(): Process the method call on a variable (static).

    Methods are registered on specific VariableSource subclasses via supported_methods.
    """

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """The method name as called in user code (e.g., 'ticked', 'valid')."""
        ...

    @staticmethod
    @abstractmethod
    def handle(var, args):
        """Handle the method call on the given variable with args.

        No type checking is needed here - the method is only called on variables
        that have it registered in their supported_methods.
        """
        raise NotImplementedError


def NativeMethod(method_name: str) -> type:
    """
    Create a method class that passes through to Numba's native method handling.
    """

    def handle(var, args):
        from numba_cfunc_compiler.variable_factory import VariableSource

        if not isinstance(var, VariableSource):
            raise TypeError(f"called {method_name} on {var.name} which is not a tracked variable")
        variable_factory = var.variable_factory
        ast_converter = variable_factory.ast_converter

        if ast_converter is not None and args:
            transformed_args = [ast_converter.visit(arg) for arg in args]
        else:
            transformed_args = list(args) if args else []

        # Return a method call AST that Numba will handle natively
        return ast.Call(
            func=ast.Attribute(
                value=var.get(),
                attr=method_name,
                ctx=ast.Load(),
            ),
            args=transformed_args,
            keywords=[],
        )

    cls_name = f"NativeMethod_{method_name}"
    return type(
        cls_name,
        (MethodBase,),
        {
            "get_name": classmethod(lambda cls: method_name),
            "handle": staticmethod(handle),
        },
    )


class Output(MethodBase):
    """Mark an output as ticked."""

    @classmethod
    def get_name(cls) -> str:
        return "output"

    @staticmethod
    def handle(var, args):
        """Create: output_ticked[output_idx] = 1"""
        output_ticked_value = AST.array_access(TICKED_OUTPUTS_ARRAY_NAME, var.array_idx)
        return AST.assignment(output_ticked_value, ast.Constant(1))


def ffi_method_factory(
    method_list: Dict[str, tuple],
    method_postfix: str = None,
) -> List[Type]:
    """
    Create method handler classes for each FFI method in the method list.

    Args:
        method_list: Dictionary mapping method names to (return_type, arg_types) tuples
        method_postfix: Optional postfix to append to FFI method names

    Returns:
        List of dynamically created method handler classes
    """
    method_classes = []

    def make_method_class(method_name: str, method_postfix: str, method_info: tuple):
        """
        Creates a method class for a given method name and postfix.
        For example, method_name='numLevels' and method_postfix='book' will create
        a method class with FFI call to 'numLevels_book'.
        """
        # In C++ different types can have the same method name. Since we export methods
        # with extern 'C' we need to create unique method names.
        if method_postfix is not None:
            ffi_method_name = f"{method_name}_{method_postfix}"
        else:
            ffi_method_name = method_name
        return_type = method_info[0]

        def handle(var, args):
            from numba_cfunc_compiler.variable_factory import LocalVariableSource

            numba_ret_type = NumbaTypeRegistry.resolve_to_numba_type(return_type)

            if isinstance(var, LocalVariableSource):
                # if the object is a local variable, we can just use the variable name
                pointer = var.get()
            else:
                # otherwise, we need to get the object pointer from the type
                pointer = var.type.runtime_value
            return FFIMethodHelper.ffi_call(
                return_type=numba_ret_type,
                obj_ptr=pointer,
                method_name=ffi_method_name,
                args=args,
            )

        cls_name = f"FFIMethod_{method_name}"
        return type(
            cls_name,
            (MethodBase,),
            {
                "get_name": classmethod(lambda cls, mn=method_name: mn),
                "OUTPUT_TYPE": return_type,
                "handle": staticmethod(handle),
            },
        )

    for mname, minfo in method_list.items():
        method_classes.append(make_method_class(mname, method_postfix, minfo))

    return method_classes


def method_handler_factory(handler_name: str, methods_to_handle: List[Type]) -> Type:
    """
    Create a method handler class that dispatches to the appropriate method.

    Each VariableSource instance creates its own handler with only the methods
    registered for that source type, so dispatch is already type-safe.

    Args:
        handler_name: Name for the handler class
        methods_to_handle: List of method classes to handle

    Returns:
        A handler class with a `handle` method that dispatches based on method name
    """
    # Index method classes by their name (no instantiation needed)
    method_classes = {m.get_name(): m for m in methods_to_handle}

    def handle(method_name, source, *args):
        if method_name in method_classes:
            return method_classes[method_name].handle(source, *args)
        return None

    return type(handler_name, (object,), {"METHODS": method_classes, "handle": handle})
