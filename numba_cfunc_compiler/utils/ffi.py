"""FFI (Foreign Function Interface) utilities."""

import ast
from typing import Any, List, Optional

from llvmlite import ir
from numba import types

from numba_cfunc_compiler.compilation_context import CompilationContext

__all__ = [
    "FFIMethodHelper",
]


class FFIMethodHelper:
    @staticmethod
    def value_from_llvm_value(llvm_value) -> int:
        return int(str(llvm_value).split(" ")[1])

    @classmethod
    def opcode_to_name(cls, opcode) -> str:
        opcode = FFIMethodHelper.value_from_llvm_value(opcode)
        ctx = CompilationContext.current()
        for name, opcode_val in ctx.ffi_opcode_cache.items():
            if opcode == opcode_val:
                return name
        raise ValueError(f"Opcode {opcode} not found in method_opcode_cache")

    @classmethod
    def name_to_opcode(cls, name: str) -> ast.Constant:
        ctx = CompilationContext.current()
        if name not in ctx.ffi_opcode_cache:
            ctx.ffi_opcode_cache[name] = ctx.ffi_next_opcode
            ctx.ffi_next_opcode += 1
        return ast.Constant(value=ctx.ffi_opcode_cache[name])

    @staticmethod
    def get_return_type(numba_type) -> ast.AST:
        """Hacky way of specifying the return type for an ffi call"""
        from numba_cfunc_compiler.utils.ast import AST

        mapping = {
            types.float64: ast.Constant(value=1.0),
            types.int64: ast.Constant(value=1),
            # Return an AST node whose inferred Numba type is voidptr.
            types.voidptr: AST.function_call("voidptr_null"),
            # if the ffi function returns an Enum, it will be represented as an int8
            types.int8: AST.function_call("make_int8"),
        }
        node = mapping.get(numba_type)
        if node is None:
            raise ValueError(f"Unsupported return type: {numba_type}")
        return node

    @staticmethod
    def ffi_call(return_type, obj_ptr, method_name: str, args: Optional[list] = None) -> ast.Call:
        from numba_cfunc_compiler.utils.ast import AST

        if args is None:
            args = []

        # Prepend object pointer if present
        call_args = [obj_ptr] + args if obj_ptr is not None else list(args)

        # Build tuple of dynamic args and call a single tuple-based intrinsic
        args_tuple = ast.Tuple(elts=call_args, ctx=ast.Load())
        ast_args = [
            FFIMethodHelper.name_to_opcode(method_name),
            FFIMethodHelper.get_return_type(return_type),
            args_tuple,
        ]

        return AST.function_call("ffi_tuple_args", *ast_args)

    @staticmethod
    def _numba_to_llvm_type(numba_type) -> Optional[Any]:
        """Translate a Numba type to an llvmlite.ir type. Returns None for unsupported/literal string types."""
        NUMBA_TO_LLVM_TYPE = {
            types.int64: ir.IntType(64),
            types.int32: ir.IntType(32),
            types.int8: ir.IntType(8),
            types.float64: ir.DoubleType(),
            types.float32: ir.FloatType(),
            types.boolean: ir.IntType(8),
            types.voidptr: ir.IntType(8).as_pointer(),
            types.CPointer: lambda x: x.dtype.as_pointer(),
        }
        return NUMBA_TO_LLVM_TYPE.get(numba_type, None)

    @staticmethod
    def numba_to_llvm_sig(numba_sig) -> ir.FunctionType:
        """Translate a Numba typing signature (or a (ret, args) tuple) to an llvmlite.ir.FunctionType.
        Supports tuple-based dynamic args where signature.args[2] is a Tuple/UniTuple of arg types.
        """
        ret_type = numba_sig.return_type
        ret_llvm = FFIMethodHelper._numba_to_llvm_type(ret_type)

        # Third arg is a tuple of dynamic arg types for ffi_tuple_args
        if len(numba_sig.args) >= 3:
            tuple_type = numba_sig.args[2]
            try:
                elem_types = list(tuple_type)
            except TypeError:
                # Handle UniTuple(dtype, count)
                dtype = getattr(tuple_type, "dtype", None)
                count = getattr(tuple_type, "count", 0)
                elem_types = [dtype] * int(count) if dtype is not None else []
        else:
            elem_types = []

        arg_llvm_types = [FFIMethodHelper._numba_to_llvm_type(t) for t in elem_types]
        return ir.FunctionType(ret_llvm, arg_llvm_types)

    @staticmethod
    def _get_or_declare_function(module, name: str, llvm_sig):
        existing = module.globals.get(name)
        if existing is not None:
            return existing
        func = ir.Function(module, llvm_sig, name=name)
        # NOTE: if a non-readonly FFI function is ever added, these attributes
        # should be made configurable (e.g. via CompilationContext).
        func.attributes.add("readonly")
        func.attributes.add("nounwind")
        return func

    @staticmethod
    def _llvm_call_from_signature(context, builder, signature, args):
        """
        Helper to perform the LLVM call for ffi intrinsics.
        Expects args layout: [method_opcode, ret_type, <dynamic args...>]
        """
        method_opcode = args[0]
        dyn_args = args[2:]
        llvm_sig = FFIMethodHelper.numba_to_llvm_sig(signature)
        method_name = FFIMethodHelper.opcode_to_name(method_opcode)
        func = FFIMethodHelper._get_or_declare_function(builder.module, method_name, llvm_sig)
        return builder.call(func, dyn_args)

    @staticmethod
    def register_ffi_symbols(symbol_names: List[str], library_module) -> None:
        """Register FFI symbols from a shared library with LLVM.

        Args:
            symbol_names: List of symbol names to register
            library_module: A module with a __file__ attribute pointing to the shared library
        """
        try:
            import ctypes

            import llvmlite.binding as llvm

            cdll = ctypes.CDLL(library_module.__file__)

            for sym in symbol_names:
                try:
                    func = getattr(cdll, sym)
                    addr = ctypes.cast(func, ctypes.c_void_p).value
                    if addr:
                        llvm.add_symbol(sym, addr)
                    else:
                        raise RuntimeError(f"Failed to register FFI symbol: {sym}")
                except Exception as e:
                    raise RuntimeError(f"Failed to register FFI symbol: {sym} {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to register FFI symbols: {e}")
