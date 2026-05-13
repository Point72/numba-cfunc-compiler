import ast
import logging
from typing import List, Union

from llvmlite import ir
from numba import TypingError, types
from numba.extending import intrinsic

from numba_cfunc_compiler.numba_config import (
    TICKED_OUTPUTS_ARRAY_NAME,
    NumbaTypeRegistry,
)

__all__ = [
    "print_ast",
    "add_statement_to_list",
    "AST",
]

logger = logging.getLogger("graph_compute")


def print_ast(value) -> None:
    ast.fix_missing_locations(value)
    tree = ast.parse(value)
    src = ast.unparse(tree)
    logger.info(src)


def add_statement_to_list(node_list: List[ast.stmt], node: Union[ast.stmt, List[ast.stmt], None]) -> None:
    if isinstance(node, list):
        node_list.extend(node)
    elif node is not None:
        node_list.append(node)


class AST:
    @staticmethod
    def array_access(name: str, index) -> ast.Subscript:
        idx_expr = index if isinstance(index, ast.AST) else ast.Constant(value=index)
        return ast.Subscript(
            value=ast.Name(id=name, ctx=ast.Load()),
            slice=idx_expr,
            ctx=ast.Load(),
        )

    @staticmethod
    def function_call(name: str, *args) -> ast.Call:
        return ast.Call(
            func=ast.Name(id=name, ctx=ast.Load()),
            args=args,
            keywords=[],
        )

    @staticmethod
    def assignment(target, value: ast.AST) -> ast.Assign:
        """Create an assignment to a target which can be a str Name or an AST (e.g., Subscript)."""
        if isinstance(target, str):
            lhs = ast.Name(id=target, ctx=ast.Store())
        elif isinstance(target, ast.Name):
            lhs = ast.Name(id=target.id, ctx=ast.Store())
        elif isinstance(target, ast.Subscript):
            lhs = ast.Subscript(value=target.value, slice=target.slice, ctx=ast.Store())
        elif isinstance(target, ast.Attribute):
            lhs = ast.Attribute(value=target.value, attr=target.attr, ctx=ast.Store())
        else:
            lhs = target

        return ast.Assign(targets=[lhs], value=value)

    @staticmethod
    def deref_pointer(name: str) -> ast.Subscript:
        return AST.array_access(name, 0)

    @staticmethod
    def cast_from_voidptr(var_name_ptr: str, numba_type_name: str) -> ast.Call:
        return AST.function_call("cast_voidptr_to_ptr", var_name_ptr, ast.Constant(value=numba_type_name))

    @staticmethod
    def set_output(variable_factory, visitor, name_node, value_node) -> List[ast.stmt]:
        """Lower set_output(name, value) into output write + tick statements.

        Only named outputs are supported. The first argument must be a string constant
        matching the declared output name in the annotation.

        Returns a list of AST statements equivalent to:
            <output_name>_ptr[0] = <value_expr>
            output_ticked[<idx>] = 1
        """
        if not isinstance(name_node, ast.Constant) or not isinstance(name_node.value, str):
            invalid_name_str = ast.unparse(name_node)
            raise TypeError(f"set_output name must be a string constant matching an output name: {invalid_name_str}")

        name = name_node.value
        output_var = variable_factory.from_name(name)
        if output_var is None:
            raise KeyError(f"set_output called with unknown output name '{name}'")
        # Verify it is actually an output
        from numba_cfunc_compiler.variable_factory import OutputSource

        if not isinstance(output_var, OutputSource):
            raise TypeError(f"{name} is not a declared output")
        idx = output_var.array_idx

        statements: List[ast.stmt] = []
        # Lower the value expression into a local variable if needed, using factory helpers
        var = variable_factory.from_ast(visitor=visitor, ast_node=value_node, statements=statements)
        value_expr = var.get()

        # Write the value and mark the output as ticked
        statements.append(output_var.write(value_expr))
        tick_lhs = AST.array_access(TICKED_OUTPUTS_ARRAY_NAME, idx)
        statements.append(AST.assignment(tick_lhs, ast.Constant(1)))
        return statements

    @staticmethod
    @intrinsic
    def cast_voidptr_to_int(typingctx, ptr):
        sig = types.int64(ptr)

        def codegen(context, builder, signature, args):
            [ptr_val] = args
            return builder.ptrtoint(ptr_val, ir.IntType(64))

        return sig, codegen

    @staticmethod
    @intrinsic
    def cast_voidptr_to_ptr(typingctx, src, target_type_const):
        if src == types.voidptr and isinstance(target_type_const, types.Literal):
            type_name = target_type_const.literal_value

            if NumbaTypeRegistry.has_numba_name(type_name):
                if type_name == "voidptr":
                    # For structs, we keep them as void pointers
                    result_type = types.voidptr
                else:
                    target_numba_type = NumbaTypeRegistry.get_numba_type(type_name)
                    result_type = types.CPointer(target_numba_type)
                sig = result_type(src, target_type_const)

                def codegen(context, builder, signature, args):
                    [src, _] = args
                    rtype = signature.return_type
                    llrtype = context.get_value_type(rtype)
                    return builder.bitcast(src, llrtype)

                return sig, codegen
            raise TypeError(f"Attempted to cast unsupported type {type_name}")

    @staticmethod
    @intrinsic
    def voidptr_null(typingctx):
        sig = types.voidptr()

        def codegen(context, builder, signature, args):
            # Create a null i8* constant
            null_ptr = ir.Constant(ir.IntType(8).as_pointer(), None)
            return null_ptr

        return sig, codegen

    @staticmethod
    @intrinsic
    def make_int8(typingctx):
        sig = types.int8()

        def codegen(context, builder, signature, args):
            return ir.Constant(ir.IntType(8), 1)

        return sig, codegen

    @staticmethod
    @intrinsic
    def ffi_tuple_args(typingctx, method_opcode, ret_type, args_tuple):
        sig = ret_type(method_opcode, ret_type, args_tuple)

        def codegen(context, builder, signature, args):
            from numba_cfunc_compiler.utils.ffi import FFIMethodHelper

            method_opcode_val = args[0]
            tuple_val = args[2]
            # Expand tuple elements into individual arguments
            n = len(signature.args[2])
            dyn_vals = [builder.extract_value(tuple_val, i) for i in range(n)]
            llvm_sig = FFIMethodHelper.numba_to_llvm_sig(signature)
            method_name = FFIMethodHelper.opcode_to_name(method_opcode_val)
            func = FFIMethodHelper._get_or_declare_function(builder.module, method_name, llvm_sig)
            return builder.call(func, dyn_vals)

        return sig, codegen

    @staticmethod
    @intrinsic
    def voidptr_to_intp(typingctx, src):
        if src != types.voidptr:
            raise TypingError(f"voidptr_to_intp expected voidptr, got {src}")
        sig = types.intp(src)

        def codegen(context, builder, signature, args):
            [vp] = args
            return builder.ptrtoint(vp, context.get_value_type(types.intp))

        return sig, codegen
