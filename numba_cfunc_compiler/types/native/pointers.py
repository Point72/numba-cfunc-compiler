from llvmlite import ir
from numba import types
from numba.extending import intrinsic

from numba_cfunc_compiler.types.registry import NumbaTypeRegistry


@intrinsic
def cast_voidptr_to_int(typingctx, ptr):
    sig = types.int64(ptr)

    def codegen(context, builder, signature, args):
        [ptr_val] = args
        return builder.ptrtoint(ptr_val, ir.IntType(64))

    return sig, codegen


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


@intrinsic
def voidptr_null(typingctx):
    sig = types.voidptr()

    def codegen(context, builder, signature, args):
        # Create a null i8* constant
        null_ptr = ir.Constant(ir.IntType(8).as_pointer(), None)
        return null_ptr

    return sig, codegen
