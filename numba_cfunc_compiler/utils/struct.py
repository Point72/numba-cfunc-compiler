from llvmlite import ir
from numba import types
from numba.extending import intrinsic

from numba_cfunc_compiler.numba_config import NumbaTypeRegistry

__all__ = [
    "StructHelper",
]


class StructHelper:
    @staticmethod
    @intrinsic
    def struct_field_access(typingctx, struct_ptr, field_offset_const, field_type_const):
        """Intrinsic for accessing struct fields by offset"""
        if struct_ptr == types.voidptr and isinstance(field_offset_const, types.Literal) and isinstance(field_type_const, types.Literal):
            field_offset = field_offset_const.literal_value
            field_type_name = field_type_const.literal_value

            # only simple numeric fields supported by this intrinsic
            if not NumbaTypeRegistry.is_numeric(field_type_name):
                raise TypeError(f"Struct field access must resolve to a simple numeric type, got {field_type_name}")

            field_numba_type = NumbaTypeRegistry.get_numba_type(field_type_name)
            sig = field_numba_type(struct_ptr, field_offset_const, field_type_const)

            def codegen(context, builder, signature, args):
                [struct_ptr, _, _] = args

                # Cast struct pointer to byte pointer for offset arithmetic
                byte_ptr_type = types.CPointer(types.int8)
                byte_ptr_llvm_type = context.get_value_type(byte_ptr_type)
                byte_ptr = builder.bitcast(struct_ptr, byte_ptr_llvm_type)

                # Add offset to get field pointer
                field_ptr = builder.gep(byte_ptr, [context.get_constant(types.intp, field_offset)])

                # Cast to field type pointer
                field_ptr_type = types.CPointer(field_numba_type)
                field_ptr_llvm_type = context.get_value_type(field_ptr_type)
                typed_field_ptr = builder.bitcast(field_ptr, field_ptr_llvm_type)

                # Load the value
                return builder.load(typed_field_ptr)

            return sig, codegen

    @staticmethod
    @intrinsic
    def struct_field_ptr(typingctx, struct_ptr, field_offset_const):
        """Intrinsic for accessing struct fields by offset, returning a voidptr to the field"""
        if struct_ptr == types.voidptr and isinstance(field_offset_const, types.Literal):
            sig = types.voidptr(struct_ptr, field_offset_const)

            def codegen(context, builder, signature, args):
                [struct_ptr, field_offset_const] = args
                field_offset = field_offset_const
                # Cast struct pointer to byte pointer for offset arithmetic
                byte_ptr_type = types.CPointer(types.int8)
                byte_ptr_llvm_type = context.get_value_type(byte_ptr_type)
                byte_ptr = builder.bitcast(struct_ptr, byte_ptr_llvm_type)
                # Add offset to get field pointer
                field_ptr = builder.gep(byte_ptr, [field_offset])
                # Cast to voidptr and return (no load)
                return builder.bitcast(field_ptr, context.get_value_type(types.voidptr))

            return sig, codegen

    @staticmethod
    @intrinsic
    def struct_field_store(typingctx, struct_ptr, field_offset_const, field_type_const, value):
        """Intrinsic for storing into struct fields by offset."""
        if struct_ptr == types.voidptr and isinstance(field_offset_const, types.Literal) and isinstance(field_type_const, types.Literal):
            field_offset = field_offset_const.literal_value
            field_type_name = field_type_const.literal_value

            if not NumbaTypeRegistry.is_numeric(field_type_name):
                raise TypeError(f"Struct field store must resolve to a simple numeric type, got {field_type_name}")

            field_numba_type = NumbaTypeRegistry.get_numba_type(field_type_name)
            sig = types.void(struct_ptr, field_offset_const, field_type_const, field_numba_type)

            def codegen(context, builder, signature, args):
                [struct_ptr_val, _offset_const, _type_const, val] = args

                # Cast struct pointer to byte pointer for offset arithmetic
                byte_ptr_type = types.CPointer(types.int8)
                byte_ptr_llvm_type = context.get_value_type(byte_ptr_type)
                byte_ptr = builder.bitcast(struct_ptr_val, byte_ptr_llvm_type)

                # Add offset to get field pointer
                field_ptr = builder.gep(byte_ptr, [context.get_constant(types.intp, field_offset)])

                # Cast to field type pointer
                field_ptr_type = types.CPointer(field_numba_type)
                field_ptr_llvm_type = context.get_value_type(field_ptr_type)
                typed_field_ptr = builder.bitcast(field_ptr, field_ptr_llvm_type)

                # Store the value
                builder.store(val, typed_field_ptr)
                return context.get_dummy_value()

            return sig, codegen

    @staticmethod
    @intrinsic
    def struct_memcpy(typingctx, dst_ptr, src_ptr, size_const):
        """Intrinsic for copying a trivially-copyable struct by raw bytes."""
        if dst_ptr == types.voidptr and src_ptr == types.voidptr and isinstance(size_const, types.Literal):
            size = size_const.literal_value
            if not isinstance(size, int) or size < 0:
                raise TypeError("struct_memcpy size must be a non-negative integer literal")

            sig = types.void(dst_ptr, src_ptr, size_const)

            def codegen(context, builder, signature, args):
                dst, src, _ = args
                # Use LLVM memcpy intrinsic for efficient copying
                i8 = ir.IntType(8)
                i8p = i8.as_pointer()
                i64 = ir.IntType(64)
                i1 = ir.IntType(1)

                dst_bytes = builder.bitcast(dst, i8p)
                src_bytes = builder.bitcast(src, i8p)

                # Get or declare the llvm.memcpy intrinsic
                module = builder.module
                memcpy_name = "llvm.memcpy.p0i8.p0i8.i64"
                if memcpy_name in module.globals:
                    memcpy_fn = module.globals[memcpy_name]
                else:
                    memcpy_ty = ir.FunctionType(ir.VoidType(), [i8p, i8p, i64, i1])
                    memcpy_fn = ir.Function(module, memcpy_ty, name=memcpy_name)

                # Call memcpy(dst, src, size, isvolatile=false)
                builder.call(
                    memcpy_fn,
                    [
                        dst_bytes,
                        src_bytes,
                        ir.Constant(i64, size),
                        ir.Constant(i1, 0),
                    ],
                )

                return None

            return sig, codegen
