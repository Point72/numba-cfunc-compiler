"""Native enum and enumset representations across the Numba FFI boundary."""

import ast
from datetime import datetime

import pytest
from llvmlite import ir
from numba import cfunc, types

from numba_cfunc_compiler.compilation_context import CompilationContext
from numba_cfunc_compiler.defaults import register_all
from numba_cfunc_compiler.defaults.primitive_support import PrimitiveType
from numba_cfunc_compiler.method_factory import ffi_method_factory
from numba_cfunc_compiler.models import VariableType
from numba_cfunc_compiler.numba_config import NumbaTypeRegistry
from numba_cfunc_compiler.type_factory import TypeFactory
from numba_cfunc_compiler.utils.enum import EnumRepresentation, enum_representation, make_enum
from numba_cfunc_compiler.utils.enumset import enumset_type, make_enumset
from numba_cfunc_compiler.utils.ffi import FFIMethodHelper
from numba_cfunc_compiler.variable_factory import LocalVariableSource


class EnumSetMarker:
    """Example host annotation with an enumset native representation."""


class EnumSetVariableType(VariableType):
    @classmethod
    def is_type_supported(cls, var_type):
        return var_type is EnumSetMarker

    def get_numba_type_name(self):
        return "enumset"


def test_enum_representation_uses_configured_native_width():
    for bit_width, numba_type in [(8, types.int8), (16, types.int16), (32, types.int32), (64, types.int64)]:
        representation = EnumRepresentation(bit_width)
        assert representation.numba_type is numba_type
        assert representation.llvm_type == ir.IntType(bit_width)
        assert representation.byte_width == bit_width // 8

    with pytest.raises(ValueError, match="ENUM_BIT_WIDTH must be one of 8, 16, 32, 64; got 7"):
        EnumRepresentation(7)


def test_enum_and_enumset_intrinsics_compile_with_their_native_abis():
    enum_func = cfunc(enum_representation.numba_type())(lambda: make_enum())
    assert enum_func.ctypes() == 0

    enumset_func = cfunc(enumset_type())(lambda: make_enumset())
    assert f"store i{enumset_type.bit_width} 0" in enumset_func.inspect_llvm()


def test_ffi_resolves_registered_return_types_and_generates_enumset_call():
    with CompilationContext():
        register_all()
        TypeFactory.register(EnumSetVariableType)

        assert FFIMethodHelper.resolve_return_type(int) is types.int64
        assert FFIMethodHelper.resolve_return_type(datetime) is types.int64
        assert FFIMethodHelper.resolve_return_type(object) is types.voidptr
        assert FFIMethodHelper.resolve_return_type(EnumSetMarker) is enumset_type
        assert NumbaTypeRegistry.get_size_for_numba_name("enumset") == enumset_type.byte_width

        method = ffi_method_factory({"contains": (EnumSetMarker, ())}, method_postfix="book")[0]
        source = LocalVariableSource(PrimitiveType(int, 1), "holder")
        call = ast.fix_missing_locations(method.handle(source, []))
        assert ast.unparse(call) == "ffi_tuple_args(1, make_enumset(), (holder,))"


def test_ffi_converts_enumset_and_pointer_types_to_llvm():
    assert FFIMethodHelper._numba_to_llvm_type(enumset_type) == enumset_type.llvm_type
    assert FFIMethodHelper._numba_to_llvm_type(types.CPointer(enumset_type)) == enumset_type.llvm_type.as_pointer()
    assert FFIMethodHelper._numba_to_llvm_type(types.CPointer(types.unicode_type)) is None

    signature = enumset_type(types.int64, types.int64, types.Tuple((types.CPointer(types.int8), enum_representation.numba_type)))
    llvm_signature = FFIMethodHelper.numba_to_llvm_sig(signature)
    assert llvm_signature.return_type == enumset_type.llvm_type
    assert llvm_signature.args == (ir.IntType(8).as_pointer(), enum_representation.llvm_type)
