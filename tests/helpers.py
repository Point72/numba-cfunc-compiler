import ast
import ctypes
from contextlib import contextmanager

from llvmlite import ir

from numba_cfunc_compiler.core.context import CompilationContext
from numba_cfunc_compiler.core.defaults import register_all
from numba_cfunc_compiler.types.builtin.struct.host import StructFieldInfo, StructHostType


class CtypesStructHostType(StructHostType):
    """Infer test struct metadata from a ctypes layout."""

    storage_type = None
    accepted_type = None

    @classmethod
    def is_type_supported(cls, var_type):
        return var_type is (cls.accepted_type or cls.storage_type)

    @classmethod
    def get_struct_fields(cls, var_type):
        native_names = {ctypes.c_double: "float64", ctypes.c_int64: "int64"}
        return {
            name: StructFieldInfo(name, getattr(cls.storage_type, name).offset, native_names[field_type], ctypes.sizeof(field_type))
            for name, field_type in cls.storage_type._fields_
        }

    @classmethod
    def get_struct_size(cls, var_type):
        return ctypes.sizeof(cls.storage_type)


@contextmanager
def default_context():
    ctx = CompilationContext()
    with ctx:
        register_all()
        yield ctx


def parse_stmt(src: str):
    return ast.parse(src).body[0]


def parse_expr(src: str):
    return ast.parse(src).body[0].value


def unparse(node: ast.AST) -> str:
    ast.fix_missing_locations(node)
    return ast.unparse(node)


def make_builder():
    module = ir.Module(name="helpers")
    func = ir.Function(module, ir.FunctionType(ir.VoidType(), []), name="test_func")
    block = func.append_basic_block("entry")
    return module, ir.IRBuilder(block)
