import ast
from contextlib import contextmanager

from llvmlite import ir

from numba_cfunc_compiler.core.context import CompilationContext
from numba_cfunc_compiler.core.defaults import register_all


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
