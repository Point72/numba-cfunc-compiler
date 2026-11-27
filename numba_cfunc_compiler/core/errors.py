"""Native callback errors, independent of Python's exception reporting runtime."""

from enum import IntEnum
from types import MethodType

from llvmlite import ir
from numba.core.compiler import Compiler


class CallbackStatus(IntEnum):
    OK = 0
    INDEX_OOB = 1
    INVALID_TAG = 2
    NULL_REF = 3
    EXCEPTION = 4


# Private codes propagated using Numba's native CPU calling convention. These
# are never the public C ABI; the wrapper maps them to CallbackStatus.
_CHECKED_ERROR_BASE = 64
STATUS_ABI_VERSION = 2


def return_checked_error(context, builder, status, message):
    """Unwind the current native function, including calls from ordinary njit.

    Keep valid exception information for Numba's standard Python wrappers.
    Our native wrapper uses only the status and never deserializes that data.
    """
    exception = IndexError if status == CallbackStatus.INDEX_OOB else ValueError
    context.call_conv.set_static_user_exc(builder, exception, (message,))
    context.call_conv._return_errcode_raw(builder, ir.Constant(ir.IntType(32), _CHECKED_ERROR_BASE + int(status)))


def _create_status_wrapper(context, library, fndesc, env, call_helper):
    module = context.create_module("ncc_status_wrapper")
    native_type = context.call_conv.get_function_type(fndesc.restype, fndesc.argtypes)
    native = ir.Function(module, native_type, fndesc.llvm_func_name)
    public_type = ir.FunctionType(ir.IntType(32), [context.get_value_type(typ) for typ in fndesc.argtypes])
    wrapper = ir.Function(module, public_type, fndesc.llvm_cfunc_wrapper_name)
    builder = ir.IRBuilder(wrapper.append_basic_block("entry"))
    status, value = context.call_conv.call_function(builder, native, fndesc.restype, fndesc.argtypes, wrapper.args, attrs=("alwaysinline",))
    first = ir.Constant(ir.IntType(32), _CHECKED_ERROR_BASE + CallbackStatus.INDEX_OOB)
    last = ir.Constant(ir.IntType(32), _CHECKED_ERROR_BASE + CallbackStatus.NULL_REF)
    checked = builder.and_(builder.icmp_signed(">=", status.code, first), builder.icmp_signed("<=", status.code, last))
    mapped = builder.select(
        checked,
        builder.sub(status.code, ir.Constant(ir.IntType(32), _CHECKED_ERROR_BASE)),
        ir.Constant(ir.IntType(32), int(CallbackStatus.EXCEPTION)),
    )
    builder.ret(builder.select(status.is_error, mapped, value))
    library.add_ir_module(module)


class StatusCompiler(Compiler):
    """Customize only this compilation's context; never patch Numba globally."""

    def define_pipelines(self):
        self.state.targetctx.create_cfunc_wrapper = MethodType(_create_status_wrapper, self.state.targetctx)
        return super().define_pipelines()
