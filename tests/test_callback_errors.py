"""Execute exported callbacks with the native status ABI."""

import ctypes
import shutil
import subprocess
from dataclasses import replace
from types import SimpleNamespace

import llvmlite.binding as llvm
import pytest
from numba import cfunc, types

from numba_cfunc_compiler.api import (
    NumbaArray,
    NumbaDict,
    NumbaList,
    State,
    create_new_array,
    create_new_dict,
    create_new_list,
    numba_method,
    set_output,
)
from numba_cfunc_compiler.compilation_cache import CacheConfig, get_cache_config, set_cache_config
from numba_cfunc_compiler.core.errors import CallbackStatus, StatusCompiler
from numba_cfunc_compiler.core.postprocess import CompilationOptions
from numba_cfunc_compiler.extension.ffi import register_ffi_method, register_ffi_tagged_method
from numba_cfunc_compiler.types.native.ffi import ffi_ref_from_voidptr, ffi_ref_type, ffi_tagged_from_voidptr, ffi_tagged_ref_type
from tests.harness import CFUNC_T, CompiledNode, Signal, compile_function, numba_node


class StatusNode(CompiledNode):
    def __init__(self, result, input_types):
        # Use the exported IR, including on cache hits. The normal test harness
        # invokes Numba's original wrapper instead.
        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()
        module = llvm.parse_assembly(result.llvm_ir)
        module.verify()
        machine = llvm.Target.from_default_triple().create_target_machine()
        self.engine = llvm.create_mcjit_compiler(module, machine)
        self.engine.finalize_object()
        address = self.engine.get_function_address(result.native_name)
        super().__init__(replace(result, compiled_func=SimpleNamespace(address=address)), input_types)
        self._func = CFUNC_T(address)
        self.status = None

    def _call(self, phase):
        self.status = self._func(self._outputs, self._out_ticked, self._state, phase, self._inputs, self._in_ticked, self._in_valid)


def test_cleanup_skips_uninitialized_and_already_freed_containers():
    @numba_node
    def containers(value: Signal[int], constants: NumbaList[int]) -> Signal[int]:
        items: State[NumbaList] = create_new_list(int)
        lookup: State[NumbaDict] = create_new_dict(int, int)
        items.append(value)
        lookup[value] = len(items)
        return lookup[value] + constants[0]

    result = compile_function(containers, constants=[10])
    node = StatusNode(result, [int])
    # A host must initialize allocation slots to null before START.
    indices = (*result.nrt_state_indices, *result.constant_container_indices)
    for index in indices:
        node._state[index] = None
    node.stop()
    assert node.status == CallbackStatus.OK
    node.start()
    assert node.execute([3])[0] == 11
    node.stop()
    assert node.status == CallbackStatus.OK
    assert all(node._state[index] is None for index in indices)
    node.stop()
    assert node.status == CallbackStatus.OK


@numba_node
def checked_array(index: Signal[int]) -> Signal[int]:
    values: State[NumbaArray] = create_new_array(int, 2)
    values[index] += 1
    return values[index]


@numba_node
def assertion(value: Signal[int]) -> Signal[int]:
    set_output("output_0", 10)
    assert value > 0
    return 20


@numba_node
def missing_key(value: Signal[int]) -> Signal[int]:
    values: State[NumbaDict] = create_new_dict(int, int)
    return values[value]


@numba_node
def tuple_index(value: Signal[int]) -> Signal[int]:
    values = (10, 20)
    return values[value * 1]


@pytest.mark.parametrize(
    "function,status,argument",
    [
        (checked_array, CallbackStatus.INDEX_OOB, 2),
        (checked_array, CallbackStatus.INDEX_OOB, -3),
        (assertion, CallbackStatus.EXCEPTION, 0),
        (missing_key, CallbackStatus.EXCEPTION, 4),
        (tuple_index, CallbackStatus.EXCEPTION, 3),
    ],
)
def test_native_errors(function, status, argument):
    result = compile_function(function)
    assert result.callback_abi_version == 2
    assert "llvm.trap" not in result.llvm_ir
    assert "numba_gil_ensure" not in result.llvm_ir
    assert "numba_do_raise" not in result.llvm_ir
    assert "PyErr_" not in result.llvm_ir
    node = StatusNode(result, [int]).start()
    assert node.status == CallbackStatus.OK
    try:
        node.execute([argument])
        assert node.status == status
    finally:
        node.stop()
        assert node.status == CallbackStatus.OK


@pytest.mark.parametrize("inline", [True, False])
def test_checked_error_unwinds_helpers(inline):
    @numba_method(force_inline=inline)
    def load(values, index):
        result = values[index]
        values[0] = 99
        return result

    @numba_method(force_inline=inline)
    def outer(values, index):
        result = load(values, index)
        values[0] = 100
        return result

    @numba_node
    def node(index: Signal[int]) -> Signal[int]:
        values: State[NumbaArray] = create_new_array(int, 2)
        return outer(values, index * 1)

    result = compile_function(node)
    compiled = StatusNode(result, [int]).start()
    compiled.execute([2])
    assert compiled.status == CallbackStatus.INDEX_OOB
    assert ctypes.cast(compiled._state[0], ctypes.POINTER(ctypes.c_int64))[0] == 0
    assert compiled.execute([0]) == (0, True)
    assert compiled.status == CallbackStatus.OK


def test_status_abi_cache_preserves_errors_and_strict_policy(tmp_path):
    previous = get_cache_config()
    set_cache_config(CacheConfig(enabled=True, directory=str(tmp_path)))
    try:
        original = compile_function(checked_array)
        cached = compile_function(checked_array)
        strict = compile_function(checked_array, options=CompilationOptions(strict_exceptions=True))
        assert cached.compiled_func is None
        assert cached.semantic_key == original.semantic_key != strict.semantic_key
        node = StatusNode(cached, [int]).start()
        assert node.execute([-1]) == (1, True)
        node.execute([2])
        assert node.status == CallbackStatus.INDEX_OOB
    finally:
        set_cache_config(previous)


@pytest.mark.parametrize("function", [assertion, missing_key, tuple_index])
def test_strict_exception_policy_rejects_reachable_errors(function):
    with pytest.raises(ValueError, match=r"test_callback_errors.py:\d+.*strict_exceptions"):
        compile_function(function, options=CompilationOptions(strict_exceptions=True))


def test_strict_exception_policy_allows_checked_errors_and_dead_raises():
    @numba_node
    def dead_raise(value: Signal[int]) -> Signal[int]:
        if False:
            raise ValueError("unreachable")
        return value

    options = CompilationOptions(strict_exceptions=True)
    for function in (checked_array, dead_raise):
        result = compile_function(function, options=options)
        compiled = StatusNode(result, [int]).start()
        compiled.execute([0])
        assert compiled.status == CallbackStatus.OK


def test_null_receivers_probes_and_runtime_tags():
    calls = []

    def native_read(pointer):
        calls.append(pointer)
        return 42 if pointer else 0

    native = ctypes.CFUNCTYPE(ctypes.c_int64, ctypes.c_void_p)(native_read)
    llvm.add_symbol("ncc_checked_read", ctypes.cast(native, ctypes.c_void_p).value)
    reference = ffi_ref_type("CheckedReceiver")
    tagged = ffi_tagged_ref_type("CheckedTaggedReceiver", ("ask", "bid"))
    register_ffi_method(reference, "read", "ncc_checked_read", types.int64)
    register_ffi_method(reference, "probe", "ncc_checked_read", types.int64, allow_null=True)
    register_ffi_tagged_method(tagged, "read", dict.fromkeys(tagged.cases, "ncc_checked_read"), types.int64)
    wrap = ffi_ref_from_voidptr(reference)
    wrap_tagged = ffi_tagged_from_voidptr(tagged)

    @cfunc(types.int32(types.voidptr, types.int32, types.CPointer(types.int64)), pipeline_class=StatusCompiler, _nrt=False)
    def callback(raw, tag, output):
        if tag == 10:
            output[0] = wrap(raw).probe()
        elif tag == 11:
            output[0] = wrap(raw).read()
        else:
            output[0] = wrap_tagged(raw, tag).read()
        return 0

    output = ctypes.c_int64(-1)
    for tag in (0, 1, 11):
        assert callback.ctypes(None, tag, ctypes.byref(output)) == CallbackStatus.NULL_REF
        assert output.value == -1
    for tag in (-1, 2, 2**31 - 1):
        assert callback.ctypes(1, tag, ctypes.byref(output)) == CallbackStatus.INVALID_TAG
        assert output.value == -1
    assert calls == []
    assert callback.ctypes(None, 10, ctypes.byref(output)) == CallbackStatus.OK
    assert output.value == 0
    assert callback.ctypes(1, 0, ctypes.byref(output)) == CallbackStatus.OK
    assert output.value == 42
    assert calls == [None, 1]


@pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc is required for native library validation")
def test_status_library_runs_without_python(tmp_path):
    result = compile_function(assertion)
    machine = llvm.Target.from_default_triple().create_target_machine(reloc="pic")
    obj = tmp_path / "callback.o"
    obj.write_bytes(machine.emit_object(llvm.parse_assembly(result.llvm_ir)))
    library = tmp_path / "callback.so"
    subprocess.run(["gcc", "-shared", str(obj), "-o", str(library)], check=True, capture_output=True)
    source = tmp_path / "caller.c"
    source.write_text("""
#include <dlfcn.h>
#include <stdint.h>
typedef int32_t (*callback_t)(void **, int8_t *, void **, int8_t, void **, int8_t *, int8_t *);
int main(int argc, char **argv) {
    void *library = dlopen(argv[1], RTLD_NOW);
    if (!library) return 1;
    callback_t callback = (callback_t)dlsym(library, argv[2]);
    if (!callback) return 2;
    int64_t output = -1, input = 0;
    void *outputs[] = {&output}, *inputs[] = {&input}, *state[] = {0};
    int8_t ticked[] = {0}, valid[] = {1};
    if (callback(outputs, ticked, state, 0, inputs, valid, valid) != 4) return 3;
    input = 1;
    if (callback(outputs, ticked, state, 0, inputs, valid, valid) != 0 || output != 20) return 4;
    dlclose(library);
    return 0;
}
""")
    executable = tmp_path / "caller"
    subprocess.run(["gcc", str(source), "-ldl", "-o", str(executable)], check=True, capture_output=True)
    subprocess.run([str(executable), str(library), result.native_name], check=True, capture_output=True)


@pytest.mark.parametrize("inline", [True, False])
def test_helper_raises_unwind_and_strict_mode_rejects_them(inline):
    @numba_method(force_inline=inline)
    def fail(value):
        if value < 0:
            raise ValueError("negative")
        return value

    @numba_method(force_inline=inline)
    def nested(value):
        return fail(value) + 10

    @numba_node
    def node(value: Signal[int]) -> Signal[int]:
        return nested(value)

    compiled = StatusNode(compile_function(node), [int]).start()
    assert compiled.execute([2]) == (12, True)
    assert compiled.execute([-1]) == (12, False)
    assert compiled.status == CallbackStatus.EXCEPTION
    with pytest.raises(ValueError, match="strict_exceptions"):
        compile_function(node, options=CompilationOptions(strict_exceptions=True))


def test_lifecycle_errors_return_status():
    import ast

    for phase in ("start", "stop"):
        options = {f"{phase}_body": ast.parse('raise ValueError("lifecycle failure")').body}
        result = compile_function(assertion, **options)
        node = StatusNode(result, [int])
        getattr(node, phase)()
        assert node.status == CallbackStatus.EXCEPTION


def test_dynamic_exception_requires_compile_time_rejection():
    from numba.core.errors import NumbaError

    @numba_node
    def dynamic_raise(value: Signal[int]) -> Signal[int]:
        raise ValueError(value * 1)
        return 0

    with pytest.raises((NumbaError, ValueError), match="dynamic_raise.*test_callback_errors.py"):
        compile_function(dynamic_raise)
