import ast
from typing import Annotated

import pytest
from llvmlite import binding as llvm, ir
from numba.core import types

from numba_cfunc_compiler.compilation_context import CompilationContext
from numba_cfunc_compiler.defaults import register_all
from numba_cfunc_compiler.output_utils import (
    collect_return_nodes,
    parse_annotated_metadata_dict,
    validate_return_arity,
    validate_single_return,
)
from numba_cfunc_compiler.post_compilation import (
    CompilationOptions,
    apply_post_compilation,
    link_ffi_bitcode,
)
from numba_cfunc_compiler.standalone.dict import StandaloneDictType
from numba_cfunc_compiler.standalone.list import StandaloneListType
from numba_cfunc_compiler.standalone.utils import (
    alloc_value_buffer,
    convert_to_i8,
    convert_to_i64,
    f64,
    get_llvm_type_for_numba_dtype,
    get_or_declare_function,
    i8,
    i8ptr,
    i32,
    i64,
    prepare_int_key_for_lookup,
    store_value_to_buffer,
)
from numba_cfunc_compiler.state_ast import (
    append_state_values_to_return,
    inject_state_params,
    is_state_annotation,
    state_annotation_target,
)
from numba_cfunc_compiler.utils.llvm import LLVMIRHelper


def unparse(node: ast.AST) -> str:
    ast.fix_missing_locations(node)
    return ast.unparse(node)


def make_builder():
    module = ir.Module(name="helpers")
    func = ir.Function(module, ir.FunctionType(ir.VoidType(), []), name="test_func")
    block = func.append_basic_block("entry")
    return module, ir.IRBuilder(block)


def test_output_utils():
    tree = ast.parse(
        """
def f(x):
    if x:
        return 1
    if x < 0:
        return (1, 2)
    return
"""
    )
    returns = collect_return_nodes(tree)
    assert len(returns) == 2
    with pytest.raises(ValueError, match="returns 2 values"):
        validate_single_return(tree)
    with pytest.raises(ValueError, match="expects 1 output"):
        validate_single_return(ast.parse("def f():\n    pass"))
    with pytest.raises(ValueError, match="annotation expects 3"):
        validate_return_arity(tree, 3)

    single = ast.parse("def f():\n    return 1")
    validate_single_return(single)
    validate_return_arity(ast.parse("def f():\n    return 1, 2"), 2)

    class SignalSet:
        pass

    metadata = parse_annotated_metadata_dict(Annotated[SignalSet, {"bid": int}], SignalSet, "SignalSet", "Annotated[SignalSet, {'bid': int}]")
    assert metadata == {"bid": int}
    assert parse_annotated_metadata_dict(int, SignalSet, "SignalSet", "example") is None
    with pytest.raises(TypeError, match="must use SignalSet"):
        parse_annotated_metadata_dict(Annotated[int, {"bid": int}], SignalSet, "SignalSet", "example")
    with pytest.raises(TypeError, match="single dict metadata"):
        parse_annotated_metadata_dict(Annotated[SignalSet, ("bad",)], SignalSet, "SignalSet", "example")


def test_state_ast():
    ann = ast.parse("state: State[int] = 1").body[0]
    assert is_state_annotation(ann)
    assert state_annotation_target(ann) == "state"
    assert not is_state_annotation(ast.parse("value: int = 1").body[0])
    with pytest.raises(ValueError, match="not a State"):
        state_annotation_target(ast.parse("value: int = 1").body[0])

    func = ast.parse("def f(x):\n    return x").body[0]
    inject_state_params(func, ["state_a", "state_b"])
    assert [arg.arg for arg in func.args.args] == ["x", "state_a", "state_b"]

    assert unparse(append_state_values_to_return(ast.Return(value=ast.Constant(1)), ["s"])) == "return (1, s)"
    assert (
        unparse(append_state_values_to_return(ast.Return(value=ast.Tuple([ast.Constant(1), ast.Constant(2)], ast.Load())), ["s"]))
        == "return (1, 2, s)"
    )
    assert unparse(append_state_values_to_return(ast.Return(value=None), ["s"])) == "return (s,)"


def test_post_compilation(caplog):
    assert LLVMIRHelper.force_inline("attributes #0 = { noinline }") == "attributes #0 = { alwaysinline }"
    renamed = llvm.parse_assembly("define void @old() { call void @old()\n ret void }")
    LLVMIRHelper.rename_exported_symbol(renamed, "old", "new")
    assert renamed.get_function("new").name == "new"
    assert "call void @new()" in str(renamed)
    same_module = llvm.parse_assembly("define void @same() { ret void }")
    LLVMIRHelper.rename_exported_symbol(same_module, "same", "same")
    assert same_module.get_function("same").name == "same"
    with pytest.raises(ValueError, match="Failed to find"):
        LLVMIRHelper.rename_exported_symbol(same_module, "old", "new")

    internalized = llvm.parse_assembly(
        "define dso_local void @entry() { ret void }\n"
        'define linkonce_odr hidden i64 @"helper.with-punctuation"() { ret i64 0 }\n'
        "define void @untouched() { ret void }\n"
        "declare void @external()\n"
    )
    LLVMIRHelper.internalize_defined_functions(internalized, only_symbols={"helper.with-punctuation", "external"}, preserve_symbols={"entry"})
    assert internalized.get_function("entry").linkage == llvm.Linkage.external
    assert internalized.get_function("helper.with-punctuation").linkage == llvm.Linkage.internal
    assert internalized.get_function("untouched").linkage == llvm.Linkage.external
    assert internalized.get_function("external").is_declaration
    internalized.verify()

    class Library:
        def get_llvm_str(self):
            return "define void @raw_name() #0 { ret void }\ndefine void @helper() { ret void }\nattributes #0 = { noinline }"

    class CompiledFunc:
        native_name = "raw_name"
        _library = Library()

    ir_text, exported = apply_post_compilation(CompiledFunc(), "abc123", CompilationOptions(force_inline=True))
    assert exported == "_ncc_numba_abc123"
    assert "@_ncc_numba_abc123" in ir_text
    assert "alwaysinline" in ir_text
    assert "define internal void @helper()" in ir_text

    module = ir.Module(name="bad_link")
    assert link_ffi_bitcode(module, b"not valid bitcode", internalize=True) is module
    assert "Failed to link FFI bitcode" in caplog.text


def test_llvm_errors():
    colliding = llvm.parse_assembly("define void @old() { ret void }\ndefine void @new() { ret void }")
    with pytest.raises(ValueError, match="Failed to rename"):
        LLVMIRHelper.rename_exported_symbol(colliding, "old", "new")

    function = llvm.parse_assembly("define void @ffi() { ret void }").get_function("ffi")

    class ChangedHeaderModule:
        functions = (function,)

        def __str__(self):
            return "define void @other() { ret void }"

    with pytest.raises(ValueError, match="Cannot mark 'ffi' alwaysinline"):
        LLVMIRHelper.mark_always_inline(ChangedHeaderModule(), {"ffi"})


@pytest.mark.parametrize("internalize", [False, True])
def test_ffi_linkage(internalize):
    module = llvm.parse_assembly('declare i64 @"ffi #0"()\ndefine i64 @entry() { %result = call i64 @"ffi #0"()\n ret i64 %result }\n')
    ffi_module = llvm.parse_assembly(
        'define i64 @"ffi #0"() #0 { ret i64 7 }\n'
        "define i64 @unused_ffi() { ret i64 9 }\n"
        'attributes #0 = { "target-cpu"="x86-64" "target-features"="+sse2" }\n'
    )

    linked = link_ffi_bitcode(module, ffi_module.as_bitcode(), internalize=internalize)

    assert linked.get_function("entry").linkage == llvm.Linkage.external
    assert linked.get_function("ffi #0").linkage == (llvm.Linkage.internal if internalize else llvm.Linkage.external)
    assert linked.get_function("unused_ffi").linkage == (llvm.Linkage.internal if internalize else llvm.Linkage.external)
    assert b"alwaysinline" in b" ".join(linked.get_function("ffi #0").attributes)
    assert '"target-cpu"' not in str(linked)
    assert '"target-features"' not in str(linked)
    linked.verify()


def test_standalone_types():
    with CompilationContext():
        register_all()
        list_type = StandaloneListType(types.int64)
        assert list_type.item_size == 8
        assert list_type.key is types.int64
        dict_type = StandaloneDictType(types.int64, types.float64)
        assert dict_type.key_size == 8
        assert dict_type.value_size == 8
        assert dict_type.key == (types.int64, types.float64)
        with pytest.raises(TypeError, match="Unsupported element type"):
            StandaloneListType(types.unicode_type)
        with pytest.raises(TypeError, match="Unsupported type"):
            StandaloneDictType(types.unicode_type, types.int64)

        assert isinstance(i8(), ir.IntType) and i8().width == 8
        assert isinstance(i32(), ir.IntType) and i32().width == 32
        assert isinstance(i64(), ir.IntType) and i64().width == 64
        assert isinstance(i8ptr(), ir.PointerType)
        assert isinstance(f64(), ir.DoubleType)
        assert isinstance(get_llvm_type_for_numba_dtype(types.int64), ir.IntType)
        assert isinstance(get_llvm_type_for_numba_dtype(types.float64), ir.DoubleType)
        assert isinstance(get_llvm_type_for_numba_dtype(types.int8), ir.IntType)
        with pytest.raises(TypeError, match="Unsupported dtype"):
            get_llvm_type_for_numba_dtype(types.unicode_type)


def test_llvm_builder():
    module, builder = make_builder()
    fnty = ir.FunctionType(i64(), [])
    declared = get_or_declare_function(module, "external_func", fnty)
    assert get_or_declare_function(module, "external_func", fnty) is declared

    small = ir.Constant(ir.IntType(32), 7)
    wide = ir.Constant(ir.IntType(64), 7)
    tiny = ir.Constant(ir.IntType(1), 1)
    byte = ir.Constant(ir.IntType(8), 1)
    assert convert_to_i64(builder, small).type.width == 64
    assert convert_to_i64(builder, wide) is wide
    assert convert_to_i8(builder, wide).type.width == 8
    assert convert_to_i8(builder, tiny).type.width == 8
    assert convert_to_i8(builder, byte) is byte

    key_ptr, hash_val = prepare_int_key_for_lookup(builder, byte, types.int8)
    assert isinstance(key_ptr.type, ir.PointerType)
    assert hash_val.type.width == 64
    key_ptr, hash_val = prepare_int_key_for_lookup(builder, wide, types.int64)
    assert isinstance(key_ptr.type, ir.PointerType)
    assert hash_val is wide

    val_ptr, val_char_ptr = alloc_value_buffer(builder, types.float64)
    assert isinstance(val_ptr.type, ir.PointerType)
    assert isinstance(val_char_ptr.type, ir.PointerType)
    int_val_ptr, int_val_char_ptr = store_value_to_buffer(builder, small, types.int64)
    assert isinstance(int_val_ptr.type, ir.PointerType)
    assert isinstance(int_val_char_ptr.type, ir.PointerType)
    bool_val_ptr, _ = store_value_to_buffer(builder, wide, types.int8)
    assert isinstance(bool_val_ptr.type, ir.PointerType)
    float_value = ir.Constant(ir.DoubleType(), 1.5)
    float_val_ptr, _ = store_value_to_buffer(builder, float_value, types.float64)
    assert isinstance(float_val_ptr.type, ir.PointerType)

    builder.ret_void()
    assert "external_func" in str(module)
