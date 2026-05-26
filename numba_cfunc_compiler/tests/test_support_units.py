import ast
import inspect
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from llvmlite import ir
from numba.core import types

from numba_cfunc_compiler.ast_handlers import ASTHandlerRegistry, HandlerPhase, HandlerResult, ast_handler, with_handlers
from numba_cfunc_compiler.compilation_context import CompilationContext
from numba_cfunc_compiler.defaults import register_all
from numba_cfunc_compiler.defaults.datetime_support import DateTimeType
from numba_cfunc_compiler.defaults.dict_support import NumbaDictType, handle_dict_contains, handle_dict_for
from numba_cfunc_compiler.defaults.list_support import NumbaListType, handle_list_for
from numba_cfunc_compiler.defaults.primitive_support import PrimitiveType
from numba_cfunc_compiler.defaults.struct_support import (
    StructFieldInfo,
    StructType,
    is_struct_type,
    struct_attr_handler,
    struct_attribute_transformer,
)
from numba_cfunc_compiler.defaults.timedelta_support import TimeDeltaType
from numba_cfunc_compiler.models import (
    CONTAINER_STATE_INIT,
    DictTypeMarker,
    InputAnalysis,
    ListTypeMarker,
    NoneType,
    OutputAnalysis,
    ParameterInfo,
    StateAnalysis,
    StateVariableInfo,
    UnknownNumbaType,
    UnknownNumbaValue,
    UnknownType,
)
from numba_cfunc_compiler.numba_config import (
    NumbaDict,
    NumbaList,
    NumbaTypeInfo,
    NumbaTypeRegistry,
    create_new_dict,
    create_new_list,
    numba_type_to_python,
)
from numba_cfunc_compiler.numba_type_inference import NumbaTypeInference
from numba_cfunc_compiler.source_registry import CfuncParam, SourceCategory, SourceInitFilter, SourceRegistry
from numba_cfunc_compiler.type_factory import TypeFactory
from numba_cfunc_compiler.utils.ast import AST, add_statement_to_list
from numba_cfunc_compiler.utils.ffi import FFIMethodHelper
from numba_cfunc_compiler.utils.types import TypeHelper
from numba_cfunc_compiler.variable_factory import (
    ConstantSource,
    ExpressionSource,
    LocalConstantSource,
    LocalVariableSource,
    OutputSource,
    VariableFactory,
    VariableSource,
    VoidPtrSource,
)


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


def test_datetime_and_timedelta_helpers_parse_and_lower():
    aware = datetime(2020, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    delta = timedelta(days=1, seconds=2, microseconds=3)

    assert DateTimeType.to_nanos(aware) == int(aware.timestamp() * 1e9)
    with pytest.raises(TypeError, match="timezone-aware"):
        DateTimeType.to_nanos(datetime(2020, 1, 1))
    assert TimeDeltaType.to_nanos(delta) == int(delta.total_seconds() * 1e9)

    assert DateTimeType.from_type(datetime, UnknownNumbaValue()).value is datetime
    assert DateTimeType.from_type(object, aware).runtime_value == aware
    assert DateTimeType.from_type(object, UnknownNumbaValue()) is None
    assert TimeDeltaType.from_type(timedelta, UnknownNumbaValue()).value is timedelta
    assert TimeDeltaType.from_type(object, delta).runtime_value == delta
    assert TimeDeltaType.from_type(object, UnknownNumbaValue()) is None

    dt_assign = parse_stmt("stamp = datetime(2020, 1, 1, tzinfo=timezone.utc)")
    lowered, var_type = DateTimeType.try_lower_assignment(dt_assign, dt_assign.value, {})
    assert var_type.value is datetime
    assert unparse(lowered).startswith("stamp = ")

    td_assign = parse_stmt("span = timedelta(seconds=5)")
    lowered, var_type = TimeDeltaType.try_lower_assignment(td_assign, td_assign.value, {})
    assert var_type.value is timedelta
    assert unparse(lowered) == "span = 5000000000"

    assert DateTimeType.try_lower_assignment(parse_stmt("x = 1"), ast.Constant(1), {}) is None
    assert TimeDeltaType.try_lower_assignment(parse_stmt("x = 1"), ast.Constant(1), {}) is None
    assert DateTimeType.try_lower_assignment(parse_stmt("obj.x = datetime(2020, 1, 1, tzinfo=timezone.utc)"), dt_assign.value, {}) is None
    assert TimeDeltaType.try_lower_assignment(parse_stmt("obj.x = timedelta(seconds=1)"), td_assign.value, {}) is None

    dt_state = parse_stmt("stamp: State[datetime] = datetime(2020, 1, 1, tzinfo=timezone.utc)")
    parsed = DateTimeType.try_parse_state(dt_state, "stamp", {})
    assert parsed == StateVariableInfo("stamp", 1577836800000000000, datetime)
    numeric_dt_state = parse_stmt("stamp: State[datetime] = 123.9")
    assert DateTimeType.try_parse_state(numeric_dt_state, "stamp", {}).initial_value == 123
    assert DateTimeType.try_parse_state(parse_stmt("x: State[int] = 1"), "x", {}) is None

    td_state = parse_stmt("span: State[timedelta] = timedelta(milliseconds=2)")
    parsed = TimeDeltaType.try_parse_state(td_state, "span", {})
    assert parsed == StateVariableInfo("span", 2000000, timedelta)
    numeric_td_state = parse_stmt("span: State[timedelta] = 456.7")
    assert TimeDeltaType.try_parse_state(numeric_td_state, "span", {}).initial_value == 456
    assert TimeDeltaType.try_parse_state(parse_stmt("x: State[int] = 1"), "x", {}) is None

    with pytest.raises(TypeError, match="Invalid initializer"):
        DateTimeType._parse_state_init(ast.Name(id="bad", ctx=ast.Load()), "stamp")
    with pytest.raises(TypeError, match="Invalid initializer"):
        TimeDeltaType._parse_state_init(ast.Name(id="bad", ctx=ast.Load()), "span")


def test_time_type_helper_accepts_safe_literals_and_rejects_other_nodes():
    assert TypeHelper.get_time_func_name(parse_expr("datetime(2020, 1, 1, tzinfo=timezone.utc)")) == "datetime"
    assert TypeHelper.get_time_func_name(parse_expr("datetime.datetime(2020, 1, 1, tzinfo=timezone.utc)")) == "datetime"
    assert TypeHelper.get_time_func_name(parse_expr("datetime.timedelta(seconds=1)")) == "timedelta"
    assert TypeHelper.get_time_func_name(ast.Name(id="x", ctx=ast.Load())) is None

    value, name = TypeHelper.eval_time_constructor(parse_expr("timedelta(seconds=1, microseconds=2)"))
    assert name == "timedelta"
    assert value == timedelta(seconds=1, microseconds=2)
    lowered = TypeHelper.lower_time_constructor(parse_expr("datetime(2020, 1, 1, tzinfo=timezone.utc)"))
    assert lowered.value == 1577836800000000000
    assert TypeHelper.lower_time_constructor(ast.Constant(1)) is None

    with pytest.raises(TypeError, match="Failed to evaluate"):
        TypeHelper.eval_time_constructor(parse_expr("timedelta(seconds=x)"))
    with pytest.raises(TypeError, match="timezone-aware"):
        TypeHelper.lower_time_constructor(parse_expr("datetime(2020, 1, 1)"))


def test_numba_list_type_parses_inputs_state_and_lowering():
    with default_context():
        marker = ListTypeMarker(int)
        list_type = NumbaListType(marker, [1, 2])
        assert list_type.get_numba_type_name() == "voidptr"
        assert list_type.is_opaque_pointer()
        assert [m.get_name() for m in list_type.get_methods()] == ["append", "pop", "clear"]
        assert NumbaListType.is_type_supported(marker)
        assert not NumbaListType.is_type_supported(int)

        create_stmts = list_type.create_new_container("items")
        assert [unparse(stmt) for stmt in create_stmts] == [
            "items_ptr = standalone_list_new(8, 0)",
            "items = standalone_list_from_voidptr(items_ptr, 'int64')",
        ]
        assert (
            unparse(list_type.from_voidptr("local_items", "items", ast.Name(id="raw", ctx=ast.Load())))
            == "local_items = standalone_list_from_voidptr(raw, 'int64')"
        )
        assert [unparse(stmt) for stmt in list_type.read_constant("const_items")][-2:] == ["const_items.append(1)", "const_items.append(2)"]
        with pytest.raises(TypeError, match="Expected ListTypeMarker"):
            NumbaListType(int, [1]).read_constant("bad")

        stmts, local_type = NumbaListType.create_local_from_type_name("items", "int")
        assert isinstance(local_type.value, ListTypeMarker)
        assert len(stmts) == 2
        with pytest.raises(TypeError, match="Unsupported List element type"):
            NumbaListType.create_local_from_type_name("items", "str")

        assign = parse_stmt("items = create_new_list(int)")
        lowered, lowered_type = NumbaListType.try_lower_assignment(assign, assign.value, {})
        assert isinstance(lowered_type, NumbaListType)
        assert len(lowered) == 2
        assert NumbaListType.try_lower_assignment(parse_stmt("items = 1"), ast.Constant(1), {}) is None
        assert NumbaListType.try_lower_assignment(parse_stmt("items = make_list(int)"), parse_expr("make_list(int)"), {}) is None
        assert NumbaListType.try_lower_assignment(parse_stmt("obj.items = create_new_list(int)"), assign.value, {}) is None
        with pytest.raises(TypeError, match="exactly 1 argument"):
            bad = parse_stmt("items = create_new_list(int, float)")
            NumbaListType.try_lower_assignment(bad, bad.value, {})
        with pytest.raises(TypeError, match="Unsupported List element type"):
            bad = parse_stmt("items = create_new_list(str())")
            NumbaListType.try_lower_assignment(bad, bad.value, {})

        assert not NumbaListType._is_create_new_list_call(ast.Constant(1))
        state = parse_stmt("items: State[NumbaList] = create_new_list(int)")
        state_info = NumbaListType.try_parse_state(state, "items", {})
        assert state_info == StateVariableInfo("items", CONTAINER_STATE_INIT, marker)
        assert NumbaListType.try_parse_state(parse_stmt("items: State[NumbaList] = []"), "items", {}) is None
        with pytest.raises(TypeError, match="exactly 1 argument"):
            NumbaListType.try_parse_state(parse_stmt("items: State[NumbaList] = create_new_list(int, float)"), "items", {})
        with pytest.raises(TypeError, match="type name"):
            NumbaListType.try_parse_state(parse_stmt("items: State[NumbaList] = create_new_list(str())"), "items", {})
        with pytest.raises(TypeError, match="Unsupported List element type"):
            NumbaListType.try_parse_state(parse_stmt("items: State[NumbaList] = create_new_list(str)"), "items", {})

        def accepts_list(values: NumbaList[int]):
            return values

        param = inspect.signature(accepts_list).parameters["values"]
        info = NumbaListType.try_parse_input(param, param.annotation)
        assert info == ParameterInfo(expected_type=marker)
        assert NumbaListType.try_parse_input(param, int) is None
        assert NumbaListType.validate_input("values", (1, 2), marker) == [1, 2]
        with pytest.raises(TypeError, match="Expected ListTypeMarker"):
            NumbaListType.validate_input("values", [1], int)
        with pytest.raises(TypeError, match="expected list or tuple"):
            NumbaListType.validate_input("values", {1}, marker)
        with pytest.raises(TypeError, match="element 1"):
            NumbaListType.validate_input("values", [1, 2.0], marker)


def test_numba_dict_type_parses_inputs_state_and_lowering():
    with default_context():
        marker = DictTypeMarker(int, float)
        dict_type = NumbaDictType(marker, {1: 2.5})
        assert dict_type.get_numba_type_name() == "voidptr"
        assert dict_type.is_opaque_pointer()
        assert [m.get_name() for m in dict_type.get_methods()] == ["get", "pop", "clear", "contains"]
        assert NumbaDictType.is_type_supported(marker)
        assert not NumbaDictType.is_type_supported(int)

        assert [unparse(stmt) for stmt in dict_type.create_new_container("mapping")] == [
            "mapping_ptr = standalone_dict_new(8, 8)",
            "mapping = standalone_dict_from_voidptr(mapping_ptr, 'int64', 'float64')",
        ]
        assert unparse(dict_type.from_voidptr("local_mapping", "mapping", ast.Name(id="raw", ctx=ast.Load()))) == (
            "local_mapping = standalone_dict_from_voidptr(raw, 'int64', 'float64')"
        )
        assert [unparse(stmt) for stmt in dict_type.read_constant("const_mapping")][-1] == "const_mapping[1] = 2.5"
        with pytest.raises(TypeError, match="Expected DictTypeMarker"):
            NumbaDictType(int, {1: 2}).read_constant("bad")

        stmts, local_type = NumbaDictType.create_local_from_type_names("mapping", "int", "bool")
        assert isinstance(local_type.value, DictTypeMarker)
        assert len(stmts) == 2
        with pytest.raises(TypeError, match="Unsupported Dict key type"):
            NumbaDictType.create_local_from_type_names("mapping", "str", "int")
        with pytest.raises(TypeError, match="Unsupported Dict value type"):
            NumbaDictType.create_local_from_type_names("mapping", "int", "str")

        assign = parse_stmt("mapping = create_new_dict(int, float)")
        lowered, lowered_type = NumbaDictType.try_lower_assignment(assign, assign.value, {})
        assert isinstance(lowered_type, NumbaDictType)
        assert len(lowered) == 2
        assert NumbaDictType.try_lower_assignment(parse_stmt("mapping = 1"), ast.Constant(1), {}) is None
        assert NumbaDictType.try_lower_assignment(parse_stmt("mapping = make_dict(int, float)"), parse_expr("make_dict(int, float)"), {}) is None
        assert NumbaDictType.try_lower_assignment(parse_stmt("obj.mapping = create_new_dict(int, float)"), assign.value, {}) is None
        with pytest.raises(TypeError, match="exactly 2 arguments"):
            bad = parse_stmt("mapping = create_new_dict(int)")
            NumbaDictType.try_lower_assignment(bad, bad.value, {})
        with pytest.raises(TypeError, match="Unsupported type"):
            bad = parse_stmt("mapping = create_new_dict(int, str())")
            NumbaDictType.try_lower_assignment(bad, bad.value, {})

        assert not NumbaDictType._is_create_new_dict_call(ast.Constant(1))
        state = parse_stmt("mapping: State[NumbaDict] = create_new_dict(int, float)")
        assert NumbaDictType.try_parse_state(state, "mapping", {}) == StateVariableInfo("mapping", CONTAINER_STATE_INIT, marker)
        assert NumbaDictType.try_parse_state(parse_stmt("mapping: State[NumbaDict] = {}"), "mapping", {}) is None
        with pytest.raises(TypeError, match="exactly 2 arguments"):
            NumbaDictType.try_parse_state(parse_stmt("mapping: State[NumbaDict] = create_new_dict(int)"), "mapping", {})
        with pytest.raises(TypeError, match="type names"):
            NumbaDictType.try_parse_state(parse_stmt("mapping: State[NumbaDict] = create_new_dict(int, str())"), "mapping", {})
        with pytest.raises(TypeError, match="Unsupported Dict value type"):
            NumbaDictType.try_parse_state(parse_stmt("mapping: State[NumbaDict] = create_new_dict(int, str)"), "mapping", {})

        def accepts_dict(values: NumbaDict[int, float]):
            return values

        param = inspect.signature(accepts_dict).parameters["values"]
        info = NumbaDictType.try_parse_input(param, param.annotation)
        assert info == ParameterInfo(expected_type=marker)
        assert NumbaDictType.try_parse_input(param, int) is None
        assert NumbaDictType.validate_input("values", {1: 2.0}, marker) == {1: 2.0}
        with pytest.raises(TypeError, match="Expected DictTypeMarker"):
            NumbaDictType.validate_input("values", {1: 2.0}, int)
        with pytest.raises(TypeError, match="expected dict"):
            NumbaDictType.validate_input("values", [(1, 2.0)], marker)
        with pytest.raises(TypeError, match="key 'bad'"):
            NumbaDictType.validate_input("values", {"bad": 2.0}, marker)
        with pytest.raises(TypeError, match="value for key"):
            NumbaDictType.validate_input("values", {1: 2}, marker)


def test_list_and_dict_for_loop_handlers_rewrite_supported_loops():
    with default_context():
        factory = VariableFactory()
        list_var = LocalVariableSource(NumbaListType(ListTypeMarker(int), None), "items")
        dict_var = LocalVariableSource(NumbaDictType(DictTypeMarker(int, int), None), "mapping")
        factory.add_variable(list_var)
        factory.add_variable(dict_var)

        class Converter:
            variable_factory = factory

            def visit(self, stmt):
                return stmt

        converter = Converter()
        list_loop = parse_stmt("for value in items:\n    total = total + value")
        rewritten = handle_list_for(converter, list_loop)
        assert isinstance(rewritten, ast.For)
        assert unparse(rewritten).startswith("for _li")
        assert handle_list_for(converter, parse_stmt("for value in range(3):\n    pass")) is None
        assert handle_list_for(converter, parse_stmt("for value in missing:\n    pass")) is None

        items_loop = parse_stmt("for key, value in mapping.items():\n    total = total + value")
        items_result = handle_dict_for(converter, items_loop)
        assert len(items_result.side_effects) == 1
        assert "_standalone_dict_iter_next_item" in unparse(items_result.node)

        keys_loop = parse_stmt("for key in mapping.keys():\n    total = total + key")
        keys_result = handle_dict_for(converter, keys_loop)
        assert "_standalone_dict_iter_next_key" in unparse(keys_result.node)
        direct_keys_result = handle_dict_for(converter, parse_stmt("for key in mapping:\n    total = total + key"))
        assert "_standalone_dict_iter_next_key" in unparse(direct_keys_result.node)
        assert handle_dict_for(converter, parse_stmt("for key in missing:\n    pass")) is None

        contains = handle_dict_contains(converter, parse_expr("1 in mapping"))
        assert unparse(contains) == "mapping.contains(1)"
        not_contains = handle_dict_contains(converter, parse_expr("1 not in mapping"))
        assert unparse(not_contains) == "not mapping.contains(1)"
        assert handle_dict_contains(converter, parse_expr("1 < 2")) is None
        assert handle_dict_contains(converter, parse_expr("1 in missing")) is None


class ExampleStruct:
    price: float
    count: int
    nested: object


class ExampleStructType(StructType):
    @classmethod
    def is_type_supported(cls, var_type):
        return var_type is ExampleStruct

    @classmethod
    def _get_struct_fields(cls, var_type):
        return {
            "price": StructFieldInfo("price", 0, "float64", 8),
            "count": StructFieldInfo("count", 8, "int64", 8),
            "nested": StructFieldInfo("nested", 16, "voidptr", 8),
        }

    @classmethod
    def _get_struct_size(cls, var_type):
        return 24


def test_struct_type_and_attribute_helpers():
    with default_context():
        TypeFactory.register(ExampleStructType, priority=0)
        struct_type = ExampleStructType.from_type(ExampleStruct, UnknownNumbaValue())
        assert struct_type.value is ExampleStruct
        assert ExampleStructType.from_type(object, ExampleStruct()).value is ExampleStruct
        assert ExampleStructType.from_type(object, UnknownNumbaValue()) is None
        assert struct_type.get_numba_type_name() == "voidptr"
        assert struct_type.is_opaque_pointer()
        assert struct_type.get_methods() == []
        assert struct_type.get_size() == 24
        assert is_struct_type(struct_type)
        assert not is_struct_type(TypeFactory.get_type(int))

        assert unparse(struct_type.get_field("order", "price")) == "struct_field_access(order, 0, 'float64')"
        assert unparse(struct_type.get_field(ast.Name(id="ptr", ctx=ast.Load()), "count")) == "struct_field_access(ptr, 8, 'int64')"
        assert unparse(struct_type.set_field("order", "count", ast.Constant(7))) == "struct_field_store(order, 8, 'int64', 7)"
        with pytest.raises(KeyError, match="missing"):
            struct_type.get_field("order", "missing")
        with pytest.raises(TypeError, match="voidptr"):
            struct_type.get_field("order", "nested")
        with pytest.raises(TypeError, match="no field metadata"):
            StructType(ExampleStruct, None).get_field("order", "price")

        factory = VariableFactory()
        struct_var = LocalVariableSource(struct_type, "order")
        factory.add_variable(struct_var)
        transformed = struct_attribute_transformer(parse_expr("order.price"), {}, factory)
        assert unparse(transformed) == "struct_field_access(order, 0, 'float64')"
        assert struct_attribute_transformer(parse_expr("missing.price"), {}, factory) is None
        assert struct_attribute_transformer(parse_expr("order.nested"), {}, factory) is None
        assert struct_attribute_transformer(parse_expr("order.price"), {}, None) is None

        class DynamicAccess:
            def get(self):
                return ast.Name(id="dynamic_order", ctx=ast.Load())

        class Container:
            key_to_child_name = {0: "order"}
            element_type = struct_type

            def create_dynamic_access(self, index, variable_factory):
                return DynamicAccess()

        factory.variable_name_map["basket"] = Container()
        dynamic = struct_attribute_transformer(parse_expr("basket[i].price"), {}, factory)
        assert unparse(dynamic) == "struct_field_access(dynamic_order, 0, 'float64')"

        inference = NumbaTypeInference(factory)
        attr_source = struct_attr_handler(inference, struct_var, "price", [])
        assert isinstance(attr_source, ExpressionSource)
        assert unparse(attr_source.get()) == "struct_field_access(order, 0, 'float64')"
        assert struct_attr_handler(inference, LocalVariableSource(TypeFactory.get_type(int), "x"), "price", []) is None
        assert struct_attr_handler(inference, LocalVariableSource(StructType(None, None), "x"), "price", []) is None
        assert struct_attr_handler(inference, struct_var, "missing", []) is None
        assert struct_attr_handler(inference, struct_var, "nested", []) is None


def test_models_type_factory_registry_and_source_registry():
    with default_context():
        inputs = InputAnalysis(
            {
                "x": (1, ParameterInfo(int, "signal")),
                "factor": (2, ParameterInfo(int)),
            }
        )
        assert inputs.get_by_category("signal") == {"x": 1}
        assert inputs.get_params_by_category("constant") == {"factor": (2, ParameterInfo(int))}

        states = StateAnalysis(
            {
                "small": StateVariableInfo("small", True, bool),
                "big": StateVariableInfo("big", 1, int),
                "alpha": StateVariableInfo("alpha", 1.0, float),
            }
        )
        assert [s.name for s in states.sorted_by_size()] == ["alpha", "big", "small"]
        assert OutputAnalysis([int], {"out": int}).named_outputs == {"out": int}

        assert ListTypeMarker(int).element_type is int
        with pytest.raises(TypeError, match="Unsupported List element type"):
            ListTypeMarker(str)
        assert DictTypeMarker(int, bool).value_type is bool
        with pytest.raises(TypeError, match="Unsupported Dict key type"):
            DictTypeMarker(str, int)
        with pytest.raises(TypeError, match="Unsupported Dict value type"):
            DictTypeMarker(int, str)

        primitive = TypeFactory.get_type(int, 3)
        assert isinstance(primitive, PrimitiveType)
        assert primitive.read_constant("x").value.value == 3
        assert primitive.prepare_voidptr_read(None) is primitive
        assert primitive.accepts_value_type(int)
        assert not primitive.accepts_value_type(float)
        assert TypeFactory.get_type(None).runtime_value.__class__ is NoneType
        unknown = TypeFactory.get_type(str)
        assert isinstance(unknown, UnknownType)
        with pytest.raises(ValueError, match="UnknownType"):
            unknown.get_numba_type_name()
        assert TypeFactory.get_type_from_ast(ast.Constant(True)).value is bool
        assert TypeFactory.get_type_from_ast(parse_expr("1 < 2")).value is bool
        assert isinstance(TypeFactory.get_type_from_ast(parse_expr("x + 1")), UnknownType)
        assert TypeFactory.get_type_size(int) == 8
        with pytest.raises(ValueError, match="No registered type class"):
            TypeFactory.get_type_size(str)

        param = inspect.Parameter("x", inspect.Parameter.POSITIONAL_ONLY, annotation=int)
        assert TypeFactory.try_parse_input(param, int)[1] == ParameterInfo(int)
        assert TypeFactory.try_parse_input(param, str) is None
        assert TypeFactory.try_parse_state(parse_stmt("x: State[int] = 1"), "x", {}) == StateVariableInfo("x", 1, int)
        assert TypeFactory.try_parse_state(parse_stmt("x: State[str] = 'a'"), "x", {}) is None

        int_info = NumbaTypeRegistry.get_by_python_type(int)
        assert int_info.numba_name == "int64"
        assert NumbaTypeRegistry.get_by_numba_name("int64") is int_info
        assert NumbaTypeRegistry.get_by_numba_type(types.int64) is int_info
        assert NumbaTypeRegistry.resolve_numba_name(str) == "voidptr"
        assert NumbaTypeRegistry.get_numba_type("int64") is types.int64
        assert NumbaTypeRegistry.resolve_to_numba_type(int) is types.int64
        assert NumbaTypeRegistry.get_size(int) == 8
        assert NumbaTypeRegistry.get_size_for_numba_type(types.float64) == 8
        assert NumbaTypeRegistry.get_size_for_numba_name("int8") == 1
        assert NumbaTypeRegistry.has_numba_name("voidptr")
        assert NumbaTypeRegistry.is_numeric("float64")
        assert not NumbaTypeRegistry.is_numeric("missing")
        assert NumbaTypeRegistry.has_python_type(bool)
        assert NumbaTypeRegistry.get_supported_type_names()["int"] is int
        assert NumbaTypeRegistry.is_supported_type(int)
        assert not NumbaTypeRegistry.is_supported_type(str)
        assert NumbaTypeRegistry.get_list_element_types() == (int, float, bool)
        assert NumbaTypeRegistry.get_dict_key_types() == (int,)
        assert NumbaTypeRegistry.get_dict_value_types() == (int, float, bool)
        assert NumbaTypeRegistry.get_numba_type_map((int, str)) == {"int64": types.int64}
        assert NumbaTypeRegistry.cpp_type_to_numba_name("DOUBLE") == "float64"
        assert NumbaTypeRegistry.cpp_type_to_numba_name("UNKNOWN") == "voidptr"
        with pytest.raises(KeyError):
            NumbaTypeRegistry.get_numba_type("missing")
        with pytest.raises(KeyError):
            NumbaTypeRegistry.get_size(str)
        with pytest.raises(KeyError):
            NumbaTypeRegistry.get_size_for_numba_type(types.unicode_type)
        with pytest.raises(KeyError):
            NumbaTypeRegistry.get_size_for_numba_name("missing")

        NumbaTypeRegistry.register_type(NumbaTypeInfo(str, "unicode", types.unicode_type, 8, False, False, "str"))
        assert NumbaTypeRegistry.get_by_python_type(str).numba_name == "unicode"

        assert numba_type_to_python(types.int64) is int
        assert numba_type_to_python(types.float64) is float
        assert numba_type_to_python(types.boolean) is bool
        assert numba_type_to_python(types.unicode_type) is types.unicode_type

        with pytest.raises(NotImplementedError):
            create_new_list(int)
        with pytest.raises(NotImplementedError):
            create_new_dict(int, int)

        params = SourceRegistry.build_cfunc_params()
        assert [p.name for p in params][:4] == ["outputs", "output_ticked", "state", "lifecycle_phase"]
        assert SourceRegistry.build_cfunc_signature().startswith('"void(CPointer(voidptr), CPointer(int8)')
        assert [arg.arg for arg in SourceRegistry.build_func_args()] == [p.name for p in params]

        class ExtraCategory(SourceCategory):
            id = "extra"
            order = 99
            init_filter = SourceInitFilter.NEVER

            @property
            def cfunc_params(self):
                return [CfuncParam("extra", "int8")]

            def create_variables(self, info, factory):
                return None

        SourceRegistry.register(ExtraCategory())
        assert SourceRegistry.get_ordered()[-1].id == "extra"
        with pytest.raises(ValueError, match="already registered"):
            SourceRegistry.register(ExtraCategory())

        class DuplicateOrder(ExtraCategory):
            id = "duplicate"

        with pytest.raises(ValueError, match="order"):
            SourceRegistry.register(DuplicateOrder())


def test_ast_utilities_and_set_output_errors():
    with default_context():
        stmts = []
        add_statement_to_list(stmts, [ast.Pass(), ast.Pass()])
        add_statement_to_list(stmts, ast.Return(value=None))
        add_statement_to_list(stmts, None)
        assert len(stmts) == 3

        assert unparse(AST.array_access("values", 2)) == "values[2]"
        assert unparse(AST.array_access("values", ast.Name(id="i", ctx=ast.Load()))) == "values[i]"
        assert unparse(AST.function_call("fn", ast.Constant(1))) == "fn(1)"
        assert unparse(AST.assignment("x", ast.Constant(1))) == "x = 1"
        assert unparse(AST.assignment(ast.Name(id="x", ctx=ast.Load()), ast.Constant(2))) == "x = 2"
        assert unparse(AST.assignment(AST.array_access("values", 0), ast.Constant(3))) == "values[0] = 3"
        assert unparse(AST.assignment(parse_expr("obj.attr"), ast.Constant(4))) == "obj.attr = 4"
        assert unparse(AST.deref_pointer("ptr")) == "ptr[0]"
        assert unparse(AST.cast_from_voidptr(ast.Name(id="raw", ctx=ast.Load()), "int64")) == "cast_voidptr_to_ptr(raw, 'int64')"

        factory = VariableFactory()
        output = OutputSource(0, TypeFactory.get_type(int), "result")
        factory.add_variable(output)
        lowered = AST.set_output(factory, SimpleNamespace(visit=lambda n: n), ast.Constant("result"), ast.Constant(5))
        assert [unparse(stmt) for stmt in lowered] == ["output_0_ptr[0] = 5", "output_ticked[0] = 1"]
        with pytest.raises(TypeError, match="string constant"):
            AST.set_output(factory, None, ast.Name(id="result", ctx=ast.Load()), ast.Constant(5))
        with pytest.raises(KeyError, match="unknown output"):
            AST.set_output(factory, None, ast.Constant("missing"), ast.Constant(5))
        factory.add_variable(LocalVariableSource(TypeFactory.get_type(int), "not_output"))
        with pytest.raises(TypeError, match="not a declared output"):
            AST.set_output(factory, None, ast.Constant("not_output"), ast.Constant(5))


def test_variable_sources_and_factory_paths():
    with default_context():
        int_type = TypeFactory.get_type(int, 1)
        source = VariableSource(int_type, "x")
        for method_name in ("local_variable_name", "get_storage_location", "read", "get"):
            with pytest.raises(NotImplementedError):
                getattr(source, method_name)()
        with pytest.raises(NotImplementedError, match="write method"):
            source.write()
        assert not source.is_opaque_pointer()
        clone = source.clone_with_name("y")
        assert clone.name == "y"

        void_source = VoidPtrSource(0, int_type, "x", "inputs")
        assert unparse(void_source.read()) == "x = cast_voidptr_to_ptr(inputs[0], 'int64')"
        assert unparse(void_source.get()) == "x[0]"
        forced = VoidPtrSource(1, int_type, "opaque", "inputs", force_opaque=True)
        assert forced.is_opaque_pointer()
        assert unparse(forced.get()) == "opaque[0]"
        assert VoidPtrSource(0, int_type, "skip", "inputs", skip_pre_read=True).read() is None

        output = OutputSource(0, int_type, "result")
        assert output.local_variable_name() == "output_0_ptr"
        assert unparse(output.write(ast.Constant(3))) == "output_0_ptr[0] = 3"
        with pytest.raises(TypeError, match="Return value"):
            output.write(ast.Constant(3.14))

        local = LocalVariableSource(int_type, "local")
        assert unparse(local.get()) == "local"
        expr = ExpressionSource(int_type, ast.BinOp(ast.Constant(1), ast.Add(), ast.Constant(2)), VariableFactory())
        assert unparse(expr.get()) == "1 + 2"
        assert LocalConstantSource(int_type, "const", 9).get().value == 9
        with pytest.raises(ValueError, match="LocalConstantSource"):
            LocalConstantSource(UnknownType(UnknownNumbaType(), object), "const", object()).get()

        const = ConstantSource(int_type, "factor")
        assert unparse(const.read()) == "factor = 1"
        assert const.get().value == 1
        with pytest.raises(ValueError, match="storage"):
            const.get_storage_location()
        opaque_const = ConstantSource(NumbaListType(ListTypeMarker(int), [1]), "items")
        assert unparse(opaque_const.get()) == "items"

        factory = VariableFactory()
        factory.add_variable(output)
        assert factory.get_source(OutputSource) == [output]
        assert factory.from_name("result") is output
        assert factory.from_name("missing") is None
        with pytest.raises(TypeError, match="subclass"):
            factory.get_source(int)
        with pytest.raises(ValueError, match="already exists"):
            factory.add_variable(OutputSource(1, int_type, "result"))
        assert factory.get_source(OutputSource) == [output]
        factory.add_variable(OutputSource(2, int_type, "other_result"))
        with pytest.raises(RuntimeError, match="array index"):
            factory.get_output_by_idx(1)

        factory = VariableFactory()
        var, assign = factory.add_local_variable(int, "x", ast.Constant(1))
        assert var.name == "x"
        assert unparse(assign) == "x = 1"
        temp = factory.create_temporary_variable(bool, ast.Constant(True), [])
        assert temp.name == "tmp_0"
        assert factory.create_temporary_variable_name() == "tmp_1"
        assert factory.from_ast(None, ast.Name(id="x", ctx=ast.Load()), []) is var
        unknown = factory.from_ast(None, ast.Name(id="new_name", ctx=ast.Load()), [])
        assert isinstance(unknown.type, UnknownType)
        literal = factory.from_ast(None, ast.Constant(10), [])
        assert isinstance(literal, LocalConstantSource)

        class FakeContainer:
            key_to_child_name = {"a": "child"}
            _idx_to_key = {0: "a"}

            def get_key_index(self, key):
                return 0

            def create_dynamic_access(self, index_expr, variable_factory):
                return ExpressionSource(int_type, ast.Subscript(ast.Name(id="bag", ctx=ast.Load()), index_expr, ctx=ast.Load()), variable_factory)

        child = LocalVariableSource(int_type, "child")
        factory.add_variable(child)
        factory.variable_name_map["bag"] = FakeContainer()
        assert factory.from_ast(None, parse_expr("bag['a']"), []) is child
        assert factory.from_ast(None, parse_expr("bag[0]"), []) is child
        with pytest.raises(KeyError, match="no key"):
            factory.from_ast(None, parse_expr("bag['missing']"), [])
        child.skip_pre_read = True
        assert isinstance(factory.from_ast(None, parse_expr("bag['a']"), []), ExpressionSource)

        class KeyVar:
            def resolve_index_expr(self, container):
                return ast.Constant(0)

        factory.variable_name_map["idx"] = KeyVar()
        assert isinstance(factory.from_ast(None, parse_expr("bag[idx]"), []), ExpressionSource)

        statements = []
        created = factory.from_ast(SimpleNamespace(visit=lambda node: ast.Constant(99)), parse_expr("x + 1"), statements)
        assert created.name.startswith("tmp_")
        assert statements

        factory.add_alias("x_alias", var)
        assert factory.from_name("x_alias") is var
        with pytest.raises(ValueError, match="already exists"):
            factory.add_alias("x", var)
        copied = factory.copy_source(var, "x_copy")
        assert copied.name == "x_copy"
        with pytest.raises(ValueError, match="already exists"):
            factory.copy_source(var, "x")


def test_ffi_method_helper_and_method_factories():
    with default_context():

        class LLVMValue:
            def __str__(self):
                return "i64 3"

        class OpcodeOne:
            def __str__(self):
                return "i64 1"

        class OpcodeMissing:
            def __str__(self):
                return "i64 99"

        assert FFIMethodHelper.value_from_llvm_value(LLVMValue()) == 3
        assert FFIMethodHelper.name_to_opcode("method").value == 1
        assert FFIMethodHelper.name_to_opcode("method").value == 1
        assert FFIMethodHelper.opcode_to_name(OpcodeOne()) == "method"
        with pytest.raises(ValueError, match="Opcode 99"):
            FFIMethodHelper.opcode_to_name(OpcodeMissing())

        assert FFIMethodHelper.get_return_type(types.float64).value == 1.0
        assert FFIMethodHelper.get_return_type(types.int64).value == 1
        assert unparse(FFIMethodHelper.get_return_type(types.voidptr)) == "voidptr_null()"
        assert unparse(FFIMethodHelper.get_return_type(types.int8)) == "make_int8()"
        with pytest.raises(ValueError, match="Unsupported return type"):
            FFIMethodHelper.get_return_type(types.unicode_type)

        call = FFIMethodHelper.ffi_call(types.int64, ast.Name(id="obj", ctx=ast.Load()), "method", [ast.Constant(5)])
        assert unparse(call) == "ffi_tuple_args(1, 1, (obj, 5))"
        call_without_obj = FFIMethodHelper.ffi_call(types.float64, None, "other", None)
        assert unparse(call_without_obj) == "ffi_tuple_args(2, 1.0, ())"

        sig = types.int64(types.int64, types.int64, types.Tuple((types.int64, types.float64)))
        llvm_sig = FFIMethodHelper.numba_to_llvm_sig(sig)
        assert isinstance(llvm_sig, ir.FunctionType)
        assert len(llvm_sig.args) == 2
        unituple_sig = types.int64(types.int64, types.int64, types.UniTuple(types.int8, 3))
        assert len(FFIMethodHelper.numba_to_llvm_sig(unituple_sig).args) == 3

        module = ir.Module()
        declared = FFIMethodHelper._get_or_declare_function(module, "demo", ir.FunctionType(ir.IntType(64), []))
        assert FFIMethodHelper._get_or_declare_function(module, "demo", ir.FunctionType(ir.IntType(64), [])) is declared
        assert "readonly" in declared.attributes
        assert "nounwind" in declared.attributes


def test_ast_handler_registry_and_decorator_wrapper():
    with CompilationContext():
        assert ASTHandlerRegistry.get_handlers("Name", HandlerPhase.PRE) == []
        calls = []

        def first(converter, node):
            calls.append("first")
            return None

        def second(converter, node):
            calls.append("second")
            return HandlerResult(ast.Name(id="handled", ctx=ast.Load()), [ast.Pass()])

        ASTHandlerRegistry.register("Name", second, HandlerPhase.PRE, priority=10)
        ASTHandlerRegistry.register("Name", first, HandlerPhase.PRE, priority=0)
        result, side_effects = ASTHandlerRegistry.run_pre_handlers("Name", None, ast.Name(id="x", ctx=ast.Load()))
        assert calls == ["first", "second"]
        assert result.id == "handled"
        assert len(side_effects) == 1

        def post_one(converter, node, result):
            return ast.Name(id=f"{result.id}_one", ctx=ast.Load())

        def post_two(converter, node, result):
            return HandlerResult(ast.Name(id=f"{result.id}_two", ctx=ast.Load()), [ast.Pass()])

        ASTHandlerRegistry.register("Name", post_one, HandlerPhase.POST, priority=0)
        ASTHandlerRegistry.register("Name", post_two, HandlerPhase.POST, priority=1)
        result, side_effects = ASTHandlerRegistry.run_post_handlers(
            "Name", None, ast.Name(id="x", ctx=ast.Load()), ast.Name(id="base", ctx=ast.Load()), []
        )
        assert result.id == "base_one_two"
        assert len(side_effects) == 1

        class Visitor:
            @with_handlers("Name")
            def visit_Name(self, node):
                return ast.Name(id="default", ctx=ast.Load())

        wrapped = Visitor().visit_Name(ast.Name(id="x", ctx=ast.Load()))
        assert isinstance(wrapped, list)
        assert wrapped[-1].id == "handled"
        ASTHandlerRegistry.clear("Name")
        assert Visitor().visit_Name(ast.Name(id="x", ctx=ast.Load())).id == "default"
        ASTHandlerRegistry.clear()
        with pytest.raises(ValueError, match="Must specify"):
            ast_handler("Name")
        with pytest.raises(ValueError, match="Cannot specify"):
            ast_handler("Name", pre=True, post=True)

        @ast_handler("Name", pre=True)
        def decorator_registered(converter, node):
            return ast.Name(id="decorated", ctx=ast.Load())

        result, _ = ASTHandlerRegistry.run_pre_handlers("Name", None, ast.Name(id="x", ctx=ast.Load()))
        assert result.id == "decorated"


def test_type_inference_assignment_and_call_paths():
    with default_context():
        factory = VariableFactory()
        inference = NumbaTypeInference(factory)
        int_type = TypeFactory.get_type(int)
        x_var = LocalVariableSource(int_type, "x")
        factory.add_variable(x_var)

        def call_handler(inf, base_var, method_name, args):
            if method_name == "double":
                return ExpressionSource(int_type, ast.BinOp(base_var.get(), ast.Mult(), ast.Constant(2)), inf.variable_factory)
            return None

        def attr_handler(inf, base_var, attr_name, args):
            if attr_name == "value":
                return ExpressionSource(int_type, ast.Constant(42), inf.variable_factory)
            return None

        NumbaTypeInference.register_call_handler(call_handler)
        NumbaTypeInference.register_attr_accessor(attr_handler)
        assert unparse(inference.handle_call_chain(parse_expr("x.double()")).get()) == "x * 2"
        assert inference.handle_call_chain(parse_expr("x.value")).get().value == 42
        assert inference.handle_call_chain(ast.Constant(1)) is None
        with pytest.raises(ValueError, match="not supported"):
            inference._dispatch_method_call(x_var, "missing", [])

        list_var = LocalVariableSource(NumbaListType(ListTypeMarker(int), None), "items")
        factory.add_variable(list_var)
        native_call = inference._dispatch_method_call(list_var, "append", [ast.Constant(1)])
        assert unparse(native_call.get()) == "items.append(1)"

        def assignment_handler(inf, node, rhs):
            if node.targets[0].id == "handled":
                return AST.assignment("handled", ast.Constant(7))
            return None

        NumbaTypeInference.register_assignment_handler(assignment_handler)
        assert unparse(inference.create_assignment_variable(parse_stmt("handled = x"), ast.Name(id="x", ctx=ast.Load()))) == "handled = 7"

        created = inference.create_assignment_variable(parse_stmt("y = x.double()"), parse_expr("x.double()"))
        assert unparse(created) == "y = x * 2"
        existing = inference.create_assignment_variable(parse_stmt("y = x.double()"), parse_expr("x.double()"))
        assert unparse(existing) == "y = x * 2"
        simple = inference.create_assignment_variable(parse_stmt("z = x"), ast.Name(id="x", ctx=ast.Load()))
        assert unparse(simple) == "z = x"
        assert factory.from_name("z").type is int_type

        opaque = LocalVariableSource(NumbaListType(ListTypeMarker(int), None), "opaque")
        factory.add_variable(opaque)
        alias_assign = inference.create_assignment_variable(parse_stmt("alias = opaque"), ast.Name(id="opaque", ctx=ast.Load()))
        assert unparse(alias_assign) == "alias = opaque"
        assert factory.from_name("alias") is opaque

        lowered = inference.try_lower_assignment(parse_stmt("new_list = create_new_list(int)"), parse_expr("create_new_list(int)"))
        assert len(lowered) == 2
        assert factory.from_name("new_list") is not None

        def bad_lowerer(node, globalns, variable_factory):
            raise ValueError("skip")

        def good_lowerer(node, globalns, variable_factory):
            if isinstance(node, ast.Attribute):
                return ast.Constant(11)
            return None

        NumbaTypeInference.register_attr_lowerer(bad_lowerer)
        NumbaTypeInference.register_attr_lowerer(good_lowerer)
        assert inference.try_attr_lowerers(parse_expr("Some.VALUE")).value == 11
        assert inference.try_attr_lowerers(ast.Constant(1)) is None

        out = OutputSource(0, int_type, "out")
        assert unparse(out.call("output", None)) == "output_ticked[0] = 1"
        assert out.call("missing") is None
