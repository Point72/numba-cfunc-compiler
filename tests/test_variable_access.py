"""Consolidated integration tests for sources."""

import ctypes
from enum import Enum, auto

import pytest
from numba import cfunc, njit, types
from numba.core.errors import TypingError
from numba.extending import register_jitable

from numba_cfunc_compiler.api import set_output
from numba_cfunc_compiler.core.analysis import FunctionAnalyzer, InputAnalysis, InputCategory, InputTypeHandler, ParameterInfo
from numba_cfunc_compiler.core.compile import create_compiled_func
from numba_cfunc_compiler.core.context import CompilationContext
from numba_cfunc_compiler.core.defaults import register_all
from numba_cfunc_compiler.core.variable_access import KeyedInputAccess, SourceAccess
from numba_cfunc_compiler.extension.callback_components import CallbackComponent, CfuncParam, ComponentInputs, ComponentRegistry, MaterializationPhase
from numba_cfunc_compiler.extension.input_sources import SLOT_INDEX, KeyFilter, SourceField, register_input_source, register_keyed_input
from numba_cfunc_compiler.types.factory import HostTypeFactory
from numba_cfunc_compiler.types.native.input import bind_input, unwrap_source
from numba_cfunc_compiler.types.native.keyed import bind_keyed
from numba_cfunc_compiler.types.policy import ValueSemantics
from tests.harness import (
    ACTIVE_SOURCE,
    PASSIVE_SOURCE,
    CompiledNode,
    HarnessInputCategory,
    Signal,
    SignalInputHandler,
    SingleSignalOutputHandler,
    compile_function,
    numba_node,
    setup_standalone_context,
)


@numba_node
def alias_status(value: Signal[int]) -> Signal[int]:
    alias = value
    computed = alias + 5
    return computed if alias.ticked() and alias.valid() else 0


@numba_node
def computed_has_no_status(value: Signal[int]) -> Signal[int]:
    computed = value + 5
    return 1 if computed.ticked() else 0


class Basket:
    pass


class MixedBasket(Basket):
    pass


class HeterogeneousBasket(Basket):
    pass


class SparseBasket(Basket):
    pass


class EmptyBasket(Basket):
    pass


class PassiveBasket(Basket):
    pass


ACTIVE_BASKET = register_keyed_input(
    identity=("tests", "basket.active"),
    element_source=ACTIVE_SOURCE,
    fields=(
        SourceField("slots", types.CPointer(types.voidptr)),
        SourceField("valid", types.CPointer(types.int8)),
        SourceField("ticked", types.CPointer(types.int8)),
        SourceField("start", types.intp),
        SourceField("length", types.intp),
    ),
    element_projection={"valid": "valid", "ticked": "ticked", "index": SLOT_INDEX},
)
ACTIVE_BASKET.iter_method("keys", filter=KeyFilter.all())
ACTIVE_BASKET.iter_method("validkeys", filter=KeyFilter.nonzero("valid"))
ACTIVE_BASKET.iter_method("tickedkeys", filter=KeyFilter.nonzero("ticked"))

PASSIVE_BASKET = register_keyed_input(
    identity=("tests", "basket.passive"),
    element_source=PASSIVE_SOURCE,
    fields=(
        SourceField("slots", types.CPointer(types.voidptr)),
        SourceField("valid", types.CPointer(types.int8)),
        SourceField("start", types.intp),
        SourceField("length", types.intp),
    ),
    element_projection={"valid": "valid", "index": SLOT_INDEX},
)
PASSIVE_BASKET.iter_method("keys", filter=KeyFilter.all())
PASSIVE_BASKET.iter_method("validkeys", filter=KeyFilter.nonzero("valid"))

TIMESTAMP_SOURCE = register_input_source(
    identity=("tests", "signal.timestamped"),
    fields=(SourceField("timestamp", types.CPointer(types.int64)), SourceField("index", types.intp)),
)


@TIMESTAMP_SOURCE.method("timestamp")
def read_timestamp(signal):
    return signal.source.timestamp[signal.source.index]


TIMESTAMP_BASKET = register_keyed_input(
    identity=("tests", "basket.timestamped"),
    element_source=TIMESTAMP_SOURCE,
    fields=(
        SourceField("slots", types.CPointer(types.voidptr)),
        SourceField("timestamp", types.CPointer(types.int64)),
        SourceField("start", types.intp),
        SourceField("length", types.intp),
    ),
    element_projection={"timestamp": "timestamp", "index": SLOT_INDEX},
)
TIMESTAMP_BASKET.iter_method("timestampedkeys", filter=KeyFilter.nonzero("timestamp"))
TIMESTAMP_BASKET.iter_method("matchingkeys", filter=KeyFilter.equals("timestamp", 5))


@TIMESTAMP_BASKET.method("has_timestamps")
def basket_has_timestamps(basket):
    return basket.length > 0


class BasketInputCategory(Enum):
    BASKET = auto()


def test_input_categories_require_enum_members():
    assert ParameterInfo(expected_type=int).category is InputCategory.CONSTANT
    basket = ParameterInfo(expected_type=Basket, category=BasketInputCategory.BASKET)
    analysis = InputAnalysis(parameters={"basket": (Basket(), basket)})
    assert isinstance(analysis.get_by_category(BasketInputCategory.BASKET)["basket"], Basket)
    with pytest.raises(TypeError, match="Enum member"):
        ParameterInfo(expected_type=Basket, category="basket")
    with pytest.raises(TypeError, match="Enum member"):
        analysis.get_by_category("basket")


def test_native_input_policies_require_enum_members():
    with pytest.raises(TypeError, match="ValueSemantics member"):
        ACTIVE_SOURCE.type_for(types.int64, types.int64, "copy")
    with pytest.raises(TypeError, match="SourceField instances"):
        register_input_source(identity=("tests", "invalid"), fields=("valid",))


class BasketInputHandler(InputTypeHandler):
    def try_parse(self, param, ann):
        if ann is Basket:
            return ParameterInfo(expected_type=Basket, category=BasketInputCategory.BASKET)
        return None

    def validate_value(self, param_name, value, expected_type):
        return value


class BasketComponent(CallbackComponent):
    id = "test.basket"
    order = 0
    materialization_phase = MaterializationPhase.EXECUTE

    @property
    def cfunc_params(self):
        return [CfuncParam("inputs", "CPointer(voidptr)"), CfuncParam("input_ticked", "CPointer(int8)"), CfuncParam("input_valid", "CPointer(int8)")]

    def create_variables(self, inputs: ComponentInputs, factory):
        int_type = HostTypeFactory.resolve(int)
        float_type = HostTypeFactory.resolve(float)
        for name, basket in inputs.input_analysis.get_by_category(BasketInputCategory.BASKET).items():
            if isinstance(basket, HeterogeneousBasket):
                children = {"x": (0, int_type), 0: (1, float_type)}
            elif isinstance(basket, MixedBasket):
                children = {"x": (0, int_type), 0: (1, int_type)}
            elif isinstance(basket, SparseBasket):
                children = {0: (0, int_type), 1: (2, int_type)}
            elif isinstance(basket, EmptyBasket):
                children = {}
            else:
                children = {0: (0, int_type), 1: (1, int_type)}
            basket_type = PASSIVE_BASKET if isinstance(basket, PassiveBasket) else ACTIVE_BASKET
            fields = {"slots": "inputs", "valid": "input_valid"}
            if basket_type is ACTIVE_BASKET:
                fields["ticked"] = "input_ticked"
            factory.add_variable(KeyedInputAccess(name, children, basket_type, fields), component=self.id)
        for name in inputs.input_analysis.get_by_category(HarnessInputCategory.SIGNAL):
            factory.add_variable(
                SourceAccess(
                    2,
                    int_type,
                    name,
                    "inputs",
                    ACTIVE_SOURCE,
                    {"valid": "input_valid", "ticked": "input_ticked", "index": 2},
                ),
                component=self.id,
            )
        return {}


class TimestampComponent(CallbackComponent):
    id = "test.timestamp"
    order = 0

    @property
    def cfunc_params(self):
        return [CfuncParam("inputs", "CPointer(voidptr)"), CfuncParam("timestamps", "CPointer(int64)")]

    def create_variables(self, inputs: ComponentInputs, factory):
        for index, (name, signal) in enumerate(inputs.input_analysis.get_by_category(HarnessInputCategory.SIGNAL).items()):
            binding = HostTypeFactory.resolve(inputs.extract_python_type_fn(signal))
            factory.add_variable(
                SourceAccess(index, binding, name, "inputs", TIMESTAMP_SOURCE, {"timestamp": "timestamps", "index": index}), component=self.id
            )
        return {}


class ValidSignalComponent(CallbackComponent):
    id = "test.valid_signal"
    order = 0

    def __init__(self, source):
        self.source = source

    @property
    def cfunc_params(self):
        return [CfuncParam("inputs", "CPointer(voidptr)"), CfuncParam("input_ticked", "CPointer(int8)"), CfuncParam("input_valid", "CPointer(int8)")]

    def create_variables(self, inputs: ComponentInputs, factory):
        for index, (name, signal) in enumerate(inputs.input_analysis.get_by_category(HarnessInputCategory.SIGNAL).items()):
            metadata = {"valid": "input_valid", "index": index}
            if self.source is ACTIVE_SOURCE:
                metadata["ticked"] = "input_ticked"
            factory.add_variable(
                SourceAccess(index, HostTypeFactory.resolve(signal.get_type()), name, "inputs", self.source, metadata), component=self.id
            )
        return {}


@numba_node
def keyed_alias(basket: Basket, index: Signal[int]) -> Signal[int]:
    child = basket.at(index)
    static_child = basket[0]
    return child + len(basket) + static_child if child.ticked() else 0


@numba_node
def keyed_iteration(basket: Basket, index: Signal[int]) -> Signal[int]:
    total = len(basket)
    for key in basket.tickedkeys():
        child = basket.at(key)
        if child.valid():
            total += child
    for key in basket.validkeys():
        total += key
    for key in basket.keys():  # noqa: SIM118 - exercise the native keys iterator
        total += key
    return total


@numba_node
def keyed_whole_alias(basket: Basket, index: Signal[int]) -> Signal[int]:
    other = basket
    return other[0] + other[1]


@numba_node
def keyed_mixed_alias(basket: Basket, index: Signal[int]) -> Signal[int]:
    other = basket
    return other["x"] + other[0]


@numba_node
def keyed_literal_consistent(basket: Basket, index: Signal[int]) -> Signal[int]:
    other = basket
    return basket[0] * 10 + other[0]


@numba_node
def keyed_heterogeneous_alias(basket: Basket, index: Signal[int]) -> Signal[float]:
    other = basket
    return other["x"] + other[0]


@numba_node
def keyed_heterogeneous_dynamic(basket: Basket, index: Signal[int]) -> Signal[float]:
    other = basket
    return other.at(index)


@numba_node
def keyed_dynamic_only(basket: Basket, index: Signal[int]) -> Signal[int]:
    return basket.at(index)


@numba_node
def keyed_implicit_dynamic_direct(basket: Basket, index: Signal[int]) -> Signal[int]:
    return basket[index]


@numba_node
def keyed_implicit_dynamic_alias(basket: Basket, index: Signal[int]) -> Signal[int]:
    other = basket
    return other[index]


@numba_node
def keyed_missing_key(basket: Basket, index: Signal[int]) -> Signal[int]:
    return basket[9]


@numba_node
def keyed_alias_missing_key(basket: Basket, index: Signal[int]) -> Signal[int]:
    other = basket
    return other[9]


@numba_node
def keyed_bool_is_not_int_key(basket: Basket, index: Signal[int]) -> Signal[int]:
    return basket[True]


@numba_node
def passive_basket_alias(basket: Basket, index: Signal[int]) -> Signal[int]:
    other = basket
    child = other.at(index)
    return child + 1 if child.valid() else 0


@numba_node
def passive_basket_iteration(basket: Basket, index: Signal[int]) -> Signal[int]:
    total = 0
    for position in basket.validkeys():
        total += position + 1
    return total


@numba_node
def passive_basket_has_no_ticked(basket: Basket, index: Signal[int]) -> Signal[int]:
    return 1 if basket[0].ticked() else 0


@numba_node
def passive_basket_has_no_tickedkeys(basket: Basket, index: Signal[int]) -> Signal[int]:
    total = 0
    for position in basket.tickedkeys():
        total += position
    return total


@numba_node
def timestamped_value(value: Signal[int]) -> Signal[int]:
    alias = value
    return alias.timestamp() + alias


@numba_node
def source_valid_only(value: Signal[int]) -> Signal[int]:
    return 1 if value.valid() else 0


@register_jitable(inline="always")
def integer_helper(value):
    return value + 1


@register_jitable(inline="always")
def float_helper(value):
    return value + 0.5


@numba_node
def int_from_float_literal(x: Signal[int]) -> Signal[int]:
    return 1.25


@numba_node
def float_from_int_literal(x: Signal[float]) -> Signal[float]:
    return 1


@numba_node
def int_from_bool_literal(x: Signal[int]) -> Signal[int]:
    return True


@numba_node
def int_from_float_expression(x: Signal[float]) -> Signal[int]:
    return x + 0.25


@numba_node
def float_from_int_expression(x: Signal[int]) -> Signal[float]:
    return x + 1


@numba_node
def int_from_bool_expression(x: Signal[int]) -> Signal[int]:
    return (x > 0) or False


@numba_node
def set_int_from_float_expression(x: Signal[float]) -> Signal[int]:
    set_output("output_0", x + 0.25)


@numba_node
def int_from_bool_source(flag: Signal[bool]) -> Signal[int]:
    return flag


@numba_node
def int_from_bool_alias(flag: Signal[bool]) -> Signal[int]:
    local = flag
    return local


@numba_node
def rebound_int_local(x: Signal[int]) -> Signal[int]:
    local = x
    local = 9
    return local


@numba_node
def rebound_bool_local(flag: Signal[bool]) -> Signal[bool]:
    local = flag
    local = False
    return local


@numba_node
def int_from_float_helper(x: Signal[int]) -> Signal[int]:
    return float_helper(x)


def _compile_helper(func):
    with setup_standalone_context():
        return create_compiled_func(
            func,
            Signal(typ=int),
            extract_python_type_fn=lambda signal: signal.get_type(),
            call_globals={"integer_helper": integer_helper, "float_helper": float_helper},
        )


def _compile_basket(func, basket):
    with CompilationContext():
        register_all()
        ComponentRegistry.register(BasketComponent())
        FunctionAnalyzer.register_input_handler(BasketInputHandler())
        FunctionAnalyzer.register_input_handler(SignalInputHandler())
        FunctionAnalyzer.register_output_handler(SingleSignalOutputHandler())
        return create_compiled_func(func, basket, Signal(typ=int), extract_python_type_fn=lambda signal: signal.get_type())


def test_input_alias_methods():
    node = CompiledNode(compile_function(alias_status), [int])
    assert node.execute([7], ticked=[1], valid=[1]) == (12, True)
    assert node.execute([7], ticked=[0], valid=[1]) == (0, True)
    assert node.execute([7], ticked=[1], valid=[0]) == (0, True)
    node._inputs[0] = None  # hosts such as CSP leave invalid slots null
    assert node.execute([0], ticked=[0], valid=[0]) == (0, True)

    payload = HostTypeFactory.resolve(int).payload
    source_type = ACTIVE_SOURCE.type_for(types.int64, types.int64, ValueSemantics.COPY, payload.key)
    bind = bind_input(source_type)

    @njit(types.int64(types.int64))
    def helper(value):
        return value + 1

    @cfunc(types.void(types.voidptr, types.CPointer(types.int8), types.CPointer(types.int64)), _nrt=False)
    def callback(raw, flags, output):
        source = bind(raw, (flags, flags, 0))
        alias = source
        output[0] = helper(unwrap_source(alias)) if alias.ticked() else 0

    value = ctypes.c_int64(12)
    flags = (ctypes.c_int8 * 1)(1)
    output = (ctypes.c_int64 * 1)()
    callback.ctypes(ctypes.c_void_p(ctypes.addressof(value)), flags, output)
    assert output[0] == 13

    with pytest.raises(TypingError, match="ticked"):
        compile_function(computed_has_no_status)


def test_keyed_basket_variants():
    for func, basket, input_types, values, expected in (
        (keyed_whole_alias, Basket(), [int, int, int], [4, 9, 0], 13),
        (keyed_mixed_alias, MixedBasket(), [int, int, int], [4, 9, 0], 13),
        (keyed_literal_consistent, MixedBasket(), [int, int, int], [4, 9, 0], 99),
        (keyed_heterogeneous_alias, HeterogeneousBasket(), [int, float, int], [4, 2.5, 0], 6.5),
    ):
        node = CompiledNode(_compile_basket(func, basket), input_types)
        assert node.execute(values) == (expected, True)

    with pytest.raises(TypingError, match="keyed_at"):
        _compile_basket(keyed_heterogeneous_dynamic, HeterogeneousBasket())

    node = CompiledNode(_compile_basket(keyed_alias, Basket()), [int, int, int])
    assert node.execute([4, 9, 1], ticked=[1, 1, 1]) == (15, True)
    assert node.execute([4, 9, 1], ticked=[1, 0, 1]) == (0, True)

    node = CompiledNode(_compile_basket(keyed_iteration, Basket()), [int, int, int])
    assert node.execute([4, 9, 0], ticked=[1, 0, 0], valid=[1, 1, 1]) == (8, True)
    assert node.execute([4, 9, 0], ticked=[0, 1, 0], valid=[0, 1, 1]) == (13, True)

    node = CompiledNode(_compile_basket(passive_basket_alias, PassiveBasket()), [int, int, int])
    assert node.execute([4, 9, 1], valid=[1, 1, 1]) == (10, True)
    assert node.execute([4, 9, 1], valid=[1, 0, 1]) == (0, True)
    node = CompiledNode(_compile_basket(passive_basket_iteration, PassiveBasket()), [int, int, int])
    assert node.execute([4, 9, 1], valid=[1, 0, 1]) == (1, True)
    with pytest.raises(TypingError, match="ticked"):
        _compile_basket(passive_basket_has_no_ticked, PassiveBasket())
    with pytest.raises(TypingError, match="tickedkeys"):
        _compile_basket(passive_basket_has_no_tickedkeys, PassiveBasket())


def test_keyed_input_invalid_access():
    with pytest.raises(TypeError, match=r"uses basket.at\(position\)"):
        _compile_basket(keyed_implicit_dynamic_direct, Basket())
    with pytest.raises(TypingError, match="getitem"):
        _compile_basket(keyed_implicit_dynamic_alias, Basket())
    with pytest.raises(KeyError, match="has no key 9"):
        _compile_basket(keyed_missing_key, Basket())
    with pytest.raises(KeyError, match="has no key True"):
        _compile_basket(keyed_bool_is_not_int_key, Basket())
    with pytest.raises(TypingError, match="getitem"):
        _compile_basket(keyed_alias_missing_key, Basket())
    with pytest.raises(TypeError, match="host slots must be contiguous"):
        _compile_basket(keyed_dynamic_only, SparseBasket())
    with pytest.raises(TypingError, match="at"):
        _compile_basket(keyed_dynamic_only, EmptyBasket())


def test_keyed_iteration_with_start_offset_and_empty_basket():
    with setup_standalone_context():
        item_type = ACTIVE_SOURCE.value_type(HostTypeFactory.resolve(int))
    basket_type = ACTIVE_BASKET.value_type((("left", 2, item_type), ("right", 3, item_type)))
    bind = bind_keyed(basket_type)
    bind_empty = bind_keyed(ACTIVE_BASKET.value_type(()))

    @cfunc(types.void(types.CPointer(types.voidptr), types.CPointer(types.int8), types.CPointer(types.int8), types.CPointer(types.int64)), _nrt=False)
    def callback(slots, valid, ticked, output):
        basket = bind((slots, valid, ticked, 2, 2))
        alias = basket
        first = basket["left"]
        second = alias["right"]
        first_dynamic = basket.at(0)
        dynamic = basket.at(1)
        output[0] = first + second + first_dynamic + dynamic
        output[1] = (1 if first.valid() else 0) + (1 if second.ticked() else 0) + (1 if dynamic.valid() else 0)
        output[2] = 0
        for position in basket.keys():  # noqa: SIM118 - exercise the registered iterator method
            output[2] += position + 1
        output[3] = 0
        for position in basket.validkeys():
            output[3] += position + 1
        output[4] = 0
        for position in basket.tickedkeys():
            output[4] += position + 1
        empty = bind_empty((slots, valid, ticked, 4, 0))
        output[5] = len(empty)
        for _ in empty.keys():  # noqa: SIM118 - exercise the empty registered iterator method
            output[5] += 1
        for _ in empty.validkeys():
            output[5] += 1

    values = (ctypes.c_int64 * 2)(4, 9)
    slots = (ctypes.c_void_p * 4)(0, 0, ctypes.addressof(values), ctypes.addressof(values) + ctypes.sizeof(ctypes.c_int64))
    valid = (ctypes.c_int8 * 4)(0, 0, 1, 0)
    ticked = (ctypes.c_int8 * 4)(0, 0, 0, 1)
    output = (ctypes.c_int64 * 6)()
    callback.ctypes(slots, valid, ticked, output)
    assert tuple(output) == (26, 2, 3, 1, 2, 0)
    valid[2], valid[3] = 0, 1
    ticked[2], ticked[3] = 1, 0
    callback.ctypes(slots, valid, ticked, output)
    assert tuple(output) == (26, 1, 3, 2, 1, 0)

    with pytest.raises(TypeError, match="host slots must be contiguous"):
        ACTIVE_BASKET.value_type((("left", 0, item_type), ("right", 2, item_type)))
    with pytest.raises(TypeError, match="host slots must be contiguous"):
        ACTIVE_BASKET.type_class((("left", 0, item_type), ("right", 2, item_type)))


def test_host_defined_timestamp_source_and_filter():
    with CompilationContext():
        register_all()
        ComponentRegistry.register(TimestampComponent())
        FunctionAnalyzer.register_input_handler(SignalInputHandler())
        FunctionAnalyzer.register_output_handler(SingleSignalOutputHandler())
        result = create_compiled_func(timestamped_value, Signal(typ=int), extract_python_type_fn=lambda signal: signal.get_type())
        int_binding = HostTypeFactory.resolve(int)
        element_type = TIMESTAMP_SOURCE.value_type(int_binding)

    active_type = ACTIVE_SOURCE.value_type(int_binding)
    passive_type = PASSIVE_SOURCE.value_type(int_binding)
    assert type(element_type) is not type(active_type)
    assert type(active_type) is not type(passive_type)
    assert len({element_type, active_type, passive_type}) == 3

    value = ctypes.c_int64(7)
    output = ctypes.c_int64()
    outputs = (ctypes.c_void_p * 1)(ctypes.addressof(output))
    output_ticked = (ctypes.c_int8 * 1)()
    inputs = (ctypes.c_void_p * 1)(ctypes.addressof(value))
    timestamps = (ctypes.c_int64 * 1)(11)
    result.compiled_func.ctypes(outputs, output_ticked, None, 0, inputs, timestamps)
    assert output.value == 18
    assert output_ticked[0] == 1

    basket_type = TIMESTAMP_BASKET.value_type((("first", 0, element_type), ("second", 1, element_type)))
    bind = bind_keyed(basket_type)

    @cfunc(types.void(types.CPointer(types.voidptr), types.CPointer(types.int64), types.CPointer(types.int64)), _nrt=False)
    def callback(slots, stamps, output_values):
        basket = bind((slots, stamps, 0, 2))
        total = 0
        for position in basket.timestampedkeys():
            total += position + 1
        matching = 0
        for position in basket.matchingkeys():
            matching += position + 1
        output_values[0] = total if basket.has_timestamps() else -1
        output_values[1] = matching

    values = (ctypes.c_int64 * 2)(4, 9)
    slots = (ctypes.c_void_p * 2)(ctypes.addressof(values), ctypes.addressof(values) + ctypes.sizeof(ctypes.c_int64))
    stamps = (ctypes.c_int64 * 2)(0, 5)
    filtered = (ctypes.c_int64 * 2)()
    callback.ctypes(slots, stamps, filtered)
    assert tuple(filtered) == (2, 2)
    stamps[0], stamps[1] = 9, 0
    callback.ctypes(slots, stamps, filtered)
    assert tuple(filtered) == (1, 0)
    stamps[0], stamps[1] = 5, 5
    callback.ctypes(slots, stamps, filtered)
    assert tuple(filtered) == (3, 3)


def test_source_and_basket_identity_registration_and_freeze():
    results = []
    for source in (PASSIVE_SOURCE, ACTIVE_SOURCE, ACTIVE_SOURCE, PASSIVE_SOURCE):
        with CompilationContext():
            register_all()
            ComponentRegistry.register(ValidSignalComponent(source))
            FunctionAnalyzer.register_input_handler(SignalInputHandler())
            FunctionAnalyzer.register_output_handler(SingleSignalOutputHandler())
            results.append(create_compiled_func(source_valid_only, Signal(typ=int), extract_python_type_fn=lambda signal: signal.get_type()))
    assert results[0].semantic_key == results[3].semantic_key
    assert results[1].semantic_key == results[2].semantic_key
    assert results[0].semantic_key != results[1].semantic_key

    identity = ("tests", "isolated.source")
    source = register_input_source(identity=identity, fields=(SourceField("index", types.intp),))
    assert register_input_source(identity=identity, fields=source.fields) is source
    with pytest.raises(ValueError, match="Duplicate metadata field"):
        register_input_source(identity=("tests", "duplicate.fields"), fields=(SourceField("index", types.intp), SourceField("index", types.intp)))
    with pytest.raises(ValueError, match="different layout"):
        register_input_source(identity=identity, fields=(SourceField("index", types.int64), SourceField("other", types.int8)))

    @source.method("index")
    def read_index(value):
        return value.source.index

    def other_index(value):
        return value.source.index + 1

    with pytest.raises(ValueError, match="already registered"):
        source.method("index")(other_index)
    registration = source.registration_key(freeze=True)
    assert source.method("index")(read_index) is read_index
    assert source.registration_key() == registration
    with pytest.raises(RuntimeError, match="frozen"):
        source.method("later")(read_index)

    fields = (
        SourceField("slots", types.CPointer(types.voidptr)),
        SourceField("start", types.intp),
        SourceField("length", types.intp),
    )
    with pytest.raises(ValueError, match="every element metadata field"):
        register_keyed_input(identity=("tests", "missing.projection"), element_source=source, fields=fields, element_projection={})
    basket_identity = ("tests", "isolated.basket")
    basket = register_keyed_input(identity=basket_identity, element_source=source, fields=fields, element_projection={"index": SLOT_INDEX})
    assert register_keyed_input(identity=basket_identity, element_source=source, fields=fields, element_projection={"index": SLOT_INDEX}) is basket
    with pytest.raises(ValueError, match="different layout"):
        register_keyed_input(
            identity=basket_identity,
            element_source=source,
            fields=(*fields, SourceField("extra", types.int8)),
            element_projection={"index": SLOT_INDEX},
        )
    with pytest.raises(ValueError, match="Unknown basket filter"):
        KeyFilter("unknown")
    with pytest.raises(TypeError, match="field name and integer"):
        KeyFilter.equals("missing", "five")
    basket.iter_method("positions", filter=KeyFilter.all())
    basket_registration = basket.registration_key(freeze=True)
    basket.iter_method("positions", filter=KeyFilter.all())
    assert basket.registration_key() == basket_registration
    with pytest.raises(RuntimeError, match="frozen"):
        basket.iter_method("later", filter=KeyFilter.all())


def test_output_typing():
    for func in (
        int_from_float_literal,
        float_from_int_literal,
        int_from_bool_literal,
        int_from_float_expression,
        float_from_int_expression,
        int_from_bool_expression,
        set_int_from_float_expression,
        int_from_bool_source,
        int_from_bool_alias,
    ):
        with pytest.raises(TypingError, match="Output .* cannot store value"):
            compile_function(func)

    assert CompiledNode(compile_function(rebound_int_local), [int]).execute([4]) == (9, True)
    assert CompiledNode(compile_function(rebound_bool_local), [bool]).execute([True]) == (False, True)

    with pytest.raises(TypingError, match="Output int64 cannot store value of type float64"):
        _compile_helper(int_from_float_helper)
