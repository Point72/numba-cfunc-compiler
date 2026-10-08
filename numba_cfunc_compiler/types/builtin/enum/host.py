import ast
from dataclasses import dataclass
from types import MappingProxyType

from numba import types

from numba_cfunc_compiler.core.names import enum_literal_name, enumset_builder_name
from numba_cfunc_compiler.extension.ast import ast_handler
from numba_cfunc_compiler.types.builtin.enum.native import EnumSetValueType, EnumValueType, enum_literal, family_types, set_builder
from numba_cfunc_compiler.types.policy import ValueSemantics
from numba_cfunc_compiler.utils.ast import AST


@dataclass(frozen=True)
class EnumFamilyBinding:
    enum_class: type
    abi_values: object
    value_type: EnumValueType
    set_type: EnumSetValueType | None
    storage_type: types.Type
    set_bits: int | None

    @property
    def key(self):
        return self.value_type.family_key

    @property
    def set_builder(self):
        if self.set_type is None:
            raise TypeError(f"Enum family {self.enum_class.__name__} has no registered enum set")
        return set_builder(self.set_type, self.value_type)

    def literal(self, member_name: str):
        return enum_literal(self.value_type, self.abi_values[member_name])

    def state_payload(self):
        from numba_cfunc_compiler.types.native.state import nominal_copy_state_payload

        return nominal_copy_state_payload(
            self.key, self.value_type, self.storage_type, self.storage_type.bitwidth // 8, self.storage_type.bitwidth // 8
        )

    def set_state_payload(self):
        from numba_cfunc_compiler.types.native.state import nominal_copy_state_payload

        if self.set_type is None:
            raise TypeError(f"Enum family {self.enum_class.__name__} has no registered enum set")
        return nominal_copy_state_payload(self.key + ("set",), self.set_type, self.set_type, self.set_bits // 8, self.set_bits // 8)

    def method(self, name: str):
        from numba_cfunc_compiler.extension.methods import register_value_method

        def decorate(implementation):
            register_value_method(self.value_type, name, implementation)
            return implementation

        return decorate

    def overload_method(self, name: str):
        from numba_cfunc_compiler.extension.methods import register_value_overload

        def decorate(factory):
            register_value_overload(self.value_type, name, factory)
            return factory

        return decorate


_FAMILIES = {}


def register_enum_family(enum_class, *, abi_values, scalar_storage="int16", set_bits=128, semantics=ValueSemantics.COPY) -> EnumFamilyBinding:
    """Register one immutable enum ABI and its nominal Numba value types."""
    if not isinstance(semantics, ValueSemantics):
        raise TypeError("Enum semantics must be a ValueSemantics member")
    if semantics is not ValueSemantics.COPY:
        raise ValueError("Enum families have copied value semantics")
    storage_type = getattr(types, scalar_storage, None)
    if not isinstance(storage_type, types.Integer) or (set_bits is not None and (set_bits < 8 or set_bits % 8)):
        raise ValueError("Invalid enum storage width or enum-set bit width")
    names = {member.name for member in enum_class}
    if set(abi_values) != names:
        raise ValueError("Enum ABI values must cover every member exactly")
    values = tuple(sorted((name, int(value)) for name, value in abi_values.items()))
    storage_max = 2 ** (storage_type.bitwidth - int(storage_type.signed))
    storage_min = -(2 ** (storage_type.bitwidth - 1)) if storage_type.signed and set_bits is None else 0
    for _, value in values:
        if value < storage_min or value >= storage_max or (set_bits is not None and value >= set_bits):
            raise ValueError(f"Enum ABI value {value} is outside registered storage")
    family_key = (enum_class.__module__, enum_class.__qualname__, values, scalar_storage, set_bits)
    existing = _FAMILIES.get(enum_class)
    if existing is not None:
        if existing.key != family_key:
            raise ValueError(f"Enum family {enum_class} was already registered with a different ABI")
        return existing
    value_type = EnumValueType(family_key, storage_type) if set_bits is None else family_types(family_key, storage_type, set_bits)[0]
    set_type = None if set_bits is None else family_types(family_key, storage_type, set_bits)[1]
    binding = EnumFamilyBinding(enum_class, MappingProxyType(dict(values)), value_type, set_type, storage_type, set_bits)
    from numba_cfunc_compiler.extension.methods import forward_value_method

    if set_type is not None:
        forward_value_method(value_type, "isin", ("candidate_set",))
        forward_value_method(set_type, "contains_any", ("other",))
        forward_value_method(set_type, "contains_all", ("other",))
    _FAMILIES[enum_class] = binding
    return binding


def enum_binding(enum_class) -> EnumFamilyBinding:
    return _FAMILIES[enum_class]


def register_ast_handlers() -> None:
    """Lower registered enum literals and set constructors in callback syntax."""

    @ast_handler("Call", pre=True)
    def _enumset_call_handler(converter, node: ast.Call):
        if not isinstance(node.func, ast.Subscript) or not isinstance(node.func.value, ast.Name) or node.func.value.id != "EnumSet":
            return None
        if not isinstance(node.func.slice, ast.Name) or len(node.args) != 1 or not isinstance(node.args[0], ast.List) or node.keywords:
            raise TypeError("EnumSet[E] requires one list literal argument")
        enum_class = converter.host_globals.get(node.func.slice.id)
        if enum_class is None:
            raise TypeError(f"Unknown enum family '{node.func.slice.id}'")
        binding = enum_binding(enum_class)
        builder_name = enumset_builder_name(binding.key)
        converter.call_globals[builder_name] = binding.set_builder
        values = ast.Tuple(elts=[converter.visit(element) for element in node.args[0].elts], ctx=ast.Load())
        return AST.function_call(builder_name, values)

    @ast_handler("Attribute", pre=True)
    def _enum_literal_handler(converter, node: ast.Attribute):
        if not isinstance(node.value, ast.Name):
            return None
        enum_class = converter.host_globals.get(node.value.id)
        if enum_class is None:
            return None
        try:
            binding = enum_binding(enum_class)
        except KeyError:
            return None
        if node.attr not in binding.abi_values:
            return None
        literal_name = enum_literal_name(binding.key, node.attr)
        converter.call_globals[literal_name] = binding.literal(node.attr)
        return AST.function_call(literal_name)
