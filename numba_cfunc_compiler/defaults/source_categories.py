from numba_cfunc_compiler.numba_config import STATE_ARRAY_NAME
from numba_cfunc_compiler.source_registry import (
    CfuncParam,
    SourceCategory,
    SourceCategoryId,
    SourceInitFilter,
    SourceRegistry,
)


class ConstantCategory(SourceCategory):
    """Constants baked into the compiled code (no cfunc params)."""

    id = SourceCategoryId.CONSTANT
    order = -4
    # Constants must be materialized before lifecycle branching so they are available in start/stop too.
    init_filter = SourceInitFilter.EVERYTIME

    @property
    def cfunc_params(self):
        return []

    def create_variables(self, info, factory):
        from numba_cfunc_compiler.type_factory import TypeFactory
        from numba_cfunc_compiler.variable_factory import ConstantSource

        for name, (value, param_info) in info.input_analysis.get_params_by_category("constant").items():
            var_type = TypeFactory.get_type(param_info.expected_type, value)
            factory.add_variable(
                ConstantSource(var_type, name),
                category=SourceCategoryId.CONSTANT,
            )


class ConstantContainerCategory(SourceCategory):
    """Constant containers backed by compiler-owned state slots."""

    id = SourceCategoryId.CONSTANT_CONTAINER
    order = -5
    init_filter = SourceInitFilter.EVERYTIME

    def create_variables(self, info, factory):
        from numba_cfunc_compiler.models import CONTAINER_STATE_INIT
        from numba_cfunc_compiler.type_factory import TypeFactory
        from numba_cfunc_compiler.variable_factory import VoidPtrSource

        for name, (value, param_info) in info.input_analysis.get_params_by_category("constant_container").items():
            var_type = TypeFactory.get_type(param_info.expected_type, value)
            var = VoidPtrSource(array_idx=len(info.state_slots), type=var_type, name=name, storage_location=STATE_ARRAY_NAME)
            info.state_slots.append((CONTAINER_STATE_INIT, var_type))
            factory.add_variable(var, category=SourceCategoryId.CONSTANT_CONTAINER)


class OutputCategory(SourceCategory):
    """Output variables (outputs, output_ticked)."""

    id = SourceCategoryId.OUTPUT
    order = -3
    init_filter = SourceInitFilter.ON_EXECUTE

    @property
    def cfunc_params(self):
        return [
            CfuncParam("outputs", "CPointer(voidptr)"),
            CfuncParam("output_ticked", "CPointer(int8)"),
        ]

    def create_variables(self, info, factory):
        from numba_cfunc_compiler.type_factory import TypeFactory
        from numba_cfunc_compiler.variable_factory import OutputSource

        if info.output_analysis.named_outputs is not None:
            for idx, (name, output_type) in enumerate(info.output_analysis.named_outputs.items()):
                var_type = TypeFactory.get_type(output_type)
                factory.add_variable(
                    OutputSource(type=var_type, name=name, array_idx=idx),
                    category=SourceCategoryId.OUTPUT,
                )
        else:
            var_type = TypeFactory.get_type(info.output_analysis.output_types[0])
            factory.add_variable(
                OutputSource(type=var_type, name="output_0", array_idx=0),
                category=SourceCategoryId.OUTPUT,
            )


class StateCategory(SourceCategory):
    """State variables (state array)."""

    id = SourceCategoryId.STATE
    order = -2
    init_filter = SourceInitFilter.EVERYTIME

    @property
    def cfunc_params(self):
        return [
            CfuncParam("state", "CPointer(voidptr)"),
        ]

    def create_variables(self, info, factory):
        from numba_cfunc_compiler.type_factory import TypeFactory
        from numba_cfunc_compiler.variable_factory import VoidPtrSource

        for state_var in info.state_analysis.sorted_by_size():
            var_type = TypeFactory.get_type(state_var.state_type)
            var = VoidPtrSource(
                array_idx=len(info.state_slots),
                type=var_type,
                name=state_var.name,
                storage_location=STATE_ARRAY_NAME,
            )
            info.state_slots.append((state_var.initial_value, var_type))
            factory.add_variable(var, category=SourceCategoryId.STATE)


class LifecycleCategory(SourceCategory):
    """Lifecycle phase scalar (always present)."""

    id = SourceCategoryId.LIFECYCLE
    order = -1
    init_filter = SourceInitFilter.NEVER

    @property
    def cfunc_params(self):
        return [
            CfuncParam("lifecycle_phase", "int8"),
        ]

    def create_variables(self, info, factory):
        pass  # handled by the AST converter directly


def register_default_categories() -> None:
    SourceRegistry.register(ConstantContainerCategory())
    SourceRegistry.register(ConstantCategory())
    SourceRegistry.register(OutputCategory())
    SourceRegistry.register(StateCategory())
    SourceRegistry.register(LifecycleCategory())
