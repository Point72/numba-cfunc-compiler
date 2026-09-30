from numba_cfunc_compiler.core.analysis import InputCategory
from numba_cfunc_compiler.core.names import STATE_ARRAY_NAME
from numba_cfunc_compiler.extension.callback_components import (
    CallbackComponent,
    CallbackComponentId,
    CfuncParam,
    ComponentInputs,
    ComponentRegistry,
    MaterializationPhase,
)


class ConstantComponent(CallbackComponent):
    """Constants baked into the compiled code (no cfunc params)."""

    id = CallbackComponentId.CONSTANT
    order = -4
    # Constants must be materialized before lifecycle branching so opaque
    # constants like standalone lists/dicts are available in start/stop too.
    materialization_phase = MaterializationPhase.ALWAYS

    @property
    def cfunc_params(self):
        return []

    def create_variables(self, inputs: ComponentInputs, factory):
        from numba_cfunc_compiler.core.variable_access import ConstantAccess
        from numba_cfunc_compiler.types.factory import HostTypeFactory

        for name, (value, param_info) in inputs.input_analysis.get_params_by_category(InputCategory.CONSTANT).items():
            binding = HostTypeFactory.resolve(param_info.expected_type)
            factory.add_variable(
                ConstantAccess(binding, name, value),
                component=CallbackComponentId.CONSTANT,
            )
        return {}


class OutputComponent(CallbackComponent):
    """Output variables (outputs, output_ticked)."""

    id = CallbackComponentId.OUTPUT
    order = -3
    materialization_phase = MaterializationPhase.EXECUTE

    @property
    def cfunc_params(self):
        return [
            CfuncParam("outputs", "CPointer(voidptr)"),
            CfuncParam("output_ticked", "CPointer(int8)"),
        ]

    def create_variables(self, inputs: ComponentInputs, factory):
        from numba_cfunc_compiler.core.variable_access import OutputAccess
        from numba_cfunc_compiler.types.factory import HostTypeFactory

        for idx, (name, output_type) in enumerate(inputs.output_analysis.outputs.items()):
            factory.add_variable(
                OutputAccess(type=HostTypeFactory.resolve(output_type), name=name if name is not None else "output_0", array_idx=idx),
                component=CallbackComponentId.OUTPUT,
            )
        return {}


class StateComponent(CallbackComponent):
    """Map declared state names to host cell pointers in the state array."""

    id = CallbackComponentId.STATE
    order = -2
    materialization_phase = MaterializationPhase.ALWAYS

    @property
    def cfunc_params(self):
        return [
            CfuncParam("state", "CPointer(voidptr)"),
        ]

    def create_variables(self, inputs: ComponentInputs, factory):
        from numba_cfunc_compiler.core.variable_access import PointerSlotAccess

        sorted_state_vars = inputs.state_analysis.sorted_by_size()
        state_metadata = {
            "nrt_state_indices": [],
            "struct_state_indices": [],
            "struct_state_sizes": [],
        }
        for idx, state_var in enumerate(sorted_state_vars):
            var = PointerSlotAccess(
                array_idx=idx,
                type=state_var.binding,
                name=state_var.name,
                storage_location=STATE_ARRAY_NAME,
            )
            var.state_plan = state_var.binding.state_plan(var)
            factory.add_variable(var, component=CallbackComponentId.STATE)
            for key, values in var.state_plan.metadata.items():
                state_metadata[key].extend(values)
        return {
            "state_values": tuple(state_var.initial_value for state_var in sorted_state_vars),
            **{key: tuple(values) for key, values in state_metadata.items()},
        }


class LifecycleComponent(CallbackComponent):
    """Lifecycle phase scalar (always present)."""

    id = CallbackComponentId.LIFECYCLE
    order = -1
    materialization_phase = MaterializationPhase.NONE

    @property
    def cfunc_params(self):
        return [
            CfuncParam("lifecycle_phase", "int8"),
        ]

    def create_variables(self, inputs: ComponentInputs, factory):
        return {}  # handled by the AST converter directly


def register_default_components() -> None:
    ComponentRegistry.register(ConstantComponent())
    ComponentRegistry.register(OutputComponent())
    ComponentRegistry.register(StateComponent())
    ComponentRegistry.register(LifecycleComponent())
