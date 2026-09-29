"""Names and lifecycle values used by generated cfunc code."""

__all__ = [
    "STATE_ARRAY_NAME",
    "TICKED_OUTPUTS_ARRAY_NAME",
    "OUTPUTS_ARRAY_NAME",
    "LIFECYCLE_PARAM_NAME",
    "LIFECYCLE_EXECUTE",
    "LIFECYCLE_START",
    "LIFECYCLE_STOP",
]

# Array name constants used in generated code
STATE_ARRAY_NAME = "state"

TICKED_OUTPUTS_ARRAY_NAME = "output_ticked"
OUTPUTS_ARRAY_NAME = "outputs"

# Lifecycle phase constants
LIFECYCLE_PARAM_NAME = "lifecycle_phase"
LIFECYCLE_EXECUTE = 0  # Normal execution
LIFECYCLE_START = 1  # Called once at node start
LIFECYCLE_STOP = 2  # Called once at node stop
