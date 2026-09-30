"""Datetime output expressions through the host callback ABI."""

from datetime import datetime

from numba_cfunc_compiler.node_api import set_output
from numba_cfunc_compiler.tests.harness import CompiledNode, Signal, compile_function, numba_node


@numba_node
def shift_timestamp(stamp: Signal[datetime]) -> Signal[datetime]:
    return stamp + 1_000_000_000


@numba_node
def set_shifted_timestamp(stamp: Signal[datetime]) -> Signal[datetime]:
    set_output("output_0", stamp + 2_000_000_000)


def test_datetime_output_expressions_use_numba_values():
    # The callback ABI stores datetimes as int64 nanoseconds. Feed the raw
    # storage value through the C callback driver to exercise both output paths.
    stamp = 1_700_000_000_000_000_000
    returned = CompiledNode(compile_function(shift_timestamp), [datetime])
    named = CompiledNode(compile_function(set_shifted_timestamp), [datetime])
    assert returned.execute([stamp]) == (stamp + 1_000_000_000, True)
    assert named.execute([stamp]) == (stamp + 2_000_000_000, True)
