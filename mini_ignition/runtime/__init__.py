"""Runtime package for Mini Ignition.

This package contains the simplified accelerator runtime layer. In a real AI
accelerator stack, a runtime moves data to device memory, submits command
buffers or kernels, synchronizes execution, reads results back, and reports
execution metadata.

The concept matters for AI accelerator enablement because generated code is not
useful until it is executed and checked against observed device behavior.
Runtime code connects compiler output to hardware state.

This toy version runs a generated straight-line ISA program on `ToyDevice` and
uses NumPy as a trusted reference. It skips asynchronous queues, host/device
transfers, streams, events, memory allocators, and driver APIs.
"""

from mini_ignition.runtime.runner import (
    RunResult,
    RuntimeInputError,
    read_matrix_from_memory,
    run_matmul,
    write_matrix_to_memory,
)
from mini_ignition.runtime.controller import (
    ControllerResult,
    StrategyReport,
    run_controller,
)

__all__ = [
    "ControllerResult",
    "RunResult",
    "RuntimeInputError",
    "StrategyReport",
    "read_matrix_from_memory",
    "run_controller",
    "run_matmul",
    "write_matrix_to_memory",
]
