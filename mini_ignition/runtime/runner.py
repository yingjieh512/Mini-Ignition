"""Runtime runner and correctness evaluation for Mini Ignition.

This file implements Stage 4: a tiny runtime that loads matrices into toy
device memory, runs generated matmul code, reads the output matrix, and checks
it against NumPy.

The concept represents a simplified accelerator runtime. Real runtimes manage
device memory, command submission, synchronization, result readback, and
profiling so generated kernels can be validated and measured on hardware.

The concept matters for AI accelerator enablement because code generation alone
is not enough. A generated kernel must execute correctly on the target device,
match a reference implementation, and produce performance counters worth
inspecting.

This toy version is synchronous and single-device. It resets the simulator,
writes `float32` matrices into flat aligned memory, runs one straight-line
program, reads the result, and reports toy cycle/instruction/arithmetic counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mini_ignition.codegen import MatmulProblem, generate_matmul_program
from mini_ignition.schemas.hw_spec import HardwareSpec


class RuntimeInputError(ValueError):
    """Raised when runtime inputs are invalid for the requested matmul run."""


@dataclass
class RunResult:
    """Outcome of one generated matmul program execution."""

    strategy: str
    passed: bool
    max_abs_error: float
    cycles: int
    instructions_executed: int
    arithmetic_ops: int
    achieved_ops_per_cycle: float
    output: np.ndarray
    expected: np.ndarray


def write_matrix_to_memory(
    device: Any,
    matrix: np.ndarray,
    base_addr: int,
    alignment: int,
) -> None:
    """Write a 2D matrix into device memory using row-major aligned layout."""

    matrix = _as_float32_matrix(matrix, "matrix")
    _validate_base_and_alignment(base_addr, alignment)

    rows, cols = matrix.shape
    for row in range(rows):
        for col in range(cols):
            address = base_addr + (row * cols + col) * alignment
            device.memory.write_scalar(address, float(matrix[row, col]))


def read_matrix_from_memory(
    device: Any,
    shape: tuple[int, int],
    base_addr: int,
    alignment: int,
) -> np.ndarray:
    """Read a 2D matrix from device memory using row-major aligned layout."""

    rows, cols = _validate_shape_tuple(shape)
    _validate_base_and_alignment(base_addr, alignment)

    result = np.zeros((rows, cols), dtype=np.float32)
    for row in range(rows):
        for col in range(cols):
            address = base_addr + (row * cols + col) * alignment
            result[row, col] = device.memory.read_scalar(address)
    return result


def run_matmul(
    device: Any,
    problem: MatmulProblem,
    spec: HardwareSpec,
    strategy: str,
    A: np.ndarray,
    B: np.ndarray,
    atol: float = 1e-4,
) -> RunResult:
    """Load inputs, run generated matmul code, and compare against NumPy."""

    if atol < 0:
        raise RuntimeInputError(f"atol must be non-negative, got {atol}.")

    a_matrix = _as_float32_matrix(A, "A")
    b_matrix = _as_float32_matrix(B, "B")
    _validate_problem_matches_inputs(problem, a_matrix, b_matrix)

    device.reset(clear_memory=True)
    write_matrix_to_memory(device, a_matrix, problem.addr_A, spec.alignment)
    write_matrix_to_memory(device, b_matrix, problem.addr_B, spec.alignment)

    program = generate_matmul_program(problem, spec, strategy)
    device.run(program)

    output = read_matrix_from_memory(
        device,
        (problem.M, problem.N),
        problem.addr_C,
        spec.alignment,
    )
    expected = (a_matrix @ b_matrix).astype(np.float32)
    abs_error = np.abs(output - expected)
    max_abs_error = float(np.max(abs_error))
    passed = bool(np.allclose(output, expected, atol=atol, rtol=0.0))

    cycles = int(device.cycle_count)
    arithmetic_ops = int(device.arithmetic_op_count)
    achieved_ops_per_cycle = arithmetic_ops / cycles if cycles > 0 else 0.0

    return RunResult(
        strategy=strategy,
        passed=passed,
        max_abs_error=max_abs_error,
        cycles=cycles,
        instructions_executed=int(device.instruction_count),
        arithmetic_ops=arithmetic_ops,
        achieved_ops_per_cycle=float(achieved_ops_per_cycle),
        output=output,
        expected=expected,
    )


def _as_float32_matrix(value: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.ndim != 2:
        raise RuntimeInputError(f"{name} must be a 2D matrix, got shape {matrix.shape}.")
    if matrix.shape[0] <= 0 or matrix.shape[1] <= 0:
        raise RuntimeInputError(
            f"{name} must have positive dimensions, got {matrix.shape}."
        )
    return matrix


def _validate_problem_matches_inputs(
    problem: MatmulProblem,
    A: np.ndarray,
    B: np.ndarray,
) -> None:
    if A.shape != (problem.M, problem.K):
        raise RuntimeInputError(
            f"A shape {A.shape} does not match problem dimensions "
            f"(M={problem.M}, K={problem.K})."
        )
    if B.shape != (problem.K, problem.N):
        raise RuntimeInputError(
            f"B shape {B.shape} does not match problem dimensions "
            f"(K={problem.K}, N={problem.N})."
        )
    if A.shape[1] != B.shape[0]:
        raise RuntimeInputError(
            f"Cannot multiply A shape {A.shape} by B shape {B.shape}: "
            "inner dimensions differ."
        )


def _validate_shape_tuple(shape: tuple[int, int]) -> tuple[int, int]:
    if len(shape) != 2:
        raise RuntimeInputError(
            f"shape must contain exactly two dimensions, got {shape}."
        )
    rows, cols = shape
    if rows <= 0 or cols <= 0:
        raise RuntimeInputError(f"shape dimensions must be positive, got {shape}.")
    return rows, cols


def _validate_base_and_alignment(base_addr: int, alignment: int) -> None:
    if base_addr < 0:
        raise RuntimeInputError(f"base_addr must be non-negative, got {base_addr}.")
    if alignment <= 0:
        raise RuntimeInputError(f"alignment must be positive, got {alignment}.")
    if base_addr % alignment != 0:
        raise RuntimeInputError(
            f"base_addr {base_addr} must be aligned to {alignment} memory slots."
        )


__all__ = [
    "RunResult",
    "RuntimeInputError",
    "read_matrix_from_memory",
    "run_matmul",
    "write_matrix_to_memory",
]
