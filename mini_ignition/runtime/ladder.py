"""Staged correctness and performance ladder for Mini Ignition.

This file implements a verification ladder: a sequence of gates that exercise
increasingly high-level behavior before trusting generated matmul kernels. In
real accelerator bring-up, teams validate basic instruction and memory behavior
before relying on compilers, runtimes, and performance claims.

The concept matters for AI accelerator enablement because failures should be
localized. If scalar arithmetic is broken, a matmul failure is not a codegen
problem yet. Each layer must pass before higher layers depend on it.

This toy version runs deterministic scalar, memory, vector, DOT, matmul, random
matmul, and performance checks. It stops on the first failure and returns a
small structured result instead of a full test framework report.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from mini_ignition.codegen import MatmulProblem
from mini_ignition.runtime.metrics import (
    compute_utilization,
    estimate_theoretical_peak,
    passes_performance_gate,
)
from mini_ignition.runtime.runner import run_matmul
from mini_ignition.schemas.hw_spec import HardwareSpec
from mini_ignition.simulator import (
    ADD,
    DOT,
    HALT,
    MOV,
    MUL,
    STORE,
    VADD,
    VLOAD,
    VMUL,
    VSTORE,
    Program,
)


@dataclass(frozen=True)
class GateResult:
    """Result for one verification ladder gate."""

    name: str
    passed: bool
    message: str


@dataclass(frozen=True)
class LadderResult:
    """Result for the full staged verification ladder."""

    passed: bool
    gates: list[GateResult]


def run_test_ladder(
    device_factory: Callable[[], Any],
    spec: HardwareSpec,
) -> LadderResult:
    """Run staged checks and stop on the first failed gate."""

    gates: list[GateResult] = []
    checks = [
        ("scalar instruction correctness", lambda: _gate_scalar(device_factory)),
        ("memory correctness", lambda: _gate_memory(device_factory)),
        ("vector instruction correctness", lambda: _gate_vector(device_factory)),
        ("DOT correctness", lambda: _gate_dot(device_factory)),
        ("fixed-shape matmul correctness", lambda: _gate_fixed_matmul(device_factory, spec)),
        (
            "randomized matmul correctness",
            lambda: _gate_randomized_matmul(device_factory, spec),
        ),
        ("performance threshold", lambda: _gate_performance(device_factory, spec)),
    ]

    for name, check in checks:
        try:
            message = check()
        except Exception as exc:
            gates.append(GateResult(name=name, passed=False, message=str(exc)))
            return LadderResult(passed=False, gates=gates)
        gates.append(GateResult(name=name, passed=True, message=message))

    return LadderResult(passed=True, gates=gates)


def _gate_scalar(device_factory: Callable[[], Any]) -> str:
    device = device_factory()
    device.run(
        Program.from_instructions(
            MOV("r0", 2.0),
            MOV("r1", 3.0),
            ADD("r2", "r0", "r1"),
            MUL("r3", "r2", "r1"),
            HALT(),
        )
    )

    _assert_close(device.read_scalar_register("r2"), 5.0, "ADD result")
    _assert_close(device.read_scalar_register("r3"), 15.0, "MUL result")
    if not device.halted:
        raise AssertionError("HALT did not stop scalar program.")
    return "Scalar MOV/ADD/MUL/HALT passed."


def _gate_memory(device_factory: Callable[[], Any]) -> str:
    device = device_factory()
    width = device.config.vector_width
    vector_addr = _align_up(1, device.config.alignment)
    device.memory.write_scalar(0, 7.0)
    _assert_close(device.memory.read_scalar(0), 7.0, "scalar memory round trip")

    values = np.arange(1, width + 1, dtype=np.float32)
    device.memory.write_vector(vector_addr, values)
    np.testing.assert_allclose(device.memory.read_vector(vector_addr, width), values)
    return "Scalar and vector memory round trips passed."


def _gate_vector(device_factory: Callable[[], Any]) -> str:
    device = device_factory()
    width = device.config.vector_width
    addr_a = 0
    addr_b = _align_up(width, device.config.alignment)
    addr_out = _align_up(addr_b + width, device.config.alignment)
    values_a = np.arange(1, width + 1, dtype=np.float32)
    values_b = np.arange(10, 10 + width, dtype=np.float32)

    device.memory.write_vector(addr_a, values_a)
    device.memory.write_vector(addr_b, values_b)
    device.run(
        Program.from_instructions(
            VLOAD("v0", addr_a),
            VLOAD("v1", addr_b),
            VADD("v2", "v0", "v1"),
            VMUL("v3", "v0", "v1"),
            VSTORE("v2", addr_out),
            HALT(),
        )
    )

    np.testing.assert_allclose(device.read_vector_register("v2"), values_a + values_b)
    np.testing.assert_allclose(device.read_vector_register("v3"), values_a * values_b)
    np.testing.assert_allclose(device.memory.read_vector(addr_out, width), values_a + values_b)
    return "Vector VLOAD/VADD/VMUL/VSTORE passed."


def _gate_dot(device_factory: Callable[[], Any]) -> str:
    device = device_factory()
    width = device.config.vector_width
    addr_b = _align_up(width, device.config.alignment)
    values_a = np.arange(1, width + 1, dtype=np.float32)
    values_b = np.arange(2, 2 + width, dtype=np.float32)

    device.memory.write_vector(0, values_a)
    device.memory.write_vector(addr_b, values_b)
    device.run(
        Program.from_instructions(
            VLOAD("v0", 0),
            VLOAD("v1", addr_b),
            DOT("r0", "v0", "v1"),
            HALT(),
        )
    )

    _assert_close(device.read_scalar_register("r0"), float(values_a @ values_b), "DOT result")
    return "DOT passed."


def _gate_fixed_matmul(
    device_factory: Callable[[], Any],
    spec: HardwareSpec,
) -> str:
    problem = MatmulProblem(M=4, N=4, K=spec.vector_width, addr_A=0, addr_B=1024, addr_C=2048)
    result = run_matmul(
        device_factory(),
        problem,
        spec,
        "scalar_naive",
        _fixed_matrix_a(problem.M, problem.K),
        _fixed_matrix_b(problem.K, problem.N),
    )
    if not result.passed:
        raise AssertionError(f"fixed matmul failed with max_abs_error={result.max_abs_error}.")
    return "Fixed-shape scalar matmul matched NumPy."


def _gate_randomized_matmul(
    device_factory: Callable[[], Any],
    spec: HardwareSpec,
) -> str:
    rng = np.random.default_rng(0)
    problem = MatmulProblem(M=3, N=3, K=spec.vector_width, addr_A=0, addr_B=1024, addr_C=2048)
    a_matrix = rng.normal(size=(problem.M, problem.K)).astype(np.float32)
    b_matrix = rng.normal(size=(problem.K, problem.N)).astype(np.float32)
    strategy = _best_vector_strategy(spec)
    result = run_matmul(device_factory(), problem, spec, strategy, a_matrix, b_matrix)
    if not result.passed:
        raise AssertionError(
            f"randomized matmul failed with max_abs_error={result.max_abs_error}."
        )
    return f"Randomized {strategy} matmul matched NumPy."


def _gate_performance(
    device_factory: Callable[[], Any],
    spec: HardwareSpec,
) -> str:
    problem = MatmulProblem(M=4, N=4, K=spec.vector_width, addr_A=0, addr_B=1024, addr_C=2048)
    strategy = _best_vector_strategy(spec)
    result = run_matmul(
        device_factory(),
        problem,
        spec,
        strategy,
        _fixed_matrix_a(problem.M, problem.K),
        _fixed_matrix_b(problem.K, problem.N),
    )
    if not result.passed:
        raise AssertionError("Performance gate requires a correct matmul result first.")

    peak = estimate_theoretical_peak(spec)
    utilization = compute_utilization(result.arithmetic_ops, result.cycles, peak)
    if not passes_performance_gate(utilization):
        raise AssertionError(
            f"utilization {utilization:.3f} is below threshold 0.100 "
            f"for strategy {strategy}."
        )
    return f"{strategy} utilization {utilization:.3f} passed performance gate."


def _best_vector_strategy(spec: HardwareSpec) -> str:
    if spec.has_vector_ops and spec.has_dot:
        return "tiled_vector_dot"
    return "scalar_naive"


def _fixed_matrix_a(rows: int, cols: int) -> np.ndarray:
    return (np.arange(1, rows * cols + 1, dtype=np.float32).reshape(rows, cols) / 7.0)


def _fixed_matrix_b(rows: int, cols: int) -> np.ndarray:
    return (np.arange(1, rows * cols + 1, dtype=np.float32).reshape(rows, cols) / 5.0)


def _align_up(value: int, alignment: int) -> int:
    remainder = value % alignment
    if remainder == 0:
        return value
    return value + alignment - remainder


def _assert_close(observed: float, expected: float, label: str) -> None:
    if not np.isclose(observed, expected):
        raise AssertionError(f"{label}: expected {expected}, got {observed}.")


__all__ = ["GateResult", "LadderResult", "run_test_ladder"]
