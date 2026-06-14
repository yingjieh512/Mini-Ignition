"""Controller loop for Mini Ignition strategy selection.

This file implements Stage 6: a simple controller that tries multiple matmul
codegen strategies, rejects candidates that fail correctness or performance
gates, and selects the best surviving strategy.

The concept represents a toy version of an agent/controller loop. In a real
Ignition-like hardware enablement system, an LLM coding agent may propose
candidate kernels or edits, but a controller, test ladder, reference
implementation, and performance gates decide what is accepted.

The concept matters for AI accelerator enablement because generated code should
not be trusted directly. The controller turns candidate strategies into
measured evidence: correctness, error, cycles, arithmetic operations, achieved
throughput, theoretical peak, and utilization.

This toy version only selects among built-in matmul strategies for `ToyDevice`.
It does not call real LLMs, edit source code, launch CUDA/Triton kernels,
manage distributed jobs, or persist experiment history.
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


DEFAULT_STRATEGIES = ["scalar_naive", "vector_dot", "tiled_vector_dot"]


@dataclass(frozen=True)
class StrategyReport:
    """Measured result for one candidate matmul strategy."""

    strategy: str
    correctness_passed: bool
    performance_passed: bool
    max_abs_error: float
    cycles: int
    arithmetic_ops: int
    achieved_ops_per_cycle: float
    theoretical_peak_ops_per_cycle: float
    utilization: float
    message: str


@dataclass(frozen=True)
class ControllerResult:
    """Final controller decision and all candidate reports."""

    selected_strategy: str | None
    reports: list[StrategyReport]
    passed: bool


def run_controller(
    device_factory: Callable[[], Any],
    spec: HardwareSpec,
    problem: MatmulProblem,
    A: np.ndarray,
    B: np.ndarray,
    strategies: list[str] | None = None,
    min_utilization: float = 0.1,
) -> ControllerResult:
    """Try candidate strategies and select the best passing implementation."""

    candidate_strategies = strategies if strategies is not None else DEFAULT_STRATEGIES
    reports: list[StrategyReport] = []
    selected_strategy: str | None = None
    selected_utilization = float("-inf")

    for strategy in candidate_strategies:
        report = _evaluate_strategy(
            device_factory,
            spec,
            problem,
            A,
            B,
            strategy,
            min_utilization,
        )
        reports.append(report)

        if (
            report.correctness_passed
            and report.performance_passed
            and report.utilization >= selected_utilization
        ):
            selected_strategy = strategy
            selected_utilization = report.utilization

    return ControllerResult(
        selected_strategy=selected_strategy,
        reports=reports,
        passed=selected_strategy is not None,
    )


def _evaluate_strategy(
    device_factory: Callable[[], Any],
    spec: HardwareSpec,
    problem: MatmulProblem,
    A: np.ndarray,
    B: np.ndarray,
    strategy: str,
    min_utilization: float,
) -> StrategyReport:
    theoretical_peak = estimate_theoretical_peak(spec)

    try:
        run_result = run_matmul(device_factory(), problem, spec, strategy, A, B)
    except Exception as exc:
        return StrategyReport(
            strategy=strategy,
            correctness_passed=False,
            performance_passed=False,
            max_abs_error=float("inf"),
            cycles=0,
            arithmetic_ops=0,
            achieved_ops_per_cycle=0.0,
            theoretical_peak_ops_per_cycle=theoretical_peak,
            utilization=0.0,
            message=f"Rejected: strategy failed to run: {exc}",
        )

    if not run_result.passed:
        return StrategyReport(
            strategy=strategy,
            correctness_passed=False,
            performance_passed=False,
            max_abs_error=run_result.max_abs_error,
            cycles=run_result.cycles,
            arithmetic_ops=run_result.arithmetic_ops,
            achieved_ops_per_cycle=run_result.achieved_ops_per_cycle,
            theoretical_peak_ops_per_cycle=theoretical_peak,
            utilization=0.0,
            message=(
                "Rejected: correctness failed "
                f"(max_abs_error={run_result.max_abs_error:.6g})."
            ),
        )

    utilization = compute_utilization(
        run_result.arithmetic_ops,
        run_result.cycles,
        theoretical_peak,
    )
    performance_passed = passes_performance_gate(utilization, min_utilization)
    if performance_passed:
        message = (
            "Accepted: correctness and performance gates passed "
            f"(utilization={utilization:.3f})."
        )
    else:
        message = (
            "Rejected: correctness passed but performance gate failed "
            f"(utilization={utilization:.3f}, threshold={min_utilization:.3f})."
        )

    return StrategyReport(
        strategy=strategy,
        correctness_passed=True,
        performance_passed=performance_passed,
        max_abs_error=run_result.max_abs_error,
        cycles=run_result.cycles,
        arithmetic_ops=run_result.arithmetic_ops,
        achieved_ops_per_cycle=run_result.achieved_ops_per_cycle,
        theoretical_peak_ops_per_cycle=theoretical_peak,
        utilization=utilization,
        message=message,
    )


__all__ = [
    "ControllerResult",
    "DEFAULT_STRATEGIES",
    "StrategyReport",
    "run_controller",
]
