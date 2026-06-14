"""Toy performance metrics for Mini Ignition.

This file implements simple metrics for evaluating generated kernels. In a
real accelerator enablement stack, performance models compare observed kernel
behavior with expected or theoretical device capability.

The concept matters for AI accelerator enablement because correctness is only
the first gate. A kernel can compute the right answer and still be too slow to
be useful, so later stages need metrics that can reject poor implementations.

This toy version uses fixed instruction latencies from `HardwareSpec` and a
single ops-per-cycle estimate. It skips bandwidth limits, occupancy, caches,
pipeline effects, tensor-core utilization, launch overhead, and statistical
benchmarking.
"""

from __future__ import annotations

from mini_ignition.schemas.hw_spec import HardwareSpec


def estimate_theoretical_peak(spec: HardwareSpec) -> float:
    """Estimate peak arithmetic operations per cycle for the toy device."""

    dot_latency = spec.latencies.get("DOT")
    if spec.has_dot and dot_latency is not None and dot_latency > 0:
        return float((2 * spec.vector_width) / dot_latency)
    return 1.0


def compute_utilization(
    arithmetic_ops: int,
    cycles: int,
    peak_ops_per_cycle: float,
) -> float:
    """Compute observed utilization as a fraction of toy theoretical peak."""

    if arithmetic_ops < 0:
        raise ValueError(f"arithmetic_ops must be non-negative, got {arithmetic_ops}.")
    if cycles <= 0:
        raise ValueError(f"cycles must be positive, got {cycles}.")
    if peak_ops_per_cycle <= 0:
        raise ValueError(
            f"peak_ops_per_cycle must be positive, got {peak_ops_per_cycle}."
        )

    achieved_ops_per_cycle = arithmetic_ops / cycles
    return float(achieved_ops_per_cycle / peak_ops_per_cycle)


def passes_performance_gate(
    utilization: float,
    min_utilization: float = 0.1,
) -> bool:
    """Return whether a kernel clears the toy utilization threshold."""

    if min_utilization < 0:
        raise ValueError(f"min_utilization must be non-negative, got {min_utilization}.")
    return utilization >= min_utilization


__all__ = [
    "compute_utilization",
    "estimate_theoretical_peak",
    "passes_performance_gate",
]
