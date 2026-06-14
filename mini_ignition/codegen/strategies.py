"""Matmul codegen strategy registry.

This file implements the small list of matmul generation strategies supported
by Stage 3. In real compiler backends, strategy selection can involve hardware
features, problem shape, memory layout, scheduling, and performance models.

The concept matters for AI accelerator enablement because different hardware
targets and matrix shapes often need different lowering decisions. A portable
baseline may be correct everywhere, while vectorized or tiled kernels depend on
observed target capabilities.

This toy version uses string names and simple validation. It skips cost models,
autotuning, dynamic dispatch, multi-kernel plans, and controller-driven
selection.
"""

from __future__ import annotations


SCALAR_NAIVE = "scalar_naive"
VECTOR_DOT = "vector_dot"
TILED_VECTOR_DOT = "tiled_vector_dot"

_STRATEGIES = (SCALAR_NAIVE, VECTOR_DOT, TILED_VECTOR_DOT)


def available_strategies() -> list[str]:
    """Return supported matmul codegen strategy names."""

    return list(_STRATEGIES)


def validate_strategy(strategy: str) -> str:
    """Return `strategy` if supported, otherwise raise a clear error."""

    if strategy not in _STRATEGIES:
        options = ", ".join(_STRATEGIES)
        raise ValueError(f"Unknown matmul strategy {strategy!r}. Available: {options}.")
    return strategy


__all__ = [
    "SCALAR_NAIVE",
    "TILED_VECTOR_DOT",
    "VECTOR_DOT",
    "available_strategies",
    "validate_strategy",
]
