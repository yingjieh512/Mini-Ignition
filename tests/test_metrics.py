import pytest

from mini_ignition.probe.characterize import characterize_device
from mini_ignition.runtime.metrics import (
    compute_utilization,
    estimate_theoretical_peak,
    passes_performance_gate,
)
from mini_ignition.simulator import ToyDevice


def test_estimate_theoretical_peak_returns_positive_value():
    spec = characterize_device(ToyDevice())

    assert estimate_theoretical_peak(spec) > 0.0


def test_compute_utilization_works():
    utilization = compute_utilization(
        arithmetic_ops=50,
        cycles=10,
        peak_ops_per_cycle=10.0,
    )

    assert utilization == 0.5


def test_compute_utilization_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="cycles"):
        compute_utilization(arithmetic_ops=10, cycles=0, peak_ops_per_cycle=1.0)


def test_passes_performance_gate_works():
    assert passes_performance_gate(0.2, min_utilization=0.1) is True
    assert passes_performance_gate(0.05, min_utilization=0.1) is False
