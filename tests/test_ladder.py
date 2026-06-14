from mini_ignition.probe.characterize import characterize_device
from mini_ignition.runtime.ladder import run_test_ladder
from mini_ignition.simulator import ToyDevice


EXPECTED_GATE_NAMES = [
    "scalar instruction correctness",
    "memory correctness",
    "vector instruction correctness",
    "DOT correctness",
    "fixed-shape matmul correctness",
    "randomized matmul correctness",
    "performance threshold",
]


def test_run_test_ladder_passes_on_default_toy_device():
    spec = characterize_device(ToyDevice())

    result = run_test_ladder(ToyDevice, spec)

    assert result.passed is True
    assert all(gate.passed for gate in result.gates)


def test_ladder_returns_ordered_gate_results():
    spec = characterize_device(ToyDevice())

    result = run_test_ladder(ToyDevice, spec)

    assert [gate.name for gate in result.gates] == EXPECTED_GATE_NAMES


def test_ladder_stops_on_failure():
    spec = characterize_device(ToyDevice())

    def broken_device_factory():
        return ToyDevice(num_scalar_registers=1)

    result = run_test_ladder(broken_device_factory, spec)

    assert result.passed is False
    assert len(result.gates) == 1
    assert result.gates[0].name == "scalar instruction correctness"
    assert result.gates[0].passed is False
