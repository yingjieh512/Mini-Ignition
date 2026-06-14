import numpy as np

from mini_ignition.codegen import MatmulProblem
from mini_ignition.probe.characterize import characterize_device
from mini_ignition.runtime.controller import ControllerResult, run_controller
from mini_ignition.simulator import ToyDevice


def test_run_controller_returns_controller_result():
    result = _run_default_controller()

    assert isinstance(result, ControllerResult)


def test_controller_tries_multiple_strategies():
    result = _run_default_controller()

    assert [report.strategy for report in result.reports] == [
        "scalar_naive",
        "vector_dot",
        "tiled_vector_dot",
    ]


def test_controller_selects_a_passing_strategy_on_default_device():
    result = _run_default_controller()

    assert result.passed is True
    assert result.selected_strategy is not None
    selected = next(
        report for report in result.reports if report.strategy == result.selected_strategy
    )
    assert selected.correctness_passed is True
    assert selected.performance_passed is True


def test_reports_contain_correctness_and_performance_info():
    result = _run_default_controller()

    for report in result.reports:
        assert isinstance(report.correctness_passed, bool)
        assert isinstance(report.performance_passed, bool)
        assert report.max_abs_error >= 0.0
        assert report.theoretical_peak_ops_per_cycle > 0.0
        assert report.utilization >= 0.0
        assert report.message


def test_controller_handles_invalid_strategy_gracefully():
    device = ToyDevice()
    spec = characterize_device(device)
    problem = MatmulProblem(M=4, N=4, K=8, addr_A=0, addr_B=1024, addr_C=2048)

    result = run_controller(
        ToyDevice,
        spec,
        problem,
        _matrix_a(),
        _matrix_b(),
        strategies=["not_a_strategy", "vector_dot"],
    )

    assert result.passed is True
    assert result.selected_strategy == "vector_dot"
    assert result.reports[0].strategy == "not_a_strategy"
    assert result.reports[0].correctness_passed is False
    assert result.reports[0].performance_passed is False
    assert "failed to run" in result.reports[0].message


def _run_default_controller() -> ControllerResult:
    device = ToyDevice()
    spec = characterize_device(device)
    problem = MatmulProblem(M=4, N=4, K=8, addr_A=0, addr_B=1024, addr_C=2048)
    return run_controller(ToyDevice, spec, problem, _matrix_a(), _matrix_b())


def _matrix_a() -> np.ndarray:
    return (np.arange(1, 33, dtype=np.float32).reshape(4, 8) / 7.0).astype(np.float32)


def _matrix_b() -> np.ndarray:
    return (np.arange(1, 33, dtype=np.float32).reshape(8, 4) / 5.0).astype(np.float32)
