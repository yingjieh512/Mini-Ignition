import numpy as np
import pytest

from mini_ignition.codegen import MatmulProblem
from mini_ignition.probe.characterize import characterize_device
from mini_ignition.runtime import (
    RuntimeInputError,
    read_matrix_from_memory,
    run_matmul,
    write_matrix_to_memory,
)
from mini_ignition.simulator import ToyDevice


def test_matrix_memory_helpers_round_trip_row_major_aligned_layout():
    device = ToyDevice()
    matrix = np.arange(1, 7, dtype=np.float32).reshape(2, 3)

    write_matrix_to_memory(device, matrix, base_addr=16, alignment=4)
    loaded = read_matrix_from_memory(device, (2, 3), base_addr=16, alignment=4)

    np.testing.assert_allclose(loaded, matrix)


def test_run_matmul_works_for_scalar_naive():
    result = _run_strategy("scalar_naive")

    _assert_successful_result(result)


def test_run_matmul_works_for_vector_dot():
    result = _run_strategy("vector_dot")

    _assert_successful_result(result)


def test_run_matmul_works_for_tiled_vector_dot():
    result = _run_strategy("tiled_vector_dot")

    _assert_successful_result(result)


def test_run_result_contains_arithmetic_ops():
    result = _run_strategy("scalar_naive")

    assert result.arithmetic_ops > 0
    assert result.achieved_ops_per_cycle > 0.0


def test_invalid_a_shape_raises_clear_error():
    device = ToyDevice()
    spec = characterize_device(device)
    problem = MatmulProblem(M=4, N=4, K=8, addr_A=0, addr_B=1024, addr_C=2048)
    bad_a = np.arange(1, 29, dtype=np.float32).reshape(4, 7)
    b = _matrix_b()

    with pytest.raises(RuntimeInputError, match="A shape"):
        run_matmul(device, problem, spec, "scalar_naive", bad_a, b)


def test_invalid_b_shape_raises_clear_error():
    device = ToyDevice()
    spec = characterize_device(device)
    problem = MatmulProblem(M=4, N=4, K=8, addr_A=0, addr_B=1024, addr_C=2048)
    a = _matrix_a()
    bad_b = np.arange(1, 33, dtype=np.float32).reshape(8, 4)[:, :3]

    with pytest.raises(RuntimeInputError, match="B shape"):
        run_matmul(device, problem, spec, "scalar_naive", a, bad_b)


def _run_strategy(strategy: str):
    device = ToyDevice()
    spec = characterize_device(device)
    problem = MatmulProblem(M=4, N=4, K=8, addr_A=0, addr_B=1024, addr_C=2048)

    return run_matmul(device, problem, spec, strategy, _matrix_a(), _matrix_b())


def _matrix_a() -> np.ndarray:
    return (np.arange(1, 33, dtype=np.float32).reshape(4, 8) / 7.0).astype(np.float32)


def _matrix_b() -> np.ndarray:
    return (np.arange(1, 33, dtype=np.float32).reshape(8, 4) / 5.0).astype(np.float32)


def _assert_successful_result(result) -> None:
    assert result.passed is True
    assert result.max_abs_error <= 1e-4
    assert result.cycles > 0
    assert result.instructions_executed > 0
    assert result.arithmetic_ops >= 0
    np.testing.assert_allclose(result.output, result.expected, atol=1e-4)
