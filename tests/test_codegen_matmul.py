import numpy as np
import pytest

from mini_ignition.codegen import (
    CodegenError,
    MatmulProblem,
    available_strategies,
    generate_matmul_program,
    pretty_print_program,
)
from mini_ignition.probe.characterize import characterize_device
from mini_ignition.simulator import Opcode, ToyDevice


def test_available_strategies_lists_stage_3_strategies():
    assert available_strategies() == [
        "scalar_naive",
        "vector_dot",
        "tiled_vector_dot",
    ]


def test_scalar_naive_generates_non_empty_halt_ending_program():
    spec = characterize_device(ToyDevice())
    problem = MatmulProblem(M=1, N=1, K=2, addr_A=0, addr_B=128, addr_C=256)

    program = generate_matmul_program(problem, spec, "scalar_naive")

    assert program
    assert program[-1].opcode is Opcode.HALT
    assert "HALT" in pretty_print_program(program)


def test_vector_dot_generates_non_empty_halt_ending_program():
    spec = characterize_device(ToyDevice())
    problem = MatmulProblem(M=1, N=1, K=8, addr_A=0, addr_B=128, addr_C=256)

    program = generate_matmul_program(problem, spec, "vector_dot")

    assert program
    assert program[-1].opcode is Opcode.HALT


def test_tiled_vector_dot_generates_non_empty_halt_ending_program():
    spec = characterize_device(ToyDevice())
    problem = MatmulProblem(M=1, N=1, K=8, addr_A=0, addr_B=128, addr_C=256)

    program = generate_matmul_program(problem, spec, "tiled_vector_dot")

    assert program
    assert program[-1].opcode is Opcode.HALT


def test_generated_scalar_naive_program_runs_and_matches_numpy():
    device = ToyDevice()
    spec = characterize_device(device)
    problem = MatmulProblem(M=2, N=2, K=3, addr_A=0, addr_B=128, addr_C=256)
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    b = np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], dtype=np.float32)

    _write_matrix(device, problem.addr_A, a, spec.alignment)
    _write_matrix(device, problem.addr_B, b, spec.alignment)
    program = generate_matmul_program(problem, spec, "scalar_naive")
    device.run(program)

    np.testing.assert_allclose(_read_matrix(device, problem.addr_C, 2, 2, spec.alignment), a @ b)


def test_generated_vector_dot_program_runs_and_matches_numpy():
    device = ToyDevice()
    spec = characterize_device(device)
    problem = MatmulProblem(M=2, N=2, K=8, addr_A=0, addr_B=512, addr_C=1024)
    a = np.arange(1, 17, dtype=np.float32).reshape(2, 8)
    b = (np.arange(1, 17, dtype=np.float32).reshape(8, 2) / 3.0).astype(np.float32)

    _write_matrix(device, problem.addr_A, a, spec.alignment)
    _write_matrix(device, problem.addr_B, b, spec.alignment)
    program = generate_matmul_program(problem, spec, "vector_dot")
    device.run(program)

    np.testing.assert_allclose(
        _read_matrix(device, problem.addr_C, 2, 2, spec.alignment),
        a @ b,
        rtol=1e-6,
    )


def test_generated_tiled_vector_dot_program_runs_and_matches_numpy():
    device = ToyDevice()
    spec = characterize_device(device)
    problem = MatmulProblem(M=2, N=2, K=8, addr_A=0, addr_B=512, addr_C=1024)
    a = (np.arange(1, 17, dtype=np.float32).reshape(2, 8) / 5.0).astype(np.float32)
    b = np.arange(1, 17, dtype=np.float32).reshape(8, 2)

    _write_matrix(device, problem.addr_A, a, spec.alignment)
    _write_matrix(device, problem.addr_B, b, spec.alignment)
    program = generate_matmul_program(problem, spec, "tiled_vector_dot")
    device.run(program)

    np.testing.assert_allclose(
        _read_matrix(device, problem.addr_C, 2, 2, spec.alignment),
        a @ b,
        rtol=1e-6,
    )


def test_invalid_strategy_raises_clear_error():
    spec = characterize_device(ToyDevice())
    problem = MatmulProblem(M=1, N=1, K=1, addr_A=0, addr_B=128, addr_C=256)

    with pytest.raises(CodegenError, match="Unknown matmul strategy"):
        generate_matmul_program(problem, spec, "surprise_me")


def _write_matrix(device, base: int, matrix: np.ndarray, alignment: int) -> None:
    rows, cols = matrix.shape
    for row in range(rows):
        for col in range(cols):
            address = base + (row * cols + col) * alignment
            device.memory.write_scalar(address, float(matrix[row, col]))


def _read_matrix(
    device,
    base: int,
    rows: int,
    cols: int,
    alignment: int,
) -> np.ndarray:
    result = np.zeros((rows, cols), dtype=np.float32)
    for row in range(rows):
        for col in range(cols):
            address = base + (row * cols + col) * alignment
            result[row, col] = device.memory.read_scalar(address)
    return result
