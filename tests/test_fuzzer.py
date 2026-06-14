from mini_ignition.probe.fuzzer import (
    fuzz_device,
    generate_scalar_program,
    generate_vector_program,
)
from mini_ignition.simulator import Opcode, ToyDevice


def test_scalar_fuzzer_generates_halt_ending_programs():
    program = generate_scalar_program(seed=123, length=8)

    assert len(program) == 8
    assert program[-1].opcode is Opcode.HALT


def test_vector_fuzzer_generates_halt_ending_programs():
    program = generate_vector_program(seed=123, length=8, vector_width=4)

    assert len(program) == 8
    assert program[-1].opcode is Opcode.HALT


def test_fuzzer_has_zero_failures_on_default_device():
    result = fuzz_device(ToyDevice, num_programs=20, seed=0)

    assert result["passed"] == 20
    assert result["failed"] == 0
