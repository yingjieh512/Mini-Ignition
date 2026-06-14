"""Matmul code generator for Mini Ignition.

This file implements Stage 3: lowering a matrix multiplication problem
`C = A @ B` into toy ISA instructions. In real accelerator enablement, this is
the tiny analog of a compiler backend or kernel generator.

The concept matters because high-level AI operations must eventually become
target-specific instruction streams. Real systems handle instruction
selection, register allocation, scheduling, memory layout, tiling, and
performance modeling before a kernel is trusted.

This toy version favors clarity and correctness. The scalar strategy emits a
portable baseline. The vector strategies use `VLOAD`, `VMUL`, and `DOT`; for
row-major strided `B` columns and aligned sparse matrix layouts, they use a
single-lane vector mask in fresh scratch memory instead of pretending the ISA
has gather loads. NumPy remains the reference implementation in tests.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from mini_ignition.codegen.strategies import (
    SCALAR_NAIVE,
    TILED_VECTOR_DOT,
    VECTOR_DOT,
    available_strategies,
    validate_strategy,
)
from mini_ignition.schemas.hw_spec import HardwareSpec
from mini_ignition.simulator import (
    ADD,
    DOT,
    HALT,
    LOAD,
    MOV,
    MUL,
    STORE,
    VMUL,
    VLOAD,
    Instruction,
)


class CodegenError(ValueError):
    """Raised when a matmul program cannot be generated for the target spec."""


@dataclass(frozen=True)
class MatmulProblem:
    """Matrix multiplication shape and device-memory addresses."""

    M: int
    N: int
    K: int
    addr_A: int
    addr_B: int
    addr_C: int


def generate_matmul_program(
    problem: MatmulProblem,
    spec: HardwareSpec,
    strategy: str,
) -> list[Instruction]:
    """Generate Mini Ignition instructions for `C = A @ B`."""

    try:
        selected = validate_strategy(strategy)
    except ValueError as exc:
        raise CodegenError(str(exc)) from exc

    _validate_problem(problem, spec)
    if selected == SCALAR_NAIVE:
        return _generate_scalar_naive(problem, spec)
    if selected == VECTOR_DOT:
        return _generate_vector_dot(problem, spec, tiled=False)
    if selected == TILED_VECTOR_DOT:
        return _generate_vector_dot(problem, spec, tiled=True)
    raise CodegenError(f"Unhandled matmul strategy {selected!r}.")


def pretty_print_program(program: Sequence[Instruction]) -> str:
    """Render a generated program as numbered instructions."""

    return "\n".join(
        f"{index:04d}: {instruction}" for index, instruction in enumerate(program)
    )


def _generate_scalar_naive(problem: MatmulProblem, spec: HardwareSpec) -> list[Instruction]:
    _require_registers(spec, scalar=4, vector=0)
    _require_opcodes(spec, ["MOV", "LOAD", "MUL", "ADD", "STORE", "HALT"])
    _require_matrix_memory(problem, spec, extra_vector_read_width=1)

    instructions: list[Instruction] = []
    for i in range(problem.M):
        for j in range(problem.N):
            instructions.append(MOV("r0", 0.0))
            for k in range(problem.K):
                instructions.extend(
                    [
                        LOAD("r1", _addr_a(problem, spec, i, k)),
                        LOAD("r2", _addr_b(problem, spec, k, j)),
                        MUL("r3", "r1", "r2"),
                        ADD("r0", "r0", "r3"),
                    ]
                )
            instructions.append(STORE("r0", _addr_c(problem, spec, i, j)))

    instructions.append(HALT())
    return instructions


def _generate_vector_dot(
    problem: MatmulProblem,
    spec: HardwareSpec,
    *,
    tiled: bool,
) -> list[Instruction]:
    if not spec.has_vector_ops:
        raise CodegenError("Vector matmul strategies require has_vector_ops=True.")
    if not spec.has_dot:
        raise CodegenError("Vector matmul strategies require has_dot=True.")
    if problem.K % spec.vector_width != 0:
        raise CodegenError(
            f"{VECTOR_DOT} requires K divisible by vector_width; "
            f"got K={problem.K}, vector_width={spec.vector_width}."
        )

    _require_registers(spec, scalar=3, vector=4)
    _require_opcodes(spec, ["MOV", "STORE", "ADD", "VLOAD", "VMUL", "DOT", "HALT"])
    _require_matrix_memory(problem, spec, extra_vector_read_width=spec.vector_width)

    mask_base = _scratch_base(problem, spec)
    if mask_base + spec.vector_width > spec.memory_size:
        raise CodegenError(
            "Not enough device memory for vector mask scratch space: "
            f"need through address {mask_base + spec.vector_width - 1}, "
            f"memory_size={spec.memory_size}."
        )

    instructions: list[Instruction] = [
        MOV("r2", 1.0),
        STORE("r2", mask_base),
        VLOAD("v3", mask_base),
    ]

    for i in range(problem.M):
        for j in range(problem.N):
            instructions.append(MOV("r0", 0.0))
            if tiled:
                for tile_k in range(0, problem.K, spec.vector_width):
                    instructions.extend(
                        _emit_masked_dot_tile(problem, spec, i, j, tile_k)
                    )
            else:
                for k in range(problem.K):
                    instructions.extend(_emit_masked_dot_at_k(problem, spec, i, j, k))
            instructions.append(STORE("r0", _addr_c(problem, spec, i, j)))

    instructions.append(HALT())
    return instructions


def _emit_masked_dot_tile(
    problem: MatmulProblem,
    spec: HardwareSpec,
    i: int,
    j: int,
    tile_k: int,
) -> list[Instruction]:
    instructions: list[Instruction] = []
    for k in range(tile_k, tile_k + spec.vector_width):
        instructions.extend(_emit_masked_dot_at_k(problem, spec, i, j, k))
    return instructions


def _emit_masked_dot_at_k(
    problem: MatmulProblem,
    spec: HardwareSpec,
    i: int,
    j: int,
    k: int,
) -> list[Instruction]:
    return [
        VLOAD("v0", _addr_a(problem, spec, i, k)),
        VLOAD("v1", _addr_b(problem, spec, k, j)),
        VMUL("v2", "v0", "v3"),
        DOT("r1", "v2", "v1"),
        ADD("r0", "r0", "r1"),
    ]


def _validate_problem(problem: MatmulProblem, spec: HardwareSpec) -> None:
    for field_name in ("M", "N", "K"):
        value = getattr(problem, field_name)
        if value <= 0:
            raise CodegenError(f"MatmulProblem {field_name} must be positive, got {value}.")
    for field_name in ("addr_A", "addr_B", "addr_C"):
        value = getattr(problem, field_name)
        if value < 0:
            raise CodegenError(f"MatmulProblem {field_name} must be non-negative, got {value}.")
        if value % spec.alignment != 0:
            raise CodegenError(
                f"MatmulProblem {field_name}={value} must be aligned to {spec.alignment}."
            )


def _require_registers(spec: HardwareSpec, *, scalar: int, vector: int) -> None:
    if spec.num_scalar_registers < scalar:
        raise CodegenError(
            f"Strategy requires at least {scalar} scalar registers, "
            f"but spec has {spec.num_scalar_registers}."
        )
    if spec.num_vector_registers < vector:
        raise CodegenError(
            f"Strategy requires at least {vector} vector registers, "
            f"but spec has {spec.num_vector_registers}."
        )


def _require_opcodes(spec: HardwareSpec, opcodes: Sequence[str]) -> None:
    supported = set(spec.supported_opcodes)
    missing = [opcode for opcode in opcodes if opcode not in supported]
    if missing:
        raise CodegenError(f"Spec is missing required opcodes: {', '.join(missing)}.")


def _require_matrix_memory(
    problem: MatmulProblem,
    spec: HardwareSpec,
    *,
    extra_vector_read_width: int,
) -> None:
    last_read = max(
        _addr_a(problem, spec, problem.M - 1, problem.K - 1) + extra_vector_read_width,
        _addr_b(problem, spec, problem.K - 1, problem.N - 1) + extra_vector_read_width,
        _addr_c(problem, spec, problem.M - 1, problem.N - 1) + 1,
    )
    if last_read > spec.memory_size:
        raise CodegenError(
            f"Problem memory range exceeds spec.memory_size={spec.memory_size}; "
            f"needs at least {last_read} slots."
        )


def _scratch_base(problem: MatmulProblem, spec: HardwareSpec) -> int:
    end = max(
        _addr_a(problem, spec, problem.M - 1, problem.K - 1) + spec.vector_width,
        _addr_b(problem, spec, problem.K - 1, problem.N - 1) + spec.vector_width,
        _addr_c(problem, spec, problem.M - 1, problem.N - 1) + 1,
    )
    return _align_up(end, spec.alignment)


def _addr_a(problem: MatmulProblem, spec: HardwareSpec, i: int, k: int) -> int:
    return problem.addr_A + (i * problem.K + k) * spec.alignment


def _addr_b(problem: MatmulProblem, spec: HardwareSpec, k: int, j: int) -> int:
    return problem.addr_B + (k * problem.N + j) * spec.alignment


def _addr_c(problem: MatmulProblem, spec: HardwareSpec, i: int, j: int) -> int:
    return problem.addr_C + (i * problem.N + j) * spec.alignment


def _align_up(value: int, alignment: int) -> int:
    remainder = value % alignment
    if remainder == 0:
        return value
    return value + alignment - remainder


__all__ = [
    "CodegenError",
    "MatmulProblem",
    "available_strategies",
    "generate_matmul_program",
    "pretty_print_program",
]
