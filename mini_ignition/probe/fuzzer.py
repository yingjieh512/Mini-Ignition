"""Structured ISA fuzzer for Mini Ignition.

This file implements random-but-valid program generation for the toy ISA. In a
real hardware enablement workflow, fuzzing is a way to stress the contract
between generated instruction streams and hardware execution.

The concept matters for AI accelerator enablement because an agent or compiler
can propose plausible programs that still fail on edge cases. A fuzzer gives
tests another source of pressure beyond hand-written examples.

This toy version generates short straight-line programs that always end in
`HALT`. It avoids branches, undefined behavior, random memory sizes, invalid
registers, and reference-model comparison; later stages can build from here.
"""

from __future__ import annotations

from collections.abc import Callable
import random
from typing import Any

from mini_ignition.simulator import (
    ADD,
    DOT,
    HALT,
    LOAD,
    MOV,
    MUL,
    STORE,
    VADD,
    VLOAD,
    VMUL,
    VSTORE,
    Program,
)


def generate_scalar_program(seed: int, length: int) -> Program:
    """Generate a valid scalar program with exactly `length` instructions."""

    _require_positive_length(length)
    rng = random.Random(seed)
    instructions = []

    if length > 1:
        instructions.append(MOV("r0", rng.randint(-8, 8)))
    if length > 2:
        instructions.append(MOV("r1", rng.randint(-8, 8)))

    choices = ("MOV", "LOAD", "STORE", "ADD", "MUL")
    scalar_registers = ("r0", "r1", "r2", "r3")
    while len(instructions) < length - 1:
        opcode = rng.choice(choices)
        if opcode == "MOV":
            instructions.append(MOV(rng.choice(scalar_registers), rng.randint(-16, 16)))
        elif opcode == "LOAD":
            instructions.append(LOAD(rng.choice(scalar_registers), 0))
        elif opcode == "STORE":
            instructions.append(STORE(rng.choice(scalar_registers), 0))
        elif opcode == "ADD":
            instructions.append(
                ADD(
                    rng.choice(scalar_registers),
                    rng.choice(scalar_registers),
                    rng.choice(scalar_registers),
                )
            )
        else:
            instructions.append(
                MUL(
                    rng.choice(scalar_registers),
                    rng.choice(scalar_registers),
                    rng.choice(scalar_registers),
                )
            )

    instructions.append(HALT())
    return Program.from_instructions(*instructions)


def generate_vector_program(seed: int, length: int, vector_width: int) -> Program:
    """Generate a valid vector program with exactly `length` instructions."""

    _require_positive_length(length)
    if vector_width <= 0:
        raise ValueError(f"vector_width must be positive, got {vector_width}.")

    rng = random.Random(seed)
    instructions = []

    if length > 1:
        instructions.append(VLOAD("v0", 0))
    if length > 2:
        instructions.append(VLOAD("v1", 0))

    choices = ("VLOAD", "VSTORE", "VADD", "VMUL", "DOT")
    vector_registers = ("v0", "v1", "v2", "v3")
    while len(instructions) < length - 1:
        opcode = rng.choice(choices)
        if opcode == "VLOAD":
            instructions.append(VLOAD(rng.choice(vector_registers), 0))
        elif opcode == "VSTORE":
            instructions.append(VSTORE(rng.choice(vector_registers), 0))
        elif opcode == "VADD":
            instructions.append(
                VADD(
                    rng.choice(vector_registers),
                    rng.choice(vector_registers),
                    rng.choice(vector_registers),
                )
            )
        elif opcode == "VMUL":
            instructions.append(
                VMUL(
                    rng.choice(vector_registers),
                    rng.choice(vector_registers),
                    rng.choice(vector_registers),
                )
            )
        else:
            instructions.append(
                DOT("r0", rng.choice(vector_registers), rng.choice(vector_registers))
            )

    instructions.append(HALT())
    return Program.from_instructions(*instructions)


def fuzz_device(
    device_factory: Callable[[], Any],
    num_programs: int = 20,
    seed: int = 0,
) -> dict[str, int]:
    """Run random valid scalar/vector programs on fresh devices."""

    if num_programs < 0:
        raise ValueError(f"num_programs must be non-negative, got {num_programs}.")

    rng = random.Random(seed)
    passed = 0
    failed = 0

    for index in range(num_programs):
        device = device_factory()
        length = rng.randint(4, 12)
        program_seed = rng.randint(0, 2**31 - 1)
        if index % 2 == 0:
            program = generate_scalar_program(program_seed, length)
        else:
            vector_width = getattr(device.config, "vector_width", 1)
            program = generate_vector_program(program_seed, length, vector_width)

        try:
            device.run(program, max_steps=length + 1)
        except Exception:
            failed += 1
        else:
            passed += 1

    return {"passed": passed, "failed": failed}


def _require_positive_length(length: int) -> None:
    if length <= 0:
        raise ValueError(f"Program length must be positive, got {length}.")


__all__ = ["fuzz_device", "generate_scalar_program", "generate_vector_program"]
