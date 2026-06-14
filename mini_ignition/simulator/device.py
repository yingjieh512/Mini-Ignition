"""Toy accelerator device simulator.

This file implements `ToyDevice`, a small stand-in for accelerator hardware or
a hardware simulator. It executes Mini Ignition ISA instructions over scalar
registers, vector registers, and flat device memory.

The concept matters for AI accelerator enablement because compilers, runtimes,
probers, and correctness tests all need something concrete to execute against.
A simulator lets learners observe behavior before adding probing, codegen, and
performance gates.

This toy version is intentionally not cycle-accurate. The cycle counter adds a
fixed latency per instruction, which is useful for teaching performance models
but does not model pipelines, memory stalls, vector lanes, hazards, caches, or
parallel execution.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace

import numpy as np

from mini_ignition.simulator.isa import (
    Instruction,
    Opcode,
    scalar_register_index,
    vector_register_index,
)
from mini_ignition.simulator.memory import Memory
from mini_ignition.simulator.program import Program


DEFAULT_LATENCIES: dict[str, int] = {
    "MOV": 1,
    "LOAD": 3,
    "STORE": 3,
    "ADD": 1,
    "MUL": 1,
    "VLOAD": 4,
    "VSTORE": 4,
    "VADD": 1,
    "VMUL": 1,
    "DOT": 3,
    "HALT": 1,
}


class DeviceExecutionError(RuntimeError):
    """Raised when the toy device cannot execute the current program."""


class RegisterError(ValueError):
    """Raised when a register name is invalid for this configured device."""


@dataclass
class DeviceConfig:
    """Configuration for a `ToyDevice` instance."""

    memory_size: int = 4096
    num_scalar_registers: int = 16
    num_vector_registers: int = 16
    vector_width: int = 8
    alignment: int = 4
    check_alignment: bool = True
    instruction_latencies: Mapping[str, int] = field(
        default_factory=lambda: dict(DEFAULT_LATENCIES)
    )

    def __post_init__(self) -> None:
        _require_positive(self.memory_size, "memory_size")
        _require_positive(self.num_scalar_registers, "num_scalar_registers")
        _require_positive(self.num_vector_registers, "num_vector_registers")
        _require_positive(self.vector_width, "vector_width")
        _require_positive(self.alignment, "alignment")

        latencies = dict(DEFAULT_LATENCIES)
        latencies.update(self.instruction_latencies)
        for opcode in Opcode:
            latency = latencies.get(opcode.value)
            if latency is None:
                raise ValueError(f"Missing latency for opcode {opcode.value}.")
            _require_positive(latency, f"latency for {opcode.value}")
        self.instruction_latencies = latencies


class ToyDevice:
    """Execute Mini Ignition programs over toy registers and memory."""

    def __init__(self, config: DeviceConfig | None = None, **overrides: object) -> None:
        if config is None:
            self.config = DeviceConfig(**overrides)
        elif overrides:
            self.config = replace(config, **overrides)
        else:
            self.config = config

        self.memory = Memory(
            self.config.memory_size,
            alignment=self.config.alignment,
            check_alignment=self.config.check_alignment,
        )
        self.scalar_registers = np.zeros(
            self.config.num_scalar_registers, dtype=np.float32
        )
        self.vector_registers = np.zeros(
            (self.config.num_vector_registers, self.config.vector_width),
            dtype=np.float32,
        )
        self.program = Program(())
        self.program_counter = 0
        self.cycle_count = 0
        self.instruction_count = 0
        self.arithmetic_op_count = 0
        self.halted = False

    def reset(self, *, clear_memory: bool = True) -> None:
        """Reset registers, counters, halt state, and optionally memory."""

        self.scalar_registers.fill(0)
        self.vector_registers.fill(0)
        if clear_memory:
            self.memory.data.fill(0)
        self.program = Program(())
        self.program_counter = 0
        self.cycle_count = 0
        self.instruction_count = 0
        self.arithmetic_op_count = 0
        self.halted = False

    def load_program(self, program: Program | Iterable[Instruction]) -> None:
        """Load a program and prepare to execute it from instruction zero."""

        self.program = program if isinstance(program, Program) else Program(program)
        self.program_counter = 0
        self.halted = False

    def run(
        self,
        program: Program | Iterable[Instruction] | None = None,
        *,
        max_steps: int = 10_000,
    ) -> None:
        """Run until `HALT` executes or `max_steps` is reached."""

        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}.")
        if program is not None:
            self.load_program(program)

        steps = 0
        while not self.halted:
            if steps >= max_steps:
                raise DeviceExecutionError(
                    f"Program did not halt within max_steps={max_steps}."
                )
            self.step()
            steps += 1

    def step(self) -> None:
        """Execute one instruction at the current program counter."""

        if self.halted:
            return
        if self.program_counter < 0 or self.program_counter >= len(self.program):
            raise DeviceExecutionError(
                f"Program counter {self.program_counter} is outside the loaded "
                "program. Did you forget HALT?"
            )

        instruction = self.program[self.program_counter]
        self._execute(instruction)
        self.program_counter += 1
        self.cycle_count += self._latency(instruction.opcode)
        self.instruction_count += 1

    def read_scalar_register(self, name: str) -> float:
        """Read a scalar register by name."""

        return float(self.scalar_registers[self._scalar_index(name)])

    def write_scalar_register(self, name: str, value: float) -> None:
        """Write a scalar register by name."""

        self.scalar_registers[self._scalar_index(name)] = np.float32(value)

    def read_vector_register(self, name: str) -> np.ndarray:
        """Read a copied vector register by name."""

        return self.vector_registers[self._vector_index(name)].copy()

    def write_vector_register(self, name: str, values: np.ndarray) -> None:
        """Write a vector register by name."""

        vector = np.asarray(values, dtype=np.float32)
        if vector.shape != (self.config.vector_width,):
            raise RegisterError(
                f"Vector register writes require shape ({self.config.vector_width},), "
                f"got {vector.shape}."
            )
        self.vector_registers[self._vector_index(name)] = vector

    def _execute(self, instruction: Instruction) -> None:
        opcode = instruction.opcode
        operands = instruction.operands

        if opcode is Opcode.MOV:
            dst, immediate = operands
            self.write_scalar_register(dst, immediate)
        elif opcode is Opcode.LOAD:
            dst, address = operands
            self.write_scalar_register(dst, self.memory.read_scalar(address))
        elif opcode is Opcode.STORE:
            src, address = operands
            self.memory.write_scalar(address, self.read_scalar_register(src))
        elif opcode is Opcode.ADD:
            dst, lhs, rhs = operands
            self.write_scalar_register(
                dst, self.read_scalar_register(lhs) + self.read_scalar_register(rhs)
            )
            self.arithmetic_op_count += 1
        elif opcode is Opcode.MUL:
            dst, lhs, rhs = operands
            self.write_scalar_register(
                dst, self.read_scalar_register(lhs) * self.read_scalar_register(rhs)
            )
            self.arithmetic_op_count += 1
        elif opcode is Opcode.HALT:
            self.halted = True
        elif opcode is Opcode.VLOAD:
            dst, address = operands
            self.write_vector_register(
                dst, self.memory.read_vector(address, self.config.vector_width)
            )
        elif opcode is Opcode.VSTORE:
            src, address = operands
            self.memory.write_vector(address, self.read_vector_register(src))
        elif opcode is Opcode.VADD:
            dst, lhs, rhs = operands
            result = self.read_vector_register(lhs) + self.read_vector_register(rhs)
            self.write_vector_register(dst, result)
            self.arithmetic_op_count += self.config.vector_width
        elif opcode is Opcode.VMUL:
            dst, lhs, rhs = operands
            result = self.read_vector_register(lhs) * self.read_vector_register(rhs)
            self.write_vector_register(dst, result)
            self.arithmetic_op_count += self.config.vector_width
        elif opcode is Opcode.DOT:
            dst, lhs, rhs = operands
            result = float(
                np.dot(self.read_vector_register(lhs), self.read_vector_register(rhs))
            )
            self.write_scalar_register(dst, result)
            self.arithmetic_op_count += (2 * self.config.vector_width) - 1
        else:
            raise DeviceExecutionError(f"Unsupported opcode {opcode!r}.")

    def _latency(self, opcode: Opcode) -> int:
        return int(self.config.instruction_latencies[opcode.value])

    def _scalar_index(self, name: str) -> int:
        index = scalar_register_index(name)
        if index >= self.config.num_scalar_registers:
            raise RegisterError(
                f"Scalar register {name!r} is out of range for "
                f"{self.config.num_scalar_registers} scalar registers."
            )
        return index

    def _vector_index(self, name: str) -> int:
        index = vector_register_index(name)
        if index >= self.config.num_vector_registers:
            raise RegisterError(
                f"Vector register {name!r} is out of range for "
                f"{self.config.num_vector_registers} vector registers."
            )
        return index


def _require_positive(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


__all__ = [
    "DEFAULT_LATENCIES",
    "DeviceConfig",
    "DeviceExecutionError",
    "RegisterError",
    "ToyDevice",
]
