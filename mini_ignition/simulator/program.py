"""Program container for toy accelerator instructions.

This file implements a small immutable sequence of ISA instructions. In a real
hardware/software stack, a program might be a compiled kernel, command buffer,
or firmware-visible instruction stream that the runtime submits to a device.

The concept matters for AI accelerator enablement because generated work must
be represented in a form that can be inspected, tested, scheduled, and executed
against a hardware contract.

This toy version stores Python `Instruction` objects directly. It skips binary
assembly, relocation, labels, branches, command queues, and synchronization.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

from mini_ignition.simulator.isa import Instruction


@dataclass(frozen=True)
class Program(Sequence[Instruction]):
    """An immutable instruction sequence."""

    instructions: tuple[Instruction, ...]

    def __init__(self, instructions: Iterable[Instruction]) -> None:
        object.__setattr__(self, "instructions", tuple(instructions))

    @classmethod
    def from_instructions(cls, *instructions: Instruction) -> "Program":
        """Build a program from positional instruction arguments."""

        return cls(instructions)

    def __getitem__(self, index: int) -> Instruction:
        return self.instructions[index]

    def __iter__(self) -> Iterator[Instruction]:
        return iter(self.instructions)

    def __len__(self) -> int:
        return len(self.instructions)

    def __str__(self) -> str:
        return "\n".join(
            f"{index:04d}: {instruction}"
            for index, instruction in enumerate(self.instructions)
        )


__all__ = ["Program"]
