"""Toy instruction set architecture for Mini Ignition.

This file implements a tiny ISA: the instruction names, instruction container,
and helper constructors used to build programs. An ISA is the simplified
contract between software and hardware: compilers and runtimes emit
instructions, while hardware or a simulator executes them.

The contract matters for AI accelerator enablement because every later layer
depends on it. Probing, code generation, correctness tests, and performance
metrics all need a precise target to talk about.

This toy version uses Python enums and dataclasses instead of binary encodings,
instruction packets, hazard rules, or real register allocation. It validates
the basic shape of operands so mistakes produce clear errors early.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from numbers import Real
import re
from typing import Any


_SCALAR_REGISTER_RE = re.compile(r"^r\d+$")
_VECTOR_REGISTER_RE = re.compile(r"^v\d+$")


class InstructionError(ValueError):
    """Raised when an instruction is malformed."""


class Opcode(str, Enum):
    """Operations supported by the toy accelerator."""

    MOV = "MOV"
    LOAD = "LOAD"
    STORE = "STORE"
    ADD = "ADD"
    MUL = "MUL"
    HALT = "HALT"
    VLOAD = "VLOAD"
    VSTORE = "VSTORE"
    VADD = "VADD"
    VMUL = "VMUL"
    DOT = "DOT"


@dataclass(frozen=True)
class Instruction:
    """A single toy instruction.

    The operands are intentionally stored as ordinary Python values so printed
    programs are easy to read while learning and debugging.
    """

    opcode: Opcode
    operands: tuple[Any, ...] = ()

    def __str__(self) -> str:
        if not self.operands:
            return self.opcode.value
        rendered = ", ".join(_format_operand(operand) for operand in self.operands)
        return f"{self.opcode.value} {rendered}"

    def __repr__(self) -> str:
        return str(self)


def scalar_register_index(name: str) -> int:
    """Return the integer index for a scalar register name like ``r0``."""

    _require_scalar_register(name, "scalar register")
    return int(name[1:])


def vector_register_index(name: str) -> int:
    """Return the integer index for a vector register name like ``v0``."""

    _require_vector_register(name, "vector register")
    return int(name[1:])


def MOV(dst_scalar: str, immediate: float) -> Instruction:
    """Move an immediate value into a scalar register."""

    _require_scalar_register(dst_scalar, "MOV destination")
    _require_number(immediate, "MOV immediate")
    return Instruction(Opcode.MOV, (dst_scalar, float(immediate)))


def LOAD(dst_scalar: str, address: int) -> Instruction:
    """Load one scalar value from memory into a scalar register."""

    _require_scalar_register(dst_scalar, "LOAD destination")
    _require_address(address, "LOAD address")
    return Instruction(Opcode.LOAD, (dst_scalar, address))


def STORE(src_scalar: str, address: int) -> Instruction:
    """Store one scalar register value into memory."""

    _require_scalar_register(src_scalar, "STORE source")
    _require_address(address, "STORE address")
    return Instruction(Opcode.STORE, (src_scalar, address))


def ADD(dst_scalar: str, lhs_scalar: str, rhs_scalar: str) -> Instruction:
    """Add two scalar registers and write the result to a scalar register."""

    _require_scalar_register(dst_scalar, "ADD destination")
    _require_scalar_register(lhs_scalar, "ADD left operand")
    _require_scalar_register(rhs_scalar, "ADD right operand")
    return Instruction(Opcode.ADD, (dst_scalar, lhs_scalar, rhs_scalar))


def MUL(dst_scalar: str, lhs_scalar: str, rhs_scalar: str) -> Instruction:
    """Multiply two scalar registers and write the result to a scalar register."""

    _require_scalar_register(dst_scalar, "MUL destination")
    _require_scalar_register(lhs_scalar, "MUL left operand")
    _require_scalar_register(rhs_scalar, "MUL right operand")
    return Instruction(Opcode.MUL, (dst_scalar, lhs_scalar, rhs_scalar))


def HALT() -> Instruction:
    """Stop device execution."""

    return Instruction(Opcode.HALT)


def VLOAD(dst_vector: str, address: int) -> Instruction:
    """Load one vector from memory into a vector register."""

    _require_vector_register(dst_vector, "VLOAD destination")
    _require_address(address, "VLOAD address")
    return Instruction(Opcode.VLOAD, (dst_vector, address))


def VSTORE(src_vector: str, address: int) -> Instruction:
    """Store one vector register into memory."""

    _require_vector_register(src_vector, "VSTORE source")
    _require_address(address, "VSTORE address")
    return Instruction(Opcode.VSTORE, (src_vector, address))


def VADD(dst_vector: str, lhs_vector: str, rhs_vector: str) -> Instruction:
    """Add two vector registers elementwise."""

    _require_vector_register(dst_vector, "VADD destination")
    _require_vector_register(lhs_vector, "VADD left operand")
    _require_vector_register(rhs_vector, "VADD right operand")
    return Instruction(Opcode.VADD, (dst_vector, lhs_vector, rhs_vector))


def VMUL(dst_vector: str, lhs_vector: str, rhs_vector: str) -> Instruction:
    """Multiply two vector registers elementwise."""

    _require_vector_register(dst_vector, "VMUL destination")
    _require_vector_register(lhs_vector, "VMUL left operand")
    _require_vector_register(rhs_vector, "VMUL right operand")
    return Instruction(Opcode.VMUL, (dst_vector, lhs_vector, rhs_vector))


def DOT(dst_scalar: str, lhs_vector: str, rhs_vector: str) -> Instruction:
    """Compute a vector dot product and write the result to a scalar register."""

    _require_scalar_register(dst_scalar, "DOT destination")
    _require_vector_register(lhs_vector, "DOT left operand")
    _require_vector_register(rhs_vector, "DOT right operand")
    return Instruction(Opcode.DOT, (dst_scalar, lhs_vector, rhs_vector))


def _require_scalar_register(name: str, role: str) -> None:
    if not isinstance(name, str) or _SCALAR_REGISTER_RE.fullmatch(name) is None:
        raise InstructionError(
            f"{role} must be a scalar register named like 'r0', got {name!r}."
        )


def _require_vector_register(name: str, role: str) -> None:
    if not isinstance(name, str) or _VECTOR_REGISTER_RE.fullmatch(name) is None:
        raise InstructionError(
            f"{role} must be a vector register named like 'v0', got {name!r}."
        )


def _require_address(address: int, role: str) -> None:
    if isinstance(address, bool) or not isinstance(address, int):
        raise InstructionError(f"{role} must be an integer memory address, got {address!r}.")


def _require_number(value: float, role: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise InstructionError(f"{role} must be a numeric value, got {value!r}.")


def _format_operand(operand: Any) -> str:
    if isinstance(operand, float):
        return f"{operand:g}"
    return str(operand)


__all__ = [
    "ADD",
    "DOT",
    "HALT",
    "Instruction",
    "InstructionError",
    "LOAD",
    "MOV",
    "MUL",
    "Opcode",
    "STORE",
    "VADD",
    "VLOAD",
    "VMUL",
    "VSTORE",
    "scalar_register_index",
    "vector_register_index",
]
