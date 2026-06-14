"""Simulator package for Mini Ignition.

This package contains the toy accelerator hardware model, its instruction set,
and its device memory. In real hardware enablement work, these pieces define
the contract that compilers, runtimes, and tests must obey.

The implementation is deliberately small: it uses Python objects and NumPy
arrays instead of hardware RTL, firmware, DMA engines, or timing-accurate
models. That makes the concepts inspectable while preserving the shape of the
real workflow.
"""

from mini_ignition.simulator.device import DeviceConfig, ToyDevice
from mini_ignition.simulator.isa import (
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
    Instruction,
    Opcode,
)
from mini_ignition.simulator.memory import Memory
from mini_ignition.simulator.program import Program

__all__ = [
    "ADD",
    "DOT",
    "DeviceConfig",
    "HALT",
    "Instruction",
    "LOAD",
    "MOV",
    "MUL",
    "Memory",
    "Opcode",
    "Program",
    "STORE",
    "ToyDevice",
    "VADD",
    "VLOAD",
    "VMUL",
    "VSTORE",
]
