"""Runtime hardware characterization for Mini Ignition.

This file implements a small probing routine that turns an executable toy
device into a `HardwareSpec`. In a real hardware enablement workflow, runtime
probing discovers observed behavior that may not be fully captured by a data
sheet or design document.

The concept matters for AI accelerator enablement because generated kernels and
controllers should target measured capabilities: supported instructions,
alignment behavior, register files, memory size, and latency hints.

This toy version runs a handful of deterministic programs against the simulator
instead of measuring real silicon. It does not model flaky devices, temperature
effects, clock changes, firmware versions, or statistical benchmarking.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mini_ignition.schemas.hw_spec import HardwareSpec
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
    Opcode,
    Program,
)
from mini_ignition.simulator.memory import MemoryAlignmentError


class CharacterizationError(RuntimeError):
    """Raised when a device fails a characterization probe."""


def characterize_device(device: Any, name: str = "toy-accelerator-v0") -> HardwareSpec:
    """Probe a device and return its observed hardware specification."""

    config = device.config
    latencies = _verify_latencies(config.instruction_latencies)

    _probe_scalar_ops(device)
    _probe_cycle_counter(device, latencies)
    has_vector_ops = _probe_vector_ops(device)
    has_dot = _probe_dot(device)
    alignment_note = _probe_alignment_behavior(device)

    supported_opcodes = [opcode.value for opcode in Opcode]
    notes = [
        "Scalar MOV/LOAD/STORE/ADD/MUL/HALT probes passed.",
        "Vector VLOAD/VSTORE/VADD/VMUL probes passed.",
        "DOT probe passed.",
        alignment_note,
        "Latencies are positive fixed instruction costs for a toy performance model.",
    ]

    return HardwareSpec(
        name=name,
        memory_size=config.memory_size,
        vector_width=config.vector_width,
        num_scalar_registers=config.num_scalar_registers,
        num_vector_registers=config.num_vector_registers,
        alignment=config.alignment,
        supported_opcodes=supported_opcodes,
        latencies=latencies,
        has_vector_ops=has_vector_ops,
        has_dot=has_dot,
        notes=notes,
    )


def _probe_scalar_ops(device: Any) -> None:
    probe = _fresh_device_like(device)
    probe.run(
        Program.from_instructions(
            MOV("r0", 2.0),
            MOV("r1", 3.0),
            ADD("r2", "r0", "r1"),
            MUL("r3", "r0", "r1"),
            STORE("r3", 0),
            LOAD("r4", 0),
            HALT(),
        )
    )

    _assert_close(probe.read_scalar_register("r2"), 5.0, "ADD probe failed")
    _assert_close(probe.read_scalar_register("r3"), 6.0, "MUL probe failed")
    _assert_close(probe.read_scalar_register("r4"), 6.0, "LOAD/STORE probe failed")
    if not probe.halted:
        raise CharacterizationError("HALT probe failed: device did not halt.")


def _probe_cycle_counter(device: Any, latencies: dict[str, int]) -> None:
    probe = _fresh_device_like(device)
    probe.run(Program.from_instructions(MOV("r0", 1.0), HALT()))
    expected = latencies["MOV"] + latencies["HALT"]
    if probe.cycle_count != expected:
        raise CharacterizationError(
            f"Cycle counter probe failed: expected {expected}, got {probe.cycle_count}."
        )


def _probe_vector_ops(device: Any) -> bool:
    probe = _fresh_device_like(device)
    width = probe.config.vector_width
    values_a = np.arange(1, width + 1, dtype=np.float32)
    values_b = np.arange(10, 10 + width, dtype=np.float32)
    addr_a = 0
    addr_b = _aligned_after(width, probe.config.alignment)
    addr_out = _aligned_after(addr_b + width, probe.config.alignment)
    _require_memory_capacity(probe, addr_out + width)

    probe.memory.write_vector(addr_a, values_a)
    probe.memory.write_vector(addr_b, values_b)
    probe.run(
        Program.from_instructions(
            VLOAD("v0", addr_a),
            VLOAD("v1", addr_b),
            VADD("v2", "v0", "v1"),
            VMUL("v3", "v0", "v1"),
            VSTORE("v2", addr_out),
            HALT(),
        )
    )

    np.testing.assert_allclose(probe.read_vector_register("v2"), values_a + values_b)
    np.testing.assert_allclose(probe.read_vector_register("v3"), values_a * values_b)
    np.testing.assert_allclose(probe.memory.read_vector(addr_out, width), values_a + values_b)
    return True


def _probe_dot(device: Any) -> bool:
    probe = _fresh_device_like(device)
    width = probe.config.vector_width
    values_a = np.arange(1, width + 1, dtype=np.float32)
    values_b = np.arange(2, 2 + width, dtype=np.float32)
    addr_b = _aligned_after(width, probe.config.alignment)
    _require_memory_capacity(probe, addr_b + width)

    probe.memory.write_vector(0, values_a)
    probe.memory.write_vector(addr_b, values_b)
    probe.run(
        Program.from_instructions(
            VLOAD("v0", 0),
            VLOAD("v1", addr_b),
            DOT("r0", "v0", "v1"),
            HALT(),
        )
    )

    _assert_close(
        probe.read_scalar_register("r0"),
        float(np.dot(values_a, values_b)),
        "DOT probe failed",
    )
    return True


def _probe_alignment_behavior(device: Any) -> str:
    probe = _fresh_device_like(device)
    alignment = probe.config.alignment
    if not probe.config.check_alignment or alignment <= 1:
        probe.memory.write_scalar(1 if probe.config.memory_size > 1 else 0, 1.0)
        return "Alignment checking is disabled or has slot alignment 1."

    misaligned = 1
    if misaligned >= probe.config.memory_size:
        raise CharacterizationError("Device memory is too small for alignment probing.")

    try:
        probe.memory.write_scalar(misaligned, 1.0)
    except MemoryAlignmentError:
        return f"Misaligned accesses are rejected for alignment {alignment}."
    raise CharacterizationError("Alignment probe failed: misaligned access succeeded.")


def _verify_latencies(raw_latencies: dict[str, int]) -> dict[str, int]:
    latencies = {str(opcode): int(latency) for opcode, latency in raw_latencies.items()}
    for opcode in Opcode:
        latency = latencies.get(opcode.value)
        if latency is None:
            raise CharacterizationError(f"Missing latency for opcode {opcode.value}.")
        if latency <= 0:
            raise CharacterizationError(
                f"Latency for opcode {opcode.value} must be positive, got {latency}."
            )
    return latencies


def _fresh_device_like(device: Any) -> Any:
    return type(device)(config=device.config)


def _aligned_after(address: int, alignment: int) -> int:
    remainder = address % alignment
    if remainder == 0:
        return address
    return address + alignment - remainder


def _require_memory_capacity(device: Any, required_size: int) -> None:
    if required_size > device.config.memory_size:
        raise CharacterizationError(
            f"Device memory size {device.config.memory_size} is too small for probe "
            f"requiring {required_size} slots."
        )


def _assert_close(observed: float, expected: float, message: str) -> None:
    if not np.isclose(observed, expected):
        raise CharacterizationError(f"{message}: expected {expected}, got {observed}.")


__all__ = ["CharacterizationError", "characterize_device"]
