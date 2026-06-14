"""Machine-readable hardware specification for Mini Ignition.

This file implements `HardwareSpec`, the schema produced after probing a toy
device. In a real hardware enablement stack, this kind of schema records what
software observed about a target: memory size, register counts, instruction
support, latency hints, and hardware quirks.

The concept matters for AI accelerator enablement because later compiler and
runtime layers need an explicit target description. Code generation should
consume observed facts such as `hw_spec.json`, not hopeful guesses.

This toy version is a Python dataclass serialized as JSON. It skips versioned
schemas, vendor extensions, topology, memory hierarchy, cache behavior,
precision modes, and detailed timing models.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HardwareSpec:
    """Observed hardware facts used by later Mini Ignition stages."""

    name: str
    memory_size: int
    vector_width: int
    num_scalar_registers: int
    num_vector_registers: int
    alignment: int
    supported_opcodes: list[str]
    latencies: dict[str, int]
    has_vector_ops: bool
    has_dot: bool
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("HardwareSpec name must be non-empty.")
        _require_positive(self.memory_size, "memory_size")
        _require_positive(self.vector_width, "vector_width")
        _require_positive(self.num_scalar_registers, "num_scalar_registers")
        _require_positive(self.num_vector_registers, "num_vector_registers")
        _require_positive(self.alignment, "alignment")
        if not self.supported_opcodes:
            raise ValueError("HardwareSpec supported_opcodes must not be empty.")
        if not self.latencies:
            raise ValueError("HardwareSpec latencies must not be empty.")

    def to_json(self, path: str | Path) -> None:
        """Write this hardware specification to a JSON file."""

        Path(path).write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "HardwareSpec":
        """Load a hardware specification from a JSON file."""

        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("HardwareSpec JSON must contain an object.")
        return cls(**_normalize_spec_dict(raw))


def _normalize_spec_dict(raw: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(raw)
    normalized["supported_opcodes"] = list(normalized["supported_opcodes"])
    normalized["latencies"] = {
        str(opcode): int(latency)
        for opcode, latency in normalized["latencies"].items()
    }
    normalized["notes"] = list(normalized.get("notes", []))
    return normalized


def _require_positive(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


__all__ = ["HardwareSpec"]
