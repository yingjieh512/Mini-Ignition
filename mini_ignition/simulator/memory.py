"""Flat memory model for the toy device.

This file implements device memory: a flat NumPy `float32` array with scalar
and vector read/write helpers. In real accelerator systems, memory behavior is
one of the first hardware/software contracts that runtimes and kernels must
understand.

The concept matters for AI accelerator enablement because generated kernels
only work when their addresses, widths, and alignment assumptions match the
device. Many real bring-up failures are memory contract failures wearing a
different hat.

This toy version has one contiguous memory space, no caches, no DMA, no paging,
and no host/device transfer latency. Alignment is optional and expressed as a
multiple of toy memory slots rather than real byte lanes.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class MemoryAccessError(ValueError):
    """Raised when a memory access is invalid."""


class MemoryAlignmentError(MemoryAccessError):
    """Raised when an access violates the configured alignment rule."""


class Memory:
    """A small flat `float32` memory array with checked access helpers."""

    def __init__(
        self,
        size: int,
        *,
        alignment: int = 1,
        check_alignment: bool = False,
    ) -> None:
        if size <= 0:
            raise ValueError(f"Memory size must be positive, got {size}.")
        if alignment <= 0:
            raise ValueError(f"Memory alignment must be positive, got {alignment}.")

        self.size = size
        self.alignment = alignment
        self.check_alignment = check_alignment
        self._data = np.zeros(size, dtype=np.float32)

    @property
    def data(self) -> np.ndarray:
        """Expose the backing array for inspection in tests and demos."""

        return self._data

    def read_scalar(self, address: int) -> float:
        """Read one `float32` memory slot as a Python float."""

        self._validate_access(address, width=1)
        return float(self._data[address])

    def write_scalar(self, address: int, value: float) -> None:
        """Write one scalar value into memory."""

        self._validate_access(address, width=1)
        self._data[address] = np.float32(value)

    def read_vector(self, address: int, width: int) -> np.ndarray:
        """Read `width` contiguous memory slots as a copied vector."""

        if width <= 0:
            raise ValueError(f"Vector width must be positive, got {width}.")
        self._validate_access(address, width=width)
        return self._data[address : address + width].copy()

    def write_vector(self, address: int, values: Sequence[float] | np.ndarray) -> None:
        """Write a contiguous vector into memory."""

        vector = np.asarray(values, dtype=np.float32)
        if vector.ndim != 1:
            raise ValueError("Vector writes require a one-dimensional value.")
        if vector.size == 0:
            raise ValueError("Vector writes require at least one value.")

        self._validate_access(address, width=int(vector.size))
        self._data[address : address + vector.size] = vector

    def _validate_access(self, address: int, *, width: int) -> None:
        if isinstance(address, bool) or not isinstance(address, int):
            raise MemoryAccessError(
                f"Memory address must be an integer slot index, got {address!r}."
            )
        if address < 0:
            raise MemoryAccessError(f"Memory address {address} is negative.")
        if address + width > self.size:
            end = address + width - 1
            raise MemoryAccessError(
                f"Memory access [{address}, {end}] is out of bounds for size {self.size}."
            )
        if self.check_alignment and address % self.alignment != 0:
            raise MemoryAlignmentError(
                f"Memory address {address} is not aligned to {self.alignment} slots."
            )


__all__ = ["Memory", "MemoryAccessError", "MemoryAlignmentError"]
