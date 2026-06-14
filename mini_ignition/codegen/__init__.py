"""Code generation package for Mini Ignition.

This package contains tiny compiler-backend pieces that lower a high-level
matrix multiplication problem into Mini Ignition ISA instructions. In a real AI
accelerator stack, codegen maps operations such as matmul onto target-specific
instructions and memory layouts.

The concept matters for AI accelerator enablement because accelerator hardware
is only useful when software can generate correct kernels for it. Hardware
specifications, tests, and reference implementations decide whether those
kernels are acceptable.

This toy version emits straight-line Python instruction objects. It skips real
IRs, register allocation, scheduling, binary assembly, cache-aware tiling,
autotuning, and production compiler passes.
"""

from mini_ignition.codegen.matmul_codegen import (
    CodegenError,
    MatmulProblem,
    available_strategies,
    generate_matmul_program,
    pretty_print_program,
)

__all__ = [
    "CodegenError",
    "MatmulProblem",
    "available_strategies",
    "generate_matmul_program",
    "pretty_print_program",
]
