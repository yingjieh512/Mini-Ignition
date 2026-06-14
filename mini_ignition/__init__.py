"""Mini Ignition package.

This package implements an educational hardware enablement stack for a toy AI
accelerator. It connects simulator pieces, later probing/codegen/runtime pieces,
and tests into one small project.

In a real system, the package boundary might separate a compiler, driver,
hardware model, and runtime. Mini Ignition keeps them together so learners can
trace the whole flow without production-scale complexity.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
