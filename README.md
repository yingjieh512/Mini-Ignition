# Mini Ignition

Mini Ignition is a learning-oriented systems and AI infrastructure project.
It implements a toy hardware enablement stack inspired by AI accelerator
bring-up workflows.

The project is intentionally small and runnable. It teaches how software
discovers, targets, tests, and gates behavior for a made-up accelerator.

## Stage 1: Toy Device Simulator

This stage implements:

- a tiny instruction set architecture (ISA)
- scalar and vector registers
- flat NumPy `float32` device memory
- a toy device simulator with a program counter
- cycle, instruction, and arithmetic operation counters
- tests for scalar instructions, vector instructions, and memory behavior

The key lesson is that generated code or controller decisions should not be
trusted directly. Tests, reference behavior, fuzzing, and performance gates
decide what is accepted.

## Install

```bash
pip install -e .
```

## Test

```bash
pytest
```

## Concept Map

| Mini Ignition component | Real-world concept |
| --- | --- |
| ToyDevice simulator | accelerator hardware or hardware simulator |
| ISA | target instruction set contract |
| Memory model | device memory and access rules |
| Cycle counter | simplified performance model |

The cycle counter is a teaching model, not cycle-accurate simulation.
