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

## Stage 2: Hardware Characterization

This stage adds hardware characterization:

- `HardwareSpec` is a machine-readable hardware description.
- `characterize_device` is runtime probing that checks observed device behavior.
- `fuzzer` is structured ISA fuzzing with valid random programs.
- `hw_spec.json` is the observed hardware schema that later codegen stages can
  consume.

The lesson is still the same: the simulator may advertise capabilities, but
small probes and fuzz tests decide what is trusted.

## Stage 3: Matmul Codegen

This stage adds a tiny matmul compiler backend:

- Matmul codegen lowers `C = A @ B` into target toy ISA instructions.
- `scalar_naive` is a portable but slow scalar baseline.
- `vector_dot` uses target vector and `DOT` instructions.
- `tiled_vector_dot` is a simplified version of tiling over the `K` dimension.
- `HardwareSpec` guides codegen decisions such as vector width, alignment, and
  available opcodes.
- NumPy is the reference implementation used by tests.

In real accelerator enablement, codegen lowers high-level operations such as
matmul into target-specific instructions. Real systems also handle register
allocation, scheduling, memory layout, tiling, and instruction selection. This
stage implements a tiny educational version of that idea.

## Stage 4: Runtime Runner + Correctness Evaluation

This stage adds a simplified accelerator runtime:

- The runtime runner handles memory loading, generated program execution,
  output retrieval, and correctness checking.
- NumPy is used as the trusted reference implementation.
- `RunResult` reports pass/fail status, maximum absolute error, cycles,
  instruction count, arithmetic operation count, and achieved ops per cycle.

Real accelerator runtimes manage device memory, command submission,
synchronization, and result readback. Code generation is not useful by itself:
a generated kernel must be executed, validated against a reference, and
measured. This runtime layer connects generated code to actual device behavior.

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
| HardwareSpec | machine-readable observed hardware schema |
| Characterization probe | runtime hardware discovery |
| Fuzzer | structured ISA stress testing |
| Matmul codegen | tiny compiler backend or kernel generator |
| Codegen strategy | target-specific lowering choice |
| Runtime runner | simplified accelerator runtime |
| NumPy comparison | trusted correctness reference |

The cycle counter is a teaching model, not cycle-accurate simulation.
