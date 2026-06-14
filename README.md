# Mini Ignition

Mini Ignition is a learning-oriented systems and AI infrastructure project. It
implements a toy hardware enablement stack inspired by AI accelerator bring-up:
a small ISA, a toy accelerator simulator, hardware characterization, matmul
code generation, runtime execution, correctness gates, performance metrics, and
a simple controller loop.

## What This Project Is

Mini Ignition is an educational miniature of the loop behind accelerator
enablement. It is small enough to read in one sitting, but it still connects the
major ideas:

- software targets an ISA
- generated programs run on a simulated device
- hardware behavior is characterized into a machine-readable spec
- matmul kernels are generated for different strategies
- a runtime executes kernels and compares outputs against NumPy
- a staged test ladder validates lower layers before higher layers
- a controller rejects failing or slow strategies and selects a passing one

## Why This Project Exists

The project is designed to teach how AI infrastructure systems become trustworthy
around hardware. It is not enough for an agent or compiler to emit plausible
code. The code must run, match a reference implementation, survive fuzzing, and
clear performance gates.

## How This Maps To Real AI Accelerator Enablement

Real accelerator enablement involves hardware simulators, ISAs, compilers,
runtimes, hardware probing, correctness test suites, profiling, and controller
logic. Mini Ignition maps each of those concepts to a runnable Python toy:

- `ToyDevice` stands in for accelerator hardware or a simulator.
- The toy ISA defines what generated programs can express.
- `HardwareSpec` records observed target capabilities.
- Matmul codegen lowers `C = A @ B` into target instructions.
- The runtime loads memory, runs programs, and reads results.
- The ladder and metrics decide whether a candidate is accepted.
- The controller is a tiny version of an agent/controller loop.

## Architecture

```text
mini-ignition/
  mini_ignition/
    simulator/      # ISA, program container, memory model, ToyDevice
    schemas/        # HardwareSpec JSON schema
    probe/          # characterization and structured ISA fuzzing
    codegen/        # matmul codegen strategies
    runtime/        # runner, metrics, ladder, controller
  examples/
    demo.py         # end-to-end demo
  tests/            # staged unit tests
```

## Stages

1. Toy ISA, memory model, and device simulator.
2. Hardware characterization and structured ISA fuzzing.
3. Matmul codegen with scalar and vector strategies.
4. Runtime runner and NumPy correctness evaluation.
5. Metrics and staged verification ladder.
6. Controller loop and end-to-end demo.

## Component Mapping

| Mini Ignition Component | Real-World Analogy | Why It Matters |
| --- | --- | --- |
| ToyDevice | accelerator hardware or simulator | executes target programs |
| ISA | target instruction set | defines what codegen can emit |
| HardwareSpec | machine-readable target description | guides codegen |
| characterize_device | hardware probing | discovers observed behavior |
| matmul_codegen | compiler backend / kernel generator | emits target-specific code |
| runner | accelerator runtime | loads memory, runs programs, reads results |
| metrics | performance model | rejects correct but slow kernels |
| ladder | verification harness | gates correctness layer by layer |
| controller | agent/controller loop | selects acceptable implementations |

## Running The Project

Install the package in editable mode:

```bash
pip install -e .
```

Run the end-to-end demo:

```bash
python examples/demo.py
```

The demo characterizes the toy accelerator, writes
`examples/generated_hw_spec.json`, runs the test ladder, evaluates matmul
strategies, and prints the selected strategy.

## Running Tests

```bash
pytest
```

The tests cover the ISA, memory, simulator, hardware spec serialization,
probing, fuzzing, codegen, runtime runner, metrics, ladder, and controller.

## Toy ISA

The ISA is the simplified contract between software and hardware.

Scalar instructions:

```text
MOV, LOAD, STORE, ADD, MUL, HALT
```

Vector instructions:

```text
VLOAD, VSTORE, VADD, VMUL, DOT
```

Instructions are Python dataclasses for readability rather than binary
encodings. Invalid registers and memory accesses raise clear exceptions.

## Hardware Characterization

`characterize_device()` probes the toy accelerator and produces a
`HardwareSpec`. The spec records memory size, vector width, register counts,
alignment, supported opcodes, latencies, vector support, and DOT support.

This is the toy version of runtime hardware probing. Later layers consume
observed facts instead of relying on guesses.

## Matmul Codegen

`generate_matmul_program()` lowers a `MatmulProblem` into toy ISA instructions
for:

- `scalar_naive`: portable scalar baseline
- `vector_dot`: vector/DOT strategy
- `tiled_vector_dot`: explicit tiling over `K` chunks

NumPy is the trusted reference implementation in tests.

## Runtime Runner

`run_matmul()` writes matrices into device memory, generates a program, runs it
on `ToyDevice`, reads the output matrix, compares it against NumPy, and returns
a `RunResult` with correctness and toy performance counters.

Real runtimes also manage device allocation, command submission, synchronization,
and result readback. Mini Ignition keeps those ideas synchronous and inspectable.

## Test Ladder

`run_test_ladder()` runs gates in order:

1. scalar instruction correctness
2. memory correctness
3. vector instruction correctness
4. DOT correctness
5. fixed-shape matmul correctness
6. randomized matmul correctness
7. performance threshold

The ladder stops on first failure. Each lower layer must pass before higher
layers are trusted.

## Controller Loop

`run_controller()` tries matmul strategies in order:

```text
scalar_naive -> vector_dot -> tiled_vector_dot
```

For each strategy, it runs the runtime, checks correctness, estimates theoretical
peak, computes utilization, applies the performance gate, and selects the
passing strategy with the highest utilization.

This is a toy version of an agent controller. In a real Ignition-like system, an
LLM coding agent may generate candidate code, but the controller and verification
gates decide what is accepted.

## Key Lesson

The central lesson is that an AI coding agent should not be trusted directly.
The agent proposes candidate code or strategies; the test ladder, reference
implementation, fuzzing, and performance gates decide whether the candidate is
accepted.

## Limitations

This project does not implement a real compiler backend, real driver, real DMA,
real CUDA/Triton kernel, real vLLM plugin, or real LLM coding agent. It is an
educational miniature of the control loop behind hardware enablement.

The cycle counter is a toy performance model, not cycle-accurate simulation.
The vector matmul strategies are intentionally simple and prioritize clarity
over real performance.

## Future Extensions

1. Add a real CUDA vector-add backend.
2. Add a Triton matmul backend.
3. Add a tiny IR before toy ISA emission.
4. Add an LLM-based code editing loop.
5. Add attention kernel generation.
6. Add KV cache simulation.
7. Add a mock OpenAI-compatible inference endpoint.
