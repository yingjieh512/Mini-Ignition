"""End-to-end Mini Ignition demo.

This script connects the full educational stack: toy accelerator simulation,
hardware characterization, staged verification, matmul codegen, runtime
execution, performance metrics, and controller-based strategy selection.

In real accelerator enablement, this kind of flow would run against silicon or
a detailed simulator. The toy version keeps everything in Python so the control
loop is easy to inspect.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mini_ignition.codegen import MatmulProblem
from mini_ignition.probe.characterize import characterize_device
from mini_ignition.runtime.controller import run_controller
from mini_ignition.runtime.ladder import run_test_ladder
from mini_ignition.simulator import ToyDevice

try:
    from rich.console import Console
    from rich.table import Table
except ImportError:  # pragma: no cover - exercised only without optional rich.
    Console = None
    Table = None


def main() -> int:
    """Run the final Mini Ignition demo."""

    printer = _Printer()
    printer.title("Mini Ignition Demo")

    printer.step("Step 1: Characterizing toy accelerator...")
    spec = characterize_device(ToyDevice())
    spec_path = Path(__file__).with_name("generated_hw_spec.json")
    spec.to_json(spec_path)
    printer.line(f"Generated {spec_path.relative_to(ROOT)}")
    printer.spec(spec)

    printer.step("Step 2: Running test ladder...")
    ladder = run_test_ladder(ToyDevice, spec)
    printer.ladder(ladder)
    if not ladder.passed:
        printer.line("Stopping: the verification ladder failed.")
        return 1

    printer.step("Step 3: Trying matmul codegen strategies...")
    problem = MatmulProblem(M=4, N=4, K=8, addr_A=0, addr_B=1024, addr_C=2048)
    a_matrix = (np.arange(1, 33, dtype=np.float32).reshape(4, 8) / 7.0).astype(
        np.float32
    )
    b_matrix = (np.arange(1, 33, dtype=np.float32).reshape(8, 4) / 5.0).astype(
        np.float32
    )
    controller = run_controller(ToyDevice, spec, problem, a_matrix, b_matrix)
    printer.controller(controller)

    if controller.selected_strategy is None:
        printer.line("No strategy passed both correctness and performance gates.")
        return 1

    printer.line(f"Selected strategy: {controller.selected_strategy}")
    printer.step("Interpretation:")
    printer.line(
        "The controller selected the fastest strategy that passed both "
        "correctness and performance gates."
    )
    printer.line(
        "In a real hardware enablement system, this loop would run against "
        "actual accelerator hardware or a simulator instead of this toy Python device."
    )
    return 0


class _Printer:
    def __init__(self) -> None:
        self.console = Console() if Console is not None else None

    def title(self, text: str) -> None:
        if self.console is not None:
            self.console.rule(f"[bold]{text}[/bold]")
        else:
            print(text)

    def step(self, text: str) -> None:
        self.line("")
        if self.console is not None:
            self.console.print(f"[bold]{text}[/bold]")
        else:
            print(text)

    def line(self, text: str) -> None:
        if self.console is not None:
            self.console.print(text)
        else:
            print(text)

    def spec(self, spec) -> None:
        rows = [
            ("name", spec.name),
            ("memory_size", spec.memory_size),
            ("vector_width", spec.vector_width),
            ("alignment", spec.alignment),
            ("has_vector_ops", spec.has_vector_ops),
            ("has_dot", spec.has_dot),
        ]
        self._table("Generated HardwareSpec", ["Field", "Value"], rows)

    def ladder(self, ladder) -> None:
        rows = []
        for index, gate in enumerate(ladder.gates, start=1):
            status = "PASS" if gate.passed else "FAIL"
            rows.append((f"Gate {index}", gate.name, status, gate.message))
        self._table("Test Ladder", ["Gate", "Name", "Status", "Message"], rows)

    def controller(self, controller) -> None:
        rows = []
        for report in controller.reports:
            correctness = "PASS" if report.correctness_passed else "FAIL"
            performance = "PASS" if report.performance_passed else "FAIL"
            rows.append(
                (
                    report.strategy,
                    correctness,
                    performance,
                    f"{report.utilization:.3f}",
                    str(report.cycles),
                    report.message,
                )
            )
        self._table(
            "Strategy Reports",
            ["Strategy", "Correct", "Perf", "Util", "Cycles", "Message"],
            rows,
        )

    def _table(self, title: str, columns: list[str], rows: list[tuple[object, ...]]) -> None:
        if self.console is None or Table is None:
            print(title)
            print(" | ".join(columns))
            for row in rows:
                print(" | ".join(str(value) for value in row))
            return

        table = Table(title=title)
        for column in columns:
            table.add_column(column)
        for row in rows:
            table.add_row(*(str(value) for value in row))
        self.console.print(table)


if __name__ == "__main__":
    raise SystemExit(main())
