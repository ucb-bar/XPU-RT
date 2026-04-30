"""Template: Custom Runtime/Execution Adapter

Implement an adapter for a specific execution runtime (device simulator,
cloud service, FPGA fabric, etc.).

See ``compgen.runtime.local_executor`` for a working example (local CPU/GPU).

Steps:
    1. Copy this file: ``cp _template.py my_runtime.py``
    2. Implement ``execute()`` and optionally ``benchmark()``
    3. Register with the runtime planner
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TemplateExecutionResult:
    """Result from executing on the template runtime."""

    outputs: list[Any]
    latency_us: float = 0.0
    memory_bytes: int = 0
    success: bool = True
    error: str = ""


class TemplateRuntimeAdapter:
    """Template runtime adapter.

    Replace with your execution backend implementation.
    """

    def __init__(self, device_config: dict[str, Any] | None = None) -> None:
        self._config = device_config or {}

    def execute(
        self,
        compiled_artifact: Any,
        inputs: dict[str, Any],
    ) -> TemplateExecutionResult:
        """Execute a compiled artifact on this runtime.

        Args:
            compiled_artifact: The compiled code/binary to execute.
            inputs: Input tensors/data.

        Returns:
            Execution result with outputs and performance data.
        """
        # TODO: Implement execution on your runtime
        # For a device simulator: load binary, push inputs, run, pull outputs
        # For a cloud service: serialize, send, receive
        # For an FPGA: program fabric, execute, read results
        return TemplateExecutionResult(
            outputs=[],
            success=False,
            error="Not implemented — replace with your runtime logic",
        )

    def benchmark(
        self,
        compiled_artifact: Any,
        inputs: dict[str, Any],
        warmup: int = 10,
        iterations: int = 100,
    ) -> TemplateExecutionResult:
        """Benchmark a compiled artifact with timing.

        Args:
            compiled_artifact: The compiled code/binary.
            inputs: Input tensors/data.
            warmup: Number of warmup iterations.
            iterations: Number of timed iterations.

        Returns:
            Result with average latency.
        """
        # TODO: Implement benchmarking loop
        return self.execute(compiled_artifact, inputs)
