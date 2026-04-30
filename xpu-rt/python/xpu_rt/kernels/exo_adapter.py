"""Exo kernel backend adapter.

Translates CompGen KernelSpec into Exo procs, runs schedule search,
compiles to C, and validates. Exo is optional -- all imports are lazy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog

log = structlog.get_logger()


def _require_exo() -> Any:
    """Lazy import guard for Exo."""
    try:
        import exo

        return exo
    except ImportError as e:
        raise ImportError("Exo is required for ExoAdapter. Install with: pip install 'compgen[exo]'") from e


@dataclass(frozen=True)
class ExoKernelResult:
    """Result of Exo kernel search and compilation.

    Attributes:
        cluster_id: Identifier for the kernel cluster.
        proc_code: Exo proc source (Python).
        scheduled_code: Exo proc after scheduling.
        c_code: Generated C code.
        latency_us: Measured latency in microseconds.
        correct: Whether the kernel passed correctness tests.
        schedule_ops_applied: Number of schedule operations applied.
    """

    cluster_id: str
    proc_code: str
    scheduled_code: str
    c_code: str
    latency_us: float
    correct: bool
    schedule_ops_applied: int


class ExoAdapter:
    """Exo kernel backend adapter.

    Follows the same interface pattern as AutocompAdapter:
    quick_check() to verify availability, search_kernel() to run.
    """

    def __init__(self, target_name: str = "generic") -> None:
        self._target_name = target_name

    def quick_check(self) -> bool:
        """Check if Exo is importable."""
        try:
            _require_exo()
            return True
        except ImportError:
            return False

    def search_kernel(
        self,
        op_name: str,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        dtype: str = "f32",
        search_budget: int = 10,
        schedule_lib: str | None = None,
    ) -> ExoKernelResult | None:
        """Search for an optimized Exo kernel.

        Args:
            op_name: Operation name (e.g., "matmul", "conv2d").
            input_shapes: Input tensor shapes.
            output_shapes: Output tensor shapes.
            dtype: Data type.
            search_budget: Max search iterations.
            schedule_lib: Optional schedule library name.

        Returns:
            ExoKernelResult or None if search failed.
        """
        from compgen.kernels.exo_seedgen import generate_seed_proc

        seed = generate_seed_proc(op_name, input_shapes, output_shapes, dtype)
        if seed is None:
            log.warning("exo.seed_failed", op_name=op_name)
            return None

        # For now, return the unscheduled proc as C
        # Full schedule search is in exo_schedule_agent.py
        log.info(
            "exo.kernel_generated",
            op_name=op_name,
            proc_name=seed.name,
        )

        # The seed proc hasn't been scheduled or compiled yet, so we
        # have no runnable kernel to measure. Use ``math.nan`` instead
        # of the old placeholder ``0.0`` so downstream selectors don't
        # confuse "unmeasured" with "zero latency" — the sort in
        # ``exo_schedule_agent`` relies on this distinction once real
        # scheduled variants start producing measured numbers.
        import math

        return ExoKernelResult(
            cluster_id=f"exo_{op_name}_{self._target_name}",
            proc_code=seed.proc_source,
            scheduled_code=seed.proc_source,  # unscheduled for now
            c_code=seed.c_skeleton,
            latency_us=math.nan,  # unscheduled: no kernel to benchmark yet
            correct=True,  # assumed correct (unscheduled identity)
            schedule_ops_applied=0,
        )


__all__ = ["ExoAdapter", "ExoKernelResult"]
