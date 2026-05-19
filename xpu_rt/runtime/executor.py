"""``XpuRtExecutor`` protocol — the substrate that closes the loop.

Until now, ``XpuRtBackend.compile_and_benchmark`` in
:mod:`xpu_rt.runtime.torch_backend` mislabeled
``torch.compile(backend="inductor")`` runs as ``"xpu_rt_compiled"``,
so any speedup-vs-baseline number derived from that path was a fixed
1.0 by construction. The XpuRtExecutor protocol is the seam where a
real substrate (Spike + Gemmini, Spike + Saturn/OPU, QNN on QRB5165,
local CPU Python sync executor, …) plugs in to actually run a bundle
and report measured cost + correctness.

The protocol is deliberately narrow: take a bundle directory, return
a :class:`~xpu_rt.runtime.measured_cost.MeasuredCost`. Concrete
executors do whatever they need internally (cross-compile, upload,
spawn simulators, …) but the bundle and the JSON artifact are the
contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from xpu_rt.runtime.errors import AdapterUnavailableError
from xpu_rt.runtime.measured_cost import MeasuredCost


class XpuRtExecutor(Protocol):
    """Substrate that runs an XPU-RT bundle and reports measured cost.

    Implementations:

    * :class:`xpu_rt.runtime.spike_executor.SpikeExecutor` — Spike +
      Gemmini RoCC extension or Saturn/OPU RVV (depending on
      ``target_id``).
    * (planned) ``QnnExecutor`` for QRB5165.
    * (planned) ``LocalCpuExecutor`` for local-machine smoke tests
      that don't need a cross-compiler.

    Attributes:
        name: Stable identifier for the executor (matches
            :attr:`MeasuredCost.executor`).
        target_id: Target id this executor dispatches against; lets
            callers pick the right substrate for the target without
            hard-coding the executor class name.
    """

    name: str
    target_id: str

    def is_available(self) -> tuple[bool, str]:
        """Return ``(True, "")`` when this executor can run today.

        When unavailable, returns ``(False, reason)`` with a short
        human-readable reason (toolchain missing, hardware absent,
        …). Callers use this to decide whether to write a
        ``skipped`` measured_cost slot rather than a ``failed`` one.
        """
        ...

    def execute(
        self,
        bundle_dir: Path,
        *,
        run_dir: Path | None = None,
        sample_inputs: tuple[Any, ...] | None = None,
    ) -> MeasuredCost:
        """Run the bundle and return measured cost.

        Args:
            bundle_dir: Bundle root (contains ``payload.mlir``,
                ``manifest.json``, ``kernel_contracts/``,
                ``generated_kernels/`` when populated).
            run_dir: Optional graph_compilation run directory; when
                supplied the executor may also write
                ``02_graph_analysis/kernel_execution/kernel_execution_report.json``
                so :mod:`xpu_rt.promotion.gates` can advance the
                ``verified_kernel`` rung. When ``None`` the executor
                still writes the bundle-level ``measured_cost.json``
                but does not touch any run directory.
            sample_inputs: Inputs originally fed through capture.
                Most executors will instead pick up
                ``golden_inputs.pt`` from the bundle; this parameter
                exists for executors that need the live Python
                tensors (e.g. for a wall-clock CPU executor).

        Returns:
            :class:`MeasuredCost`. The executor is responsible for
            populating ``cycles_total`` / ``latency_us_p50_total``
            from its samples. The caller writes the artifact to disk
            via :meth:`MeasuredCost.write_json`.

        Raises:
            AdapterUnavailableError: When :meth:`is_available` would
                have returned False and the caller invoked
                :meth:`execute` anyway. Executors must NOT raise
                bare exceptions for "toolchain missing" — that's a
                first-class outcome, surfaced as
                :class:`AdapterUnavailableError`.
        """
        ...


__all__ = ["AdapterUnavailableError", "XpuRtExecutor"]
