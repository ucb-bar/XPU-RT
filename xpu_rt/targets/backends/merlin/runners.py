"""Runner protocols for merlin-compiled artifacts.

Profiling a compiled ``.vmfb`` (or chipyard image) needs a runtime
target — these can be local (``spike``, ``host``), simulated
(``firesim``), or remote (``board``). Each runner returns a flat
``{dispatch_id: mean_us}`` mapping so :func:`profile_dispatch_matrix`
stays target-agnostic.

This module defines the runner protocol and three minimal concrete
implementations:

* :class:`NoopRunner` — fills measurements from a static table (used
  by tests and ``--dry-run`` flows).
* :class:`HostRunner` — invokes ``iree-benchmark-module`` on the host.
* :class:`SpikeRunner` / :class:`FiresimRunner` — stubs that delegate
  to merlin's ``tools/chipyard.py``.

Concrete on-board runners (QRB5165 + ``qnn-net-run``) live under
``backends/qnn/`` and are NOT plumbed through here on purpose: that
flow is QNN-specific and already typed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


class Runner(Protocol):
    """Anything that can measure ``.vmfb`` dispatches in microseconds."""

    kind: str

    def run(
        self,
        *,
        artifact: Path,
        dispatch_ids: list[str],
        iterations: int,
    ) -> dict[str, float]:
        """Return ``{dispatch_id: mean_us}`` for each requested id."""


@dataclass(frozen=True)
class NoopRunner:
    """Fill measurements from a pre-supplied table; never executes anything.

    Useful for tests and the ``--dry-run`` MCP path. Missing keys
    fall back to ``default_us``.
    """

    kind: str = "noop"
    table: dict[str, float] = field(default_factory=dict)
    default_us: float = 1000.0

    def run(
        self,
        *,
        artifact: Path,  # noqa: ARG002
        dispatch_ids: list[str],
        iterations: int,  # noqa: ARG002
    ) -> dict[str, float]:
        return {d: float(self.table.get(d, self.default_us)) for d in dispatch_ids}


@dataclass(frozen=True)
class HostRunner:
    """Drives ``iree-benchmark-module`` on the local machine."""

    kind: str = "host"
    benchmark_binary: str = "iree-benchmark-module"

    def run(
        self,
        *,
        artifact: Path,
        dispatch_ids: list[str],
        iterations: int,
    ) -> dict[str, float]:
        if not shutil.which(self.benchmark_binary):
            return {d: float("nan") for d in dispatch_ids}
        out: dict[str, float] = {}
        for disp in dispatch_ids:
            cmd = [
                self.benchmark_binary,
                f"--module={artifact}",
                f"--function={disp}",
                f"--benchmark_repetitions={iterations}",
                "--benchmark_format=json",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                out[disp] = float("nan")
                continue
            try:
                report = json.loads(proc.stdout)
            except json.JSONDecodeError:
                out[disp] = float("nan")
                continue
            # Pick first "real_time" mean — google benchmark schema.
            benches = report.get("benchmarks") or []
            mean_ns = next(
                (b.get("real_time") for b in benches if b.get("name") and b.get("real_time")),
                None,
            )
            out[disp] = (float(mean_ns) / 1000.0) if mean_ns else float("nan")
        return out


@dataclass(frozen=True)
class SpikeRunner:
    """Stub: defers to merlin's ``tools/chipyard.py spike`` driver.

    The actual subprocess call is performed by the caller through
    :class:`MerlinBridge`; this dataclass is metadata only.
    """

    kind: str = "spike"
    isa: str = "rv64gcv"
    pk_path: Path | None = None

    def run(
        self,
        *,
        artifact: Path,
        dispatch_ids: list[str],
        iterations: int,  # noqa: ARG002
    ) -> dict[str, float]:
        return {d: float("nan") for d in dispatch_ids}


@dataclass(frozen=True)
class FiresimRunner:
    """Stub: firesim measurements come from merlin's chipyard flow."""

    kind: str = "firesim"
    target_name: str = ""

    def run(
        self,
        *,
        artifact: Path,
        dispatch_ids: list[str],
        iterations: int,  # noqa: ARG002
    ) -> dict[str, float]:
        return {d: float("nan") for d in dispatch_ids}
