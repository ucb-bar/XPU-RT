"""Measured-cost evidence emitted by XPU-RT runtime executors.

This is the canonical artifact that closes the outer loop ``input
facts → decision → applied transform → executed bundle → measured
delta``. The bundle's ``payload.mlir`` / ``execution_plan.yaml`` /
``generated_kernels/*`` describe what should run; an
:class:`~xpu_rt.runtime.executor.XpuRtExecutor` actually runs it on
real hardware (or a simulator) and writes a :class:`MeasuredCost`
back into the bundle as ``measured_cost.json``.

Why this lives alongside ``compile_baseline.json`` rather than under
``02_graph_analysis/compiled_bottleneck/``: ``compile_baseline.json``
is per-bundle baseline data; ``measured_cost.json`` is per-bundle
post-execution data. Both belong to the bundle the user can publish.
The ``compiled_bottleneck``/``profiler_evidence`` reports in the run
directory remain the canonical AOT cost-analysis surfaces; this
artifact records what one specific run measured.

The promotion gate (:mod:`xpu_rt.promotion.gates`) accepts a
bundle-level :class:`MeasuredCost` as evidence for the
``characterized`` level alongside the existing ``compiled_bottleneck``
and ``profiler_evidence`` paths.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


CorrectnessOutcome = Literal["pass", "fail", "skipped", "error"]
"""Outcome of comparing executor output against the eager reference.

``pass``: bit-equality or within tolerance.
``fail``: produced output but it disagreed with eager.
``skipped``: executor ran but correctness wasn't checked (no eager ref
    available, or the correctness path itself was disabled).
``error``: executor crashed before correctness could be evaluated.
"""


@dataclass(frozen=True)
class MeasurementSample:
    """One per-region measurement that backs a :class:`MeasuredCost`.

    For a single-kernel bundle this list has one entry. For a fused
    multi-region bundle there is one entry per region the executor
    actually ran (which may differ from the count in
    ``kernel_contracts/`` if some regions failed the executor's
    op-family support).

    Attributes:
        region_id: ``KernelContract.region_id`` the sample corresponds
            to (may be empty when the executor doesn't carry region
            metadata).
        op_family: ``KernelContract.op_family`` (``"matmul"``, …).
        cycles: Executor-reported cycle count, when the substrate
            exposes one (Spike, Gemmini counters). ``None`` for
            substrates that only measure wall-clock.
        latency_us_p50: Median wall-clock latency in microseconds
            when measured.
        latency_us_p95: 95th percentile wall-clock latency.
        correctness: Per-region correctness outcome.
        mismatches: Number of mismatched elements when
            ``correctness != "pass"``. ``None`` when the executor
            doesn't expose per-element counts.
        total_elements: Output element count compared against eager.
        extras: Executor-specific extras (counter readings, exit
            codes, …) — surfaced for debugging but not gated on.
    """

    region_id: str = ""
    op_family: str = ""
    cycles: int | None = None
    latency_us_p50: float | None = None
    latency_us_p95: float | None = None
    correctness: CorrectnessOutcome = "skipped"
    mismatches: int | None = None
    total_elements: int | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "op_family": self.op_family,
            "cycles": self.cycles,
            "latency_us_p50": self.latency_us_p50,
            "latency_us_p95": self.latency_us_p95,
            "correctness": self.correctness,
            "mismatches": self.mismatches,
            "total_elements": self.total_elements,
            "extras": dict(self.extras),
        }


SCHEMA_VERSION = "measured_cost_v1"


@dataclass(frozen=True)
class MeasuredCost:
    """Aggregate executor measurement attached to a bundle.

    Attributes:
        executor: Stable name of the executor that produced this
            artifact (``"spike_gemmini"``, ``"spike_saturn_opu_v128"``,
            ``"qnn_qrb5165"``, ``"local_cpu_python_sync"``, …). Lets
            consumers identify which substrate the numbers are from
            without parsing extras.
        hardware_key: Canonical hardware identifier (matches
            ``KernelContract.hardware_key`` when available). Used to
            join with target-card derivation rules.
        target_id: Target id the executor was dispatched against
            (``"gemmini"``, ``"saturn_opu_v128"``, …).
        cycles_total: Sum of per-sample cycles when every sample
            reports cycles, else ``None``. The single number a gate
            check or comparison plot consumes.
        latency_us_p50_total: Sum of per-sample p50 latencies (in
            microseconds) — None unless every sample measured
            wall-clock.
        correctness_vs_eager: Roll-up correctness across all samples.
            ``"pass"`` iff every sample is ``"pass"``; otherwise the
            most severe outcome wins (``error > fail > skipped >
            pass``).
        samples: Per-region :class:`MeasurementSample` rows.
        run_id: Free-form executor-run id (workdir name, batch tag,
            …). Lets a multi-run study group results.
        notes: Optional free-text caveats (e.g. "fell back to scalar
            ref because Gemmini extension not built into spike").
            Surfaced verbatim in the trust report; do not encode
            structured data here — use ``samples[*].extras``.
    """

    executor: str
    hardware_key: str = ""
    target_id: str = ""
    cycles_total: int | None = None
    latency_us_p50_total: float | None = None
    correctness_vs_eager: CorrectnessOutcome = "skipped"
    samples: tuple[MeasurementSample, ...] = ()
    run_id: str = ""
    notes: str = ""

    def as_json_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "executor": self.executor,
            "hardware_key": self.hardware_key,
            "target_id": self.target_id,
            "cycles_total": self.cycles_total,
            "latency_us_p50_total": self.latency_us_p50_total,
            "correctness_vs_eager": self.correctness_vs_eager,
            "samples": [s.as_dict() for s in self.samples],
            "run_id": self.run_id,
            "notes": self.notes,
        }

    def write_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.as_json_dict(), indent=2))

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "MeasuredCost":
        version = data.get("schema_version")
        if version not in (None, SCHEMA_VERSION):
            raise ValueError(
                f"measured_cost.json schema_version={version!r} unsupported; "
                f"this build understands {SCHEMA_VERSION!r}"
            )
        samples = tuple(
            MeasurementSample(
                region_id=str(s.get("region_id", "")),
                op_family=str(s.get("op_family", "")),
                cycles=_as_int_or_none(s.get("cycles")),
                latency_us_p50=_as_float_or_none(s.get("latency_us_p50")),
                latency_us_p95=_as_float_or_none(s.get("latency_us_p95")),
                correctness=_as_correctness(s.get("correctness", "skipped")),
                mismatches=_as_int_or_none(s.get("mismatches")),
                total_elements=_as_int_or_none(s.get("total_elements")),
                extras=dict(s.get("extras") or {}),
            )
            for s in data.get("samples", []) or ()
        )
        return cls(
            executor=str(data.get("executor", "")),
            hardware_key=str(data.get("hardware_key", "")),
            target_id=str(data.get("target_id", "")),
            cycles_total=_as_int_or_none(data.get("cycles_total")),
            latency_us_p50_total=_as_float_or_none(data.get("latency_us_p50_total")),
            correctness_vs_eager=_as_correctness(
                data.get("correctness_vs_eager", "skipped")
            ),
            samples=samples,
            run_id=str(data.get("run_id", "")),
            notes=str(data.get("notes", "")),
        )

    @classmethod
    def read_json(cls, path: Path) -> "MeasuredCost":
        return cls.from_json_dict(json.loads(path.read_text()))


def aggregate_correctness(
    samples: tuple[MeasurementSample, ...] | list[MeasurementSample],
) -> CorrectnessOutcome:
    """Roll up per-sample outcomes into a single correctness verdict.

    Order of severity (worst → best): ``error`` > ``fail`` > ``skipped``
    > ``pass``. If any sample is ``error`` the rollup is ``error``;
    otherwise the worst non-pass wins; if every sample is ``pass`` the
    rollup is ``pass``. An empty sample list rolls up to ``skipped`` —
    nothing was measured.
    """
    if not samples:
        return "skipped"
    severity = {"pass": 0, "skipped": 1, "fail": 2, "error": 3}
    worst = max(samples, key=lambda s: severity.get(s.correctness, 1))
    return worst.correctness


# --------------------------------------------------------------------------- #
# Internal coercion helpers
# --------------------------------------------------------------------------- #


def _as_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


_VALID_CORRECTNESS = frozenset({"pass", "fail", "skipped", "error"})


def _as_correctness(value: Any) -> CorrectnessOutcome:
    s = str(value).lower()
    if s in _VALID_CORRECTNESS:
        return s  # type: ignore[return-value]
    return "skipped"


__all__ = [
    "CorrectnessOutcome",
    "MeasuredCost",
    "MeasurementSample",
    "SCHEMA_VERSION",
    "aggregate_correctness",
]
