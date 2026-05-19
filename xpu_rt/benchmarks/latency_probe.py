"""CPU-latency probe for graph-compilation run directories.

Given a finished ``run_graph_compilation`` run that produced
``03_recipe_planning/post_lowering/transformed_payload.mlir`` plus the
capture-time artefacts in ``00_graph_capture/`` (``exported_program.pt2``
and ``golden_inputs.pt``), time N invocations of the transformed
payload through ``xpu_rt.runtime.cpu_executor.execute`` and return
median / p50 / p90 wall-clock latency.

The probe is mask-agnostic by design: it doesn't know about
``SubsystemMask``. Its input is the *output* of a compile run, which
already reflects whichever mask was active when the compile ran. The
prove-or-kill harness pairs it with a control + treatment run to get
the latency delta.

Why use the bundle path rather than ``xpu_rt.api.compile_model``? The
subsystem mask is wired at
``xpu_rt.graph_compilation.action_space.build_action_space``, which is
only called from ``run_graph_compilation``. ``compile_model`` uses a
separate xDSL + equality-saturation pipeline that doesn't consult the
mask, so its latency numbers would be identical for control and
treatment. Reading the post-lowering payload from disk lets us
measure the actual mask-affected artefact.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median, stdev
from typing import Any

import torch


# --------------------------------------------------------------------------- #
# Result type
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LatencyProbeResult:
    """Per-cell latency measurement.

    Both ``latency_median_us`` and ``latency_min_us`` are reported.
    The min is the cleaner signal on noisy CPU workloads — the
    median is dominated by per-run variance (thermal throttling,
    GC pauses, context switches) on sub-100us workloads, which can
    fabricate apparent 50%+ deltas that vanish under a min-based
    re-analysis. ``latency_stddev_us`` is reported so the caller
    can flag noisy distributions.

    Recommended kill-rule signal: min-based delta. Median is kept
    for backward compat + as a divergence indicator (when median
    and min disagree by more than the noise floor, the distribution
    has a long tail and the median should not be trusted).
    """

    status: str  # "ok" | "skipped" | "error"
    detail: str  # human-readable note when status != "ok"
    n_warmup: int
    n_iters: int
    latency_median_us: float
    latency_p50_us: float
    latency_p90_us: float
    latency_min_us: float
    latency_stddev_us: float
    per_run_us: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "detail": self.detail,
            "n_warmup": self.n_warmup,
            "n_iters": self.n_iters,
            "latency_median_us": self.latency_median_us,
            "latency_p50_us": self.latency_p50_us,
            "latency_p90_us": self.latency_p90_us,
            "latency_min_us": self.latency_min_us,
            "latency_stddev_us": self.latency_stddev_us,
            "per_run_us": list(self.per_run_us),
        }


def _skipped(reason: str) -> LatencyProbeResult:
    return LatencyProbeResult(
        status="skipped", detail=reason, n_warmup=0, n_iters=0,
        latency_median_us=0.0, latency_p50_us=0.0,
        latency_p90_us=0.0, latency_min_us=0.0,
        latency_stddev_us=0.0, per_run_us=(),
    )


def _error(reason: str) -> LatencyProbeResult:
    return LatencyProbeResult(
        status="error", detail=reason, n_warmup=0, n_iters=0,
        latency_median_us=0.0, latency_p50_us=0.0,
        latency_p90_us=0.0, latency_min_us=0.0,
        latency_stddev_us=0.0, per_run_us=(),
    )


# --------------------------------------------------------------------------- #
# Probe
# --------------------------------------------------------------------------- #


def measure_run_dir_latency(
    run_dir: Path,
    *,
    n_warmup: int = 3,
    n_iters: int = 25,
) -> LatencyProbeResult:
    """Time the transformed payload from a finished compile run.

    Returns a typed result. Failures (missing artefacts, parse errors,
    executor errors) are reported as ``status="skipped"`` or
    ``status="error"`` — the probe never raises.
    """
    run_dir = Path(run_dir).resolve()

    payload_path = (
        run_dir / "03_recipe_planning" / "post_lowering"
        / "transformed_payload.mlir"
    )
    if not payload_path.is_file():
        # Fall back to the un-transformed payload (the post-lowering
        # stage only writes transformed_payload.mlir when at least one
        # transform-like artifact applied). The un-transformed payload
        # is identical to the source — for runs where the mask removed
        # all transforms we still want a baseline number.
        payload_path = (
            run_dir / "01_payload_lowering" / "export_program" / "payload.mlir"
        )
        if not payload_path.is_file():
            return _skipped(f"no payload.mlir under {run_dir}")

    ep_path = run_dir / "00_graph_capture" / "exported_program.pt2"
    if not ep_path.is_file():
        return _skipped(f"no exported_program.pt2 under {ep_path.parent}")

    gi_path = run_dir / "00_graph_capture" / "golden_inputs.pt"
    if not gi_path.is_file():
        return _skipped(f"no golden_inputs.pt under {gi_path.parent}")

    try:
        exported_program = torch.export.load(str(ep_path))
    except Exception as exc:  # noqa: BLE001
        return _error(f"torch.export.load: {type(exc).__name__}: {exc}")

    try:
        raw_inputs = torch.load(gi_path, weights_only=False)
    except Exception as exc:  # noqa: BLE001
        return _error(f"torch.load(golden_inputs): {type(exc).__name__}: {exc}")
    inputs: tuple[torch.Tensor, ...] = (
        tuple(raw_inputs) if isinstance(raw_inputs, (list, tuple)) else (raw_inputs,)
    )

    try:
        # Reuse the bundle runner's context builder + parser. Keeps
        # the dialect set canonical with the bundle path.
        from xpu_rt.runtime.bundle_runner import _build_payload_context
        from xdsl.parser import Parser
        ctx = _build_payload_context()
        payload_module = Parser(ctx, payload_path.read_text()).parse_module()
    except Exception as exc:  # noqa: BLE001
        return _error(f"xdsl parse({payload_path.name}): {type(exc).__name__}: {exc}")

    try:
        from xpu_rt.runtime.cpu_executor import execute as _xpu_rt_execute
    except Exception as exc:  # noqa: BLE001
        return _error(f"import cpu_executor: {type(exc).__name__}: {exc}")

    # Warmup
    try:
        with torch.no_grad():
            for _ in range(n_warmup):
                _xpu_rt_execute(payload_module, exported_program, inputs)
    except Exception as exc:  # noqa: BLE001
        return _error(f"warmup: {type(exc).__name__}: {exc}")

    per_run_us: list[float] = []
    try:
        with torch.no_grad():
            for _ in range(n_iters):
                t0 = time.perf_counter()
                _xpu_rt_execute(payload_module, exported_program, inputs)
                per_run_us.append((time.perf_counter() - t0) * 1e6)
    except Exception as exc:  # noqa: BLE001
        return _error(f"iter: {type(exc).__name__}: {exc}")

    samples = sorted(per_run_us)
    n = len(samples)
    p50 = samples[n // 2]
    # NumPy-style nearest-rank p90 with linear interpolation between
    # the two adjacent samples (close enough for n=25; exact p90 is
    # samples[int(0.9 * (n-1))] under nearest-rank).
    k = int(0.9 * (n - 1))
    p90 = samples[k]
    return LatencyProbeResult(
        status="ok",
        detail="",
        n_warmup=n_warmup,
        n_iters=n_iters,
        latency_median_us=median(samples),
        latency_p50_us=p50,
        latency_p90_us=p90,
        latency_min_us=samples[0],
        latency_stddev_us=stdev(samples) if len(samples) > 1 else 0.0,
        per_run_us=tuple(per_run_us),
    )
