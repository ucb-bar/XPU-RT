"""Subsystem-ablation harness.

Runs the existing graph-compilation pipeline twice for each model —
once with `mask_a` active, once with `mask_b` active — and emits a
typed diff row. The default comparison is ``all_on`` (control) vs.
``<subsystem>=off`` (treatment); the mask is propagated via the
`XPU_RT_SUBSYSTEM_MASK` env var so subsystems can opt in without
threading a parameter through every layer.

This harness reuses `pass_pool_ablation.run_one_cell` for the actual
run — it already classifies outcomes, extracts agent picks, reads
validation reports, and detects promoted-candidate hits.

A run dir for cell *(model, mask_label)* lives at
``<out_root>/<model>__<mask_label>/`` and is byte-pinned by
`run_one_cell`'s `shutil.rmtree` + fresh-build contract. After the
run the harness writes ``subsystem_mask.json`` into the run dir so
each cell is self-describing.

The diff between two cells of the same model is captured in
`SubsystemAblationRow`, which extends `AblationResult` with three
delta columns:

- ``decision_seconds_delta`` — control - treatment
- ``candidate_changed``     — True iff selected_candidate_id differs
- ``outcome_changed``       — True iff typed_outcome differs

Latency / memory deltas are *not* derived from the request-only path
that `run_one_cell` exercises (it stops at agent-decision-request).
Phase-1 PRs that need real latency must either extend
`run_one_cell` to `stop_after="execution-plan-emit"` or layer a
torchbench measurement on top.

The mask off-path for any given subsystem is wired in its phase PR.
The foundation harness can already run (mask_a=all_on,
mask_b=all_on) — useful for noise-floor calibration — but a real
on/off run will raise `SubsystemMaskUnwiredError` from the
subsystem entry point until the phase PR ships.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from xpu_rt.benchmarks.latency_probe import (
    LatencyProbeResult,
    measure_run_dir_latency,
)
from xpu_rt.benchmarks.pass_pool_ablation import AblationResult, run_one_cell
from xpu_rt.benchmarks.subsystem_mask import (
    SubsystemMask,
    _ACTIVE_MASK_ENV,
)


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# Per-cell + per-diff result
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SubsystemAblationCell:
    """One (model, mask_label) cell — wraps an `AblationResult`."""

    mask_label: str  # "control" | "treatment" | custom
    mask: SubsystemMask
    result: AblationResult
    run_dir: str  # path to the cell's run directory, relative to out_root
    latency: LatencyProbeResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mask_label": self.mask_label,
            "mask": self.mask.to_dict(),
            "disabled_flags": list(self.mask.disabled_flags()),
            "result": self.result.to_dict(),
            "run_dir": self.run_dir,
            "latency": self.latency.to_dict() if self.latency else None,
        }


@dataclass(frozen=True)
class SubsystemAblationRow:
    """A control-vs-treatment diff for one model under one subsystem flag."""

    model_id: str
    target_id: str
    subsystem_flag: str  # e.g. "kernels.codegen_fallback"
    control: SubsystemAblationCell
    treatment: SubsystemAblationCell

    @property
    def candidate_changed(self) -> bool:
        return (
            self.control.result.selected_candidate_id
            != self.treatment.result.selected_candidate_id
        )

    @property
    def outcome_changed(self) -> bool:
        return self.control.result.typed_outcome != self.treatment.result.typed_outcome

    @property
    def decision_seconds_delta(self) -> float:
        return self.control.result.decision_seconds - self.treatment.result.decision_seconds

    @property
    def latency_median_us_delta(self) -> float | None:
        """Treatment - control (positive = treatment slower = control wins).

        Returns None when either side's latency probe didn't run.
        """
        if (
            self.control.latency is None
            or self.treatment.latency is None
            or self.control.latency.status != "ok"
            or self.treatment.latency.status != "ok"
        ):
            return None
        return self.treatment.latency.latency_median_us - self.control.latency.latency_median_us

    @property
    def control_speedup_pct(self) -> float | None:
        """Median-based percentage speedup of control vs treatment.

        ``(treatment_median - control_median) / treatment_median * 100``.
        Positive = control (subsystem on) is faster.
        Returns None when latency probes didn't run.

        **Use ``control_speedup_min_pct`` as the primary kill signal**
        on noisy executors; median can be dominated by per-run
        variance on sub-100us workloads.
        """
        if (
            self.control.latency is None
            or self.treatment.latency is None
            or self.control.latency.status != "ok"
            or self.treatment.latency.status != "ok"
        ):
            return None
        c = self.control.latency.latency_median_us
        t = self.treatment.latency.latency_median_us
        if t == 0.0:
            return None
        return (t - c) / t * 100.0

    @property
    def control_speedup_min_pct(self) -> float | None:
        """Min-based percentage speedup of control vs treatment.

        ``(treatment_min - control_min) / treatment_min * 100``. The
        min is the cleaner signal on noisy CPU workloads — it
        approximates the inherent op cost without the long tail of
        thermal/GC outliers that pull the median around.
        Positive = control (subsystem on) is faster.
        """
        if (
            self.control.latency is None
            or self.treatment.latency is None
            or self.control.latency.status != "ok"
            or self.treatment.latency.status != "ok"
        ):
            return None
        c = self.control.latency.latency_min_us
        t = self.treatment.latency.latency_min_us
        if t == 0.0:
            return None
        return (t - c) / t * 100.0

    @property
    def latency_noise_divergence_pp(self) -> float | None:
        """Absolute gap between median- and min-based speedup, in pp.

        Large gap (>5pp by default in the summary) means the median
        is being dragged around by outliers — treat the median delta
        as noise and trust the min-based result.
        """
        a = self.control_speedup_pct
        b = self.control_speedup_min_pct
        if a is None or b is None:
            return None
        return abs(a - b)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "target_id": self.target_id,
            "subsystem_flag": self.subsystem_flag,
            "control": self.control.to_dict(),
            "treatment": self.treatment.to_dict(),
            "deltas": {
                "candidate_changed": self.candidate_changed,
                "outcome_changed": self.outcome_changed,
                "decision_seconds_delta": self.decision_seconds_delta,
                "latency_median_us_delta": self.latency_median_us_delta,
                "control_speedup_pct": self.control_speedup_pct,
                "control_speedup_min_pct": self.control_speedup_min_pct,
                "latency_noise_divergence_pp": self.latency_noise_divergence_pp,
            },
        }


@dataclass
class SubsystemAblationPack:
    """Aggregate diff report across (model × subsystem_flag) pairs."""

    schema_version: str = "subsystem_ablation_pack_v1"
    generated_at_utc: str = field(default_factory=_utc_now)
    commit: str = ""
    subsystem_flag: str = ""
    rows: list[SubsystemAblationRow] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        n = len(self.rows)

        def _count(label: str, side: str) -> int:
            return sum(
                1 for r in self.rows
                if getattr(r, side).result.typed_outcome == label
            )

        control_verified = _count("verified", "control")
        treatment_verified = _count("verified", "treatment")

        # Latency aggregation. A row is "measured" iff both control
        # and treatment have an ok LatencyProbeResult. The primary
        # kill signal is min-based; median is reported alongside for
        # context + as a noise-divergence indicator.
        measured = [r for r in self.rows if r.control_speedup_pct is not None]
        n_meas = len(measured)

        def _med_pct(r: SubsystemAblationRow) -> float:
            return r.control_speedup_pct or 0.0

        def _min_pct(r: SubsystemAblationRow) -> float:
            return r.control_speedup_min_pct or 0.0

        # Min-based counts (primary kill signal).
        ge5_min = sum(1 for r in measured if _min_pct(r) >= 5.0)
        le_neg5_min = sum(1 for r in measured if _min_pct(r) <= -5.0)
        # Median-based counts (legacy / context).
        ge5_med = sum(1 for r in measured if _med_pct(r) >= 5.0)
        le_neg5_med = sum(1 for r in measured if _med_pct(r) <= -5.0)
        # Noise-divergence: count rows where median and min disagree
        # by more than 5pp. These rows' median values cannot be
        # trusted as a kill-rule signal.
        noisy = sum(
            1 for r in measured
            if (r.latency_noise_divergence_pp or 0.0) > 5.0
        )
        median_min_speedup_pct = (
            sorted(_min_pct(r) for r in measured)[n_meas // 2]
            if n_meas else 0.0
        )
        median_med_speedup_pct = (
            sorted(_med_pct(r) for r in measured)[n_meas // 2]
            if n_meas else 0.0
        )

        return {
            "subsystem_flag": self.subsystem_flag,
            "row_count": n,
            "candidate_changed_count": sum(1 for r in self.rows if r.candidate_changed),
            "outcome_changed_count": sum(1 for r in self.rows if r.outcome_changed),
            "latency_measured_count": n_meas,
            # Min-based (primary).
            "control_speedup_min_ge5pct_count": ge5_min,
            "control_speedup_min_le_neg5pct_count": le_neg5_min,
            "median_control_speedup_min_pct": median_min_speedup_pct,
            # Median-based (legacy; keep until callers migrate).
            "control_speedup_ge5pct_count": ge5_med,
            "control_speedup_le_neg5pct_count": le_neg5_med,
            "median_control_speedup_pct": median_med_speedup_pct,
            # Noise quality.
            "noise_divergent_row_count": noisy,
            # Codegen-success columns: verified > verification_fail
            # > typed_blocked > error. The kill rule for fusion calls
            # for ">=2 model uplift in codegen success" — i.e.
            # `control_verified_count - treatment_verified_count`.
            "control_verified_count": control_verified,
            "treatment_verified_count": treatment_verified,
            "codegen_success_uplift": control_verified - treatment_verified,
            "control_verification_fail_count": _count("verification_fail", "control"),
            "treatment_verification_fail_count": _count("verification_fail", "treatment"),
            "control_typed_blocked_count": _count("typed_blocked", "control"),
            "treatment_typed_blocked_count": _count("typed_blocked", "treatment"),
            "control_error_count": _count("error", "control"),
            "treatment_error_count": _count("error", "treatment"),
            "mean_decision_seconds_delta": (
                sum(r.decision_seconds_delta for r in self.rows) / n
                if n else 0.0
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at_utc": self.generated_at_utc,
            "commit": self.commit,
            "subsystem_flag": self.subsystem_flag,
            "summary": self.summary(),
            "rows": [r.to_dict() for r in self.rows],
        }


# --------------------------------------------------------------------------- #
# Cell runner: set env, run, write sidecar
# --------------------------------------------------------------------------- #


def _write_sidecar(run_dir: Path, mask: SubsystemMask, mask_label: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    sidecar = {
        "schema_version": "subsystem_mask_sidecar_v1",
        "mask_label": mask_label,
        "mask": mask.to_dict(),
        "disabled_flags": list(mask.disabled_flags()),
        "wired_flags": sorted(SubsystemMask._WIRED_FLAGS),
    }
    (run_dir / "subsystem_mask.json").write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_cell(
    *,
    model_yaml: Path,
    target_yaml: Path,
    out_dir: Path,
    mask: SubsystemMask,
    mask_label: str,
    mode: str = "greedy",
    agent_response_path: Path | None = None,
    stop_after: str = "agent-decision-request",
    latency_iters: int = 0,
    latency_warmup: int = 3,
) -> SubsystemAblationCell:
    """Run one (model, mask) cell.

    The mask is exported to `XPU_RT_SUBSYSTEM_MASK` (as a
    comma-separated disable list) for the duration of the call.
    Subsystem entry points read the env var via
    `subsystem_mask.active_mask_from_env`. The env var is restored
    on exit so the caller's environment is unchanged.
    """
    disable_list = ",".join(mask.disabled_flags())
    prior = os.environ.get(_ACTIVE_MASK_ENV)
    if disable_list:
        os.environ[_ACTIVE_MASK_ENV] = disable_list
    else:
        os.environ.pop(_ACTIVE_MASK_ENV, None)
    try:
        result = run_one_cell(
            model_yaml=model_yaml,
            target_yaml=target_yaml,
            out_dir=out_dir,
            mode=mode,
            agent_response_path=agent_response_path,
            stop_after=stop_after,
        )
    finally:
        if prior is None:
            os.environ.pop(_ACTIVE_MASK_ENV, None)
        else:
            os.environ[_ACTIVE_MASK_ENV] = prior

    _write_sidecar(out_dir, mask, mask_label)

    latency: LatencyProbeResult | None = None
    if latency_iters > 0 and result.typed_outcome in ("verified", "verification_fail"):
        # Only probe latency when the run produced something runnable
        # (verified or verification_fail — both have a transformed
        # payload). Skipped for typed_blocked / error since there's
        # nothing to time.
        latency = measure_run_dir_latency(
            out_dir, n_warmup=latency_warmup, n_iters=latency_iters,
        )

    return SubsystemAblationCell(
        mask_label=mask_label,
        mask=mask,
        result=result,
        run_dir=str(out_dir),
        latency=latency,
    )


# --------------------------------------------------------------------------- #
# Suite runner
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SubsystemAblationSpec:
    """One model + one subsystem-flag pair to ablate."""

    model_yaml: Path
    target_yaml: Path
    subsystem_flag: str
    mode: str = "greedy"
    agent_response_path: Path | None = None
    stop_after: str = "agent-decision-request"
    latency_iters: int = 0  # 0 disables the probe
    latency_warmup: int = 3


def run_subsystem_ablation(
    specs: Iterable[SubsystemAblationSpec],
    *,
    out_root: Path,
    commit: str = "",
    subsystem_flag: str = "",
) -> SubsystemAblationPack:
    """Run control + treatment cells for each spec and aggregate.

    Control mask: `SubsystemMask.all_on()`.
    Treatment mask: `all_on().from_disable_list([spec.subsystem_flag])`.

    Cells live at:
        ``<out_root>/<model>__control/``
        ``<out_root>/<model>__treatment__<flag>/``
    """
    out_root = Path(out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    pack = SubsystemAblationPack(commit=commit, subsystem_flag=subsystem_flag)

    control_mask = SubsystemMask.all_on()
    for spec in specs:
        treatment_mask = SubsystemMask.from_disable_list([spec.subsystem_flag])
        flag_slug = spec.subsystem_flag.replace(".", "_")

        control_dir = out_root / f"{spec.model_yaml.stem}__control"
        treatment_dir = out_root / f"{spec.model_yaml.stem}__treatment__{flag_slug}"

        control_cell = run_cell(
            model_yaml=spec.model_yaml,
            target_yaml=spec.target_yaml,
            out_dir=control_dir,
            mask=control_mask,
            mask_label="control",
            mode=spec.mode,
            agent_response_path=spec.agent_response_path,
            stop_after=spec.stop_after,
            latency_iters=spec.latency_iters,
            latency_warmup=spec.latency_warmup,
        )
        treatment_cell = run_cell(
            model_yaml=spec.model_yaml,
            target_yaml=spec.target_yaml,
            out_dir=treatment_dir,
            mask=treatment_mask,
            mask_label=f"treatment__{spec.subsystem_flag}",
            mode=spec.mode,
            agent_response_path=spec.agent_response_path,
            stop_after=spec.stop_after,
            latency_iters=spec.latency_iters,
            latency_warmup=spec.latency_warmup,
        )
        pack.rows.append(SubsystemAblationRow(
            model_id=control_cell.result.model_id,
            target_id=control_cell.result.target_id,
            subsystem_flag=spec.subsystem_flag,
            control=control_cell,
            treatment=treatment_cell,
        ))
    return pack


def emit_pack(pack: SubsystemAblationPack, *, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(pack.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out_path


# --------------------------------------------------------------------------- #
# Noise-floor calibration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class NoiseFloorEntry:
    model_id: str
    target_id: str
    n_repeats: int
    mean_decision_seconds: float
    stddev_decision_seconds: float
    decision_seconds_samples: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "target_id": self.target_id,
            "n_repeats": self.n_repeats,
            "mean_decision_seconds": self.mean_decision_seconds,
            "stddev_decision_seconds": self.stddev_decision_seconds,
            "decision_seconds_samples": list(self.decision_seconds_samples),
        }


def _stddev(samples: list[float]) -> float:
    if len(samples) < 2:
        return 0.0
    mean = sum(samples) / len(samples)
    var = sum((x - mean) ** 2 for x in samples) / (len(samples) - 1)
    return var ** 0.5


def calibrate_noise_floor(
    model_yamls: list[Path],
    target_yaml: Path,
    *,
    out_root: Path,
    n_repeats: int = 3,
    mode: str = "greedy",
) -> list[NoiseFloorEntry]:
    """Run all-on N times per model; record per-model decision-seconds stddev.

    Writes per-repeat run dirs at ``out_root/<model>__noise_<i>/`` and
    a summary at ``out_root/noise_floor.json``.
    """
    out_root = Path(out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    mask = SubsystemMask.all_on()
    entries: list[NoiseFloorEntry] = []
    for model_yaml in model_yamls:
        samples: list[float] = []
        target_id = target_yaml.stem
        for i in range(n_repeats):
            run_dir = out_root / f"{model_yaml.stem}__noise_{i}"
            cell = run_cell(
                model_yaml=model_yaml,
                target_yaml=target_yaml,
                out_dir=run_dir,
                mask=mask,
                mask_label=f"noise_{i}",
                mode=mode,
            )
            samples.append(cell.result.decision_seconds)
        mean = sum(samples) / len(samples) if samples else 0.0
        entries.append(NoiseFloorEntry(
            model_id=model_yaml.stem,
            target_id=target_id,
            n_repeats=n_repeats,
            mean_decision_seconds=mean,
            stddev_decision_seconds=_stddev(samples),
            decision_seconds_samples=tuple(samples),
        ))
    payload = {
        "schema_version": "noise_floor_v1",
        "generated_at_utc": _utc_now(),
        "n_repeats": n_repeats,
        "mode": mode,
        "entries": [e.to_dict() for e in entries],
    }
    (out_root / "noise_floor.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return entries


def below_noise_floor(
    *,
    delta_seconds: float,
    noise_stddev: float,
    relative_floor: float = 0.05,
    sigma_floor: float = 2.0,
) -> bool:
    """Is the observed delta inside the noise band?

    A delta is "no signal" if it's smaller than max(sigma_floor *
    stddev, relative_floor * |delta|). Returns True iff the delta is
    below the floor and the result should be treated as a kill-rule
    failure (per the plan's per-component criterion).
    """
    floor = max(sigma_floor * noise_stddev, relative_floor * abs(delta_seconds))
    return abs(delta_seconds) <= floor
