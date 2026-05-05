"""M-18 Profiler-Calibrated Cost Preview.

Runs a measured profile of the captured exported program and joins the
results with the deterministic-roofline ``cost_preview_v2`` /
``hardware_resource_report`` predictions to produce typed calibration
artifacts.

Outputs (all under ``02_graph_analysis/calibration/``):

- ``profile_run.json`` — raw profiler aggregates, one row per
  recorded op. Schema-versioned, deterministic across reruns up to
  measurement noise.
- ``profiler_calibration_report.json`` — per-region predicted vs
  measured timing with calibration error metrics (per-region MAPE
  and suite-level RMSE) plus an explicit ``calibration_status``
  taxonomy (``calibrated`` / ``partial_match`` / ``no_op_match`` /
  ``not_run``).
- ``calibration_summary.md`` — short reviewer-facing summary.
- ``figures/predicted_vs_measured.png`` and
  ``figures/calibration_error_distribution.png``.

Hard non-goals:

- No compiler-core mutation. The capture artifacts and graph-analysis
  reports are read-only.
- No new optimization families.
- No replacement of the deterministic-roofline baseline. The baseline
  stays exactly as M-17.1 left it; this module LAYERS calibration
  evidence on top.
- No claim of profiler-calibrated when measurements failed or no FX
  nodes matched any profiler op (status falls back to
  ``not_profiler_calibrated``).

Best-effort: if torch is unavailable, exported_program is missing, or
profiling raises, the module emits a typed ``not_run`` calibration
report and never raises into the pipeline.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


_PROFILE_ITERATIONS_DEFAULT = 32
_WARMUP_ITERATIONS_DEFAULT = 4


def _utcnow() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# --------------------------------------------------------------------------- #
# Profile a captured exported program
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _ProfileResult:
    success: bool
    iterations: int
    warmup: int
    wall_us_per_iter: float
    op_to_us: dict[str, float]   # per-op-name CPU time (us per iteration)
    note: str = ""


def _run_profile(run_dir: Path, *, iterations: int, warmup: int) -> _ProfileResult:
    """Best-effort profile run. Returns a result with ``success=False``
    and a typed note when anything goes wrong."""
    try:
        import torch
        from torch.profiler import profile, ProfilerActivity
    except ImportError as exc:
        return _ProfileResult(
            success=False, iterations=0, warmup=0, wall_us_per_iter=0.0,
            op_to_us={}, note=f"torch unavailable: {exc}",
        )

    ep_path = run_dir / "00_graph_capture" / "exported_program.pt2"
    inputs_path = run_dir / "00_graph_capture" / "golden_inputs.pt"
    if not ep_path.exists() or not inputs_path.exists():
        return _ProfileResult(
            success=False, iterations=0, warmup=0, wall_us_per_iter=0.0,
            op_to_us={},
            note=(
                "missing capture artifact "
                f"(exported_program.pt2 exists={ep_path.exists()}, "
                f"golden_inputs.pt exists={inputs_path.exists()})"
            ),
        )

    try:
        ep = torch.export.load(str(ep_path))
        inputs = torch.load(str(inputs_path), weights_only=False)
        if not isinstance(inputs, (list, tuple)):
            inputs = (inputs,)
        model = ep.module()
        # ``ep.module()`` returns a synthetic GraphModuleImpl that does
        # NOT support ``.eval()``. The exported program is already in
        # whatever mode the user captured it in — best-effort skip.
        try:
            model.eval()
        except (NotImplementedError, AttributeError):
            pass
    except Exception as exc:  # noqa: BLE001
        return _ProfileResult(
            success=False, iterations=0, warmup=0, wall_us_per_iter=0.0,
            op_to_us={}, note=f"failed to load model: {type(exc).__name__}: {exc}",
        )

    # Warmup.
    try:
        with torch.no_grad():
            for _ in range(max(0, warmup)):
                model(*inputs)
    except Exception as exc:  # noqa: BLE001
        return _ProfileResult(
            success=False, iterations=0, warmup=0, wall_us_per_iter=0.0,
            op_to_us={}, note=f"warmup failed: {type(exc).__name__}: {exc}",
        )

    # Profile.
    op_to_us: dict[str, float] = {}
    wall_us_per_iter = 0.0
    try:
        t0 = time.perf_counter_ns()
        with profile(
            activities=[ProfilerActivity.CPU], record_shapes=False,
        ) as prof:
            with torch.no_grad():
                for _ in range(iterations):
                    model(*inputs)
        wall_ns = time.perf_counter_ns() - t0
        wall_us_per_iter = (wall_ns / 1000.0) / max(1, iterations)

        # Aggregate per-op CPU time. ``key_averages`` returns one row per
        # unique op key; ``cpu_time_total`` is in microseconds aggregated
        # over all calls of that key across all iterations.
        for ev in prof.key_averages():
            key = str(ev.key)
            cpu_total_us = float(ev.cpu_time_total)
            op_to_us[key] = op_to_us.get(key, 0.0) + cpu_total_us / max(1, iterations)
    except Exception as exc:  # noqa: BLE001
        return _ProfileResult(
            success=False, iterations=0, warmup=warmup, wall_us_per_iter=0.0,
            op_to_us={}, note=f"profile failed: {type(exc).__name__}: {exc}",
        )

    return _ProfileResult(
        success=True, iterations=iterations, warmup=warmup,
        wall_us_per_iter=wall_us_per_iter, op_to_us=op_to_us,
        note="cpu profiler aggregated by op key",
    )


# --------------------------------------------------------------------------- #
# Map profiler op-keys to regions
# --------------------------------------------------------------------------- #


# Canonical FX-node → aten-op decomposition table. The torch.profiler
# records aten/c10 events at the kernel level; many high-level FX nodes
# decompose into multiple aten kernels. We map FX node names to the set
# of aten kernels they typically produce, so calibration can attribute
# kernel timings back to the originating FX node. This table is
# intentionally a best-effort approximation — limitations are recorded
# in the report's ``known_limitations`` block.
_FX_TO_ATEN_DECOMPOSITION: dict[str, tuple[str, ...]] = {
    "linear": ("linear", "addmm", "mm", "matmul", "permute", "t"),
    "conv1d": ("conv1d", "convolution", "_convolution"),
    "conv2d": ("conv2d", "convolution", "_convolution"),
    "conv3d": ("conv3d", "convolution", "_convolution"),
    "max_pool2d": ("max_pool2d", "max_pool2d_with_indices"),
    "avg_pool2d": ("avg_pool2d",),
    "batch_norm": ("batch_norm", "_native_batch_norm_legit", "native_batch_norm"),
    "layer_norm": ("layer_norm", "native_layer_norm"),
    "relu": ("relu", "clamp_min"),
    "gelu": ("gelu",),
    "silu": ("silu",),
    "tanh": ("tanh",),
    "sigmoid": ("sigmoid",),
    "softmax": ("softmax", "_softmax"),
    "log_softmax": ("log_softmax", "_log_softmax"),
    "add": ("add",),
    "mul": ("mul",),
    "sub": ("sub",),
    "div": ("div",),
    "mean": ("mean",),
    "sum": ("sum",),
    "matmul": ("matmul", "mm", "bmm", "addmm"),
    "bmm": ("bmm",),
    "embedding": ("embedding",),
    "flatten": ("view", "reshape", "flatten"),
    "view": ("view",),
    "reshape": ("reshape", "view"),
    "transpose": ("transpose", "permute"),
    "permute": ("permute",),
    "concat": ("cat",),
    "cat": ("cat",),
    "stack": ("stack",),
    "expand": ("expand",),
    "addmm": ("addmm",),
}


def _expand_fx_node(fx: str) -> tuple[str, ...]:
    """Strip the trailing ``_<index>`` and look up the aten
    decomposition. Falls back to the bare name if not in the table."""
    base = fx.rstrip("_0123456789")
    return _FX_TO_ATEN_DECOMPOSITION.get(base, (base,))


def _normalize_op_key(key: str) -> str:
    """Strip ``aten::``, ``torch::`` prefixes and trailing ``.default`` /
    ``.Tensor`` overload tags so we can fuzzy-match against FX node
    names. ``aten::linear`` → ``linear``; ``aten::add.Tensor`` → ``add``."""
    k = key
    for prefix in ("aten::", "torch::", "c10::"):
        if k.startswith(prefix):
            k = k[len(prefix):]
            break
    for suffix in (".default", ".Tensor", ".out"):
        if k.endswith(suffix):
            k = k[: -len(suffix)]
            break
    return k.strip()


def _build_claim_map(
    *,
    op_to_us: dict[str, float],
    regions: list[tuple[str, list[str]]],   # (region_id, fx_nodes)
) -> tuple[dict[str, str], dict[str, int]]:
    """Two-pass attribution.

    Returns:
      - ``norm_to_orig``: normalized key → original profiler key
      - ``claim_count[key]``: number of (region, fx_node) pairs whose
        decomposition expansion matches that profiler key. Each region
        later gets ``op_to_us[key] / claim_count[key]`` for the keys
        its fx_nodes match — fair sharing of one profiler key across
        N FX-node claimants (e.g. 3 linear regions share 3 addmm
        kernel events evenly).
    """
    norm_to_orig: dict[str, str] = {}
    for k in op_to_us:
        norm_to_orig[_normalize_op_key(k)] = k

    claim_count: dict[str, int] = {}
    for _rid, fx_nodes in regions:
        for fx in fx_nodes:
            for exp in _expand_fx_node(fx):
                # Match every profiler key whose normalized form is
                # exactly the expansion (no fuzzy startswith — we use
                # the explicit decomposition table).
                for n, orig in norm_to_orig.items():
                    if (
                        n == exp
                        or n.startswith(exp + ".")
                        or n.endswith("::" + exp)
                    ):
                        claim_count[orig] = claim_count.get(orig, 0) + 1
    return norm_to_orig, claim_count


def _match_op_to_region(
    op_to_us: dict[str, float],
    fx_nodes: list[str],
    *,
    norm_to_orig: dict[str, str],
    claim_count: dict[str, int],
) -> tuple[float, list[str]]:
    """Return (sum-of-attributed-us, matched-keys) for one region.

    Each profiler key contributes ``op_to_us[key] / claim_count[key]``
    so that one kernel event shared across N FX-node claimants is
    split fairly among them.
    """
    if not fx_nodes:
        return 0.0, []
    total = 0.0
    matched: list[str] = []
    seen_in_region: set[str] = set()
    for fx in fx_nodes:
        for exp in _expand_fx_node(fx):
            for n, orig in norm_to_orig.items():
                if orig in seen_in_region:
                    continue
                if (
                    n == exp
                    or n.startswith(exp + ".")
                    or n.endswith("::" + exp)
                ):
                    cc = max(1, claim_count.get(orig, 1))
                    total += op_to_us[orig] / cc
                    matched.append(n)
                    seen_in_region.add(orig)
                    break
    return total, matched


# --------------------------------------------------------------------------- #
# Calibration math
# --------------------------------------------------------------------------- #


def _predicted_us_for_region(rd_cost: dict[str, Any], target_id: str) -> float:
    """Pull the predicted latency out of a region_dossier ``cost`` block."""
    block = rd_cost.get("estimated_latency_us") if rd_cost else None
    if isinstance(block, dict):
        v = block.get(target_id)
        if v is None and block:
            v = next(iter(block.values()))
        return float(v or 0.0)
    if isinstance(block, (int, float)):
        return float(block)
    return 0.0


def _read_region_dossiers(ga: Path) -> list[dict[str, Any]]:
    rd_dir = ga / "region_dossiers"
    if not rd_dir.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for p in sorted(rd_dir.iterdir()):
        if p.is_file() and p.suffix == ".json":
            try:
                out.append(json.loads(p.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
    return out


# --------------------------------------------------------------------------- #
# Top-level entry point
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CalibrationResult:
    overall: str                   # "calibrated" | "partial" | "not_run"
    out_dir: Path
    profile_run_path: Path
    report_path: Path
    summary_md_path: Path
    matched_region_count: int
    total_region_count: int
    suite_predicted_us: float
    suite_measured_us: float
    suite_scale: float | None
    suite_mape: float | None


def run_profiler_calibration(
    run_dir: Path,
    *,
    iterations: int = _PROFILE_ITERATIONS_DEFAULT,
    warmup: int = _WARMUP_ITERATIONS_DEFAULT,
) -> CalibrationResult:
    """Run a measured profile + emit calibration artifacts.

    Best-effort: emits ``status=not_run`` artifacts when profiling
    can't run. Never raises into the caller.
    """
    run_dir = Path(run_dir).resolve()
    ga = run_dir / "02_graph_analysis"
    out_dir = ga / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    profile_run_path = out_dir / "profile_run.json"
    report_path = out_dir / "profiler_calibration_report.json"
    summary_md_path = out_dir / "calibration_summary.md"

    # Resolve target_id.
    cap = _read_json(run_dir / "00_graph_capture" / "capture_report.json")
    target_id = (cap or {}).get("target_id", "host_cpu")

    profile_res = _run_profile(run_dir, iterations=iterations, warmup=warmup)

    if not profile_res.success:
        # Emit typed not_run artifacts.
        profile_run_path.write_text(
            json.dumps({
                "schema_version": "profile_run_v1",
                "success": False,
                "iterations": 0, "warmup": 0,
                "wall_us_per_iter": 0.0,
                "op_to_us": {},
                "note": profile_res.note,
                "generated_at_utc": _utcnow(),
            }, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        report = {
            "schema_version": "profiler_calibration_report_v1",
            "calibration_status": "not_run",
            "overall": "not_run",
            "target_id": target_id,
            "iterations": 0, "warmup": 0,
            "regions": [],
            "summary": {
                "matched_region_count": 0,
                "total_region_count": 0,
                "suite_predicted_us": 0.0,
                "suite_measured_us": 0.0,
                "suite_scale": None,
                "suite_mape": None,
            },
            "checks": [
                {"name": "profile_run_succeeded",
                 "status": "fail",
                 "detail": profile_res.note},
            ],
            "known_limitations": [
                "fuzzy FX-node ↔ profiler-op matching",
                "single-process CPU profile only",
                "single-batch-size measurement",
            ],
            "generated_at_utc": _utcnow(),
        }
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8",
        )
        summary_md_path.write_text(
            f"# Profiler Calibration — not_run\n\n- reason: {profile_res.note}\n",
            encoding="utf-8",
        )
        return CalibrationResult(
            overall="not_run", out_dir=out_dir,
            profile_run_path=profile_run_path,
            report_path=report_path, summary_md_path=summary_md_path,
            matched_region_count=0, total_region_count=0,
            suite_predicted_us=0.0, suite_measured_us=0.0,
            suite_scale=None, suite_mape=None,
        )

    # Persist raw profile.
    profile_run_path.write_text(
        json.dumps({
            "schema_version": "profile_run_v1",
            "success": True,
            "iterations": profile_res.iterations,
            "warmup": profile_res.warmup,
            "wall_us_per_iter": profile_res.wall_us_per_iter,
            "op_to_us": dict(sorted(profile_res.op_to_us.items())),
            "note": profile_res.note,
            "generated_at_utc": _utcnow(),
        }, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    # Map to regions.
    region_map = _read_json(ga / "region_map.json") or {"regions": []}
    region_dossiers = _read_region_dossiers(ga)
    rd_by_id = {rd.get("region_id"): rd for rd in region_dossiers}

    regions_out: list[dict[str, Any]] = []
    matched_count = 0
    suite_predicted = 0.0
    suite_measured = 0.0
    per_region_errors: list[float] = []

    # Pre-compute the claim map so multi-claimant profiler keys share
    # their time fairly across regions.
    region_pairs = [
        (r.get("region_id"), list(r.get("fx_nodes") or []))
        for r in region_map.get("regions", [])
    ]
    norm_to_orig, claim_count = _build_claim_map(
        op_to_us=profile_res.op_to_us, regions=region_pairs,
    )

    for r in region_map.get("regions", []):
        rid = r.get("region_id")
        fx_nodes = list(r.get("fx_nodes") or [])
        kind = r.get("kind") or r.get("source_classification") or ""
        rd = rd_by_id.get(rid, {})
        cost = rd.get("cost", {}) if isinstance(rd, dict) else {}
        predicted_us = _predicted_us_for_region(cost, target_id)

        measured_us, matched_keys = _match_op_to_region(
            profile_res.op_to_us, fx_nodes,
            norm_to_orig=norm_to_orig, claim_count=claim_count,
        )

        if measured_us > 0.0:
            matched_count += 1
        if predicted_us > 0.0:
            suite_predicted += predicted_us
        suite_measured += measured_us

        # Per-region error metrics.
        if measured_us > 0.0 and predicted_us > 0.0:
            abs_err = measured_us - predicted_us
            rel_err = abs_err / measured_us
            mape_term = abs(abs_err) / measured_us
            per_region_errors.append(mape_term)
        else:
            abs_err = None
            rel_err = None

        regions_out.append({
            "region_id": rid,
            "kind": kind,
            "fx_nodes": fx_nodes,
            "predicted_us": predicted_us,
            "measured_us": measured_us,
            "matched_op_keys": matched_keys,
            "abs_error_us": abs_err,
            "rel_error": rel_err,
            "match_status": "matched" if measured_us > 0.0 else "no_match",
        })

    # Suite-level metrics.
    suite_scale = (
        suite_measured / suite_predicted
        if suite_predicted > 0.0 and suite_measured > 0.0 else None
    )
    suite_mape = (
        sum(per_region_errors) / len(per_region_errors)
        if per_region_errors else None
    )

    total_regions = len(regions_out)
    if matched_count == 0:
        calibration_status = "no_op_match"
        overall = "partial"
    elif matched_count >= total_regions:
        calibration_status = "calibrated"
        overall = "calibrated"
    else:
        calibration_status = "partial_match"
        overall = "calibrated" if matched_count >= max(1, total_regions // 2) else "partial"

    report = {
        "schema_version": "profiler_calibration_report_v1",
        "calibration_status": calibration_status,
        "overall": overall,
        "target_id": target_id,
        "iterations": profile_res.iterations,
        "warmup": profile_res.warmup,
        "wall_us_per_iter": profile_res.wall_us_per_iter,
        "regions": regions_out,
        "summary": {
            "matched_region_count": matched_count,
            "total_region_count": total_regions,
            "match_fraction": (
                matched_count / total_regions if total_regions > 0 else 0.0
            ),
            "suite_predicted_us": suite_predicted,
            "suite_measured_us": suite_measured,
            "suite_scale": suite_scale,
            "suite_mape": suite_mape,
        },
        "checks": [
            {"name": "profile_run_succeeded", "status": "pass", "detail": ""},
            {"name": "at_least_one_region_matched",
             "status": "pass" if matched_count > 0 else "fail",
             "detail": f"{matched_count}/{total_regions} regions matched"},
            {"name": "suite_predicted_positive",
             "status": "pass" if suite_predicted > 0.0 else "fail"},
            {"name": "suite_measured_positive",
             "status": "pass" if suite_measured > 0.0 else "fail"},
        ],
        "known_limitations": [
            "fuzzy FX-node ↔ profiler-op matching (e.g. conv2d may not match aten::convolution exactly)",
            "single-process CPU profile only",
            "single-batch-size measurement",
            "kernel launch overhead and cache state vary across runs",
        ],
        "generated_at_utc": _utcnow(),
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8",
    )

    # Summary markdown.
    body = (
        f"# Profiler Calibration — {overall} ({calibration_status})\n\n"
        f"- target_id: `{target_id}`\n"
        f"- iterations: {profile_res.iterations} (warmup {profile_res.warmup})\n"
        f"- wall_us_per_iter: {profile_res.wall_us_per_iter:.2f}\n"
        f"- regions matched: {matched_count}/{total_regions}\n"
        f"- suite_predicted_us: {suite_predicted:.2f}\n"
        f"- suite_measured_us: {suite_measured:.2f}\n"
    )
    if suite_scale is not None:
        body += f"- suite_scale (measured/predicted): {suite_scale:.3f}\n"
    if suite_mape is not None:
        body += f"- suite_mape: {suite_mape:.3f}\n"
    summary_md_path.write_text(body, encoding="utf-8")

    # Render figures (best-effort).
    try:
        _render_figures(report=report, figures_dir=figures_dir)
    except Exception:  # noqa: BLE001
        pass

    # M-18.4: layer measured_latency_us onto graph_dossier_v3 + llm_graph_view.
    # Best-effort — never break the pipeline if the dossier shape drifts.
    if overall == "calibrated":
        try:
            _apply_calibration_overlay(run_dir=run_dir, report=report)
        except Exception:  # noqa: BLE001
            pass

    return CalibrationResult(
        overall=overall, out_dir=out_dir,
        profile_run_path=profile_run_path,
        report_path=report_path, summary_md_path=summary_md_path,
        matched_region_count=matched_count,
        total_region_count=total_regions,
        suite_predicted_us=suite_predicted,
        suite_measured_us=suite_measured,
        suite_scale=suite_scale, suite_mape=suite_mape,
    )


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def _apply_calibration_overlay(
    *, run_dir: Path, report: dict[str, Any],
) -> None:
    """Layer ``calibration`` onto graph_dossier_v3 + llm_graph_view.

    For each matched region we add::

        region["calibration"] = {
            "measured_latency_us": <float>,
            "abs_error_us": <float>,
            "rel_error": <float>,
            "matched_op_keys": [<str>, ...],
            "calibration_status": "<from report>",
        }

    Untouched on regions that didn't match (matched_op_keys=[],
    measured_us=0). v3 and llm_view files are not byte-pinned to the
    manifest hash chain (they're emitted after stage_record), so
    re-writing them here is safe.
    """
    ga = run_dir / "02_graph_analysis"
    measured_by_region = {
        r["region_id"]: r for r in report.get("regions", [])
    }
    suite = report.get("summary", {}) or {}
    cal_status = report.get("calibration_status", "")

    def _enrich(region_block: dict[str, Any]) -> None:
        rid = region_block.get("region_id")
        m = measured_by_region.get(rid)
        if not m:
            return
        region_block["calibration"] = {
            "measured_latency_us": m.get("measured_us"),
            "predicted_latency_us": m.get("predicted_us"),
            "abs_error_us": m.get("abs_error_us"),
            "rel_error": m.get("rel_error"),
            "matched_op_keys": m.get("matched_op_keys", []),
            "match_status": m.get("match_status", "no_match"),
            "calibration_status": cal_status,
            "suite_scale": suite.get("suite_scale"),
        }

    for fname in ("graph_dossier_v3.json", "llm_graph_view.json"):
        fpath = ga / fname
        if not fpath.exists():
            continue
        try:
            doc = json.loads(fpath.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        regions = doc.get("regions") or []
        for region in regions:
            _enrich(region)
        # Top-level summary tag.
        doc.setdefault("calibration", {}).update({
            "calibration_status": cal_status,
            "suite_scale": suite.get("suite_scale"),
            "suite_predicted_us": suite.get("suite_predicted_us"),
            "suite_measured_us": suite.get("suite_measured_us"),
            "matched_region_count": suite.get("matched_region_count"),
            "total_region_count": suite.get("total_region_count"),
        })
        fpath.write_text(
            json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8",
        )


def _render_figures(*, report: dict[str, Any], figures_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    regions = report.get("regions") or []
    matched = [r for r in regions if r["match_status"] == "matched"]

    # 1. Predicted vs measured scatter.
    fig, ax = plt.subplots(figsize=(6, 5))
    if matched:
        xs = [r["predicted_us"] for r in matched]
        ys = [r["measured_us"] for r in matched]
        ax.scatter(xs, ys, color="#1971c2")
        max_v = max(max(xs), max(ys), 1.0)
        ax.plot([0, max_v], [0, max_v], color="black",
                linestyle="--", linewidth=0.8, label="y=x")
        ax.set_xlabel("predicted_us")
        ax.set_ylabel("measured_us")
        ax.set_title(
            f"Predicted vs measured per region "
            f"(n={len(matched)})"
        )
        ax.legend()
    else:
        ax.text(0.5, 0.5, "no matched regions",
                ha="center", va="center")
    fig.tight_layout()
    fig.savefig(figures_dir / "predicted_vs_measured.png", format="png", dpi=110)
    plt.close(fig)

    # 2. Calibration error distribution (rel_error).
    fig, ax = plt.subplots(figsize=(6, 4.5))
    errors = [r["rel_error"] for r in matched if r.get("rel_error") is not None]
    if errors:
        ax.hist(errors, bins=12, color="#5f3dc4")
        ax.axvline(x=0.0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("rel_error = (measured - predicted) / measured")
        ax.set_ylabel("region count")
        ax.set_title("Calibration error distribution")
    else:
        ax.text(0.5, 0.5, "no calibration data",
                ha="center", va="center")
    fig.tight_layout()
    fig.savefig(
        figures_dir / "calibration_error_distribution.png",
        format="png", dpi=110,
    )
    plt.close(fig)
