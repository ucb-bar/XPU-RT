"""Per-Tile-Candidate Measured Cost.

Layered on top of (region-level calibration). For each legal
``set_tile_params`` candidate, this module:

1. Reconstructs the matmul shape (M, N, K) and the tile
   parameters (tM, tN, tK) from the candidate's ``recipe_delta`` +
   the cost_preview_v2 diagnostics.
2. Generates a deterministic input pair on CPU.
3. Times an un-tiled baseline (``torch.matmul``) and the
   ``_tiled_matmul_eval`` boundary-aware tiled evaluator, both with
   warmup + N iterations.
4. Computes per-candidate ``measured_speedup``, ``rel_error`` vs
   the predicted ``candidate_static_latency_us``.
5. Emits ``02_graph_analysis/candidate_calibration/
   candidate_calibration_report.json``.
6. Layers ``calibration`` blocks onto each tile candidate's entry in
   ``cost_preview_v2.json`` and ``llm_graph_view.json``.

This turns the dossier from "calibrated region facts" into
"calibrated candidate consequences" — the agent now sees, per
candidate, both the deterministic prediction AND the measured cost.

Hard non-goals:

- No new candidate generation, no new transforms, no kernel codegen.
- Source payload artifacts unchanged.
- No compiler-core mutation.
- Best-effort: torch unavailable / cost_preview missing → typed
  ``not_run`` report; never raises into the pipeline.
- SetTileParams only for the MVP. Fusion candidates (which produce
  identical numerical output) are out of scope here — measuring them
  per-candidate would only show launch-overhead noise.

Opt-in via ``COMPGEN_CALIBRATE_CANDIDATES=1``. Default OFF so suite
runs stay deterministic.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


_DEFAULT_ITERATIONS = 16
_DEFAULT_WARMUP = 3


def _utcnow() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


@dataclass(frozen=True)
class CandidateCalibrationResult:
    overall: str           # "calibrated" | "no_candidates" | "not_run"
    out_dir: Path
    report_path: Path
    summary_md_path: Path
    candidate_count: int
    candidates_calibrated: int


# --------------------------------------------------------------------------- #
# Per-candidate measurement
# --------------------------------------------------------------------------- #


def _extract_shape_and_tile(
    candidate: dict[str, Any], cost_preview: dict[str, Any] | None,
) -> tuple[tuple[int, int, int], tuple[int, int, int]] | None:
    """Return ``((M, N, K), (tM, tN, tK))`` for a set_tile_params
    candidate, or None when the metadata is incomplete."""
    delta_list = candidate.get("recipe_delta") or []
    if not delta_list:
        return None
    delta = delta_list[0]
    if delta.get("op") != "SetTileParams":
        return None
    tile = delta.get("tile") or {}
    try:
        tM = int(tile["M"]); tN = int(tile["N"]); tK = int(tile["K"])
    except (KeyError, ValueError, TypeError):
        return None
    if cost_preview is None:
        return None
    diag = (cost_preview.get("diagnostics") or {})
    iters = (diag.get("candidate") or {}).get("iters") or {}
    try:
        iM = int(iters.get("M") or 1)
        iN = int(iters.get("N") or 1)
        iK = int(iters.get("K") or 1)
    except (TypeError, ValueError):
        return None
    M = tM * iM
    N = tN * iN
    K = tK * iK
    if M <= 0 or N <= 0 or K <= 0:
        return None
    return (M, N, K), (tM, tN, tK)


def _measure_one_candidate(
    *,
    M: int, N: int, K: int,
    tM: int, tN: int, tK: int,
    iterations: int, warmup: int,
):  # type: ignore[no-untyped-def]
    """Time both the un-tiled baseline (torch.matmul) and the
    tiled evaluator. Returns (baseline_us_per_iter, tiled_us_per_iter)."""
    import torch
    from compgen.graph_compilation.real_transform_differential import (
        _tiled_matmul_eval,
    )

    g = torch.Generator()
    g.manual_seed(0xCA11B0)
    A = torch.randn(M, K, dtype=torch.float32, generator=g)
    B = torch.randn(K, N, dtype=torch.float32, generator=g)

    # Baseline.
    with torch.no_grad():
        for _ in range(warmup):
            torch.matmul(A, B)
        t0 = time.perf_counter_ns()
        for _ in range(iterations):
            torch.matmul(A, B)
        baseline_ns = time.perf_counter_ns() - t0
    baseline_us = (baseline_ns / 1000.0) / max(1, iterations)

    # Tiled (boundary-aware) evaluator.
    with torch.no_grad():
        for _ in range(warmup):
            _tiled_matmul_eval(A, B, tile_M=tM, tile_N=tN, tile_K=tK)
        t0 = time.perf_counter_ns()
        for _ in range(iterations):
            _tiled_matmul_eval(A, B, tile_M=tM, tile_N=tN, tile_K=tK)
        tiled_ns = time.perf_counter_ns() - t0
    tiled_us = (tiled_ns / 1000.0) / max(1, iterations)

    return baseline_us, tiled_us


# --------------------------------------------------------------------------- #
# Cost-preview / LLM-view overlay
# --------------------------------------------------------------------------- #


def _apply_candidate_overlay(
    *, run_dir: Path, results_by_id: dict[str, dict[str, Any]],
) -> None:
    """Layer ``calibration`` onto each cost_preview_v2.cost_previews[]
    and llm_graph_view.regions[].legal_candidates[]."""
    ga = run_dir / "02_graph_analysis"

    # cost_preview_v2.json — direct list of cost preview entries.
    cp_path = ga / "cost_preview_v2.json"
    if cp_path.exists():
        try:
            doc = json.loads(cp_path.read_text(encoding="utf-8"))
            for cp in doc.get("cost_previews", []):
                cid = cp.get("candidate_id")
                cal = results_by_id.get(cid)
                if cal is not None:
                    cp["calibration"] = {
                        "measured_baseline_us": cal["measured_baseline_us"],
                        "measured_tiled_us": cal["measured_tiled_us"],
                        "measured_speedup": cal["measured_speedup"],
                        "rel_error": cal["rel_error"],
                        "calibration_status": "calibrated",
                    }
            cp_path.write_text(
                json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8",
            )
        except (OSError, json.JSONDecodeError):
            pass

    # llm_graph_view.json — regions[].legal_candidates[] entries.
    lv_path = ga / "llm_graph_view.json"
    if lv_path.exists():
        try:
            doc = json.loads(lv_path.read_text(encoding="utf-8"))
            for region in doc.get("regions", []) or []:
                for lc in region.get("legal_candidates", []) or []:
                    cid = lc.get("candidate_id")
                    cal = results_by_id.get(cid)
                    if cal is not None:
                        lc["calibration"] = {
                            "measured_baseline_us": cal["measured_baseline_us"],
                            "measured_tiled_us": cal["measured_tiled_us"],
                            "measured_speedup": cal["measured_speedup"],
                            "rel_error": cal["rel_error"],
                        }
            lv_path.write_text(
                json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8",
            )
        except (OSError, json.JSONDecodeError):
            pass


# --------------------------------------------------------------------------- #
# Top-level entry point
# --------------------------------------------------------------------------- #


def run_candidate_calibration(
    run_dir: Path,
    *,
    iterations: int = _DEFAULT_ITERATIONS,
    warmup: int = _DEFAULT_WARMUP,
) -> CandidateCalibrationResult:
    """Run per-tile-candidate calibration. Best-effort; emits a typed
    ``not_run`` report if torch is missing or the inputs are incomplete."""
    run_dir = Path(run_dir).resolve()
    ga = run_dir / "02_graph_analysis"
    out_dir = ga / "candidate_calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "candidate_calibration_report.json"
    summary_md_path = out_dir / "candidate_calibration_summary.md"

    cas = _read_json(ga / "candidate_actions.json")
    cp_doc = _read_json(ga / "cost_preview_v2.json")
    if cas is None or cp_doc is None:
        report = {
            "schema_version": "candidate_calibration_report_v1",
            "calibration_status": "not_run",
            "overall": "not_run",
            "candidate_count": 0,
            "candidates_calibrated": 0,
            "candidates": [],
            "note": (
                "missing candidate_actions.json or cost_preview_v2.json "
                "(stop_after must be ≥ cost-preview-v2)"
            ),
            "generated_at_utc": _utcnow(),
        }
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8",
        )
        summary_md_path.write_text(
            "# Candidate Calibration — not_run\n\n"
            "- reason: missing graph-analysis inputs\n",
            encoding="utf-8",
        )
        return CandidateCalibrationResult(
            overall="not_run", out_dir=out_dir,
            report_path=report_path, summary_md_path=summary_md_path,
            candidate_count=0, candidates_calibrated=0,
        )

    try:
        import torch  # noqa: F401
    except ImportError as exc:
        report = {
            "schema_version": "candidate_calibration_report_v1",
            "calibration_status": "not_run",
            "overall": "not_run",
            "candidate_count": 0,
            "candidates_calibrated": 0,
            "candidates": [],
            "note": f"torch unavailable: {exc}",
            "generated_at_utc": _utcnow(),
        }
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8",
        )
        return CandidateCalibrationResult(
            overall="not_run", out_dir=out_dir,
            report_path=report_path, summary_md_path=summary_md_path,
            candidate_count=0, candidates_calibrated=0,
        )

    cp_by_id = {p["candidate_id"]: p for p in cp_doc.get("cost_previews", [])}
    legal_tile = [
        c for c in cas.get("candidates", [])
        if c.get("kind") == "set_tile_params"
        and (c.get("legality") or {}).get("ok")
    ]

    if not legal_tile:
        report = {
            "schema_version": "candidate_calibration_report_v1",
            "calibration_status": "no_candidates",
            "overall": "no_candidates",
            "candidate_count": 0,
            "candidates_calibrated": 0,
            "candidates": [],
            "note": "no legal SetTileParams candidates to calibrate",
            "generated_at_utc": _utcnow(),
        }
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8",
        )
        summary_md_path.write_text(
            "# Candidate Calibration — no_candidates\n", encoding="utf-8",
        )
        return CandidateCalibrationResult(
            overall="no_candidates", out_dir=out_dir,
            report_path=report_path, summary_md_path=summary_md_path,
            candidate_count=0, candidates_calibrated=0,
        )

    per_candidate: list[dict[str, Any]] = []
    results_by_id: dict[str, dict[str, Any]] = {}
    for c in legal_tile:
        cid = c["candidate_id"]
        cp = cp_by_id.get(cid)
        shape_tile = _extract_shape_and_tile(c, cp)
        entry: dict[str, Any] = {
            "candidate_id": cid,
            "region_id": c.get("region_id"),
            "label": c.get("label"),
            "predicted_us": (
                cp.get("candidate_static_latency_us")
                if cp is not None else None
            ),
        }
        if shape_tile is None:
            entry.update({
                "calibration_status": "not_run",
                "note": "could not extract shape/tile from metadata",
            })
            per_candidate.append(entry)
            continue
        (M, N, K), (tM, tN, tK) = shape_tile
        try:
            baseline_us, tiled_us = _measure_one_candidate(
                M=M, N=N, K=K, tM=tM, tN=tN, tK=tK,
                iterations=iterations, warmup=warmup,
            )
        except Exception as exc:  # noqa: BLE001
            entry.update({
                "calibration_status": "fail",
                "note": f"{type(exc).__name__}: {exc}",
            })
            per_candidate.append(entry)
            continue
        predicted_us = entry["predicted_us"]
        rel_error = (
            (tiled_us - predicted_us) / tiled_us
            if predicted_us is not None and tiled_us > 0 else None
        )
        speedup = baseline_us / tiled_us if tiled_us > 0 else None
        entry.update({
            "calibration_status": "calibrated",
            "matmul_shape": {"M": M, "N": N, "K": K},
            "tile": {"M": tM, "N": tN, "K": tK},
            "iters": {
                "M": M // tM if tM > 0 else 0,
                "N": N // tN if tN > 0 else 0,
                "K": K // tK if tK > 0 else 0,
            },
            "iterations_per_measurement": iterations,
            "warmup_iterations": warmup,
            "measured_baseline_us": baseline_us,
            "measured_tiled_us": tiled_us,
            "measured_speedup": speedup,
            "rel_error": rel_error,
        })
        per_candidate.append(entry)
        results_by_id[cid] = entry

    candidates_calibrated = sum(
        1 for e in per_candidate
        if e.get("calibration_status") == "calibrated"
    )
    overall = "calibrated" if candidates_calibrated > 0 else "not_run"

    report = {
        "schema_version": "candidate_calibration_report_v1",
        "calibration_status": "calibrated" if candidates_calibrated > 0 else "no_candidates",
        "overall": overall,
        "candidate_count": len(legal_tile),
        "candidates_calibrated": candidates_calibrated,
        "iterations_per_measurement": iterations,
        "warmup_iterations": warmup,
        "candidates": per_candidate,
        "summary": {
            "mean_speedup": (
                sum(
                    e.get("measured_speedup") or 0.0
                    for e in per_candidate
                    if e.get("measured_speedup") is not None
                ) / max(1, candidates_calibrated)
                if candidates_calibrated > 0 else None
            ),
            "min_speedup": (
                min(
                    (e["measured_speedup"] for e in per_candidate
                     if e.get("measured_speedup") is not None),
                    default=None,
                )
            ),
            "max_speedup": (
                max(
                    (e["measured_speedup"] for e in per_candidate
                     if e.get("measured_speedup") is not None),
                    default=None,
                )
            ),
            "mean_rel_error": (
                sum(
                    abs(e.get("rel_error") or 0.0)
                    for e in per_candidate
                    if e.get("rel_error") is not None
                ) / max(1, sum(
                    1 for e in per_candidate
                    if e.get("rel_error") is not None
                ))
                if any(e.get("rel_error") is not None for e in per_candidate)
                else None
            ),
        },
        "known_limitations": [
            "CPU only; uses M-16's _tiled_matmul_eval (boundary-aware Python loop)",
            "single-batch-size; no autotune or kernel selection",
            "rel_error is per-candidate predicted vs measured tiled_us",
            "SetTileParams only — fusion candidates not included in MVP",
        ],
        "generated_at_utc": _utcnow(),
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8",
    )

    s = report["summary"]
    summary_md = (
        f"# Candidate Calibration — {overall}\n\n"
        f"- candidates: {candidates_calibrated}/{len(legal_tile)}\n"
        f"- iterations: {iterations} (warmup {warmup})\n"
    )
    if s.get("mean_speedup") is not None:
        summary_md += f"- mean speedup: {s['mean_speedup']:.3f}\n"
        summary_md += f"- min/max speedup: {s['min_speedup']:.3f} / {s['max_speedup']:.3f}\n"
    if s.get("mean_rel_error") is not None:
        summary_md += f"- mean |rel_error|: {s['mean_rel_error']:.3f}\n"
    summary_md_path.write_text(summary_md, encoding="utf-8")

    # Layer onto cost_preview_v2 + llm_graph_view.
    if results_by_id:
        try:
            _apply_candidate_overlay(run_dir=run_dir, results_by_id=results_by_id)
        except Exception:  # noqa: BLE001
            pass

    return CandidateCalibrationResult(
        overall=overall, out_dir=out_dir,
        report_path=report_path, summary_md_path=summary_md_path,
        candidate_count=len(legal_tile),
        candidates_calibrated=candidates_calibrated,
    )
