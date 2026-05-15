"""Per-Region Compiled Differential.

Extends 's single-region compiled-kernel foundation to every
matmul region in a model that has a legal ``set_tile_params``
candidate. For each region, picks the lowest-cost legal tile (greedy
per-region) and runs both the GPU and CPU compile+execute+verify
tracks 's primitives. Aggregates per-region results into a
single ``region_compiled_differential_report.json`` that downstream
tools (retry detector, evidence pack) can consume.

Layered alongside (whose single-region artifact remains) and
alongside the FX-level evidence; never
mutates any of those tracks.

Hard non-goals:

- No new candidate generation, no new transforms.
- No compiler-core imports.
- Best-effort: per-region failures emit typed entries; the aggregate
  never raises into the pipeline.
SetTileParams only. FuseProducerConsumer fan-out is territory.

Output layout::

    02_graph_analysis/kernel_execution/
        regions/<region_id>/
            compiled_kernel_run_gpu.json
            compiled_kernel_run_cpu.json
            triton_kernel_<region>.py
            cpu_kernel_<region>.c
            cffi_build_<region>/
        region_compiled_differential_report.json
        region_compiled_differential_summary.md

Opt-in via the same ``COMPGEN_RUN_KERNELS=1`` env var as. Default
OFF so suite runs stay deterministic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


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
# Per-region tile selection
# --------------------------------------------------------------------------- #


def _per_region_set_tile_picks(
    *,
    candidate_actions: dict[str, Any],
    cost_preview: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """For each region with at least one legal ``set_tile_params``
    candidate, return the lowest-cost legal candidate's metadata
    sufficient to drive 's GPU/CPU tracks.

    Returns a list of dicts, one per region::

        {"region_id": ..., "candidate_id": ..., "candidate_kind": ...,
         "matmul_shape": {"M": ..., "N": ..., "K": ...},
         "tile": {"M": ..., "N": ..., "K": ...},
         "static_relative_cost": ..., "recipe_op_id": "recipe_0000"}

    M/N/K are derived from cost_preview_v2's
    ``diagnostics.candidate.{tile,iters}`` (same approach uses).
    """
    cp_by_id: dict[str, dict[str, Any]] = {}
    if cost_preview is not None:
        for cp in cost_preview.get("cost_previews", []):
            cp_by_id[cp["candidate_id"]] = cp

    by_region: dict[str, list[dict[str, Any]]] = {}
    for c in candidate_actions.get("candidates", []) or []:
        if c.get("kind") != "set_tile_params":
            continue
        if not (c.get("legality") or {}).get("ok"):
            continue
        rid = c.get("region_id", "")
        cp = cp_by_id.get(c["candidate_id"], {})
        diag = (cp.get("diagnostics") or {}).get("candidate") or {}
        tile = diag.get("tile") or (
            (c.get("recipe_delta") or [{}])[0].get("tile") or {}
        )
        iters = diag.get("iters") or {}
        try:
            tM = int(tile.get("M") or 0)
            tN = int(tile.get("N") or 0)
            tK = int(tile.get("K") or 0)
            iM = int(iters.get("M") or 1)
            iN = int(iters.get("N") or 1)
            iK = int(iters.get("K") or 1)
        except (TypeError, ValueError):
            continue
        if tM <= 0 or tN <= 0 or tK <= 0:
            continue
        M = tM * iM; N = tN * iN; K = tK * iK
        if M <= 0 or N <= 0 or K <= 0:
            continue
        cost = float(
            (c.get("cost_preview") or {}).get("static_relative_cost", 1.0)
            or 1.0,
        )
        by_region.setdefault(rid, []).append({
            "region_id": rid,
            "candidate_id": c["candidate_id"],
            "candidate_kind": "set_tile_params",
            "matmul_shape": {"M": M, "N": N, "K": K},
            "tile": {"M": tM, "N": tN, "K": tK},
            "static_relative_cost": cost,
            "label": c.get("label", ""),
        })

    picks: list[dict[str, Any]] = []
    for rid, cands in by_region.items():
        cands.sort(key=lambda c: (c["static_relative_cost"], c["candidate_id"]))
        picks.append(cands[0])
    # Stable order across reruns.
    picks.sort(key=lambda c: c["region_id"])
    return picks


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def _aggregate_region_results(
    region_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Roll per-region GPU+CPU results into top-level counts."""
    total = len(region_results)
    gpu_compiled = sum(
        1 for r in region_results
        if (r.get("gpu") or {}).get("compile_status") == "compiled"
    )
    cpu_compiled = sum(
        1 for r in region_results
        if (r.get("cpu") or {}).get("compile_status") == "compiled"
    )

    refinement_counts: dict[str, int] = {}
    for r in region_results:
        for track_name in ("gpu", "cpu"):
            t = r.get(track_name) or {}
            if t.get("compile_status") != "compiled":
                continue
            ref = (t.get("numerical") or {}).get("refinement_status") or "n/a"
            key = f"{track_name}::{ref}"
            refinement_counts[key] = refinement_counts.get(key, 0) + 1

    bit_eq_count = sum(
        v for k, v in refinement_counts.items()
        if k.endswith("discharged_compiled_bit_equality")
    )
    tol_eq_count = sum(
        v for k, v in refinement_counts.items()
        if k.endswith("discharged_tolerance_eps")
    )
    fail_count = sum(
        v for k, v in refinement_counts.items()
        if k.endswith("fail_outside_tolerance")
    )

    # Suite-level summary scale (mean of GPU + CPU measured_us per region).
    gpu_us_total = 0.0
    cpu_us_total = 0.0
    gpu_n = 0; cpu_n = 0
    for r in region_results:
        g = r.get("gpu") or {}
        if g.get("compile_status") == "compiled" and g.get("measured_us_per_iter"):
            gpu_us_total += float(g["measured_us_per_iter"])
            gpu_n += 1
        c = r.get("cpu") or {}
        if c.get("compile_status") == "compiled" and c.get("measured_us_per_iter"):
            cpu_us_total += float(c["measured_us_per_iter"])
            cpu_n += 1

    return {
        "region_count": total,
        "gpu_compiled_count": gpu_compiled,
        "cpu_compiled_count": cpu_compiled,
        "compiled_bit_equality_count": bit_eq_count,
        "tolerance_eps_count": tol_eq_count,
        "fail_outside_tolerance_count": fail_count,
        "refinement_breakdown": refinement_counts,
        "mean_gpu_us": (gpu_us_total / gpu_n) if gpu_n > 0 else None,
        "mean_cpu_us": (cpu_us_total / cpu_n) if cpu_n > 0 else None,
    }


# --------------------------------------------------------------------------- #
# Top-level entry point
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RegionDifferentialResult:
    overall: str            # "ok" | "partial" | "no_candidates" | "not_run"
    out_dir: Path
    report_path: Path
    summary_md_path: Path
    region_count: int
    gpu_compiled_count: int
    cpu_compiled_count: int
    fail_outside_tolerance_count: int


def run_region_compiled_differential(
    run_dir: Path,
    *,
    iterations: int = 32,
    warmup: int = 4,
) -> RegionDifferentialResult:
    """Per-region fan-out of. Best-effort; never raises."""
    run_dir = Path(run_dir).resolve()
    ga = run_dir / "02_graph_analysis"
    out_dir = ga / "kernel_execution"
    out_dir.mkdir(parents=True, exist_ok=True)
    regions_dir = out_dir / "regions"
    regions_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "region_compiled_differential_report.json"
    summary_md_path = out_dir / "region_compiled_differential_summary.md"

    cas = _read_json(ga / "candidate_actions.json")
    cp = _read_json(ga / "cost_preview_v2.json")
    cap = _read_json(run_dir / "00_graph_capture" / "capture_report.json")
    if cas is None or cp is None:
        report = {
            "schema_version": "region_compiled_differential_report_v1",
            "status": "not_run",
            "overall": "not_run",
            "note": (
                "missing candidate_actions.json or cost_preview_v2.json "
                "(stop-after must be ≥ cost-preview-v2)"
            ),
            "regions": [],
            "summary": {},
            "generated_at_utc": _utcnow(),
        }
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8",
        )
        summary_md_path.write_text(
            "# Region-level Compiled Differential — not_run\n\n"
            "- reason: graph-analysis inputs missing\n",
            encoding="utf-8",
        )
        return RegionDifferentialResult(
            overall="not_run", out_dir=out_dir,
            report_path=report_path, summary_md_path=summary_md_path,
            region_count=0, gpu_compiled_count=0,
            cpu_compiled_count=0, fail_outside_tolerance_count=0,
        )

    picks = _per_region_set_tile_picks(
        candidate_actions=cas, cost_preview=cp,
    )
    if not picks:
        report = {
            "schema_version": "region_compiled_differential_report_v1",
            "status": "no_candidates",
            "overall": "no_candidates",
            "note": "no regions with legal SetTileParams candidates",
            "regions": [],
            "summary": {},
            "generated_at_utc": _utcnow(),
        }
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8",
        )
        summary_md_path.write_text(
            "# Region-level Compiled Differential — no_candidates\n",
            encoding="utf-8",
        )
        return RegionDifferentialResult(
            overall="no_candidates", out_dir=out_dir,
            report_path=report_path, summary_md_path=summary_md_path,
            region_count=0, gpu_compiled_count=0,
            cpu_compiled_count=0, fail_outside_tolerance_count=0,
        )

    target_id = (cap or {}).get("target_id", "")
    model_id = (cap or {}).get("model_id", "")

    # Reuse 's GPU + CPU primitives.
    from compgen.graph_compilation.kernel_execution_gpu import run_gpu_track
    from compgen.graph_compilation.kernel_execution_cpu import run_cpu_track

    region_results: list[dict[str, Any]] = []
    for pick in picks:
        rid = pick["region_id"]
        safe_region = "".join(
            c if c.isalnum() or c == "_" else "_" for c in rid
        ) or "matmul"
        region_out_dir = regions_dir / safe_region
        region_out_dir.mkdir(parents=True, exist_ok=True)

        common = {
            "schema_version": "compiled_kernel_run_v1",
            "model_id": model_id,
            "target_id": target_id,
            "region_id": rid,
            "candidate_id": pick["candidate_id"],
            "recipe_op_id": "recipe_0000",
            "matmul_shape": pick["matmul_shape"],
            "tile": pick["tile"],
            "transformed_payload_real_mlir": (
                "03_recipe_planning/real_lowering/transformed_payload.real.mlir"
            ),
            "transformed_payload_real_mlir_sha256": "",
            "iterations": iterations,
            "warmup": warmup,
        }

        # GPU + CPU tracks.
        gpu_out: dict[str, Any] = {}
        cpu_out: dict[str, Any] = {}
        try:
            gpu_path = run_gpu_track(out_dir=region_out_dir, common=common)
            gpu_out = _read_json(gpu_path) or {}
        except Exception as exc:  # noqa: BLE001
            gpu_out = {
                "compile_status": "internal_error",
                "run_status": "not_run",
                "note": f"{type(exc).__name__}: {exc}",
            }
        try:
            cpu_path = run_cpu_track(out_dir=region_out_dir, common=common)
            cpu_out = _read_json(cpu_path) or {}
        except Exception as exc:  # noqa: BLE001
            cpu_out = {
                "compile_status": "internal_error",
                "run_status": "not_run",
                "note": f"{type(exc).__name__}: {exc}",
            }

        region_results.append({
            "region_id": rid,
            "candidate_id": pick["candidate_id"],
            "matmul_shape": pick["matmul_shape"],
            "tile": pick["tile"],
            "static_relative_cost": pick["static_relative_cost"],
            "label": pick["label"],
            "out_dir": str(region_out_dir.relative_to(run_dir)),
            "gpu": {
                "compile_status": gpu_out.get("compile_status"),
                "run_status": gpu_out.get("run_status"),
                "measured_us_per_iter": gpu_out.get("measured_us_per_iter"),
                "measured_us_stddev": gpu_out.get("measured_us_stddev"),
                "numerical": gpu_out.get("numerical") or {},
                "device": gpu_out.get("device") or {},
                "note": gpu_out.get("note", ""),
            },
            "cpu": {
                "compile_status": cpu_out.get("compile_status"),
                "run_status": cpu_out.get("run_status"),
                "measured_us_per_iter": cpu_out.get("measured_us_per_iter"),
                "measured_us_stddev": cpu_out.get("measured_us_stddev"),
                "numerical": cpu_out.get("numerical") or {},
                "compiler": cpu_out.get("compiler", ""),
                "note": cpu_out.get("note", ""),
            },
        })

    summary = _aggregate_region_results(region_results)

    # Determine top-level status.
    status = "fail" if summary["fail_outside_tolerance_count"] > 0 else "pass"
    if summary["gpu_compiled_count"] == 0 and summary["cpu_compiled_count"] == 0:
        status = "not_run"

    report = {
        "schema_version": "region_compiled_differential_report_v1",
        "status": status,
        "overall": status,
        "model_id": model_id,
        "target_id": target_id,
        "iterations": iterations,
        "warmup": warmup,
        "regions": region_results,
        "summary": summary,
        "known_limitations": [
            "per-region greedy: lowest static_relative_cost legal SetTileParams",
            "FuseProducerConsumer regions excluded (M-23 territory)",
            "fp32 fp32 fp32 only; mixed-precision is M-22+",
            "single batch size",
        ],
        "generated_at_utc": _utcnow(),
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8",
    )

    body = (
        f"# Region-level Compiled Differential — {status}\n\n"
        f"- regions covered: {summary['region_count']}\n"
        f"- gpu compiled: {summary['gpu_compiled_count']}\n"
        f"- cpu compiled: {summary['cpu_compiled_count']}\n"
        f"- bit-equality discharged: "
        f"{summary['compiled_bit_equality_count']}\n"
        f"- tolerance_eps discharged: {summary['tolerance_eps_count']}\n"
        f"- fail_outside_tolerance: "
        f"{summary['fail_outside_tolerance_count']}\n"
    )
    if summary.get("mean_gpu_us") is not None:
        body += f"- mean_gpu_us: {summary['mean_gpu_us']:.2f}\n"
    if summary.get("mean_cpu_us") is not None:
        body += f"- mean_cpu_us: {summary['mean_cpu_us']:.2f}\n"
    body += "\n## Per-region\n\n"
    body += "| region | candidate | tile | gpu_us | cpu_us | gpu_ref | cpu_ref |\n"
    body += "|---|---|---|---|---|---|---|\n"
    for r in region_results:
        t = r["tile"]
        gpu = r["gpu"]; cpu = r["cpu"]
        body += (
            f"| `{r['region_id']}` "
            f"| `{r['label']}` "
            f"| ({t['M']},{t['N']},{t['K']}) "
            f"| {gpu.get('measured_us_per_iter') or '—'} "
            f"| {cpu.get('measured_us_per_iter') or '—'} "
            f"| {(gpu.get('numerical') or {}).get('refinement_status', '—')} "
            f"| {(cpu.get('numerical') or {}).get('refinement_status', '—')} |\n"
        )
    summary_md_path.write_text(body, encoding="utf-8")

    return RegionDifferentialResult(
        overall=status, out_dir=out_dir,
        report_path=report_path, summary_md_path=summary_md_path,
        region_count=summary["region_count"],
        gpu_compiled_count=summary["gpu_compiled_count"],
        cpu_compiled_count=summary["cpu_compiled_count"],
        fail_outside_tolerance_count=summary["fail_outside_tolerance_count"],
    )
