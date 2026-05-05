"""M-21 Per-Candidate Deterministic Analytical Cost Model.

A pure-function analytical cost model rooted in the target hardware
spec (target YAML) and the graph dossier facts (region shape, tile
geometry, working-set fit). For each legal SetTileParams candidate it
emits a deterministic per-candidate cost breakdown — same inputs
produce byte-identical output across reruns, no measurement, no
randomness, no system calls.

Layered alongside the existing cost tracks:

- ``cost_preview_v2.json`` (M-13) — earlier static cost preview;
  similar roofline; preserved.
- ``candidate_calibration_report.json`` (M-18.3) — Python-evaluator
  measured.
- ``compiled_kernel_run_gpu.json`` / ``compiled_kernel_run_cpu.json``
  (M-19) and ``region_compiled_differential_report.json`` (M-20) —
  real compiled-kernel measured.

M-21's contribution: an EXPLICIT, EXPLAINABLE, DETERMINISTIC
per-candidate cost rooted in three sources only:

1. Target YAML (peak_compute_gflops, peak_bandwidth_gb_s,
   memory_tiers, supported_dtypes).
2. Per-region matmul shape (from region_dossier or matmul_signature).
3. Per-candidate tile (from candidate_actions recipe_delta).

The model is a **blocked-matmul roofline** with per-tier bandwidth
multipliers and explicit reload counts (LHS reloaded N/tN times,
RHS reloaded M/tM times), giving the agent visibility into WHY a
particular tile choice has a particular cost prediction.

When M-19/M-20 measurements are present on disk, the per-candidate
entry includes a ``calibration_delta`` block (predicted/measured
ratio). The model itself is unchanged — calibration is read-only
ingestion.

Hard non-goals:

- No empirical measurement.
- No new candidate generation, no new transforms.
- No compiler-core imports.
- No mutation of region_dossiers / candidate_actions / region_map
  (which the integrity suite enforces).
- SetTileParams only. FuseProducerConsumer cost model is M-23 territory.

M-21 *does* layer additive ``m21_analytical_cost`` blocks onto
``cost_preview_v2.cost_previews[]`` and
``llm_graph_view.regions[].legal_candidates[]`` (same pattern as
M-18.3's ``calibration`` overlay) so the agent sees all per-candidate
cost columns in one place.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


_DTYPE_BYTES: dict[str, int] = {
    "f32": 4, "fp32": 4, "float32": 4,
    "f16": 2, "fp16": 2, "float16": 2, "bf16": 2,
    "f8": 1, "fp8": 1, "fp8_e4m3": 1, "fp8_e5m2": 1,
    "i32": 4, "i16": 2, "i8": 1,
}

# Per-tier bandwidth multipliers relative to peak_bandwidth_gb_s
# (which represents DRAM/system bandwidth in the target YAML). These
# are deterministic conventions used when the target YAML doesn't
# provide explicit per-tier values. Documented in the report's
# ``model_inputs`` block; agents can recompute exactly.
_TIER_BW_MULTIPLIER: dict[str, float] = {
    "system": 1.0,
    "l3": 2.0,
    "l2": 4.0,
    "scratchpad": 8.0,
}

# Per-iter overhead (in microseconds) reflecting loop bookkeeping +
# branch + index arithmetic. Same value cost_preview_v2 uses (~10ns).
_PER_ITER_OVERHEAD_US: float = 0.01


def _utcnow() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _load_target_yaml(target_id: str, repo_root: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        return {}
    candidates = [
        repo_root / "configs" / "targets" / f"{target_id}.yaml",
        repo_root / "configs" / "targets" / "host_cpu.yaml",
    ]
    for c in candidates:
        if c.exists():
            try:
                return yaml.safe_load(c.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError:
                continue
    return {}


# --------------------------------------------------------------------------- #
# Pure-function cost model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _ModelInputs:
    matmul_shape: tuple[int, int, int]    # (M, N, K)
    tile: tuple[int, int, int]            # (tM, tN, tK)
    dtype_bytes: int
    peak_compute_gflops: float
    peak_bandwidth_gb_s: float
    memory_tiers: dict[str, int]
    tier_bw_multiplier: dict[str, float]


def _select_memory_tier(
    *, working_set_bytes: int, memory_tiers: dict[str, int],
) -> str:
    """Return the smallest (highest-bandwidth) tier the working set
    fits in. Falls back to ``system`` if working_set exceeds even L3."""
    # Order matters: smallest (fastest) → largest (slowest).
    for name, key in (
        ("scratchpad", "scratchpad_bytes"),
        ("l2", "l2_bytes"),
        ("l3", "l3_bytes"),
    ):
        cap = int(memory_tiers.get(key, 0) or 0)
        if cap > 0 and working_set_bytes <= cap:
            return name
    return "system"


def predict_candidate_cost(
    *,
    matmul_shape: tuple[int, int, int],
    tile: tuple[int, int, int],
    dtype_bytes: int,
    peak_compute_gflops: float,
    peak_bandwidth_gb_s: float,
    memory_tiers: dict[str, int],
    tier_bw_multiplier: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Pure function. Returns a deterministic cost breakdown for a
    blocked-matmul tile. Same inputs → byte-identical JSON output.

    Model:
        compute_time_us = total_flops / (peak_compute_gflops * 1e9) * 1e6
        memory_time_us  = total_bytes_moved / (effective_bw * 1e9) * 1e6
        predicted_us    = max(compute_time_us, memory_time_us)
                          + iters_total * per_iter_overhead_us
        effective_bw    = peak_bandwidth_gb_s * tier_multiplier[<best fitting tier>]

    Blocked-matmul reload counts (textbook):
        lhs_reloads     = ceil(N / tN)        (LHS reloaded once per N-tile)
        rhs_reloads     = ceil(M / tM)        (RHS reloaded once per M-tile)
        out_writes      = 1                   (final tile written once)
        total_bytes_moved = (lhs_reloads * M*K + rhs_reloads * K*N + M*N) * dtype_bytes
    """
    M, N, K = matmul_shape
    tM, tN, tK = tile
    if min(M, N, K, tM, tN, tK) <= 0:
        raise ValueError(f"non-positive shape/tile: {matmul_shape=} {tile=}")

    tier_mult = dict(tier_bw_multiplier or _TIER_BW_MULTIPLIER)

    # Iteration counts (round up for boundary tiles).
    iM = (M + tM - 1) // tM
    iN = (N + tN - 1) // tN
    iK = (K + tK - 1) // tK
    iter_total = iM * iN * iK

    # FLOPs (matmul: 2 * M * N * K — independent of tiling).
    total_flops = 2 * M * N * K

    # Per-tile working set (LHS tile + RHS tile + Out tile, all in
    # registers/scratchpad simultaneously for the inner kernel).
    lhs_tile_bytes = tM * tK * dtype_bytes
    rhs_tile_bytes = tK * tN * dtype_bytes
    out_tile_bytes = tM * tN * dtype_bytes
    tile_working_set_bytes = lhs_tile_bytes + rhs_tile_bytes + out_tile_bytes

    # Pick best-fitting memory tier from spec.
    tier = _select_memory_tier(
        working_set_bytes=tile_working_set_bytes,
        memory_tiers=memory_tiers,
    )
    bw_multiplier = float(tier_mult.get(tier, 1.0))
    effective_bw_gbps = peak_bandwidth_gb_s * bw_multiplier

    # Reload-count memory model.
    lhs_reloads = iN
    rhs_reloads = iM
    out_writes = 1
    bytes_moved_lhs = lhs_reloads * M * K * dtype_bytes
    bytes_moved_rhs = rhs_reloads * K * N * dtype_bytes
    bytes_moved_out = out_writes * M * N * dtype_bytes
    total_bytes_moved = bytes_moved_lhs + bytes_moved_rhs + bytes_moved_out

    # Roofline.
    if peak_compute_gflops > 0:
        compute_time_us = (total_flops / (peak_compute_gflops * 1e9)) * 1e6
    else:
        compute_time_us = float("inf")
    if effective_bw_gbps > 0:
        memory_time_us = (total_bytes_moved / (effective_bw_gbps * 1e9)) * 1e6
    else:
        memory_time_us = float("inf")

    overhead_us = iter_total * _PER_ITER_OVERHEAD_US

    if compute_time_us > memory_time_us:
        bottleneck_resource = "compute"
        bottleneck_time = compute_time_us
    else:
        bottleneck_resource = "memory"
        bottleneck_time = memory_time_us
    predicted_us = bottleneck_time + overhead_us

    arithmetic_intensity = (
        total_flops / total_bytes_moved if total_bytes_moved > 0 else 0.0
    )

    return {
        "matmul_shape": {"M": M, "N": N, "K": K},
        "tile": {"M": tM, "N": tN, "K": tK},
        "iters": {
            "M": iM, "N": iN, "K": iK, "total": iter_total,
        },
        "compute": {
            "flops": total_flops,
            "peak_compute_gflops": peak_compute_gflops,
            "compute_time_us": compute_time_us,
        },
        "working_set": {
            "lhs_tile_bytes": lhs_tile_bytes,
            "rhs_tile_bytes": rhs_tile_bytes,
            "out_tile_bytes": out_tile_bytes,
            "tile_working_set_bytes": tile_working_set_bytes,
            "memory_tier": tier,
            "memory_tier_bandwidth_multiplier": bw_multiplier,
            "effective_bandwidth_gb_s": effective_bw_gbps,
        },
        "memory": {
            "lhs_reloads": lhs_reloads,
            "rhs_reloads": rhs_reloads,
            "out_writes": out_writes,
            "bytes_moved_lhs": bytes_moved_lhs,
            "bytes_moved_rhs": bytes_moved_rhs,
            "bytes_moved_out": bytes_moved_out,
            "total_bytes_moved": total_bytes_moved,
            "memory_time_us": memory_time_us,
        },
        "overhead": {
            "iter_count": iter_total,
            "per_iter_overhead_us": _PER_ITER_OVERHEAD_US,
            "total_overhead_us": overhead_us,
        },
        "predicted_us": predicted_us,
        "bottleneck_resource": bottleneck_resource,
        "bottleneck_tier": tier,
        "arithmetic_intensity": arithmetic_intensity,
        "model_kind": "blocked_matmul_roofline_v1",
        "deterministic": True,
    }


# --------------------------------------------------------------------------- #
# Region matmul-shape resolution
# --------------------------------------------------------------------------- #


def _matmul_shape_for_region(
    *,
    region_id: str,
    candidate: dict[str, Any],
    cost_preview_entry: dict[str, Any] | None,
    region_dossier: dict[str, Any] | None,
) -> tuple[int, int, int] | None:
    """Best-effort: extract (M, N, K) for a region. Sources tried in
    order: cost_preview's diagnostics tile×iters, candidate's recipe_delta
    tile, region_dossier's source block."""
    if cost_preview_entry is not None:
        diag = cost_preview_entry.get("diagnostics") or {}
        # `tile` is at diagnostics-level (shared between baseline + candidate);
        # `iters` is under diagnostics.candidate.
        tile = diag.get("tile") or {}
        cand_diag = diag.get("candidate") or {}
        iters = cand_diag.get("iters") or {}
        try:
            tM = int(tile.get("M") or 0); tN = int(tile.get("N") or 0); tK = int(tile.get("K") or 0)
            iM = int(iters.get("M") or 1); iN = int(iters.get("N") or 1); iK = int(iters.get("K") or 1)
            if tM > 0 and tN > 0 and tK > 0:
                return (tM * iM, tN * iN, tK * iK)
        except (TypeError, ValueError):
            pass

    delta = (candidate.get("recipe_delta") or [{}])[0]
    tile = delta.get("tile") or {}
    if tile and region_dossier is not None:
        # We need iters from somewhere; if cost_preview missed, fall back
        # to region_dossier's flops/bytes to compute M*N*K.
        cost = region_dossier.get("cost") or {}
        flops = int(cost.get("flops") or 0)
        # M*N*K = flops/2 — but we still need to know the actual M/N/K
        # individually. region_dossier doesn't carry them explicitly.
        # Without iters, we can't recover; fall through.
    return None


# --------------------------------------------------------------------------- #
# Per-region dossier loader
# --------------------------------------------------------------------------- #


def _load_region_dossiers(ga: Path) -> dict[str, dict[str, Any]]:
    rd_dir = ga / "region_dossiers"
    out: dict[str, dict[str, Any]] = {}
    if not rd_dir.is_dir():
        return out
    for p in sorted(rd_dir.iterdir()):
        if p.suffix != ".json":
            continue
        d = _read_json(p)
        if d is None:
            continue
        rid = d.get("region_id")
        if rid:
            out[str(rid)] = d
    return out


# --------------------------------------------------------------------------- #
# Calibration cross-reference (read-only ingest of M-19 / M-20 measurements)
# --------------------------------------------------------------------------- #


def _calibration_cross_ref(
    *, run_dir: Path, candidate_id: str, region_id: str,
) -> dict[str, Any]:
    """When M-19 / M-20 measurements are on disk for this candidate
    or region, compute the predicted/measured ratio. Read-only;
    returns ``{"present": False}`` when no measurements exist."""
    base = run_dir / "02_graph_analysis" / "kernel_execution"
    if not base.is_dir():
        return {"present": False}

    # M-19 single-region (the SELECTED candidate's compiled run).
    measured_gpu = None
    measured_cpu = None
    for fname in ("compiled_kernel_run_gpu.json", "compiled_kernel_run_cpu.json"):
        d = _read_json(base / fname)
        if d is None:
            continue
        if d.get("candidate_id") != candidate_id:
            continue
        if d.get("compile_status") != "compiled":
            continue
        us = d.get("measured_us_per_iter")
        if us is None:
            continue
        if "gpu" in fname:
            measured_gpu = float(us)
        else:
            measured_cpu = float(us)

    # M-20 per-region fan-out (matched by region_id, since M-20 picks
    # one tile per region).
    if measured_gpu is None or measured_cpu is None:
        m20_path = base / "region_compiled_differential_report.json"
        m20 = _read_json(m20_path)
        if m20 is not None:
            for r in m20.get("regions", []) or []:
                if r.get("region_id") != region_id:
                    continue
                if r.get("candidate_id") != candidate_id:
                    continue
                if measured_gpu is None:
                    g = (r.get("gpu") or {})
                    if g.get("compile_status") == "compiled":
                        gv = g.get("measured_us_per_iter")
                        if gv is not None:
                            measured_gpu = float(gv)
                if measured_cpu is None:
                    c = (r.get("cpu") or {})
                    if c.get("compile_status") == "compiled":
                        cv = c.get("measured_us_per_iter")
                        if cv is not None:
                            measured_cpu = float(cv)

    if measured_gpu is None and measured_cpu is None:
        return {"present": False}

    return {
        "present": True,
        "measured_gpu_us": measured_gpu,
        "measured_cpu_us": measured_cpu,
    }


# --------------------------------------------------------------------------- #
# Overlay: layer m21_analytical_cost onto cost_preview_v2 + llm_graph_view
# --------------------------------------------------------------------------- #


def _compact_overlay_block(entry: dict[str, Any]) -> dict[str, Any]:
    """Project a per-candidate analytical-cost entry into the compact
    overlay block that lives inside cost_preview_v2 / llm_graph_view.
    Carries the headline numbers an agent needs without duplicating the
    full breakdown (which stays in the standalone report)."""
    block: dict[str, Any] = {
        "model_kind": entry.get("model_kind", "blocked_matmul_roofline_v1"),
        "model_version": 1,
        "deterministic": entry.get("deterministic", True),
        "predicted_us": entry.get("predicted_us"),
        "bottleneck_resource": entry.get("bottleneck_resource"),
        "bottleneck_tier": entry.get("bottleneck_tier"),
        "compute_time_us": (entry.get("compute") or {}).get("compute_time_us"),
        "memory_time_us": (entry.get("memory") or {}).get("memory_time_us"),
        "overhead_us": (entry.get("overhead") or {}).get("total_overhead_us"),
        "arithmetic_intensity": entry.get("arithmetic_intensity"),
        "matmul_shape": entry.get("matmul_shape"),
        "tile": entry.get("tile"),
        "iters_total": (entry.get("iters") or {}).get("total"),
        "effective_bandwidth_gb_s": (
            (entry.get("working_set") or {}).get("effective_bandwidth_gb_s")
        ),
        "tile_working_set_bytes": (
            (entry.get("working_set") or {}).get("tile_working_set_bytes")
        ),
    }
    cal = entry.get("calibration_delta")
    if cal:
        block["calibration_delta"] = cal
    return block


def _apply_analytical_cost_overlay(
    *, run_dir: Path, results_by_id: dict[str, dict[str, Any]],
) -> None:
    """Layer ``m21_analytical_cost`` onto each cost_preview_v2.cost_previews[]
    and llm_graph_view.regions[].legal_candidates[] entry whose candidate_id
    has an analytical cost result. Overwrites prior M-21 overlay if it
    exists (re-runs are byte-stable because the per-candidate result is
    deterministic). Leaves M-18.3's ``calibration`` block untouched."""
    ga = run_dir / "02_graph_analysis"

    cp_path = ga / "cost_preview_v2.json"
    if cp_path.exists():
        try:
            doc = json.loads(cp_path.read_text(encoding="utf-8"))
            for cp in doc.get("cost_previews", []):
                cid = cp.get("candidate_id")
                entry = results_by_id.get(cid)
                if entry is not None and entry.get("model_status") == "ok":
                    cp["m21_analytical_cost"] = _compact_overlay_block(entry)
            cp_path.write_text(
                json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8",
            )
        except (OSError, json.JSONDecodeError):
            pass

    lv_path = ga / "llm_graph_view.json"
    if lv_path.exists():
        try:
            doc = json.loads(lv_path.read_text(encoding="utf-8"))
            for region in doc.get("regions", []) or []:
                for lc in region.get("legal_candidates", []) or []:
                    cid = lc.get("candidate_id")
                    entry = results_by_id.get(cid)
                    if entry is not None and entry.get("model_status") == "ok":
                        lc["m21_analytical_cost"] = _compact_overlay_block(entry)
            lv_path.write_text(
                json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8",
            )
        except (OSError, json.JSONDecodeError):
            pass


# --------------------------------------------------------------------------- #
# Top-level entry point
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AnalyticalCostResult:
    overall: str                    # "ok" | "no_candidates" | "not_run"
    out_dir: Path
    report_path: Path
    summary_md_path: Path
    candidate_count: int
    candidates_modeled: int


def run_analytical_cost(
    run_dir: Path, *, repo_root: Path | None = None,
) -> AnalyticalCostResult:
    """Build the M-21 deterministic per-candidate analytical cost
    report. Best-effort; never raises."""
    run_dir = Path(run_dir).resolve()
    repo_root = repo_root or Path(__file__).resolve().parents[3]

    ga = run_dir / "02_graph_analysis"
    out_dir = ga / "analytical_cost"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "per_candidate_analytical_cost.json"
    summary_md_path = out_dir / "analytical_cost_summary.md"

    cas = _read_json(ga / "candidate_actions.json")
    cp_doc = _read_json(ga / "cost_preview_v2.json")
    if cas is None or cp_doc is None:
        body = {
            "schema_version": "per_candidate_analytical_cost_v1",
            "overall": "not_run",
            "note": (
                "missing candidate_actions.json or cost_preview_v2.json"
            ),
            "candidates": [],
            "generated_at_utc": _utcnow(),
        }
        report_path.write_text(
            json.dumps(body, indent=2, sort_keys=True), encoding="utf-8",
        )
        summary_md_path.write_text(
            "# Analytical Cost — not_run\n", encoding="utf-8",
        )
        return AnalyticalCostResult(
            overall="not_run", out_dir=out_dir,
            report_path=report_path, summary_md_path=summary_md_path,
            candidate_count=0, candidates_modeled=0,
        )

    cap = _read_json(run_dir / "00_graph_capture" / "capture_report.json")
    target_id = (cap or {}).get("target_id", "host_cpu")
    target = _load_target_yaml(target_id, repo_root)
    peak_compute = float(target.get("peak_compute_gflops", 0.0) or 0.0)
    peak_bandwidth = float(target.get("peak_bandwidth_gb_s", 0.0) or 0.0)
    memory_tiers = dict(target.get("memory_tiers", {}) or {})
    tier_mult = dict(target.get("tier_bandwidth_multipliers")
                     or _TIER_BW_MULTIPLIER)

    cp_by_id = {p["candidate_id"]: p for p in cp_doc.get("cost_previews", [])}
    region_dossiers = _load_region_dossiers(ga)

    candidates_out: list[dict[str, Any]] = []
    set_tile = [
        c for c in cas.get("candidates", [])
        if c.get("kind") == "set_tile_params"
        and (c.get("legality") or {}).get("ok")
    ]

    for c in set_tile:
        cid = c["candidate_id"]
        rid = c.get("region_id", "")
        cp_entry = cp_by_id.get(cid)
        rdoss = region_dossiers.get(rid)
        shape = _matmul_shape_for_region(
            region_id=rid, candidate=c,
            cost_preview_entry=cp_entry,
            region_dossier=rdoss,
        )
        delta = (c.get("recipe_delta") or [{}])[0]
        tile = delta.get("tile") or {}
        try:
            tM = int(tile.get("M") or 0)
            tN = int(tile.get("N") or 0)
            tK = int(tile.get("K") or 0)
        except (TypeError, ValueError):
            tM = tN = tK = 0
        if shape is None or min(shape) <= 0 or min(tM, tN, tK) <= 0:
            candidates_out.append({
                "candidate_id": cid, "region_id": rid,
                "model_status": "skipped",
                "skip_reason": "could not resolve matmul shape or tile",
            })
            continue

        # Determine dtype_bytes from candidate / cp / region dossier.
        dtype = "f32"
        if cp_entry is not None:
            features = cp_entry.get("features") or {}
            dtype = features.get("dtype") or dtype
        dtype_bytes = _DTYPE_BYTES.get(str(dtype).lower(), 4)

        try:
            cost_breakdown = predict_candidate_cost(
                matmul_shape=shape,
                tile=(tM, tN, tK),
                dtype_bytes=dtype_bytes,
                peak_compute_gflops=peak_compute,
                peak_bandwidth_gb_s=peak_bandwidth,
                memory_tiers=memory_tiers,
                tier_bw_multiplier=tier_mult,
            )
        except Exception as exc:  # noqa: BLE001
            candidates_out.append({
                "candidate_id": cid, "region_id": rid,
                "model_status": "error",
                "skip_reason": f"{type(exc).__name__}: {exc}",
            })
            continue

        cal = _calibration_cross_ref(
            run_dir=run_dir, candidate_id=cid, region_id=rid,
        )
        if cal.get("present"):
            predicted_us = cost_breakdown["predicted_us"]
            calibration_delta: dict[str, Any] = {}
            if cal.get("measured_gpu_us") is not None:
                m = float(cal["measured_gpu_us"])
                calibration_delta["measured_gpu_us"] = m
                calibration_delta["predicted_vs_gpu_ratio"] = (
                    predicted_us / m if m > 0 else None
                )
            if cal.get("measured_cpu_us") is not None:
                m = float(cal["measured_cpu_us"])
                calibration_delta["measured_cpu_us"] = m
                calibration_delta["predicted_vs_cpu_ratio"] = (
                    predicted_us / m if m > 0 else None
                )
            cost_breakdown["calibration_delta"] = calibration_delta

        candidates_out.append({
            "candidate_id": cid,
            "region_id": rid,
            "candidate_kind": "set_tile_params",
            "label": c.get("label", ""),
            "static_relative_cost": (
                (c.get("cost_preview") or {}).get("static_relative_cost")
            ),
            "model_status": "ok",
            **cost_breakdown,
            "confidence": 0.6,
            "known_limitations": [
                "blocked-matmul roofline; no register pressure model",
                "tier bandwidth uses target multipliers, not measured",
                "no boundary-tile penalty; iter count assumes uniform tiles",
                "no fusion benefit modeled (M-23 territory)",
                "no cache associativity / replacement modeled",
            ],
            "model_inputs": {
                "target_id": target_id,
                "peak_compute_gflops": peak_compute,
                "peak_bandwidth_gb_s": peak_bandwidth,
                "memory_tiers": memory_tiers,
                "tier_bw_multiplier": tier_mult,
                "dtype": dtype,
                "dtype_bytes": dtype_bytes,
            },
        })

    candidates_modeled = sum(
        1 for x in candidates_out if x.get("model_status") == "ok"
    )

    # Aggregate.
    summary: dict[str, Any] = {
        "candidates_total": len(candidates_out),
        "candidates_modeled": candidates_modeled,
        "candidates_skipped": len(candidates_out) - candidates_modeled,
        "compute_bound_count": sum(
            1 for x in candidates_out
            if x.get("bottleneck_resource") == "compute"
        ),
        "memory_bound_count": sum(
            1 for x in candidates_out
            if x.get("bottleneck_resource") == "memory"
        ),
        "tier_breakdown": {},
        "min_predicted_us": None, "max_predicted_us": None,
        "mean_predicted_us": None,
    }
    tier_counter: dict[str, int] = {}
    predicted_values: list[float] = []
    for x in candidates_out:
        if x.get("model_status") != "ok":
            continue
        tier = x.get("bottleneck_tier", "unknown")
        tier_counter[tier] = tier_counter.get(tier, 0) + 1
        predicted_values.append(float(x["predicted_us"]))
    summary["tier_breakdown"] = tier_counter
    if predicted_values:
        summary["min_predicted_us"] = min(predicted_values)
        summary["max_predicted_us"] = max(predicted_values)
        summary["mean_predicted_us"] = (
            sum(predicted_values) / len(predicted_values)
        )

    overall = "ok" if candidates_modeled > 0 else (
        "no_candidates" if not set_tile else "not_run"
    )

    # Layer the per-candidate analytical cost onto cost_preview_v2 +
    # llm_graph_view (additive `m21_analytical_cost` block; same pattern
    # as M-18.3's `calibration` overlay). Done before persisting the
    # standalone report so the overlay write is part of the same logical
    # M-21 step.
    results_by_id = {
        x["candidate_id"]: x for x in candidates_out
        if x.get("model_status") == "ok"
    }
    if results_by_id:
        _apply_analytical_cost_overlay(
            run_dir=run_dir, results_by_id=results_by_id,
        )

    body = {
        "schema_version": "per_candidate_analytical_cost_v1",
        "overall": overall,
        "model_kind": "blocked_matmul_roofline_v1",
        "model_version": 1,
        "deterministic": True,
        "target_id": target_id,
        "model_inputs_used": {
            "peak_compute_gflops": peak_compute,
            "peak_bandwidth_gb_s": peak_bandwidth,
            "memory_tiers": memory_tiers,
            "tier_bw_multiplier": tier_mult,
            "per_iter_overhead_us": _PER_ITER_OVERHEAD_US,
        },
        "summary": summary,
        "candidates": candidates_out,
        "known_limitations": [
            "blocked-matmul roofline only; no fusion benefit modeled",
            "tier bandwidth multipliers are conventional defaults unless target YAML overrides",
            "no register-pressure / occupancy model",
            "ignores boundary-tile inefficiency",
            "ignores cache associativity / replacement effects",
        ],
        "generated_at_utc": _utcnow(),
    }
    report_path.write_text(
        json.dumps(body, indent=2, sort_keys=True), encoding="utf-8",
    )

    md_lines = [
        f"# Analytical Cost (M-21) — {overall}\n",
        f"- target: `{target_id}`",
        f"- peak_compute_gflops: {peak_compute}",
        f"- peak_bandwidth_gb_s: {peak_bandwidth}",
        f"- model_kind: blocked_matmul_roofline_v1 (deterministic)",
        f"- candidates: {candidates_modeled}/{len(candidates_out)} modeled",
        "",
        "| candidate | tile | matmul | tier | bottleneck | predicted_us |"
        " gpu_us | cpu_us |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for x in sorted(
        candidates_out, key=lambda c: c.get("predicted_us") or 0,
    ):
        if x.get("model_status") != "ok":
            continue
        sh = x["matmul_shape"]; t = x["tile"]
        cal = x.get("calibration_delta") or {}
        md_lines.append(
            f"| `{x['label']}` | ({t['M']},{t['N']},{t['K']}) "
            f"| ({sh['M']},{sh['N']},{sh['K']}) "
            f"| `{x['bottleneck_tier']}` "
            f"| `{x['bottleneck_resource']}` "
            f"| {x['predicted_us']:.4f} "
            f"| {cal.get('measured_gpu_us', '—')} "
            f"| {cal.get('measured_cpu_us', '—')} |"
        )
    summary_md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return AnalyticalCostResult(
        overall=overall, out_dir=out_dir,
        report_path=report_path, summary_md_path=summary_md_path,
        candidate_count=len(candidates_out),
        candidates_modeled=candidates_modeled,
    )
