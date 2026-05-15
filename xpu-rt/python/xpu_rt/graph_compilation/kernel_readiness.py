"""Kernel Section Readiness Lock.

Parallel of for kernel-level evidence. turns the FX-level
"Snapshot of Graph Analysis" slide's 6 rows into hard typed artifacts;
does the same for the kernel-level twin slide where every row is
backed by REAL COMPILED-KERNEL EVIDENCE ////.

Read-only aggregator. No new measurement, no compiler-core changes,
no candidate generation, no mutation of source artifacts.

The 6 slide rows:

1. compiled_precision — /refinement_status per region
2. compiled_working_set — utilizations (compute/bandwidth)
3. compiled_lifetime — per-kernel CUDA events
                                (ready_for_m24_1 — Nsight ncu integration
                                 needed for register-pressure / occupancy)
4. compiled_candidate_evidence — /measurements per legal candidate
5. compiled_agent_view — agent_decision_request × cross-ref
6. compiled_bottleneck — kernel_calibration_status

Output: 02_graph_analysis/kernel_readiness/{matrix, 6 reports, summary}.

Hard non-goals:
- No new measurement.
- No new candidate generation.
- No compiler-core imports.
No mutation of ////reports.
- fp32 only.
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
# Status constants
# --------------------------------------------------------------------------- #

_NOT_RUN = "not_run"
_READY = "ready"
_READY_FOR_M24_1 = "ready_for_m24_1"
_PARTIAL = "partial"
_NOT_READY = "not_ready"


def _kernels_were_on(run_dir: Path) -> bool:
    """Heuristic: the report exists with overall=ok iff
    COMPGEN_RUN_KERNELS=1 actually reached with measurements."""
    cb = _read_json(
        run_dir / "02_graph_analysis" / "compiled_bottleneck"
        / "compiled_bottleneck_report.json"
    )
    return cb is not None and cb.get("overall") == "ok"


# --------------------------------------------------------------------------- #
# Row 1: Compiled precision (/refinement_status)
# --------------------------------------------------------------------------- #


def _build_compiled_precision_report(run_dir: Path) -> dict[str, Any]:
    base = run_dir / "02_graph_analysis" / "kernel_execution"
    m20 = _read_json(base / "region_compiled_differential_report.json")
    if m20 is None:
        return {
            "schema_version": "compiled_precision_report_v1",
            "row": 1, "claim": "compiled_precision",
            "status": _NOT_RUN,
            "reason": "M-20 region_compiled_differential not_run",
            "regions": [],
            "summary": {
                "regions_total": 0,
                "bit_equality_count": 0,
                "tolerance_eps_count": 0,
                "fail_outside_tolerance_count": 0,
            },
            "known_limitations": [
                "fp32 only",
                "tolerance_eps refinement is the steady state for "
                "compiled fp32 matmul (bit-equality requires "
                "K_iters == 1)",
            ],
            "generated_at_utc": _utcnow(),
        }

    regions_out: list[dict[str, Any]] = []
    bit_eq = tol = fail = 0
    for r in m20.get("regions", []) or []:
        rid = r.get("region_id") or ""
        gpu = (r.get("gpu") or {}).get("numerical") or {}
        cpu = (r.get("cpu") or {}).get("numerical") or {}
        gpu_status = gpu.get("refinement_status")
        cpu_status = cpu.get("refinement_status")

        # Region status: pass if any track discharges; fail if any
        # exceeds tolerance.
        region_status = "ok"
        for s in (gpu_status, cpu_status):
            if s == "fail_outside_tolerance":
                region_status = "fail"
        if region_status == "ok":
            for s in (gpu_status, cpu_status):
                if s == "discharged_compiled_bit_equality":
                    bit_eq += 1
                    break
            else:
                for s in (gpu_status, cpu_status):
                    if s == "discharged_tolerance_eps":
                        tol += 1
                        break
        else:
            fail += 1

        regions_out.append({
            "region_id": rid,
            "candidate_id": r.get("candidate_id"),
            "gpu_refinement_status": gpu_status,
            "cpu_refinement_status": cpu_status,
            "gpu_max_abs_error": gpu.get("max_abs_error"),
            "gpu_max_rel_error": gpu.get("max_rel_error"),
            "cpu_max_abs_error": cpu.get("max_abs_error"),
            "cpu_max_rel_error": cpu.get("max_rel_error"),
            "status": region_status,
        })

    if not regions_out:
        status = _NOT_RUN
        reason = "no compiled regions"
    elif fail > 0:
        status = _NOT_READY
        reason = f"{fail} region(s) exceed tolerance"
    elif bit_eq + tol == len(regions_out):
        status = _READY
        reason = f"{bit_eq} bit_equality + {tol} tolerance_eps"
    else:
        status = _PARTIAL
        reason = "some regions missing refinement_status"

    return {
        "schema_version": "compiled_precision_report_v1",
        "row": 1, "claim": "compiled_precision",
        "status": status, "reason": reason,
        "regions": regions_out,
        "summary": {
            "regions_total": len(regions_out),
            "bit_equality_count": bit_eq,
            "tolerance_eps_count": tol,
            "fail_outside_tolerance_count": fail,
        },
        "known_limitations": [
            "fp32 only",
            "tolerance_eps refinement is the steady state for "
            "compiled fp32 matmul (bit-equality requires K_iters == 1)",
        ],
        "generated_at_utc": _utcnow(),
    }


# --------------------------------------------------------------------------- #
# Row 2: Compiled working-set (utilizations)
# --------------------------------------------------------------------------- #


def _build_compiled_working_set_report(run_dir: Path) -> dict[str, Any]:
    cb = _read_json(
        run_dir / "02_graph_analysis" / "compiled_bottleneck"
        / "compiled_bottleneck_report.json"
    )
    if cb is None or cb.get("overall") != "ok":
        return {
            "schema_version": "compiled_working_set_report_v1",
            "row": 2, "claim": "compiled_working_set",
            "status": _NOT_RUN,
            "reason": "M-22 compiled_bottleneck_report not_ok",
            "regions": [],
            "kernel_calibration_status": (cb or {}).get(
                "kernel_calibration_status", "not_kernel_calibrated"
            ),
            "generated_at_utc": _utcnow(),
        }

    regions_out: list[dict[str, Any]] = []
    fully_populated = 0
    for r in cb.get("regions", []) or []:
        if r.get("model_status") != "ok":
            continue
        gpu = r.get("gpu") or {}
        cpu = r.get("cpu") or {}
        c_util = (
            gpu.get("compute_utilization")
            if gpu.get("compute_utilization") is not None
            else (cpu or {}).get("compute_utilization")
        )
        b_util = (
            gpu.get("bandwidth_utilization")
            if gpu.get("bandwidth_utilization") is not None
            else (cpu or {}).get("bandwidth_utilization")
        )
        populated = c_util is not None and b_util is not None
        if populated:
            fully_populated += 1
        regions_out.append({
            "region_id": r.get("region_id"),
            "candidate_id": r.get("candidate_id"),
            "compute_utilization": c_util,
            "bandwidth_utilization": b_util,
            "measured_bottleneck": r.get("measured_bottleneck"),
            "populated": populated,
        })

    kc = cb.get("kernel_calibration_status", "not_kernel_calibrated")
    if not regions_out:
        status = _NOT_RUN
        reason = "no M-22 evidence regions"
    elif kc == "kernel_calibrated" and fully_populated == len(regions_out):
        status = _READY
        reason = "every non-opaque region has measured utilizations"
    elif kc == "partial_kernel_calibration" and fully_populated > 0:
        status = _READY
        reason = (
            f"{fully_populated}/{len(regions_out)} regions populated "
            f"(M-22 partial_kernel_calibration)"
        )
    else:
        status = _PARTIAL
        reason = (
            f"{fully_populated}/{len(regions_out)} regions populated"
        )

    return {
        "schema_version": "compiled_working_set_report_v1",
        "row": 2, "claim": "compiled_working_set",
        "status": status, "reason": reason,
        "regions": regions_out,
        "kernel_calibration_status": kc,
        "summary": {
            "regions_total": len(regions_out),
            "fully_populated_count": fully_populated,
        },
        "known_limitations": [
            "achieved_compute / achieved_bandwidth derived post-hoc "
            "from M-22 (M-19/M-20 measured time × M-21 analytical "
            "flops/bytes); does not measure DRAM traffic directly",
            "tier classification is from analytical working_set "
            "vs target memory_tiers, not measured cache residency",
        ],
        "generated_at_utc": _utcnow(),
    }


# --------------------------------------------------------------------------- #
# Row 3: Compiled lifetime (per-kernel CUDA events; ready_for_m24_1)
# --------------------------------------------------------------------------- #


def _build_compiled_lifetime_report(run_dir: Path) -> dict[str, Any]:
    pe = _read_json(
        run_dir / "02_graph_analysis" / "profiler_evidence"
        / "profiler_evidence_report.json"
    )
    kl = _read_json(
        run_dir / "02_graph_analysis" / "kernel_lifetime"
        / "kernel_lifetime_evidence_report.json"
    )

    if pe is None or pe.get("overall") != "ok":
        return {
            "schema_version": "compiled_lifetime_report_v1",
            "row": 3, "claim": "compiled_lifetime",
            "status": _NOT_RUN,
            "reason": "M-22.1 profiler_evidence not_ok",
            "regions": [],
            "known_limitations": [
                "register-pressure / SM-occupancy / shared-memory "
                "are now sourced from triton.compiler.CompiledKernel "
                "introspection (M-24.1) when present",
                "dynamic counters (SM throughput, achieved occupancy, "
                "cache hit rates) require ncu with non-admin "
                "perf-counter access (RmProfilingAdminOnly=0)",
            ],
            "generated_at_utc": _utcnow(),
        }

    # Index lifetime evidence by region_id (when present).
    lifetime_by_region: dict[str, dict[str, Any]] = {}
    if kl is not None and kl.get("overall") == "ok":
        for r in kl.get("regions", []) or []:
            rid = str(r.get("region_id") or "")
            if rid:
                lifetime_by_region[rid] = r

    regions_out: list[dict[str, Any]] = []
    has_kernel_events = 0
    has_static_lifetime = 0
    for r in pe.get("regions", []) or []:
        rid = str(r.get("region_id") or "")
        gpu = r.get("gpu") or {}
        kernel_events = gpu.get("kernel_events") or []
        total_calls = gpu.get("total_cuda_calls", 0)
        self_cuda_us = gpu.get("self_cuda_us_per_iter")
        if kernel_events:
            has_kernel_events += 1

        # static lifetime fields (Triton introspection).
        kl_r = lifetime_by_region.get(rid) or {}
        ti = kl_r.get("triton_introspection") or {}
        register_pressure = (
            ti.get("register_pressure")
            if ti.get("introspection_status") == "introspected"
            else None
        )
        register_spills = (
            ti.get("register_spills")
            if ti.get("introspection_status") == "introspected"
            else None
        )
        shared_memory_bytes = (
            ti.get("shared_memory_bytes")
            if ti.get("introspection_status") == "introspected"
            else None
        )
        occ = ti.get("theoretical_occupancy") or {}
        theoretical_occupancy = occ.get("occupancy_fraction")
        ncu_status = ((kl_r.get("ncu_evidence") or {})
                      .get("ncu_status", "not_collected"))

        # Row-3 "ready" requires the 4 static fields populated.
        static_complete = (
            register_pressure is not None
            and register_spills is not None
            and shared_memory_bytes is not None
            and theoretical_occupancy is not None
        )
        if static_complete:
            has_static_lifetime += 1

        regions_out.append({
            "region_id": rid,
            "candidate_id": r.get("candidate_id"),
            "kernel_event_count": len(kernel_events),
            "total_cuda_calls": total_calls,
            "self_cuda_us_per_iter": self_cuda_us,
            "top_kernels": [
                {
                    "key": ev.get("key"),
                    "count": ev.get("count"),
                    "self_cuda_time_us": ev.get("self_cuda_time_us"),
                }
                for ev in kernel_events[:3]
            ],
            # fields (None when introspection didn't run):
            "register_pressure": register_pressure,
            "register_spills": register_spills,
            "shared_memory_bytes": shared_memory_bytes,
            "theoretical_occupancy": theoretical_occupancy,
            "occupancy_limit": occ.get("limit"),
            "target_arch": ti.get("target_arch"),
            "ncu_status": ncu_status,
        })

    # ready when introspection populated static fields on every
    # region with kernel events. ready_for_m24_1 is the previous fallback
    # (we have CUDA events but no static lifetime). not_run when no
    # regions at all.
    if not regions_out:
        status = _NOT_RUN
        reason = "no M-22.1 evidence regions"
    elif has_static_lifetime == has_kernel_events and has_static_lifetime > 0:
        status = _READY
        reason = (
            f"{has_static_lifetime}/{len(regions_out)} regions have "
            f"static lifetime fields (register_pressure, "
            f"register_spills, shared_memory_bytes, "
            f"theoretical_occupancy) from M-24.1 Triton introspection"
        )
    elif has_kernel_events > 0:
        status = _READY_FOR_M24_1
        reason = (
            f"{has_kernel_events}/{len(regions_out)} regions have "
            f"per-kernel CUDA events from torch.profiler; "
            f"M-24.1 Triton introspection partially populated "
            f"({has_static_lifetime}/{len(regions_out)})"
        )
    else:
        status = _PARTIAL
        reason = "regions present but no kernel_events captured"

    return {
        "schema_version": "compiled_lifetime_report_v1",
        "row": 3, "claim": "compiled_lifetime",
        "status": status, "reason": reason,
        "regions": regions_out,
        "summary": {
            "regions_total": len(regions_out),
            "regions_with_kernel_events": has_kernel_events,
            "regions_with_static_lifetime": has_static_lifetime,
        },
        "known_limitations": [
            "static fields (registers, shared_mem, theoretical "
            "occupancy) come from triton.compiler.CompiledKernel "
            "introspection (M-24.1); deterministic, no admin needed",
            "dynamic counters (achieved occupancy, SM throughput, "
            "cache hit rates) require ncu with "
            "RmProfilingAdminOnly=0 (root or kernel param)",
            "no instruction-mix breakdown (compute vs memory vs sync)",
            "torch.profiler aggregates per kernel_key, not per region",
        ],
        "generated_at_utc": _utcnow(),
    }


# --------------------------------------------------------------------------- #
# Row 4: Compiled candidate evidence (every legal SetTileParams covered)
# --------------------------------------------------------------------------- #


def _build_compiled_candidate_evidence_report(
    run_dir: Path,
) -> dict[str, Any]:
    ga = run_dir / "02_graph_analysis"
    cas = _read_json(ga / "candidate_actions.json")
    m20 = _read_json(
        ga / "kernel_execution"
        / "region_compiled_differential_report.json"
    )

    if cas is None:
        return {
            "schema_version": "compiled_candidate_evidence_report_v1",
            "row": 4, "claim": "compiled_candidate_evidence",
            "status": _NOT_RUN,
            "reason": "candidate_actions.json missing",
            "candidates": [],
            "generated_at_utc": _utcnow(),
        }

    legal_set_tile = [
        c for c in cas.get("candidates", []) or []
        if c.get("kind") == "set_tile_params"
        and (c.get("legality") or {}).get("ok")
    ]

    measured_region_ids: set[str] = set()
    if m20 is not None:
        for r in m20.get("regions", []) or []:
            rid = str(r.get("region_id") or "")
            if not rid:
                continue
            gpu = r.get("gpu") or {}
            cpu = r.get("cpu") or {}
            if (gpu.get("compile_status") == "compiled"
                    or cpu.get("compile_status") == "compiled"):
                measured_region_ids.add(rid)

    candidates_out: list[dict[str, Any]] = []
    by_region: dict[str, list[str]] = {}
    for c in legal_set_tile:
        rid = str(c.get("region_id") or "")
        cid = c.get("candidate_id", "")
        by_region.setdefault(rid, []).append(cid)

    region_evidence_count = 0
    region_total = len(by_region)
    for rid, cids in sorted(by_region.items()):
        has_evidence = rid in measured_region_ids
        if has_evidence:
            region_evidence_count += 1
        candidates_out.append({
            "region_id": rid,
            "legal_candidate_ids": sorted(cids),
            "has_compiled_evidence": has_evidence,
        })

    m20_present = m20 is not None
    if region_total == 0:
        status = _NOT_RUN
        reason = "no legal SetTileParams candidates"
    elif region_evidence_count == region_total:
        status = _READY
        reason = (
            f"every region with a legal SetTileParams candidate has "
            f"at least one compiled measurement"
        )
    elif region_evidence_count > 0:
        status = _PARTIAL
        reason = (
            f"{region_evidence_count}/{region_total} regions covered"
        )
    elif not m20_present:
        # /didn't run at all (kernels off). Honest not_run,
        # NOT not_ready — we never tried to measure these.
        status = _NOT_RUN
        reason = (
            "M-19/M-20 region_compiled_differential not on disk "
            "(kernels off)"
        )
    else:
        # /ran but produced no compiled regions (every track
        # failed to compile). That's a real coverage failure.
        status = _NOT_READY
        reason = "M-20 ran but no regions compiled successfully"

    return {
        "schema_version": "compiled_candidate_evidence_report_v1",
        "row": 4, "claim": "compiled_candidate_evidence",
        "status": status, "reason": reason,
        "regions": candidates_out,
        "summary": {
            "regions_total": region_total,
            "regions_with_evidence": region_evidence_count,
            "legal_candidate_count": len(legal_set_tile),
        },
        "known_limitations": [
            "M-20 fans out per-region but uses greedy tile pick; "
            "per-tile-candidate compiled cost is M-21 analytical + "
            "M-22 measured for the selected tile only",
        ],
        "generated_at_utc": _utcnow(),
    }


# --------------------------------------------------------------------------- #
# Row 5: Compiled agent-view completeness
# --------------------------------------------------------------------------- #


def _build_compiled_agent_view_report(run_dir: Path) -> dict[str, Any]:
    candidates_paths = [
        run_dir / "03_recipe_planning" / "agent_decision"
        / "agent_decision_request.json",
        run_dir / "agent_decision_request.json",
    ]
    req = next((_read_json(p) for p in candidates_paths if p.exists()), None)
    cb = _read_json(
        run_dir / "02_graph_analysis" / "compiled_bottleneck"
        / "compiled_bottleneck_report.json"
    )

    if req is None:
        return {
            "schema_version": "compiled_agent_view_report_v1",
            "row": 5, "claim": "compiled_agent_view",
            "status": _NOT_RUN,
            "reason": "agent_decision_request.json missing",
            "generated_at_utc": _utcnow(),
        }

    cids_allowed = set(req.get("candidate_ids_allowed", []) or [])
    measured_cids: set[str] = set()
    if cb is not None:
        for r in cb.get("regions", []) or []:
            if r.get("model_status") != "ok":
                continue
            cid = str(r.get("candidate_id") or "")
            if cid:
                measured_cids.add(cid)

    leaks = sorted(measured_cids - cids_allowed)
    if not measured_cids:
        status = _NOT_RUN
        reason = "no measured candidates (M-22 had no evidence)"
    elif not leaks:
        status = _READY
        reason = (
            f"all {len(measured_cids)} measured candidates are in "
            f"candidate_ids_allowed"
        )
    else:
        status = _NOT_READY
        reason = (
            f"{len(leaks)} measured candidate(s) not in "
            f"candidate_ids_allowed: {leaks[:3]}"
        )

    return {
        "schema_version": "compiled_agent_view_report_v1",
        "row": 5, "claim": "compiled_agent_view",
        "status": status, "reason": reason,
        "summary": {
            "candidate_ids_allowed_count": len(cids_allowed),
            "measured_candidate_count": len(measured_cids),
            "leaks": leaks,
        },
        "known_limitations": [
            "this row checks the action surface contains every "
            "measured candidate; it does NOT verify the agent's "
            "rationale or selection (M-25 evidence pack territory)",
        ],
        "generated_at_utc": _utcnow(),
    }


# --------------------------------------------------------------------------- #
# Row 6: Compiled bottleneck classification
# --------------------------------------------------------------------------- #


def _build_compiled_bottleneck_report(run_dir: Path) -> dict[str, Any]:
    cb = _read_json(
        run_dir / "02_graph_analysis" / "compiled_bottleneck"
        / "compiled_bottleneck_report.json"
    )
    if cb is None:
        return {
            "schema_version": "compiled_bottleneck_report_v1",
            "row": 6, "claim": "compiled_bottleneck",
            "status": _NOT_RUN,
            "reason": "M-22 compiled_bottleneck not_run",
            "generated_at_utc": _utcnow(),
        }

    kc = cb.get("kernel_calibration_status", "not_kernel_calibrated")
    s = cb.get("summary", {}) or {}
    a = s.get("agreement_with_analytical", {}) or {}

    if kc == "kernel_calibrated":
        status = _READY
        reason = "every non-opaque region has compiled_evidence"
    elif kc == "partial_kernel_calibration":
        status = _READY
        reason = (
            "some regions calibrated; rest report typed unavailability "
            "honestly"
        )
    elif kc == "not_kernel_calibrated":
        status = _NOT_RUN
        reason = "kernels did not produce evidence on any region"
    else:
        status = _PARTIAL
        reason = f"unknown kernel_calibration_status: {kc!r}"

    return {
        "schema_version": "compiled_bottleneck_report_v1",
        "row": 6, "claim": "compiled_bottleneck",
        "status": status, "reason": reason,
        "kernel_calibration_status": kc,
        "summary": {
            "regions_with_evidence": s.get("regions_with_evidence", 0),
            "non_opaque_total_regions": s.get("non_opaque_total_regions", 0),
            "agreement_count": a.get("agreement_count", 0),
            "disagreement_count": a.get("disagreement_count", 0),
        },
        "known_limitations": [
            "M-22 derives bottleneck post-hoc from measured time × "
            "analytical flops/bytes; cache_evidence is from M-22.1 "
            "torch.profiler (CUDA) when available",
            "disagreement with M-21 analytical bottleneck is the "
            "honest launch-overhead-regime signal, not a system bug",
        ],
        "generated_at_utc": _utcnow(),
    }


# --------------------------------------------------------------------------- #
# Top-level matrix + summary
# --------------------------------------------------------------------------- #


_ROW_BUILDERS: tuple[
    tuple[int, str, str, "callable[[Path], dict[str, Any]]"], ...
] = (
    (1, "compiled_precision",
     "precision_report.json",
     _build_compiled_precision_report),
    (2, "compiled_working_set",
     "working_set_report.json",
     _build_compiled_working_set_report),
    (3, "compiled_lifetime",
     "lifetime_report.json",
     _build_compiled_lifetime_report),
    (4, "compiled_candidate_evidence",
     "candidate_evidence_report.json",
     _build_compiled_candidate_evidence_report),
    (5, "compiled_agent_view",
     "agent_view_report.json",
     _build_compiled_agent_view_report),
    (6, "compiled_bottleneck",
     "bottleneck_report.json",
     _build_compiled_bottleneck_report),
)


@dataclass(frozen=True)
class KernelReadinessResult:
    overall: str
    out_dir: Path
    matrix_path: Path
    summary_md_path: Path
    ready_count: int
    ready_for_m24_1_count: int
    partial_count: int
    not_ready_count: int


def run_kernel_section_readiness(run_dir: Path) -> KernelReadinessResult:
    """Build 's 6 typed kernel-readiness reports + matrix +
    summary. Best-effort; never raises."""
    run_dir = Path(run_dir).resolve()
    out_dir = run_dir / "02_graph_analysis" / "kernel_readiness"
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = out_dir / "kernel_section_readiness_matrix.json"
    summary_md_path = out_dir / "kernel_section_readiness_summary.md"

    kernels_on = _kernels_were_on(run_dir)

    rows_out: list[dict[str, Any]] = []
    for row_num, claim, fname, fn in _ROW_BUILDERS:
        try:
            body = fn(run_dir)
        except Exception as exc:  # noqa: BLE001
            body = {
                "schema_version": (
                    f"compiled_{claim}_report_v1"
                ),
                "row": row_num, "claim": claim,
                "status": _NOT_RUN,
                "reason": f"builder error: {type(exc).__name__}: {exc}",
                "generated_at_utc": _utcnow(),
            }
        # Persist the per-row report.
        (out_dir / fname).write_text(
            json.dumps(body, indent=2, sort_keys=True), encoding="utf-8",
        )
        rows_out.append({
            "row": row_num,
            "claim": claim,
            "report": fname,
            "status": body.get("status", _NOT_RUN),
            "reason": body.get("reason", ""),
        })

    ready_count = sum(1 for r in rows_out if r["status"] == _READY)
    ready_m24_1 = sum(
        1 for r in rows_out if r["status"] == _READY_FOR_M24_1
    )
    partial = sum(1 for r in rows_out if r["status"] == _PARTIAL)
    not_ready = sum(1 for r in rows_out if r["status"] == _NOT_READY)
    not_run = sum(1 for r in rows_out if r["status"] == _NOT_RUN)

    # overall status semantics (mirrors ):
    #   pass    iff every row in {ready, ready_for_m24_1}
    #   partial iff some rows ready and some not_run/partial
    #   fail    iff any row not_ready
    #   not_run iff every row not_run (kernels off)
    if not_ready > 0:
        overall = "fail"
    elif (ready_count + ready_m24_1) == len(rows_out):
        overall = "pass"
    elif not_run == len(rows_out):
        overall = "not_run"
    else:
        overall = "partial"

    matrix = {
        "schema_version": "kernel_section_readiness_matrix_v1",
        "overall": overall,
        "kernels_enabled": kernels_on,
        "slide_rows": rows_out,
        "ready_count": ready_count,
        "ready_for_m24_1_count": ready_m24_1,
        "partial_count": partial,
        "not_ready_count": not_ready,
        "not_run_count": not_run,
        "honest_non_claims": [
            "row 3 (compiled_lifetime) is intentionally "
            "ready_for_m24_1 — register-pressure / SM-occupancy / "
            "stack-usage await Nsight Compute (ncu) integration "
            "(parallel to M-17.1's ready_for_m18 before M-18 shipped)",
            "fp32 only",
            "CUDA-only kernel evidence; CPU perf is honest "
            "perf_unavailable when paranoid >= 3",
            "single target per run; cross-target readiness is M-25",
        ],
        "generated_at_utc": _utcnow(),
    }
    matrix_path.write_text(
        json.dumps(matrix, indent=2, sort_keys=True), encoding="utf-8",
    )

    md_lines = [
        f"# Kernel Section Readiness (M-24) — {overall}\n",
        f"- kernels_enabled: {kernels_on}",
        f"- ready: {ready_count}",
        f"- ready_for_m24_1: {ready_m24_1}",
        f"- partial: {partial}",
        f"- not_ready: {not_ready}",
        f"- not_run: {not_run}",
        "",
        "| row | claim | status | reason |",
        "|---|---|---|---|",
    ]
    for r in rows_out:
        md_lines.append(
            f"| {r['row']} | `{r['claim']}` | `{r['status']}` "
            f"| {r['reason']} |"
        )
    summary_md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return KernelReadinessResult(
        overall=overall, out_dir=out_dir,
        matrix_path=matrix_path, summary_md_path=summary_md_path,
        ready_count=ready_count,
        ready_for_m24_1_count=ready_m24_1,
        partial_count=partial,
        not_ready_count=not_ready,
    )
