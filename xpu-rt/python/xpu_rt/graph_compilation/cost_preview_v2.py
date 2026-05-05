"""Cost Preview V2 (Milestone 13).

Read-only, deterministic, target-and-tile-sensitive cost-preview
augmentation that lets the agent reason about consequences instead of
just legality. For every legal candidate, M-13 emits a
``cost_preview_v2`` record with:

- ``baseline_static_latency_us`` (pre-transform roofline)
- ``candidate_static_latency_us`` (post-transform roofline)
- ``relative_cost`` (candidate / baseline)
- ``confidence`` (lower for opaque/contract; higher for structured
  linalg with M-12 verification evidence)
- ``features`` (flops, bytes, arithmetic_intensity, tile, fits_*,
  real_transform_verified)
- ``known_limitations`` (explicit honesty about what the static model
  doesn't capture)
- ``evidence`` (paths to region_dossier and, when available, the M-12
  ``real_differential_report.json``)

Hard non-goals:

- No benchmarking, profiling, learned surrogate, or hardware simulator.
- No new candidate generation.
- No real LLM calls.
- No mutation of ``candidate_actions.json`` or ``action_space.mlir``
  (both pinned by ``graph_analysis.output_hash``). Cost-preview-v2 is a
  derived join artifact under ``02_graph_analysis/`` byte-pinned via
  its own internal source SHAs (same pattern as M-10B v3).

The cost model is a deliberately simple roofline:

::

    flops = 2 * M * N * K       (matmul; element-wise scaled by numel)
    bytes = sum(input + output bytes)
    compute_time = flops / peak_compute
    memory_time = bytes / (peak_bw * tier_multiplier)
    latency = max(compute_time, memory_time)

Where ``tier_multiplier`` is greater when the working set fits a faster
tier (scratchpad, L2). Candidate-side ``tile_bytes`` change the tier
selection — that's how the model becomes tile-sensitive. Switching
``peak_compute_gflops`` or ``peak_bandwidth_gb_s`` in the target YAML
changes the latency — that's how the model becomes target-sensitive.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


# --------------------------------------------------------------------------- #
# Result + helpers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CostPreviewV2Result:
    overall: str
    out_dir: Path
    cost_preview_path: Path
    validation_path: Path
    summary_md_path: Path
    candidate_count: int
    legal_candidate_count: int
    failures: tuple[str, ...]


def _utcnow() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> dict[str, Any]:
    obj: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return obj


def _read_json_or_none(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _read_json(path)


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_or_none(path: Path) -> str | None:
    return _sha256_file(path) if path.exists() else None


# --------------------------------------------------------------------------- #
# Target profile reader
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _TargetProfile:
    target_id: str
    peak_compute_flops: float
    peak_bandwidth_bytes_per_sec: float
    scratchpad_bytes: int
    l2_bytes: int
    l3_bytes: int


_DEFAULT_TARGET = _TargetProfile(
    target_id="host_cpu",
    peak_compute_flops=100e9,
    peak_bandwidth_bytes_per_sec=30e9,
    scratchpad_bytes=32 * 1024,
    l2_bytes=512 * 1024,
    l3_bytes=16 * 1024 * 1024,
)


def _load_target_profile(target_yaml: Path | None) -> _TargetProfile:
    if target_yaml is None or not target_yaml.exists():
        return _DEFAULT_TARGET
    raw = yaml.safe_load(target_yaml.read_text(encoding="utf-8"))
    tiers = raw.get("memory_tiers", {}) or {}
    return _TargetProfile(
        target_id=str(raw.get("target_id", "host_cpu")),
        peak_compute_flops=float(raw.get("peak_compute_gflops", 100.0)) * 1e9,
        peak_bandwidth_bytes_per_sec=float(raw.get("peak_bandwidth_gb_s", 30.0)) * 1e9,
        scratchpad_bytes=int(tiers.get("scratchpad_bytes", 32 * 1024)),
        l2_bytes=int(tiers.get("l2_bytes", 512 * 1024)),
        l3_bytes=int(tiers.get("l3_bytes", 16 * 1024 * 1024)),
    )


def _bandwidth_tier_multiplier(
    *, working_set_bytes: int, target: _TargetProfile,
) -> tuple[float, str]:
    """A coarse roofline-cache model: smaller working sets get higher
    effective bandwidth. Returns (multiplier, tier_label)."""
    if working_set_bytes <= target.scratchpad_bytes:
        return 4.0, "scratchpad"
    if working_set_bytes <= target.l2_bytes:
        return 2.0, "l2"
    if working_set_bytes <= target.l3_bytes:
        return 1.2, "l3"
    return 1.0, "system"


# --------------------------------------------------------------------------- #
# Roofline cost
# --------------------------------------------------------------------------- #


def _roofline_latency_us(
    *, flops: float, bytes_moved: int, working_set_bytes: int,
    target: _TargetProfile,
) -> tuple[float, dict[str, Any]]:
    bw_mul, tier = _bandwidth_tier_multiplier(
        working_set_bytes=working_set_bytes, target=target,
    )
    effective_bw = target.peak_bandwidth_bytes_per_sec * bw_mul
    compute_time = flops / max(target.peak_compute_flops, 1.0)
    memory_time = bytes_moved / max(effective_bw, 1.0)
    latency_s = max(compute_time, memory_time)
    return latency_s * 1e6, {
        "tier": tier,
        "bw_multiplier": bw_mul,
        "compute_time_us": compute_time * 1e6,
        "memory_time_us": memory_time * 1e6,
        "bottleneck": (
            "compute" if compute_time >= memory_time else "memory"
        ),
    }


def _matmul_baseline_cost(
    *, M: int, N: int, K: int, target: _TargetProfile,
) -> tuple[float, dict[str, Any]]:
    """Baseline: treat the full matmul as one big operation. Working
    set = full LHS + RHS + OUT (since nothing is tiled)."""
    flops = 2.0 * M * N * K
    bytes_moved = (M * K + K * N + M * N) * 4  # f32
    working_set = bytes_moved
    return _roofline_latency_us(
        flops=flops, bytes_moved=bytes_moved,
        working_set_bytes=working_set, target=target,
    )


# Per-tile-iteration loop overhead (seconds). Models the cost of
# bookkeeping (slice computation, branch, etc.) for each tile-loop
# step. Realistic ballpark for a vectorized inner loop on a modern CPU.
_PER_ITER_OVERHEAD_S = 5e-9  # 5 ns / iteration


def _matmul_tiled_cost(
    *, M: int, N: int, K: int, tM: int, tN: int, tK: int,
    target: _TargetProfile,
) -> tuple[float, dict[str, Any]]:
    """Tiled: same total flops + bytes, but per-tile working set
    determines effective bandwidth, AND a small per-iteration loop
    overhead distinguishes tiles that produce different iteration
    counts (e.g. 2 iters vs 1 iter for the same matmul). This makes
    the cost model tile-sensitive even when all tiles fit the same
    memory tier."""
    flops = 2.0 * M * N * K
    bytes_moved = (M * K + K * N + M * N) * 4
    # Per-tile working set: LHS tile + RHS tile + OUT tile.
    tile_bytes = (tM * tK + tK * tN + tM * tN) * 4
    base_us, diag = _roofline_latency_us(
        flops=flops, bytes_moved=bytes_moved,
        working_set_bytes=tile_bytes, target=target,
    )
    # Iter counts: ceil(dim / tile), but capped at 1 if tile >= dim
    # (the M-11B degenerate-single-iter case).
    iters_M = max(1, (M + tM - 1) // tM)
    iters_N = max(1, (N + tN - 1) // tN)
    iters_K = max(1, (K + tK - 1) // tK)
    total_iters = iters_M * iters_N * iters_K
    overhead_us = total_iters * _PER_ITER_OVERHEAD_S * 1e6
    diag["iters"] = {
        "M": iters_M, "N": iters_N, "K": iters_K, "total": total_iters,
    }
    diag["overhead_us"] = overhead_us
    return base_us + overhead_us, diag


# --------------------------------------------------------------------------- #
# Per-candidate cost preview
# --------------------------------------------------------------------------- #


_CANDIDATE_CONFIDENCE = {
    "set_tile_params": 0.55,
    "fuse_producer_consumer": 0.45,
    "create_kernel_contract": 0.20,
    "create_payload_lowering_extension": 0.20,
    "keep_as_fallback": 0.15,
    "quantize_fp8": 0.30,
    "set_accumulator": 0.30,
    "enable_fast_math": 0.30,
    "assign_device": 0.30,
}


_TILE_LABEL_RE = re.compile(
    r"tile_M(?P<M>\d+)_N(?P<N>\d+)_K(?P<K>\d+)"
)


def _parse_tile_from_label(label: str) -> tuple[int, int, int] | None:
    m = _TILE_LABEL_RE.search(label)
    if not m:
        return None
    return int(m.group("M")), int(m.group("N")), int(m.group("K"))


def _build_candidate_cost_preview(
    *,
    candidate: dict[str, Any],
    region_dossier: dict[str, Any] | None,
    target: _TargetProfile,
    target_yaml_rel: str,
    region_dossier_ref: str | None,
    real_diff_report: dict[str, Any] | None,
    real_diff_report_rel: str | None,
    selected_candidate_id: str,
) -> dict[str, Any]:
    candidate_id = candidate.get("candidate_id", "")
    candidate_kind = candidate.get("kind", "")
    region_id = candidate.get("region_id", "")
    legality_ok = bool((candidate.get("legality") or {}).get("ok"))
    cost_legacy = candidate.get("cost_preview") or {}

    # Region-level facts.
    region_kind = ""
    region_flops = 0
    region_bytes = 0
    if region_dossier is not None:
        region_kind = region_dossier.get("kind", "")
        rcost = region_dossier.get("cost", {}) or {}
        region_flops = int(rcost.get("flops", 0) or 0)
        region_bytes = int(rcost.get("bytes", 0) or 0)

    confidence = _CANDIDATE_CONFIDENCE.get(candidate_kind, 0.30)

    # Verification evidence: only flag when M-12 has actually
    # discharged this candidate's obligation (not just any pass).
    real_transform_verified = False
    verification_evidence_path: str | None = None
    if (
        candidate_id == selected_candidate_id
        and real_diff_report is not None
        and real_diff_report.get("status") == "pass"
        and real_diff_report.get("mode") == "executable_real_transform"
    ):
        real_transform_verified = True
        verification_evidence_path = real_diff_report_rel
        # M-12 evidence raises confidence on the verified candidate.
        confidence = max(confidence, 0.75)

    # Compute baseline + candidate latency.
    baseline_us = 0.0
    candidate_us = 0.0
    cost_diag: dict[str, Any] = {}
    unavailable_reason = ""

    is_matmul_like = region_kind in {"matmul", "conv"}
    if is_matmul_like and region_dossier is not None:
        # Need M/N/K. Region dossier doesn't always carry them in a
        # dedicated field; derive from inputs/outputs in `reuse`.
        reuse = region_dossier.get("reuse", {}) or {}
        in_shapes = [
            tuple(t.get("shape", []) or []) for t in reuse.get("inputs", [])
        ]
        out_shapes = [
            tuple(t.get("shape", []) or []) for t in reuse.get("outputs", [])
        ]
        # Identify (M, K) and (K, N) inputs and (M, N) output.
        M = N = K = 0
        for s in in_shapes:
            if len(s) == 2:
                if M == 0:
                    M, K = s
                elif K == 0:
                    K, N = s
                elif s[1] == K:
                    # Probably (K, N)
                    _, N = s
        if out_shapes and len(out_shapes[0]) == 2:
            M_out, N_out = out_shapes[0]
            if M_out and N_out:
                M = M or M_out
                N = N or N_out
        if M and N and K:
            baseline_us, base_diag = _matmul_baseline_cost(
                M=M, N=N, K=K, target=target,
            )
            cost_diag["baseline"] = base_diag
            # Choose tile based on candidate kind.
            tM = tN = tK = 0
            if candidate_kind == "set_tile_params":
                parsed = _parse_tile_from_label(candidate.get("label", ""))
                if parsed:
                    tM, tN, tK = parsed
                else:
                    # Try recipe_delta.tile (selected candidate carries it).
                    for d in candidate.get("recipe_delta", []) or []:
                        t = d.get("tile") or {}
                        if all(k in t for k in ("M", "N", "K")):
                            tM, tN, tK = int(t["M"]), int(t["N"]), int(t["K"])
                            break
            if tM and tN and tK and candidate_kind == "set_tile_params":
                candidate_us, cand_diag = _matmul_tiled_cost(
                    M=M, N=N, K=K, tM=tM, tN=tN, tK=tK, target=target,
                )
                cost_diag["candidate"] = cand_diag
                cost_diag["tile"] = {"M": tM, "N": tN, "K": tK}
            elif candidate_kind == "fuse_producer_consumer":
                # Fusion saves one round-trip of intermediate tensor I/O.
                # Approximate: subtract one M*N output write and one read.
                saved_bytes = max(M * N * 4, 0)
                eff_bytes = max((M * K + K * N + M * N) * 4 - 2 * saved_bytes, 1)
                candidate_us, cand_diag = _roofline_latency_us(
                    flops=2.0 * M * N * K, bytes_moved=eff_bytes,
                    working_set_bytes=eff_bytes, target=target,
                )
                cost_diag["candidate"] = cand_diag
            elif candidate_kind == "quantize_fp8":
                # Assume halved memory bytes (8-bit vs 32-bit).
                eff_bytes = max((M * K + K * N + M * N) // 2, 1) * 4
                candidate_us, cand_diag = _roofline_latency_us(
                    flops=2.0 * M * N * K, bytes_moved=eff_bytes,
                    working_set_bytes=eff_bytes, target=target,
                )
                cost_diag["candidate"] = cand_diag
            else:
                # Other kinds (contract, fallback, placement, numerics):
                # treat as baseline (no perf claim from M-13).
                candidate_us = baseline_us
                cost_diag["candidate"] = {
                    "comment": "no perf delta claimed for this candidate kind",
                }
        else:
            unavailable_reason = (
                f"could not derive M/N/K from region_dossier "
                f"(in_shapes={in_shapes}, out_shapes={out_shapes})"
            )
    elif region_dossier is None:
        unavailable_reason = "region_dossier missing"
    else:
        # Non-matmul region: use region_dossier's flops + bytes directly.
        if region_flops > 0 or region_bytes > 0:
            baseline_us, base_diag = _roofline_latency_us(
                flops=float(region_flops), bytes_moved=region_bytes,
                working_set_bytes=max(region_bytes, 1),
                target=target,
            )
            cost_diag["baseline"] = base_diag
            # Generic candidate kinds default to baseline cost.
            candidate_us = baseline_us
            cost_diag["candidate"] = {
                "comment": "no perf delta claimed for non-matmul candidate",
            }
        else:
            unavailable_reason = "region has zero flops and zero bytes"

    relative_cost = (
        candidate_us / baseline_us if baseline_us > 0 else 1.0
    )

    features = {
        "flops": region_flops,
        "bytes": region_bytes,
        "arithmetic_intensity": (
            (region_dossier or {}).get("cost", {}).get(
                "arithmetic_intensity", 0.0
            )
        ),
        "bottleneck_resource": (
            (region_dossier or {}).get("cost", {}).get(
                "bottleneck_resource", {}
            )
        ),
        "tile": cost_diag.get("tile"),
        "fits_scratchpad": cost_legacy.get("fits_scratchpad"),
        "fits_l2": cost_legacy.get("fits_l2"),
        "real_transform_verified": real_transform_verified,
    }

    known_limitations = [
        "static roofline estimate only",
        "no cache contention model",
        "no launch overhead model",
        "no measured hardware calibration",
    ]
    if not real_transform_verified:
        known_limitations.append(
            "no differential verification on this candidate"
        )

    cp: dict[str, Any] = {
        "schema_version": "candidate_cost_preview_v2",
        "model": "static_roofline_v2",
        "candidate_id": candidate_id,
        "candidate_kind": candidate_kind,
        "region_id": region_id,
        "legality_ok": legality_ok,
        "baseline_static_latency_us": round(baseline_us, 6),
        "candidate_static_latency_us": round(candidate_us, 6),
        "relative_cost": round(relative_cost, 6),
        "confidence": round(confidence, 4),
        "features": features,
        "known_limitations": known_limitations,
        "evidence": {
            "region_dossier": region_dossier_ref,
            "real_differential_report": verification_evidence_path,
            "target_yaml": target_yaml_rel,
        },
        "diagnostics": cost_diag,
    }
    if unavailable_reason:
        cp["unavailable_reason"] = unavailable_reason
    return cp


# --------------------------------------------------------------------------- #
# Validators
# --------------------------------------------------------------------------- #


def _validate(
    *, cost_previews: list[dict[str, Any]],
    candidate_actions: dict[str, Any],
    real_diff_report: dict[str, Any] | None,
    selected_candidate_id: str,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    fails: dict[str, list[str]] = {}

    def _add(rule: str, fail_details: list[str]) -> None:
        checks.append(
            {
                "id": rule,
                "status": "pass" if not fail_details else "fail",
                "fail_count": len(fail_details),
                "details": fail_details,
            }
        )

    cp_by_id: dict[str, dict[str, Any]] = {
        cp["candidate_id"]: cp for cp in cost_previews
    }

    # CPV2R001 — every legal candidate has cost_preview_v2.
    fdetails: list[str] = []
    for c in candidate_actions.get("candidates", []):
        if (c.get("legality") or {}).get("ok") is True:
            cid = c["candidate_id"]
            if cid not in cp_by_id:
                fdetails.append(f"legal candidate {cid!r} missing cost_preview_v2")
    _add("CPV2R001_legal_has_cost_preview_v2", fdetails)

    # CPV2R002 — non-degeneracy: when ≥2 SetTileParams candidates with
    # different tile geometries are present, the cost model must
    # produce ≥2 distinct relative_costs. For models whose legal
    # candidates are exclusively non-perf-delta kinds (contract /
    # fallback / extension / placement), identical 1.0 cost is honest
    # — those candidate kinds carry no static perf prediction without
    # a chosen kernel, so the rule is satisfied vacuously.
    fdetails = []
    tile_costs = [
        cp["relative_cost"] for cp in cost_previews
        if cp["candidate_kind"] == "set_tile_params"
        and cp.get("legality_ok") is True
    ]
    if len(tile_costs) >= 2:
        if len(set(round(c, 6) for c in tile_costs)) < 2:
            fdetails.append(
                f"all {len(tile_costs)} legal SetTileParams candidates have "
                f"identical relative_cost={tile_costs[0]} despite different "
                f"tile geometries"
            )
    _add("CPV2R002_tile_costs_not_constant", fdetails)

    # CPV2R003 — opaque/contract confidence < linalg structured confidence.
    structured_confs = [
        cp["confidence"] for cp in cost_previews
        if cp["candidate_kind"] in {"set_tile_params", "fuse_producer_consumer"}
        and cp.get("legality_ok") is True
    ]
    opaque_confs = [
        cp["confidence"] for cp in cost_previews
        if cp["candidate_kind"] in {
            "create_kernel_contract", "create_payload_lowering_extension",
            "keep_as_fallback",
        }
    ]
    fdetails = []
    if structured_confs and opaque_confs:
        max_opaque = max(opaque_confs)
        min_structured = min(structured_confs)
        # M-12-verified candidates can boost above the default; check
        # that the typical structured baseline (without verification)
        # is still > opaque max. Use the unboosted SetTileParams default.
        unboosted_structured = [
            cp["confidence"] for cp in cost_previews
            if cp["candidate_kind"] == "set_tile_params"
            and not (cp["features"] or {}).get("real_transform_verified")
            and cp.get("legality_ok") is True
        ]
        baseline_structured = (
            min(unboosted_structured) if unboosted_structured
            else min_structured
        )
        if baseline_structured <= max_opaque:
            fdetails.append(
                f"opaque/contract max confidence {max_opaque} >= "
                f"structured baseline {baseline_structured}"
            )
    _add("CPV2R003_structured_confidence_higher_than_opaque", fdetails)

    # CPV2R004 — real_transform_verified flag is grounded in M-12.
    fdetails = []
    for cp in cost_previews:
        rtv = (cp.get("features") or {}).get("real_transform_verified")
        if not rtv:
            continue
        if real_diff_report is None:
            fdetails.append(
                f"candidate {cp['candidate_id']!r} claims "
                f"real_transform_verified but no M-12 report exists"
            )
            continue
        if (
            real_diff_report.get("status") != "pass"
            or real_diff_report.get("mode") != "executable_real_transform"
        ):
            fdetails.append(
                f"candidate {cp['candidate_id']!r} claims verified but "
                f"M-12 report status={real_diff_report.get('status')} "
                f"mode={real_diff_report.get('mode')}"
            )
            continue
        if cp["candidate_id"] != selected_candidate_id:
            fdetails.append(
                f"candidate {cp['candidate_id']!r} claims verified but is "
                f"not the selected candidate ({selected_candidate_id!r})"
            )
    _add("CPV2R004_verification_evidence_grounded", fdetails)

    # CPV2R005 — every cost_preview has all required keys.
    required_keys = {
        "schema_version", "candidate_id", "candidate_kind", "region_id",
        "baseline_static_latency_us", "candidate_static_latency_us",
        "relative_cost", "confidence", "features", "known_limitations",
        "evidence",
    }
    fdetails = []
    for cp in cost_previews:
        missing = required_keys - set(cp)
        if missing:
            fdetails.append(
                f"candidate {cp.get('candidate_id', '?')!r} missing keys: "
                f"{sorted(missing)}"
            )
    _add("CPV2R005_required_keys_present", fdetails)

    overall = "pass" if all(c["status"] == "pass" for c in checks) else "fail"
    return {
        "schema_version": "cost_preview_v2_validation_v1",
        "overall": overall,
        "checks": checks,
        "counts": {
            "candidates_total": len(candidate_actions.get("candidates", [])),
            "legal_candidates": sum(
                1 for c in candidate_actions.get("candidates", [])
                if (c.get("legality") or {}).get("ok") is True
            ),
            "cost_previews_emitted": len(cost_previews),
        },
    }


# --------------------------------------------------------------------------- #
# v3 + llm_graph_view re-emission with cost_preview_v2 inlined
# --------------------------------------------------------------------------- #


def _inline_cost_preview_into_v3(
    *,
    v3_path: Path,
    llm_view_path: Path,
    cost_previews: dict[str, dict[str, Any]],
) -> None:
    if not v3_path.exists() or not llm_view_path.exists():
        return
    v3 = _read_json(v3_path)
    llm = _read_json(llm_view_path)

    for region in v3.get("regions", []):
        for c in region.get("legal_candidates", []):
            cp = cost_previews.get(c["candidate_id"])
            if cp is not None:
                c["cost_preview_v2"] = cp
        for c in region.get("illegal_candidates", []):
            cp = cost_previews.get(c["candidate_id"])
            if cp is not None:
                c["cost_preview_v2"] = cp
        sel = region.get("selected")
        if sel is not None:
            cp = cost_previews.get(sel.get("candidate_id", ""))
            if cp is not None:
                sel["cost_preview_v2"] = cp

    v3.setdefault("source", {})["cost_preview_v2"] = (
        "02_graph_analysis/cost_preview_v2.json"
    )
    v3_path.write_text(json.dumps(v3, indent=2, sort_keys=True), encoding="utf-8")

    # llm_graph_view: sort each region's legal candidates by relative_cost
    # ascending; inline the v2 cost preview for visible candidates.
    for region in llm.get("regions", []):
        legal_list = region.get("legal_candidates", [])
        for c in legal_list:
            cp = cost_previews.get(c["candidate_id"])
            if cp is not None:
                c["cost_preview_v2"] = {
                    "baseline_static_latency_us": cp["baseline_static_latency_us"],
                    "candidate_static_latency_us":
                        cp["candidate_static_latency_us"],
                    "relative_cost": cp["relative_cost"],
                    "confidence": cp["confidence"],
                    "real_transform_verified":
                        cp["features"].get("real_transform_verified", False),
                }
        legal_list.sort(
            key=lambda c: (
                (c.get("cost_preview_v2") or {}).get("relative_cost", 1.0),
                c.get("candidate_id", ""),
            )
        )
    llm_view_path.write_text(
        json.dumps(llm, indent=2, sort_keys=True), encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# Top-level entry point
# --------------------------------------------------------------------------- #


def run_cost_preview_v2(
    run_dir: Path, *, target_yaml_path: Path | None = None,
) -> CostPreviewV2Result:
    run_dir = Path(run_dir).resolve()
    ga = run_dir / "02_graph_analysis"
    rp = run_dir / "03_recipe_planning"
    if not ga.is_dir():
        raise FileNotFoundError(f"02_graph_analysis/ missing under {run_dir}")

    region_map = _read_json(ga / "region_map.json")
    candidate_actions = _read_json(ga / "candidate_actions.json")
    region_dossiers_map = _read_json(
        ga / "graph_dossier_v2.json"
    ).get("region_dossiers", {}) or {}

    real_diff_report = _read_json_or_none(
        rp / "real_verification" / "real_differential_report.json"
    )
    real_diff_report_rel = (
        "03_recipe_planning/real_verification/real_differential_report.json"
        if real_diff_report is not None else None
    )
    candidate_selection = _read_json_or_none(rp / "candidate_selection.json")
    selected_candidate_id = (
        (candidate_selection or {}).get("selected_candidate_id", "") or ""
    )

    target = _load_target_profile(target_yaml_path)
    target_yaml_rel = (
        target_yaml_path.name if target_yaml_path is not None
        else "(default-host_cpu)"
    )

    # Index region dossiers by region_id.
    region_dossiers: dict[str, dict[str, Any]] = {}
    region_dossier_refs: dict[str, str] = {}
    for region_id, ref in region_dossiers_map.items():
        path = run_dir / ref
        if path.exists():
            region_dossiers[region_id] = _read_json(path)
            region_dossier_refs[region_id] = ref

    # Build cost previews per candidate.
    cost_previews: list[dict[str, Any]] = []
    cost_previews_by_id: dict[str, dict[str, Any]] = {}
    for c in candidate_actions.get("candidates", []):
        rid = c.get("region_id", "")
        rd = region_dossiers.get(rid)
        ref = region_dossier_refs.get(rid)
        cp = _build_candidate_cost_preview(
            candidate=c, region_dossier=rd, target=target,
            target_yaml_rel=target_yaml_rel, region_dossier_ref=ref,
            real_diff_report=real_diff_report,
            real_diff_report_rel=real_diff_report_rel,
            selected_candidate_id=selected_candidate_id,
        )
        cost_previews.append(cp)
        cost_previews_by_id[cp["candidate_id"]] = cp

    legal_count = sum(
        1 for c in candidate_actions.get("candidates", [])
        if (c.get("legality") or {}).get("ok") is True
    )

    cost_preview_obj = {
        "schema_version": "cost_preview_v2_v1",
        "model_id": (region_map.get("model_id") or "")
        or (candidate_actions.get("model_id") or ""),
        "target_id": target.target_id,
        "generated_at_utc": _utcnow(),
        "summary": {
            "candidates_total": len(cost_previews),
            "legal_candidates": legal_count,
            "min_relative_cost": min(
                (cp["relative_cost"] for cp in cost_previews if cp.get("legality_ok")),
                default=None,
            ),
            "max_relative_cost": max(
                (cp["relative_cost"] for cp in cost_previews if cp.get("legality_ok")),
                default=None,
            ),
            "verified_candidates": sum(
                1 for cp in cost_previews
                if (cp["features"] or {}).get("real_transform_verified")
            ),
        },
        "source": {
            "candidate_actions": "02_graph_analysis/candidate_actions.json",
            "candidate_actions_sha256": _sha256_or_none(
                ga / "candidate_actions.json"
            ),
            "region_map": "02_graph_analysis/region_map.json",
            "region_map_sha256": _sha256_or_none(ga / "region_map.json"),
            "graph_dossier_v2": "02_graph_analysis/graph_dossier_v2.json",
            "graph_dossier_v2_sha256": _sha256_or_none(
                ga / "graph_dossier_v2.json"
            ),
            "real_differential_report": real_diff_report_rel,
            "real_differential_report_sha256": _sha256_or_none(
                rp / "real_verification" / "real_differential_report.json"
            ),
            "target_yaml": target_yaml_rel,
        },
        "cost_previews": cost_previews,
    }
    cost_preview_path = ga / "cost_preview_v2.json"
    cost_preview_path.write_text(
        json.dumps(cost_preview_obj, indent=2, sort_keys=True), encoding="utf-8",
    )

    # Validation.
    validation = _validate(
        cost_previews=cost_previews,
        candidate_actions=candidate_actions,
        real_diff_report=real_diff_report,
        selected_candidate_id=selected_candidate_id,
    )
    validation_path = ga / "cost_preview_v2_validation.json"
    validation_path.write_text(
        json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8",
    )

    # Re-emit graph_dossier_v3.json + llm_graph_view.json with cost
    # preview v2 inlined.
    _inline_cost_preview_into_v3(
        v3_path=ga / "graph_dossier_v3.json",
        llm_view_path=ga / "llm_graph_view.json",
        cost_previews=cost_previews_by_id,
    )

    # Markdown summary.
    md_lines: list[str] = []
    md_lines.append("# Cost Preview V2\n")
    md_lines.append(f"_Generated_: {_utcnow()}\n")
    md_lines.append(f"- target: `{target.target_id}`")
    md_lines.append(
        f"- peak_compute: `{target.peak_compute_flops / 1e9:.1f} GFLOPS`  "
        f"\n- peak_bandwidth: `{target.peak_bandwidth_bytes_per_sec / 1e9:.1f} GB/s`"
    )
    md_lines.append(f"\n## Summary\n")
    md_lines.append(f"- candidates_total: {len(cost_previews)}")
    md_lines.append(f"- legal_candidates: {legal_count}")
    md_lines.append(
        f"- verified_candidates (M-12): "
        f"{cost_preview_obj['summary']['verified_candidates']}\n"
    )
    md_lines.append("## Validation\n")
    md_lines.append("| rule | status | fail_count |")
    md_lines.append("|---|---|---:|")
    for c in validation["checks"]:
        md_lines.append(f"| {c['id']} | {c['status']} | {c['fail_count']} |")
    md_lines.append("\n## Top legal candidates by relative_cost\n")
    md_lines.append(
        "| candidate_id | kind | region | rel_cost | confidence | verified |"
    )
    md_lines.append("|---|---|---|---:|---:|---|")
    legal_sorted = sorted(
        (cp for cp in cost_previews if cp.get("legality_ok")),
        key=lambda c: c["relative_cost"],
    )
    for cp in legal_sorted[:10]:
        verified = (cp["features"] or {}).get("real_transform_verified", False)
        md_lines.append(
            f"| `{cp['candidate_id'][:48]}` | {cp['candidate_kind']} | "
            f"`{cp['region_id']}` | {cp['relative_cost']:.4f} | "
            f"{cp['confidence']:.2f} | {verified} |"
        )
    summary_md_path = ga / "cost_preview_v2_summary.md"
    summary_md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return CostPreviewV2Result(
        overall=validation["overall"],
        out_dir=ga,
        cost_preview_path=cost_preview_path,
        validation_path=validation_path,
        summary_md_path=summary_md_path,
        candidate_count=len(cost_previews),
        legal_candidate_count=legal_count,
        failures=tuple(
            f"{c['id']}: {d}" for c in validation["checks"]
            if c["status"] != "pass" for d in c["details"]
        ),
    )
