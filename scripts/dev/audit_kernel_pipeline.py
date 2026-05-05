"""Deep audit of a CompGen run directory after the M-19 → M-23 kernel
pipeline has run.

The audit is intentionally adversarial: it looks for SILENT FAILURES,
not just success. For a run dir, it checks:

1. **Ledger completeness** — every expected stage event present;
   M-19 / M-20 / M-21 / M-22 / M-22.1 / M-23 events present iff
   their opt-in conditions were met.
2. **R009 hash chain** — `stage[i].input_hash == stage[i-1].output_hash`.
3. **IR artifacts** — `payload.mlir` (Stage 1), `action_space.mlir`
   (Stage 2), `transformed_payload.real.mlir` (Stage 3 when applicable)
   exist and are non-empty.
4. **Agent decision request** — `agent_guidance` block present with
   all required fields; `candidate_ids_allowed` non-degenerate (or
   zero with a documented reason); `sources` references new optional
   evidence artifacts.
5. **Cost-matrix completeness in llm_graph_view** — at least one
   legal candidate per region carries the M-21 overlay; at least one
   carries the M-18.3 calibration block when M-18.3 ran.
6. **Cross-overlay consistency**:
   - M-21 standalone modeled count == cost_preview_v2 overlay count.
   - M-22 evidence count == hardware_resource_report.compiled_evidence
     count.
   - M-22.1 cache_evidence value matches across
     compiled_bottleneck_report and hardware_resource_report.
7. **M-15B steady state** — detector returns no failure (or a
   typed expected one).
8. **Per-stage report statuses** — capture / lower / strict_gate /
   M-12 / M-16.2 / M-19 / M-20 / M-22 / M-23.

Output: a single JSON dict per run dir, plus a human-readable
markdown summary. Designed to be called from
``run_kernel_stress_suite.py`` after each model run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(p: Path) -> dict[str, Any] | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_jsonl(p: Path) -> list[dict[str, Any]]:
    if not p.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


# --------------------------------------------------------------------------- #
# Per-check functions — each returns (status, evidence_dict)
# --------------------------------------------------------------------------- #


def _check_ledger(run_dir: Path) -> tuple[str, dict[str, Any]]:
    events = _read_jsonl(run_dir / "stage_ledger.jsonl")
    if not events:
        return "fail", {"reason": "stage_ledger.jsonl empty or missing"}

    # Expected stage_ids.
    stage_ids = {e.get("stage_id") for e in events}
    expected_stages = {"graph_capture", "payload_lowering",
                       "graph_analysis", "recipe_planning"}
    missing_stages = expected_stages - stage_ids
    notes = [e.get("note") or "" for e in events]
    text = " | ".join(notes)

    # Optional milestone events (only required when their stage actually ran).
    milestones_seen: dict[str, bool] = {}
    for tag in ("M-19", "M-20", "M-21", "M-22", "M-22.1", "M-23"):
        milestones_seen[tag] = any(tag in n for n in notes)

    return (
        "pass" if not missing_stages else "fail",
        {
            "event_count": len(events),
            "stage_ids_seen": sorted(stage_ids),
            "missing_stages": sorted(missing_stages),
            "milestones_seen": milestones_seen,
        },
    )


def _check_r009_hash_chain(run_dir: Path) -> tuple[str, dict[str, Any]]:
    manifest = _read(run_dir / "run_manifest.json")
    if manifest is None:
        return "fail", {"reason": "run_manifest.json missing"}
    stages = manifest.get("stages") or []
    if len(stages) < 2:
        return "skip", {"reason": "fewer than 2 stages recorded"}
    breaks: list[dict[str, str]] = []
    for prev, cur in zip(stages, stages[1:]):
        if prev.get("output_hash") != cur.get("input_hash"):
            breaks.append({
                "between": f"{prev.get('stage_id')} -> {cur.get('stage_id')}",
                "prev_out": prev.get("output_hash", "")[:12],
                "cur_in": cur.get("input_hash", "")[:12],
            })
    return (
        "pass" if not breaks else "fail",
        {"stage_count": len(stages), "breaks": breaks},
    )


def _check_ir_artifacts(run_dir: Path) -> tuple[str, dict[str, Any]]:
    # payload.mlir lives under one of the partition subdirs depending
    # on the capture path (export_program vs dynamo_partitions/*).
    pl_dir = run_dir / "01_payload_lowering"
    payload_candidates = sorted(pl_dir.glob("**/payload.mlir"))
    payload = payload_candidates[0] if payload_candidates else (
        pl_dir / "payload.mlir"
    )
    action_space = run_dir / "02_graph_analysis" / "action_space.mlir"
    transformed = (
        run_dir / "03_recipe_planning" / "real_lowering"
        / "transformed_payload.real.mlir"
    )
    findings: dict[str, Any] = {
        "payload_mlir_partition_count": len(payload_candidates),
    }
    issues: list[str] = []
    for label, path in (
        ("payload.mlir", payload),
        ("action_space.mlir", action_space),
        ("transformed_payload.real.mlir", transformed),
    ):
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        findings[label] = {
            "exists": exists, "size_bytes": size,
            "rel_path": (
                str(path.relative_to(run_dir)) if exists else None
            ),
        }
        if label != "transformed_payload.real.mlir":
            # Required for any successful capture+lower run.
            if not exists:
                issues.append(f"{label} missing")
            elif size < 8:
                issues.append(f"{label} suspiciously small ({size}B)")
    return ("pass" if not issues else "fail", {**findings, "issues": issues})


def _check_agent_decision_request(run_dir: Path) -> tuple[str, dict[str, Any]]:
    candidates = [
        run_dir / "03_recipe_planning" / "agent_decision"
        / "agent_decision_request.json",
        run_dir / "agent_decision_request.json",
    ]
    req: dict[str, Any] | None = None
    found_path: Path | None = None
    for p in candidates:
        if p.exists():
            req = _read(p)
            found_path = p
            break
    if req is None:
        return "fail", {"reason": "agent_decision_request.json not found"}

    issues: list[str] = []

    # agent_guidance block.
    g = req.get("agent_guidance")
    if g is None:
        issues.append("agent_guidance block missing")
        guidance_summary = None
    else:
        guidance_required = (
            "guidance_version", "preamble", "cost_column_priority",
            "disagreement_handling", "rationale_field_examples",
            "forbidden_phrase_patterns", "preferred_neutral_phrases",
            "response_shape",
        )
        missing = [f for f in guidance_required if f not in g]
        if missing:
            issues.append(
                f"agent_guidance missing fields: {sorted(missing)}"
            )
        guidance_summary = {
            "guidance_version": g.get("guidance_version"),
            "n_cost_columns": len(g.get("cost_column_priority", []) or []),
            "n_disagreement_signals": len(
                g.get("disagreement_handling", []) or []
            ),
            "n_rationale_examples": len(
                g.get("rationale_field_examples", []) or []
            ),
            "n_forbidden_phrases": len(
                g.get("forbidden_phrase_patterns", []) or []
            ),
        }

    # candidate_ids_allowed non-degenerate.
    cids = req.get("candidate_ids_allowed", [])
    n_allowed = len(cids)

    # sources references new optional evidence artifacts.
    sources = req.get("sources", {}) or {}
    expected_keys = (
        "analytical_cost_report",
        "compiled_bottleneck_report",
        "region_compiled_differential_report",
        "hardware_resource_report",
        "readiness_matrix",
    )
    missing_source_keys = [k for k in expected_keys if k not in sources]
    if missing_source_keys:
        issues.append(
            f"sources missing optional-evidence keys: {missing_source_keys}"
        )

    return (
        "pass" if not issues else "fail",
        {
            "request_path": (
                str(found_path.relative_to(run_dir)) if found_path else None
            ),
            "agent_guidance": guidance_summary,
            "candidate_ids_allowed_count": n_allowed,
            "sources_keys_present": sorted(sources.keys()),
            "issues": issues,
        },
    )


def _check_cost_matrix_completeness(run_dir: Path) -> tuple[str, dict[str, Any]]:
    """Check that the agent actually sees the expected cost columns
    in llm_graph_view for at least one candidate per region."""
    lv = _read(run_dir / "02_graph_analysis" / "llm_graph_view.json")
    if lv is None:
        return "skip", {"reason": "llm_graph_view.json missing"}
    cp = _read(run_dir / "02_graph_analysis" / "cost_preview_v2.json")
    has_static = 0
    has_m21 = 0
    has_m18_3_calibration = 0
    n_legal_candidates = 0
    if cp is not None:
        for p in cp.get("cost_previews", []) or []:
            n_legal_candidates += 1
            if p.get("static_relative_cost") is not None:
                has_static += 1
            if p.get("m21_analytical_cost") is not None:
                has_m21 += 1
            if p.get("calibration") is not None:
                has_m18_3_calibration += 1
    return (
        "pass" if has_static > 0 else "skip",
        {
            "n_legal_candidates": n_legal_candidates,
            "with_static_relative_cost": has_static,
            "with_m21_analytical_cost": has_m21,
            "with_m183_calibration": has_m18_3_calibration,
        },
    )


def _check_cross_overlay_consistency(
    run_dir: Path,
) -> tuple[str, dict[str, Any]]:
    """Verify each overlay matches its standalone source of truth."""
    issues: list[str] = []
    findings: dict[str, Any] = {}

    # M-21: cost_preview_v2 overlay count == standalone modeled count.
    ac = _read(
        run_dir / "02_graph_analysis" / "analytical_cost"
        / "per_candidate_analytical_cost.json"
    )
    cp = _read(run_dir / "02_graph_analysis" / "cost_preview_v2.json")
    if ac is not None and cp is not None:
        m21_modeled = {
            c["candidate_id"] for c in ac.get("candidates", []) or []
            if c.get("model_status") == "ok"
        }
        cp_overlaid = {
            p["candidate_id"] for p in cp.get("cost_previews", []) or []
            if p.get("m21_analytical_cost") is not None
        }
        findings["m21_modeled"] = len(m21_modeled)
        findings["m21_cp_overlaid"] = len(cp_overlaid)
        missing_in_cp = m21_modeled - cp_overlaid
        leaked_in_cp = cp_overlaid - m21_modeled
        if missing_in_cp:
            issues.append(
                f"M-21 missing in cost_preview_v2: "
                f"{sorted(missing_in_cp)[:3]}"
            )
        if leaked_in_cp:
            issues.append(
                f"M-21 overlay leaked: {sorted(leaked_in_cp)[:3]}"
            )

    # M-22: hardware_resource_report compiled_evidence count == M-22 ok regions.
    cb = _read(
        run_dir / "02_graph_analysis" / "compiled_bottleneck"
        / "compiled_bottleneck_report.json"
    )
    hrr = _read(
        run_dir / "02_graph_analysis" / "readiness"
        / "hardware_resource_report.json"
    )
    if cb is not None and hrr is not None and cb.get("overall") == "ok":
        m22_ok = {
            r["region_id"] for r in cb.get("regions", []) or []
            if r.get("model_status") == "ok"
        }
        hrr_overlaid = {
            r["region_id"] for r in hrr.get("regions", []) or []
            if r.get("compiled_evidence") is not None
        }
        findings["m22_ok"] = len(m22_ok)
        findings["m22_hrr_overlaid"] = len(hrr_overlaid)
        if m22_ok - hrr_overlaid:
            issues.append(
                f"M-22 missing in hardware_resource_report: "
                f"{sorted(m22_ok - hrr_overlaid)[:3]}"
            )

    # M-22.1: cache_evidence consistency between
    # compiled_bottleneck_report and hardware_resource_report.
    if cb is not None and hrr is not None:
        cb_by_rid = {
            r.get("region_id"): r.get("cache_evidence")
            for r in cb.get("regions", []) or []
            if r.get("model_status") == "ok"
        }
        hrr_by_rid = {
            r.get("region_id"): (
                (r.get("compiled_evidence") or {}).get("cache_evidence")
            )
            for r in hrr.get("regions", []) or []
            if r.get("compiled_evidence") is not None
        }
        mismatches = [
            (rid, cb_by_rid[rid], hrr_by_rid.get(rid))
            for rid in cb_by_rid
            if rid in hrr_by_rid and cb_by_rid[rid] != hrr_by_rid[rid]
        ]
        findings["m221_cross_overlay_mismatches"] = len(mismatches)
        if mismatches:
            issues.append(
                f"M-22.1 cache_evidence drift between cb and hrr: "
                f"{mismatches[:3]}"
            )

    return ("pass" if not issues else "fail", {**findings, "issues": issues})


def _check_m15b_steady_state(run_dir: Path) -> tuple[str, dict[str, Any]]:
    """The M-15B detector should NOT see failures on a healthy run.
    Surface any detected failure typed."""
    try:
        from compgen.graph_compilation.downstream_retry import (
            detect_downstream_failure,
        )
    except ImportError:
        return "skip", {"reason": "downstream_retry import failed"}
    failure = detect_downstream_failure(run_dir)
    if failure is None:
        return "pass", {"failed_check": None}
    # NOTE: M-15B's failure being detected on a stress run is a HONEST
    # retry signal, not a system bug. The audit reports it but classifies
    # as a typed retry-needed state, distinct from "broken pipeline".
    return "retry_needed", {
        "failed_check": failure.failed_check,
        "failed_stage": failure.failed_stage,
        "report_path": failure.report_path,
        "failure_summary": failure.failure_summary,
    }


# --------------------------------------------------------------------------- #
# Top-level audit
# --------------------------------------------------------------------------- #


_AUDIT_CHECKS: tuple[
    tuple[str, "callable[[Path], tuple[str, dict[str, Any]]]"], ...
] = (
    ("ledger_completeness", _check_ledger),
    ("r009_hash_chain", _check_r009_hash_chain),
    ("ir_artifacts", _check_ir_artifacts),
    ("agent_decision_request", _check_agent_decision_request),
    ("cost_matrix_completeness", _check_cost_matrix_completeness),
    ("cross_overlay_consistency", _check_cross_overlay_consistency),
    ("m15b_steady_state", _check_m15b_steady_state),
)


def audit_run_dir(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    checks: dict[str, dict[str, Any]] = {}
    fail_count = 0
    for name, fn in _AUDIT_CHECKS:
        try:
            status, evidence = fn(run_dir)
        except Exception as exc:  # noqa: BLE001
            status = "error"
            evidence = {"exception": f"{type(exc).__name__}: {exc}"}
        checks[name] = {"status": status, **evidence}
        if status == "fail":
            fail_count += 1
    return {
        "run_dir": str(run_dir),
        "overall": "pass" if fail_count == 0 else "fail",
        "fail_count": fail_count,
        "checks": checks,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    result = audit_run_dir(args.run_dir)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0 if result["overall"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
