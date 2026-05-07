"""Downstream-Gate Rejection Retry (Milestone 15B).

When an agent_decision response passes the M-14A validator and the
recipe is committed, but a downstream gate (M-08 post-lowering, M-09
differential, M-11B real-lowering, M-12 real-transform-differential)
reports ``status=fail``, this module:

1. Detects which stage failed and what the failed candidate was.
2. Emits a typed ``downstream_retry_request.json`` carrying:
   - failed_stage, failed_check, failure_summary
   - failed_candidate_id (= candidate_selection.selected_candidate_id)
   - report_path (the actual stage report so Claude Code can read it)
   - retry_policy (must_choose_different_candidate, exclude list)
   - candidate_ids_allowed (legal set MINUS failed candidate)
3. Snapshots the failed attempt to
   ``downstream_retry/attempts/attempt_<N>/`` for audit.

The compiler does NOT auto-retry — Claude Code reads the request,
selects an alternative legal candidate, and re-invokes the pipeline
with a new agent_decision_response. The unchanged M-14A validator + the
action_space.mlir resolver still gate the retry.

Failures that count:

- M-08 post_lowering_verification_report.status == "fail"
- M-09 differential_verification_report.status == "fail"
- M-11B real_transform_validation.overall == "fail"
- M-12 real_differential_report.status == "fail"

Failures that do NOT count (these are path-aware skipped/blocked):

- M-11A eligibility=false (the candidate kind doesn't qualify; not a
  "candidate is bad" signal — typical for FuseProducerConsumer or
  CreateKernelContract recipes)
- M-11B real_transform_kind="unsupported_real_transform" (skipped)
- M-11B real_transform_kind="non_executable_structural_ir" (the
  artifact is structural-only by design)
- M-12 status="blocked" (no executable evaluator)

Hard non-goals:

- No automatic candidate generation.
- No new transforms.
- No edits to verifier reports or transformed payloads.
- No compiler-core changes.

There is no test-only failure injection. M-15B is exercised by tests
that produce *real* downstream failures (e.g. greedy on tiny_mlp
selects tile_16, which makes M-12's bit-equality check fail honestly).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        obj: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return obj
    except (OSError, json.JSONDecodeError):
        return None


def _utcnow() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# Failure detection
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DownstreamFailure:
    failed_stage: str          # e.g. "real_transform_differential"
    failed_check: str          # e.g. "real_transform_differential_check"
    report_path: str           # rel-path to run_dir
    failure_summary: str
    semantic_obligation: str   # obligation id, when known
    remaining: tuple[str, ...]


_DOWNSTREAM_REPORTS: tuple[
    tuple[str, str, str, str], ...
] = (
    # (stage_id, report_path_relative, status_field, failed_check_label)
    ("post_lowering_verification",
     "03_recipe_planning/post_lowering/post_lowering_verification_report.json",
     "status", "post_lowering_structural_check"),
    ("differential_verification",
     "03_recipe_planning/differential_verification/differential_verification_report.json",
     "status", "metadata_noop_equivalence"),
    ("real_transform_validation",
     "03_recipe_planning/real_lowering/real_transform_validation.json",
     "overall", "real_transform_validation"),
    ("real_transform_differential",
     "03_recipe_planning/real_verification/real_differential_report.json",
     "status", "real_transform_differential_check"),
    # M-16.2 fusion-side counterpart. Same retry semantics: a fail
    # status means the agent's selected fusion candidate did not
    # discharge the obligation, and the agent must pick a different
    # candidate. The retry filter excludes the failed candidate id.
    ("real_fusion_differential",
     "03_recipe_planning/real_verification/real_fusion_differential_report.json",
     "status", "real_fusion_differential_check"),
    # M-20 region-level compiled differential. status=fail means at
    # least one matmul region produced an out-of-tolerance compiled
    # kernel. The retry detector treats this the same as M-12: the
    # agent should pick a different SetTileParams candidate (or a
    # different family) to unblock. Note: status=pass with
    # tolerance_eps does NOT trigger retry — that's expected for
    # compiled fp32 matmul.
    ("region_compiled_differential",
     "02_graph_analysis/kernel_execution/region_compiled_differential_report.json",
     "status", "compiled_kernel_differential_check"),
    # M-49 (Phase C) glue-emit differential. status=fail means the
    # M-47-emitted plan executor produced a numerical mismatch vs
    # eager torch.matmul that exceeds the contract's claimed
    # refinement (bit_equality or Higham-bounded tolerance_eps).
    # The retry filter excludes the failed candidate id; the agent
    # should pick a different SetTileParams candidate to unblock.
    ("glue_differential",
     "06_glue_emit/glue_differential_report.json",
     "status", "glue_differential_check"),
    # M-23 compiled fusion verification. status=fail means the fused
    # producer→consumer kernel produced a numerical mismatch vs eager
    # unfused chain (a real correctness regression). The retry filter
    # excludes the failed candidate id; the agent should pick a
    # different fusion candidate or fall back to the SetTileParams /
    # CreateKernelContract path. Note: status=pass with tolerance_eps
    # does NOT trigger retry; bit-equality is the steady-state target
    # for pointwise→pointwise but not strictly required.
    ("compiled_fusion_differential",
     "02_graph_analysis/compiled_fusion/compiled_fusion_differential_report.json",
     "status", "compiled_fusion_differential_check"),
)


def detect_downstream_failure(run_dir: Path) -> DownstreamFailure | None:
    """Scan downstream stage reports in pipeline order. Return the
    FIRST stage whose report says ``status/overall == "fail"``. Skipped
    / blocked / not-applicable paths are ignored — they're not the
    candidate's fault.
    """
    run_dir = Path(run_dir).resolve()
    for stage_id, rel, status_field, check_label in _DOWNSTREAM_REPORTS:
        report_path = run_dir / rel
        body = _read_json(report_path)
        if body is None:
            continue
        status = body.get(status_field, "")
        if status != "fail":
            continue
        # Pull a brief failure summary + obligation context.
        failure_summary = ""
        if "failure_reasons" in body and body["failure_reasons"]:
            failure_summary = "; ".join(body["failure_reasons"][:3])
        elif "checks" in body:
            failed_checks = [
                c for c in body["checks"] if c.get("status") == "fail"
            ]
            if failed_checks:
                failure_summary = "; ".join(
                    c.get("detail", "") or c.get("name", "")
                    for c in failed_checks[:3]
                )
        obligation = ""
        remaining: list[str] = []
        for ostat in body.get("semantic_status", []) or []:
            if ostat.get("status", "").startswith("fail"):
                obligation = ostat.get("obligation", "")
                remaining = list(ostat.get("remaining", []) or [])
                break
        # M-12 has its own obligation block.
        if not obligation:
            for ob in body.get("obligations", []) or []:
                if ob.get("status", "") in ("remaining", "fail"):
                    obligation = ob.get("obligation", "")
                    remaining = list(ob.get("remaining", []) or [])
                    break
        return DownstreamFailure(
            failed_stage=stage_id,
            failed_check=check_label,
            report_path=rel,
            failure_summary=failure_summary or f"{stage_id} reported status=fail",
            semantic_obligation=obligation,
            remaining=tuple(remaining),
        )
    return None


# --------------------------------------------------------------------------- #
# Retry-request emission
# --------------------------------------------------------------------------- #


def emit_downstream_retry_request(
    run_dir: Path,
    *,
    failure: DownstreamFailure,
    attempt_index: int = 0,
) -> Path:
    """Write ``03_recipe_planning/downstream_retry/downstream_retry_request.json``
    with the typed retry payload. Also snapshot the failed attempt to
    ``downstream_retry/attempts/attempt_<N>/``.

    ``candidate_ids_allowed`` is computed by intersecting the bounded
    set Claude Code originally saw (from ``agent_decision_request.json``)
    with the legal set in ``candidate_actions.json``, then subtracting
    the failed ``selected_candidate_id``. The agent must pick from this
    reduced surface in its next attempt.
    """
    run_dir = Path(run_dir).resolve()
    out_dir = run_dir / "03_recipe_planning" / "downstream_retry"
    out_dir.mkdir(parents=True, exist_ok=True)
    attempt_dir = out_dir / "attempts" / f"attempt_{attempt_index:03d}"
    attempt_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the failed candidate from M-05's commit.
    selection = _read_json(
        run_dir / "03_recipe_planning" / "candidate_selection.json"
    ) or {}
    failed_candidate_id = selection.get("selected_candidate_id", "")
    failed_recipe_op = "recipe_0000"  # single-recipe MVP

    # Compute the next-attempt allowed set: bounded view ∩ legal,
    # minus failed candidate. When the bounded view is absent (greedy
    # mode never emits agent_decision_request.json), fall back to the
    # full legal set so the retry surface is non-empty for the next
    # attempt.
    request = _read_json(
        run_dir / "03_recipe_planning" / "agent_decision"
        / "agent_decision_request.json"
    ) or {}
    bounded_allowed = set(request.get("candidate_ids_allowed", []) or [])
    candidate_actions = _read_json(
        run_dir / "02_graph_analysis" / "candidate_actions.json"
    ) or {}
    legal_ids = {
        c["candidate_id"] for c in candidate_actions.get("candidates", [])
        if (c.get("legality") or {}).get("ok") is True
    }
    if bounded_allowed:
        candidate_set = bounded_allowed & legal_ids
    else:
        candidate_set = legal_ids
    next_allowed = sorted(candidate_set - {failed_candidate_id})

    # Build the typed retry-request payload.
    retry: dict[str, Any] = {
        "schema_version": "downstream_retry_request_v1",
        "status": "retry_required",
        "attempt_index": attempt_index,
        "model_id": request.get("model_id", "")
        or selection.get("model_id", ""),
        "target_id": request.get("target_id", "")
        or selection.get("target_id", ""),
        "failed_stage": failure.failed_stage,
        "failed_check": failure.failed_check,
        "failed_candidate_id": failed_candidate_id,
        "failed_recipe_op": failed_recipe_op,
        "failure_summary": failure.failure_summary,
        "evidence": {
            "report_path": failure.report_path,
            "report_sha256": (
                _sha256_file(run_dir / failure.report_path)
                if (run_dir / failure.report_path).exists() else None
            ),
            "semantic_obligation": failure.semantic_obligation,
            "remaining": list(failure.remaining),
        },
        "retry_policy": {
            "must_choose_different_candidate": True,
            "allowed_candidate_source": "agent_decision_request.candidate_ids_allowed",
            "exclude_candidate_ids": [failed_candidate_id]
            if failed_candidate_id else [],
            "prefer_candidates_with": [
                "legality.ok=true",
                "cost_preview_v2.real_transform_verified=true",
                "lower relative_cost",
                "same or stronger refinement status",
            ],
        },
        "candidate_ids_allowed": next_allowed,
        "generated_at_utc": _utcnow(),
    }

    # Write top-level + per-attempt copy.
    retry_path = out_dir / "downstream_retry_request.json"
    retry_path.write_text(
        json.dumps(retry, indent=2, sort_keys=True), encoding="utf-8",
    )
    (attempt_dir / "downstream_retry_request.json").write_text(
        json.dumps(retry, indent=2, sort_keys=True), encoding="utf-8",
    )

    # Snapshot the failed candidate id + the underlying stage report
    # under the attempt dir so it survives the next pipeline wipe.
    (attempt_dir / "selected_candidate_id.txt").write_text(
        failed_candidate_id + "\n", encoding="utf-8",
    )
    src_report = run_dir / failure.report_path
    if src_report.exists():
        shutil.copy2(src_report, attempt_dir / "failed_stage_report.json")

    # Failed-candidate context: features the agent should consider when
    # picking an alternative.
    failed_cand_record: dict[str, Any] = {}
    for c in candidate_actions.get("candidates", []):
        if c.get("candidate_id") == failed_candidate_id:
            failed_cand_record = c
            break
    cost_preview_v2 = _read_json(
        run_dir / "02_graph_analysis" / "cost_preview_v2.json"
    ) or {}
    failed_cp = {}
    for cp in cost_preview_v2.get("cost_previews", []):
        if cp.get("candidate_id") == failed_candidate_id:
            failed_cp = cp
            break
    failed_context = {
        "schema_version": "failed_candidate_context_v1",
        "failed_candidate_id": failed_candidate_id,
        "candidate_kind": failed_cand_record.get("kind", ""),
        "region_id": failed_cand_record.get("region_id", ""),
        "label": failed_cand_record.get("label", ""),
        "cost_preview_v2": failed_cp,
        "failed_stage": failure.failed_stage,
        "failed_check": failure.failed_check,
        "report_excerpt": (
            (_read_json(run_dir / failure.report_path) or {}).get(
                "failure_reasons", []
            )[:3]
        ),
        "generated_at_utc": _utcnow(),
    }
    (out_dir / "failed_candidate_context.json").write_text(
        json.dumps(failed_context, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return retry_path


def emit_downstream_retry_summary(
    run_dir: Path,
    *,
    attempts: list[dict[str, Any]],
    max_attempts: int,
    final_status: str,
    final_selected_candidate_id: str | None,
    recipe_committed: bool,
) -> Path:
    """Append/overwrite ``downstream_retry_summary.json`` with the
    full attempt history for an in-process iterative retry."""
    out_dir = run_dir / "03_recipe_planning" / "downstream_retry"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": "downstream_retry_summary_v1",
        "status": final_status,
        "max_attempts": max_attempts,
        "attempts": attempts,
        "final_selected_candidate_id": final_selected_candidate_id,
        "recipe_committed": recipe_committed,
        "generated_at_utc": _utcnow(),
    }
    p = out_dir / "downstream_retry_summary.json"
    p.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return p


# --------------------------------------------------------------------------- #
# Top-level entry point: scan + emit (called from run_graph_compilation)
# --------------------------------------------------------------------------- #


def detect_and_emit(run_dir: Path) -> DownstreamFailure | None:
    """Scan for a downstream failure; if found, emit retry artifacts.

    Returns the ``DownstreamFailure`` (or None). Callers use this to
    decide whether to raise / abort / mark the run as fail-but-typed.
    """
    failure = detect_downstream_failure(run_dir)
    if failure is None:
        return None
    emit_downstream_retry_request(run_dir, failure=failure, attempt_index=0)
    return failure
