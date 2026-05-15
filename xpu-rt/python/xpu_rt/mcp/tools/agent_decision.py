"""MCP tools for the agent-decision protocol (Claude-Code-first).

These tools let Claude Code (or any MCP client — Codex etc.) drive a
full CompGen compilation end-to-end without reaching out to an
external LLM provider. The agent reads the bounded view, picks a
candidate, and the compiler validates + commits the choice through
the unchanged 11-check validator.

Tools exposed:

- ``compgen_emit_agent_decision_request`` — run the pipeline up to
  ``--stop-after agent-decision-request`` and return the bounded view
  (request + llm_graph_view summary + cost previews for legal
  candidates + greedy's pick).
- ``compgen_commit_agent_decision_response`` — given a response dict,
  re-run the pipeline with ``--selection-mode agent-file`` and report
  what landed (validation, recipe.mlir excerpt, downstream stage
  status). On failure, returns typed retry hints
  (``failed_stage``, ``failed_check``, ``failed_candidate_id``,
  ``retry_options``) so the agent can pick a different candidate
  without re-reading files.
- ``compgen_inspect_pipeline_run`` — read-only summary of a finished
  run directory: per-stage status, agent decision verdict, redaction
  audit, any failure reasons.
- ``compgen_pipeline_status`` — mid-run progress reader. Reads
  ``stage_ledger.jsonl`` and returns the latest event per stage so the
  agent can give the user updates while a long compile is in flight.

Implementation notes:

- The handlers prefer **in-process** execution
  (``run_graph_compilation`` direct call) when feasible — saves the
  Python startup cost on each invocation. Falls back to a subprocess
  with **streamed stderr** (written to a tail file at
  ``<repo>/.compgen/last_run.stderr.log``) only when the caller forces
  it via ``force_subprocess=True`` for isolation.
- Tools take a ``SessionManager`` to satisfy the MCP server contract
  but do NOT use session state — the ``run_dir`` path is the state.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from compgen.mcp.session import SessionManager


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


_STDERR_TAIL_REL = ".compgen/last_run.stderr.log"
_STDOUT_TAIL_REL = ".compgen/last_run.stdout.log"
_TAIL_BYTES = 4_000


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        obj: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return obj
    except (OSError, json.JSONDecodeError):
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except OSError:
        return []
    return out


def _resolve_repo_root() -> Path:
    # python/compgen/mcp/tools/agent_decision.py → 4 parents up = repo root.
    return Path(__file__).resolve().parents[4]


def _tail(text: str) -> str:
    return (text or "")[-_TAIL_BYTES:]


# --------------------------------------------------------------------------- #
# Pipeline invocation: in-process by default; subprocess with streamed
# stderr as fallback.
# --------------------------------------------------------------------------- #


def _run_in_process(
    *,
    model_config: str,
    target_config: str,
    out_dir: Path,
    stop_after: str,
    selection_mode: str,
    agent_decision_response_path: Path | None = None,
) -> dict[str, Any]:
    """Call ``run_graph_compilation`` directly (no subprocess).

    Returns ``{ok, returncode, error, exception_type}``. ``returncode``
    is 0 on success, 1 on a typed compiler error
    (``RuntimeError`` etc.), 2 on unexpected exception.
    """
    try:
        from compgen.graph_compilation.run import run_graph_compilation

        kwargs: dict[str, Any] = {
            "model_config_path": Path(model_config),
            "target_config_path": Path(target_config),
            "out_dir": out_dir,
            "stop_after": stop_after,
            "selection_mode": selection_mode,
        }
        if agent_decision_response_path is not None:
            kwargs["agent_decision_response_paths"] = [
                Path(agent_decision_response_path),
            ]
        run_graph_compilation(**kwargs)
        return {"ok": True, "returncode": 0, "in_process": True}
    except RuntimeError as exc:
        # Typed compiler-side failure (e.g. downstream rejection,
        # validation reject, agent_max_retries exhausted). The
        # run dir's reports carry the structured detail.
        return {
            "ok": False, "returncode": 1, "in_process": True,
            "exception_type": "RuntimeError",
            "error": str(exc),
        }
    except Exception as exc:  # noqa: BLE001 - want to surface every failure typed
        return {
            "ok": False, "returncode": 2, "in_process": True,
            "exception_type": type(exc).__name__,
            "error": str(exc),
        }


def _run_subprocess_with_tail(
    *, args: list[str], cwd: Path, timeout_sec: int,
) -> dict[str, Any]:
    """Invoke the CLI as a subprocess, streaming stderr to a tail file.

    The tail file at ``<cwd>/.compgen/last_run.stderr.log`` is updated
    line-by-line so ``compgen_pipeline_status`` can read mid-run. Only
    used as a fallback when ``force_subprocess=True``; the in-process
    path is preferred.
    """
    cmd = [sys.executable, "-m", "compgen.graph_compilation", *args]
    tail_dir = cwd / ".compgen"
    tail_dir.mkdir(parents=True, exist_ok=True)
    stderr_tail = tail_dir / "last_run.stderr.log"
    stdout_tail = tail_dir / "last_run.stdout.log"
    stderr_tail.write_text("", encoding="utf-8")
    stdout_tail.write_text("", encoding="utf-8")

    stderr_buf: list[str] = []
    stdout_buf: list[str] = []

    def _drain(stream: Any, sink: list[str], path: Path) -> None:
        with path.open("a", encoding="utf-8") as f:
            for line in iter(stream.readline, ""):
                sink.append(line)
                f.write(line)
                f.flush()
            stream.close()

    try:
        proc = subprocess.Popen(
            cmd, cwd=str(cwd),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
    except OSError as exc:
        return {
            "ok": False, "returncode": -1, "in_process": False,
            "error": f"failed to spawn subprocess: {exc}",
            "command": shlex.join(cmd),
        }

    t_out = threading.Thread(
        target=_drain, args=(proc.stdout, stdout_buf, stdout_tail), daemon=True,
    )
    t_err = threading.Thread(
        target=_drain, args=(proc.stderr, stderr_buf, stderr_tail), daemon=True,
    )
    t_out.start(); t_err.start()
    try:
        rc = proc.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
        rc = -2
        return {
            "ok": False, "returncode": rc, "in_process": False,
            "error": f"timeout after {timeout_sec}s",
            "stdout_tail": _tail("".join(stdout_buf)),
            "stderr_tail": _tail("".join(stderr_buf)),
            "command": shlex.join(cmd),
            "stderr_log_path": str(stderr_tail.relative_to(cwd)),
        }
    t_out.join(timeout=5); t_err.join(timeout=5)
    return {
        "ok": rc == 0, "returncode": rc, "in_process": False,
        "stdout_tail": _tail("".join(stdout_buf)),
        "stderr_tail": _tail("".join(stderr_buf)),
        "command": shlex.join(cmd),
        "stderr_log_path": str(stderr_tail.relative_to(cwd)),
    }


def _per_stage_status(run_dir: Path) -> dict[str, str]:
    manifest = _read_json(run_dir / "run_manifest.json")
    if manifest is None:
        return {}
    return {
        s.get("stage_id", ""): s.get("status", "")
        for s in manifest.get("stages", []) or []
    }


# --------------------------------------------------------------------------- #
# Failure inspection: read structured reports + return typed retry hints.
# --------------------------------------------------------------------------- #


def _detect_failure_with_retry_hint(run_dir: Path) -> dict[str, Any] | None:
    """Inspect the run dir's structured reports for a downstream failure
    and produce a typed retry hint the agent can act on without
    re-reading files. Returns ``None`` if no failure is detected.

    Returns ``{failed_stage, failed_check, failed_candidate_id,
    failure_summary, retry_options[]}``.
    """
    # already writes a typed downstream_retry_request.json when
    # any downstream stage reports status=fail; reuse that.
    rr = _read_json(
        run_dir / "03_recipe_planning" / "downstream_retry"
        / "downstream_retry_request.json"
    )
    if rr is not None and rr.get("status") == "retry_required":
        retry_options = list(rr.get("candidate_ids_allowed", []))
        return {
            "failed_stage": rr.get("failed_stage", ""),
            "failed_check": rr.get("failed_check", ""),
            "failed_candidate_id": rr.get("failed_candidate_id", ""),
            "failure_summary": rr.get("evidence", {}).get(
                "failure_summary", ""
            ),
            "retry_options": retry_options,
            "report_path": rr.get("evidence", {}).get("report_path", ""),
        }

    # validation failure (caught before recipe.mlir commit).
    val = _read_json(
        run_dir / "03_recipe_planning" / "agent_decision"
        / "agent_decision_validation.json"
    )
    if val is not None and val.get("overall") != "pass":
        failed_checks = [
            c for c in (val.get("checks") or [])
            if c.get("status") != "pass"
        ]
        if failed_checks:
            return {
                "failed_stage": "agent_decision_validation",
                "failed_check": failed_checks[0].get("name", ""),
                "failed_candidate_id": val.get("selected_candidate_id", ""),
                "failure_summary": "; ".join(
                    f"{c.get('name')}: {c.get('detail') or 'fail'}"
                    for c in failed_checks
                ),
                # Agent should consult candidate_ids_allowed in the
                # request and pick a different one.
                "retry_options": [],
            }

    return None


# --------------------------------------------------------------------------- #
# Tool: emit agent-decision request
# --------------------------------------------------------------------------- #


def compgen_emit_agent_decision_request(
    sm: SessionManager,  # noqa: ARG001 - filesystem-stateful, no session state
    *,
    model_config: str,
    target_config: str,
    out_dir: str,
    timeout_sec: int = 600,
    force_subprocess: bool = False,
) -> dict[str, Any]:
    """Run the CompGen pipeline up to ``agent-decision-request`` and
    return the bounded view the agent should reason over.

    Args:
        model_config: path to a model YAML.
        target_config: path to a target YAML.
        out_dir: output run directory (will be replaced).
        timeout_sec: pipeline timeout in seconds.
        force_subprocess: if True, run via subprocess (slower; only use
            for isolation testing).

    Returns:
        ``{ok, in_process, run_dir, request_path, candidate_ids_allowed,
        visible_regions_summary, legal_set_tile_params, greedy_pick,
        stage_status}``. On failure: ``{ok: false, error, exception_type
        | stderr_tail}``.
    """
    repo = _resolve_repo_root()
    out_path = Path(out_dir).resolve()

    if force_subprocess:
        proc = _run_subprocess_with_tail(
            args=[
                "run",
                "--model", model_config,
                "--target", target_config,
                "--out", str(out_path),
                "--stop-after", "agent-decision-request",
                "--selection-mode", "greedy",
            ],
            cwd=repo, timeout_sec=timeout_sec,
        )
    else:
        proc = _run_in_process(
            model_config=model_config, target_config=target_config,
            out_dir=out_path, stop_after="agent-decision-request",
            selection_mode="greedy",
        )
    request_path = (
        out_path / "03_recipe_planning" / "agent_decision"
        / "agent_decision_request.json"
    )
    request = _read_json(request_path)

    # Special case: greedy's probe pick may itself fail (e.g.
    # tiny_mlp tile_16 → K_iters=4 → bit-equality fails). When that
    # happens, the pipeline raises but the bounded-view artifacts are
    # already on disk (they're emitted 's check). Surface
    # the greedy_pick_warning so the agent knows to pick something
    # other than what greedy chose. This is "soft failure" — the emit
    # itself is successful (the agent has everything to make a good
    # decision), but the typed retry hint tells the agent which
    # candidate is known-bad.
    greedy_pick_warning: dict[str, Any] | None = None
    if not proc["ok"] and request is not None:
        hint = _detect_failure_with_retry_hint(out_path)
        if hint is not None:
            greedy_pick_warning = hint
            # Treat the emit as a soft success — the agent can read the
            # bounded view AND knows greedy's pick is bad.
        else:
            return {
                "ok": False,
                "stage": "pipeline_emit_request",
                "error": "pipeline failed before agent-decision-request",
                **proc,
            }
    elif not proc["ok"]:
        return {
            "ok": False,
            "stage": "pipeline_emit_request",
            "error": "pipeline failed before agent-decision-request",
            **proc,
        }

    if request is None:
        return {
            "ok": False,
            "stage": "request_emit",
            "error": "agent_decision_request.json not produced",
            "run_dir": str(out_path),
            **proc,
        }

    cost_preview = _read_json(out_path / "02_graph_analysis" / "cost_preview_v2.json")
    cost_by_id: dict[str, dict[str, Any]] = {}
    if cost_preview is not None:
        for cp in cost_preview.get("cost_previews", []):
            cost_by_id[cp["candidate_id"]] = cp

    legal_set_tile = []
    for cp in cost_by_id.values():
        if (
            cp.get("candidate_kind") == "set_tile_params"
            and cp.get("legality_ok") is True
        ):
            legal_set_tile.append(
                {
                    "candidate_id": cp["candidate_id"],
                    "region_id": cp["region_id"],
                    "tile": cp.get("features", {}).get("tile"),
                    "relative_cost": cp["relative_cost"],
                    "confidence": cp["confidence"],
                    "real_transform_verified": cp["features"].get(
                        "real_transform_verified", False,
                    ),
                }
            )
    legal_set_tile.sort(
        key=lambda c: (c["relative_cost"], c["candidate_id"])
    )

    visible_summary = []
    for r in request.get("visible_regions", []):
        legal_count = len(r.get("legal_candidates") or [])
        visible_summary.append(
            {
                "region_id": r["region_id"],
                "kind": r.get("kind", ""),
                "site_id": r.get("site_id", ""),
                "priority": r.get("priority"),
                "legal_candidate_count": legal_count,
            }
        )

    greedy_pick = _read_json(
        out_path / "03_recipe_planning" / "candidate_selection.json"
    )

    result = {
        "ok": True,
        "in_process": proc.get("in_process", False),
        "run_dir": str(out_path),
        "request_path": str(request_path.relative_to(out_path)),
        "model_id": request.get("model_id", ""),
        "target_id": request.get("target_id", ""),
        "candidate_ids_allowed": request.get("candidate_ids_allowed", []),
        "visible_regions_summary": visible_summary,
        "legal_set_tile_params": legal_set_tile,
        "greedy_pick": (
            {
                "candidate_id": greedy_pick.get("selected_candidate_id"),
                "candidate_kind": greedy_pick.get("candidate_kind"),
                "region_id": greedy_pick.get("region_id"),
                "rationale_primary": (
                    greedy_pick.get("rationale", {}) or {}
                ).get("primary_reason", ""),
            }
            if greedy_pick is not None else None
        ),
        "stage_status": _per_stage_status(out_path),
    }
    if greedy_pick_warning is not None:
        # The greedy-probe pick will fail downstream — the agent should
        # NOT ratify it. Refresh candidate_ids_allowed to exclude the
        # failed candidate so the agent picks from the safe set.
        bad = greedy_pick_warning.get("failed_candidate_id", "")
        result["greedy_pick_warning"] = greedy_pick_warning
        if bad:
            result["candidate_ids_allowed"] = [
                c for c in result["candidate_ids_allowed"] if c != bad
            ]
    return result


# --------------------------------------------------------------------------- #
# Tool: commit agent-decision response
# --------------------------------------------------------------------------- #


def compgen_commit_agent_decision_response(
    sm: SessionManager,  # noqa: ARG001
    *,
    model_config: str,
    target_config: str,
    out_dir: str,
    response: dict[str, Any],
    stop_after: str = "cost-preview-v2",
    timeout_sec: int = 600,
    force_subprocess: bool = False,
) -> dict[str, Any]:
    """Re-run the pipeline with ``--selection-mode agent-file`` using
    the supplied response. The compiler's validator runs the
    11 typed checks before any recipe.mlir commit.

    On failure the response includes typed retry hints (``failed_stage``,
    ``failed_check``, ``failed_candidate_id``, ``retry_options``) so
    the agent can pick a different candidate without re-parsing files.
    """
    repo = _resolve_repo_root()
    out_path = Path(out_dir).resolve()
    # Persist the response OUTSIDE the run dir (which `run` wipes) so
    # the file survives the next invocation.
    tmp_dir = out_path.parent / f".{out_path.name}.agent_decision_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    response_path = tmp_dir / "agent_decision_response.json"
    response_path.write_text(
        json.dumps(response, indent=2, sort_keys=True), encoding="utf-8",
    )

    if force_subprocess:
        proc = _run_subprocess_with_tail(
            args=[
                "run",
                "--model", model_config,
                "--target", target_config,
                "--out", str(out_path),
                "--stop-after", stop_after,
                "--selection-mode", "agent-file",
                "--agent-decision-response", str(response_path),
            ],
            cwd=repo, timeout_sec=timeout_sec,
        )
    else:
        proc = _run_in_process(
            model_config=model_config, target_config=target_config,
            out_dir=out_path, stop_after=stop_after,
            selection_mode="agent-file",
            agent_decision_response_path=response_path,
        )

    out: dict[str, Any] = {
        "ok": proc["ok"],
        "in_process": proc.get("in_process", False),
        "run_dir": str(out_path),
        "stop_after": stop_after,
        "subprocess": proc,
    }

    validation = _read_json(
        out_path / "03_recipe_planning" / "agent_decision"
        / "agent_decision_validation.json"
    )
    out["validation_overall"] = (
        validation.get("overall") if validation is not None else None
    )
    out["validation_failed_checks"] = (
        [c for c in validation["checks"] if c["status"] != "pass"]
        if validation is not None else None
    )
    out["selected_candidate_id"] = (
        validation.get("selected_candidate_id") if validation else None
    )
    audit = _read_json(
        out_path / "03_recipe_planning" / "agent_decision"
        / "provider_redaction_audit.json"
    )
    out["redaction_audit_status"] = (
        audit.get("status") if audit is not None else None
    )

    recipe_path = out_path / "03_recipe_planning" / "recipe.mlir"
    if recipe_path.exists():
        recipe_text = recipe_path.read_text(encoding="utf-8")
        out["recipe_mlir_excerpt"] = "\n".join(
            line for line in recipe_text.splitlines()
            if "source_candidate" in line or "selection_mode" in line
            or "recipe.set_tile_params" in line
            or "recipe.fuse_producer_consumer" in line
            or "recipe.create_kernel_contract" in line
        )
    else:
        out["recipe_mlir_excerpt"] = None

    out["stage_status"] = _per_stage_status(out_path)

    pl_report = _read_json(
        out_path / "03_recipe_planning" / "post_lowering"
        / "post_lowering_verification_report.json"
    )
    diff_report = _read_json(
        out_path / "03_recipe_planning" / "differential_verification"
        / "differential_verification_report.json"
    )
    real_diff = _read_json(
        out_path / "03_recipe_planning" / "real_verification"
        / "real_differential_report.json"
    )
    out["downstream_reports"] = {
        "post_lowering_verification": (
            pl_report.get("status") if pl_report else None
        ),
        "differential_verification": (
            diff_report.get("status") if diff_report else None
        ),
        "real_transform_differential": (
            {
                "status": real_diff.get("status"),
                "mode": real_diff.get("mode"),
                "cases_passed": (real_diff.get("cases") or {}).get("passed"),
                "cases_total": (real_diff.get("cases") or {}).get("total"),
                "max_abs_error": (real_diff.get("error") or {}).get(
                    "max_abs_error"
                ),
                "refinement_status": (real_diff.get("error") or {}).get(
                    "refinement_status"
                ),
            } if real_diff else None
        ),
    }

    # On failure: attach typed retry hints.
    if not proc["ok"]:
        hint = _detect_failure_with_retry_hint(out_path)
        if hint is not None:
            out["retry_hint"] = hint

    return out


# --------------------------------------------------------------------------- #
# Tool: inspect a finished pipeline run
# --------------------------------------------------------------------------- #


def compgen_inspect_pipeline_run(
    sm: SessionManager,  # noqa: ARG001
    *,
    out_dir: str,
) -> dict[str, Any]:
    """Read-only health summary of an existing run dir."""
    out_path = Path(out_dir).resolve()
    if not out_path.is_dir():
        return {
            "ok": False, "run_dir": str(out_path), "exists": False,
            "error": "run dir does not exist",
        }

    validation = _read_json(
        out_path / "03_recipe_planning" / "agent_decision"
        / "agent_decision_validation.json"
    )
    audit = _read_json(
        out_path / "03_recipe_planning" / "agent_decision"
        / "provider_redaction_audit.json"
    )
    pl_report = _read_json(
        out_path / "03_recipe_planning" / "post_lowering"
        / "post_lowering_verification_report.json"
    )
    diff_report = _read_json(
        out_path / "03_recipe_planning" / "differential_verification"
        / "differential_verification_report.json"
    )
    real_diff = _read_json(
        out_path / "03_recipe_planning" / "real_verification"
        / "real_differential_report.json"
    )

    validate_overall: str | None = None
    validate_failures: list[str] = []
    try:
        from compgen.graph_compilation.validate import validate_run

        rep = validate_run(out_path)
        validate_overall = rep.overall
        validate_failures = [
            f"{r.rule_id}: {r.detail[:120]}"
            for r in rep.rules if r.status != "pass"
        ]
    except Exception as exc:  # noqa: BLE001
        validate_overall = "error"
        validate_failures = [f"{type(exc).__name__}: {exc}"]

    result = {
        "ok": True,
        "run_dir": str(out_path),
        "exists": True,
        "stage_status": _per_stage_status(out_path),
        "validation_overall": (
            validation.get("overall") if validation else None
        ),
        "validation_failed_checks": (
            [c for c in validation["checks"] if c["status"] != "pass"]
            if validation is not None else []
        ),
        "redaction_audit_status": (
            audit.get("status") if audit else None
        ),
        "downstream_reports": {
            "post_lowering_verification": (
                pl_report.get("status") if pl_report else None
            ),
            "differential_verification": (
                diff_report.get("status") if diff_report else None
            ),
            "real_transform_differential": (
                {
                    "status": real_diff.get("status"),
                    "mode": real_diff.get("mode"),
                    "cases_passed": (real_diff.get("cases") or {}).get("passed"),
                    "cases_total": (real_diff.get("cases") or {}).get("total"),
                } if real_diff else None
            ),
        },
        "validate_run_overall": validate_overall,
        "validate_run_failures": validate_failures,
    }

    # Attach retry hint if the run failed.
    hint = _detect_failure_with_retry_hint(out_path)
    if hint is not None:
        result["retry_hint"] = hint

    return result


# --------------------------------------------------------------------------- #
# Tool: pipeline status (mid-run progress)
# --------------------------------------------------------------------------- #


def compgen_pipeline_status(
    sm: SessionManager,  # noqa: ARG001
    *,
    out_dir: str,
    tail_lines: int = 20,
) -> dict[str, Any]:
    """Read mid-run progress from a CompGen run directory.

    Returns the latest event per stage from ``stage_ledger.jsonl``,
    plus tail lines of the most recent CLI subprocess (if any). Use
    this to give the user updates while a long compile is running:
    *"capture done… payload-lowering done… running (12/16
    cases)…"*. Does NOT block; reads whatever's on disk now.
    """
    out_path = Path(out_dir).resolve()
    if not out_path.is_dir():
        return {
            "ok": False, "run_dir": str(out_path), "exists": False,
            "error": "run dir does not exist (compile may not have started)",
        }
    ledger = _read_jsonl(out_path / "stage_ledger.jsonl")

    # Per-stage latest event (start | artifact_written | finish).
    latest_per_stage: dict[str, dict[str, Any]] = {}
    for ev in ledger:
        sid = ev.get("stage_id", "")
        if sid:
            latest_per_stage[sid] = ev

    # Stage status from manifest if it's been written (post-completion).
    stage_status = _per_stage_status(out_path)

    # Tail of the last subprocess invocation, if any.
    repo = _resolve_repo_root()
    stderr_tail_path = repo / _STDERR_TAIL_REL
    stderr_lines: list[str] = []
    if stderr_tail_path.exists():
        try:
            stderr_lines = stderr_tail_path.read_text(
                encoding="utf-8", errors="replace",
            ).splitlines()[-tail_lines:]
        except OSError:
            stderr_lines = []

    return {
        "ok": True,
        "run_dir": str(out_path),
        "exists": True,
        "ledger_event_count": len(ledger),
        "latest_event_per_stage": latest_per_stage,
        "stage_status": stage_status,
        "manifest_written": (out_path / "run_manifest.json").exists(),
        "recipe_mlir_written": (
            out_path / "03_recipe_planning" / "recipe.mlir"
        ).exists(),
        "stderr_tail": stderr_lines,
    }


# --------------------------------------------------------------------------- #
# MCP tool registration
# --------------------------------------------------------------------------- #


_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "schema_version": {"type": "string"},
        "selected_candidate_id": {"type": "string"},
        "rationale": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "evidence": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "field": {"type": "string"},
                            "value": {},
                            "reason": {"type": "string"},
                        },
                        "required": ["field"],
                    },
                },
                "rejected_alternatives": {
                    "type": "array",
                    "items": {"type": "object"},
                },
            },
            "required": ["summary", "evidence"],
        },
    },
    "required": ["selected_candidate_id", "rationale"],
}


AGENT_DECISION_TOOLS: list[dict[str, Any]] = [
    {
        "name": "compgen_emit_agent_decision_request",
        "description": (
            "Run the CompGen pipeline up to --stop-after "
            "agent-decision-request and return the bounded view: "
            "candidate_ids_allowed, per-region brief, cost previews "
            "for legal SetTileParams candidates, and greedy's pick. "
            "Step 1 of a Claude-Code-driven (or Codex-driven) "
            "compilation: read the bounded view, reason about "
            "candidates, then call compgen_commit_agent_decision_response. "
            "Runs in-process by default (saves Python startup); pass "
            "force_subprocess=true for isolation."
        ),
        "phase": "lifecycle",
        "handler": compgen_emit_agent_decision_request,
        "input_schema": {
            "type": "object",
            "properties": {
                "model_config": {
                    "type": "string",
                    "description": "Path to a model YAML.",
                },
                "target_config": {
                    "type": "string",
                    "description": "Path to a target YAML.",
                },
                "out_dir": {
                    "type": "string",
                    "description": "Output run directory (will be replaced).",
                },
                "timeout_sec": {
                    "type": "integer",
                    "default": 600,
                },
                "force_subprocess": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "Run via subprocess instead of in-process. "
                        "Slower (pays Python startup) but isolates "
                        "imports. Default: False."
                    ),
                },
            },
            "required": ["model_config", "target_config", "out_dir"],
        },
    },
    {
        "name": "compgen_commit_agent_decision_response",
        "description": (
            "Commit the agent's pick. Re-runs the pipeline with "
            "--selection-mode agent-file using the supplied response. "
            "The 11-check M-14A validator runs before recipe.mlir "
            "is written; if any check fails the run aborts before "
            "commit. On failure the response includes a typed "
            "retry_hint {failed_stage, failed_check, "
            "failed_candidate_id, retry_options[]} so the agent can "
            "pick a different candidate without re-reading files."
        ),
        "phase": "transform",
        "handler": compgen_commit_agent_decision_response,
        "input_schema": {
            "type": "object",
            "properties": {
                "model_config": {"type": "string"},
                "target_config": {"type": "string"},
                "out_dir": {"type": "string"},
                "response": _RESPONSE_SCHEMA,
                "stop_after": {
                    "type": "string",
                    "default": "cost-preview-v2",
                    "description": (
                        "How far to run after the agent's commit. "
                        "Common: cost-preview-v2, "
                        "real-transform-differential, gap-discovery."
                    ),
                },
                "timeout_sec": {"type": "integer", "default": 600},
                "force_subprocess": {"type": "boolean", "default": False},
            },
            "required": ["model_config", "target_config", "out_dir", "response"],
        },
    },
    {
        "name": "compgen_inspect_pipeline_run",
        "description": (
            "Read-only health summary of a CompGen run directory. "
            "Returns per-stage status, agent-decision validation "
            "verdict, redaction audit status, downstream reports, "
            "validate_run's R001-R012 manifest hash-chain result, "
            "and (on failure) a typed retry_hint."
        ),
        "phase": "inspect",
        "handler": compgen_inspect_pipeline_run,
        "input_schema": {
            "type": "object",
            "properties": {
                "out_dir": {"type": "string"},
            },
            "required": ["out_dir"],
        },
    },
    {
        "name": "compgen_pipeline_status",
        "description": (
            "Mid-run progress reader. Reads stage_ledger.jsonl + "
            "run_manifest.json (if written) and the last subprocess's "
            "stderr tail file. Use during a long compile to give "
            "the user updates ('capture done, lowering done, running "
            "M-12...'). Non-blocking; returns whatever's on disk now."
        ),
        "phase": "inspect",
        "handler": compgen_pipeline_status,
        "input_schema": {
            "type": "object",
            "properties": {
                "out_dir": {"type": "string"},
                "tail_lines": {"type": "integer", "default": 20},
            },
            "required": ["out_dir"],
        },
    },
]
