"""MCP tools for the kernel-codegen provider protocol.

Four tools, parallel to 's agent-decision pair:

- ``compgen_emit_kernel_codegen_request`` — runs the pipeline to
  ``--stop-after kernel-codegen-request`` and returns the task surface
  Claude Code or another provider needs to fulfil.
- ``compgen_run_kernel_codegen_task`` — convenience helper. Spawns a
  Claude Code subagent (or any callable provider) on the task and
  collects its response. Optional; the operator-driven flow (write
  the response file by hand) still works without it.
- ``compgen_commit_kernel_codegen_response`` — validates the provider
  response against the task contract, writes the attempt trail,
  and returns the typed next_action. On accept, routes to
  (verifier — pending until lands). On recoverable fail, emits a
  retry_request. On fatal / exhausted, emits a downstream_retry_request.
- ``compgen_inspect_kernel_codegen_task`` — read-only view of the
  task surface (request, attempts, validation reports, certificates,
  failure reports).

The compiler trusts artifacts and certificates only. The MCP tools
are convenience wrappers over the file-based protocol implemented in
``compgen.graph_compilation.kernel_codegen`` and
``compgen.graph_compilation.kernel_codegen_response``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from compgen.graph_compilation.kernel_codegen_response import (
    DEFAULT_MAX_ATTEMPTS,
    commit_response,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_or_none(path: Path) -> dict[str, Any] | None:
    """Best-effort JSON read. Returns None if the file is missing OR
    if the contents do not parse as JSON (a malformed response.json
    written by a rejected attempt is honest evidence — surface it as
    None rather than crashing the inspect tool)."""
    if not path.exists():
        return None
    try:
        return _read_json(path)
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------------- #
# 1. emit_kernel_codegen_request
# --------------------------------------------------------------------------- #


def compgen_emit_kernel_codegen_request(
    *,
    model_config: str,
    target_config: str,
    out_dir: str,
    timeout_sec: int = 600,
    force_subprocess: bool = False,
) -> dict[str, Any]:
    """Run the pipeline through the kernel-codegen-request boundary
    and return the task surface a provider needs.

    Returns ``{ok, run_dir, request_path, task_id, contract_hash,
    contract_paths, allowed_backends, required_outputs, forbidden,
    artifact_dir, request_kind, not_applicable_reason}``.

    Convenience layer over driving the CLI; the result also lands on
    disk under ``04_kernel_codegen/requests/<task_id>.request.json``.
    """
    # Reuse the agent_decision helpers — same pattern.
    from compgen.mcp.tools.agent_decision import (
        _resolve_repo_root,
        _run_in_process,
        _run_subprocess_with_tail,
    )

    repo_root = _resolve_repo_root()
    out = Path(out_dir).resolve()
    if force_subprocess:
        proc_result = _run_subprocess_with_tail(
            repo_root=repo_root,
            args=[
                "run",
                "--model", str(model_config),
                "--target", str(target_config),
                "--out", str(out),
                "--stop-after", "kernel-codegen-request",
                "--selection-mode", "greedy",
            ],
            timeout_sec=timeout_sec,
        )
        if proc_result.get("returncode", 1) != 0:
            return {
                "ok": False,
                "error": proc_result.get("stderr_tail", ""),
                "stdout_tail": proc_result.get("stdout_tail", ""),
            }
    else:
        result = _run_in_process(
            model=Path(model_config), target=Path(target_config),
            out_dir=out, stop_after="kernel-codegen-request",
            selection_mode="greedy",
        )
        if not result.get("ok"):
            return {"ok": False, "error": result.get("error", "")}

    # Read the emitted request.
    requests_dir = out / "04_kernel_codegen" / "requests"
    request_files = sorted(requests_dir.glob("*.request.json")) if requests_dir.is_dir() else []
    if not request_files:
        return {
            "ok": False,
            "error": (
                f"no kernel-codegen request was emitted at "
                f"{requests_dir.relative_to(out)}; check the pipeline "
                f"output for typed-blocked reasons"
            ),
        }
    request_path = request_files[0]
    request = _read_json(request_path)
    return {
        "ok": True,
        "run_dir": str(out),
        "request_path": str(request_path.relative_to(out)),
        "task_id": request["task_id"],
        "contract_hash": request["contract_hash"],
        "contract_paths": request["contract_paths"],
        "allowed_backends": request["allowed_backends"],
        "required_outputs": request["required_outputs"],
        "forbidden": request["forbidden"],
        "artifact_dir": request["artifact_dir"],
        "request_kind": request["request_kind"],
        "not_applicable_reason": request.get("not_applicable_reason", ""),
    }


# --------------------------------------------------------------------------- #
# 2. run_kernel_codegen_task — spawn a Claude Code subagent
# --------------------------------------------------------------------------- #


def compgen_run_kernel_codegen_task(
    *,
    run_dir: str,
    task_id: str,
    provider: str = "claude_code",
    timeout_sec: int = 900,
) -> dict[str, Any]:
    """Convenience: spawn a Claude Code subagent (or other provider)
    on a task and collect its response.

    ships this as a stub that returns a placeholder pointing at
    the operator-driven flow. The actual subagent invocation lands
    when this is wired to the parent MCP harness's Agent tool. Until
    then, the operator writes the response JSON by hand.

    Returns ``{ok, run_dir, task_id, provider, response_path,
    operator_action_required}``.
    """
    run_dir_path = Path(run_dir).resolve()
    request_path = (
        run_dir_path / "04_kernel_codegen" / "requests" / f"{task_id}.request.json"
    )
    if not request_path.exists():
        return {
            "ok": False,
            "error": (
                f"task not found at {request_path.relative_to(run_dir_path)}; "
                f"call compgen_emit_kernel_codegen_request first"
            ),
        }
    response_path = (
        run_dir_path / "04_kernel_codegen" / "responses" / f"{task_id}.response.json"
    )
    if response_path.exists():
        return {
            "ok": True,
            "run_dir": str(run_dir_path),
            "task_id": task_id,
            "provider": provider,
            "response_path": str(response_path.relative_to(run_dir_path)),
            "operator_action_required": False,
            "note": "response already exists; pass to compgen_commit_kernel_codegen_response",
        }
    # ships the operator-driven flow only. The harness-driven
    # subagent spawn lands when the parent MCP server wires this to
    # the Agent tool — flag the outstanding work.
    return {
        "ok": False,
        "operator_action_required": True,
        "task_id": task_id,
        "provider": provider,
        "request_path": str(request_path.relative_to(run_dir_path)),
        "expected_response_path": str(response_path.relative_to(run_dir_path)),
        "note": (
            "M-43 ships the file-based protocol; the harness-driven "
            "subagent spawn lands in M-43.1. Operator: read the request, "
            "write the response per kernel_codegen_response_v1, then call "
            "compgen_commit_kernel_codegen_response."
        ),
    }


# --------------------------------------------------------------------------- #
# 3. commit_kernel_codegen_response
# --------------------------------------------------------------------------- #


def compgen_commit_kernel_codegen_response(
    *,
    run_dir: str,
    task_id: str,
    response: dict[str, Any] | str,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> dict[str, Any]:
    """Validate + commit a provider response. Returns the typed
    CommitResult dict (see kernel_codegen_response.CommitResult)."""
    run_dir_path = Path(run_dir).resolve()
    result = commit_response(
        run_dir=run_dir_path,
        task_id=task_id,
        response=response,
        max_attempts=max_attempts,
    )
    return {"ok": True, **result.to_dict()}


# --------------------------------------------------------------------------- #
# 4. inspect_kernel_codegen_task — read-only view
# --------------------------------------------------------------------------- #


def compgen_inspect_kernel_codegen_task(
    *,
    run_dir: str,
    task_id: str,
) -> dict[str, Any]:
    """Read-only view of one task's full state — the request, every
    attempt + its validation report, the attempts log, any certificate,
    and any retry/failure reports."""
    run_dir_path = Path(run_dir).resolve()
    out_dir = run_dir_path / "04_kernel_codegen"
    request_path = out_dir / "requests" / f"{task_id}.request.json"
    request = _read_json_or_none(request_path)
    if request is None:
        return {"ok": False, "error": f"task {task_id!r} not found"}

    attempts_root = out_dir / "attempts" / task_id
    attempts: list[dict[str, Any]] = []
    if attempts_root.is_dir():
        for attempt_dir in sorted(attempts_root.iterdir()):
            if not attempt_dir.is_dir():
                continue
            response_path = attempt_dir / "response.json"
            response = _read_json_or_none(response_path)
            if response is None and response_path.exists():
                # Malformed response — capture the raw bytes preview so
                # the audit can see what the provider wrote.
                response = {
                    "_malformed": True,
                    "preview": response_path.read_text(encoding="utf-8", errors="replace")[:512],
                }
            attempts.append({
                "attempt_dir": str(attempt_dir.relative_to(run_dir_path)),
                "response": response,
                "validation_report": _read_json_or_none(
                    attempt_dir / "validation_report.json"
                ),
            })

    return {
        "ok": True,
        "run_dir": str(run_dir_path),
        "task_id": task_id,
        "request": request,
        "attempts": attempts,
        "attempts_log": _read_json_or_none(out_dir / "kernel_codegen_attempts.json"),
        "retry_request": _read_json_or_none(
            out_dir / "kernel_codegen_retry_request.json"
        ),
        "failure_report": _read_json_or_none(
            out_dir / "kernel_codegen_failure_report.json"
        ),
        # kernel certificate, indexed by contract_hash.
        "certificate": _resolve_certificate(run_dir_path, request),
    }


# --------------------------------------------------------------------------- #
# compgen_compare_kernel_bids — read-only auction summary
# --------------------------------------------------------------------------- #


def compgen_compare_kernel_bids(
    *,
    run_dir: str,
    task_id: str,
) -> dict[str, Any]:
    """Read-only ranked summary of an auction.

    Returns one row per provider that bid on the task, with:

    * ``provider_name``
    * ``rank`` (1 = best by score)
    * ``score`` (lower is better; perf / confidence)
    * ``confidence``, ``perf_estimate_us``, ``time_to_generate_s_estimate``,
      ``cache_hit``, ``rationale``
    * ``fulfilled``, ``verifier_status``, ``paper_claimable``
    * ``certificate_path`` (when verified)

    The agent uses this to make pruning / re-ranking decisions without
    reading every full ProviderResult or BidPreview JSON. When no
    auction has run for the task, returns ``{"ok": False, ...}``.
    """
    run_dir_path = Path(run_dir).resolve()
    auction_root = run_dir_path / "04_kernel_codegen" / "auction" / task_id
    report_path = auction_root / "auction_report.json"
    if not report_path.exists():
        return {
            "ok": False,
            "task_id": task_id,
            "error": "no_auction_report",
        }

    report = _read_json(report_path)
    bids = report.get("bids", []) or []
    fulfilled = {f["provider_name"]: f for f in (report.get("fulfilled") or [])}
    verified = {v["provider_name"]: v for v in (report.get("verified") or [])}

    rows: list[dict[str, Any]] = []
    for record in bids:
        bid = record.get("bid", {}) or {}
        name = record.get("provider_name", "")
        ful = fulfilled.get(name, {})
        ver = verified.get(name, {})
        rows.append(
            {
                "provider_name": name,
                "rank": record.get("rank"),
                "score": record.get("score"),
                "confidence": bid.get("confidence"),
                "perf_estimate_us": bid.get("perf_estimate_us"),
                "time_to_generate_s_estimate": bid.get("time_to_generate_s_estimate"),
                "cache_hit": bid.get("cache_hit", False),
                "rationale": bid.get("rationale", ""),
                "fulfilled": bool(ful.get("found", False)),
                "fulfill_error": ful.get("error", ""),
                "verifier_status": ver.get("overall", "skipped"),
                "verifier_failure_kind": ver.get("failure_kind", ""),
                "certificate_path": ver.get("certificate_path", ""),
                "paper_claimable": (
                    ver.get("overall") == "pass"
                    and not bid.get("cache_hit", False)
                ),
            }
        )

    return {
        "ok": True,
        "task_id": task_id,
        "contract_hash": report.get("contract_hash", ""),
        "mode": report.get("mode", ""),
        "overall": report.get("overall", ""),
        "winner_provider": report.get("winner_provider", ""),
        "rows": rows,
    }


def _resolve_certificate(run_dir: Path, request: dict[str, Any]) -> dict[str, Any] | None:
    """Best-effort: load the kernel certificate keyed by the request's
    contract_hash. Returns None when no cert exists yet."""
    contract_hash = request.get("contract_hash") or ""
    if not contract_hash:
        return None
    cert_path = (
        run_dir / "04_kernel_codegen" / "certificates" / f"{contract_hash}.json"
    )
    body = _read_json_or_none(cert_path)
    if body is None:
        return None
    # Validate the certificate is still consistent with its artifacts
    # (catches post-cert mutation per the negative-control
    # pattern).
    try:
        from compgen.kernels.kernel_certificate import (
            KernelCertificate,
            validate_certificate,
        )
        cert = KernelCertificate.from_dict(body)
        v = validate_certificate(run_dir=run_dir, cert=cert)
        body = dict(body)
        body["__validation"] = {
            "valid": v.valid,
            "failure_kind": v.failure_kind,
            "failure_summary": v.failure_summary,
            "drifted": dict(v.drifted),
        }
    except Exception:  # noqa: BLE001
        body = dict(body)
        body["__validation"] = {"valid": False, "failure_kind": "load_error"}
    return body
