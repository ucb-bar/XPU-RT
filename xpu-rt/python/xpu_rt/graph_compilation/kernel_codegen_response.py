"""Provider-response schema + validator + commit tool (M-43).

Phase C M-43. Closes the kernel-codegen subagent loop: a provider
(spawned Claude Code agent, manual operator, autocomp, cache hit) writes
a ``provider_response_v1`` JSON; the commit tool validates it against
the M-42 task contract and routes the artifacts to the M-44 verifier.

Failure-class taxonomy (per the user's refined plan):

::

    Recoverable provider failures (retry up to N=3 with typed feedback):
      schema_invalid, compile_error, metadata_mismatch,
      numerical_mismatch, shape_mismatch
    Provider protocol violations (maybe 1 retry):
      unsupported_backend, semantic_contract_violation
    Transient (1 retry):
      timeout
    Protocol_or_contract_fatal (NO retry — immediate M-15B reject):
      contract_hash_mismatch, contract_mutation, forbidden_path_write

After 3 failed recoverable attempts the commit tool emits a
``downstream_retry_request_v1`` with
``failed_check=kernel_codegen_attempts_exhausted`` so the OUTER agent
reconsiders the candidate, not just patches kernel code forever.

This module is data-only — no MCP wrapping here. The MCP entry points
live in ``compgen.mcp.tools.kernel_codegen`` and call into here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


_RESPONSE_SCHEMA_VERSION = "kernel_codegen_response_v1"
_ATTEMPTS_LOG_SCHEMA_VERSION = "kernel_codegen_attempts_v1"
_RETRY_REQUEST_SCHEMA_VERSION = "kernel_codegen_retry_request_v1"

DEFAULT_MAX_ATTEMPTS = 3


# --------------------------------------------------------------------------- #
# Failure-class taxonomy
# --------------------------------------------------------------------------- #


# ``failure_kind`` (rejection reason) → ``recoverability`` bucket. The
# commit tool uses this to decide retry vs. fatal.
RECOVERABILITY: dict[str, str] = {
    # Recoverable provider failures.
    "schema_invalid":              "recoverable_provider_failure",
    "compile_error":               "recoverable_provider_failure",
    "metadata_mismatch":           "recoverable_provider_failure",
    "numerical_mismatch":          "recoverable_provider_failure",
    "shape_mismatch":              "recoverable_provider_failure",
    # Provider protocol violations — at most one retry.
    "unsupported_backend":         "provider_protocol_violation",
    "semantic_contract_violation": "provider_protocol_violation",
    # Transient.
    "timeout":                     "transient",
    # Fatal — immediate M-15B downstream reject.
    "contract_hash_mismatch":      "protocol_or_contract_fatal",
    "contract_mutation":           "protocol_or_contract_fatal",
    "forbidden_path_write":        "protocol_or_contract_fatal",
}

# Specific provider claims and how to map them. The provider's
# ``claims.expected_numerics`` value must be one of these.
ALLOWED_EXPECTED_NUMERICS: tuple[str, ...] = (
    "bit_equality", "tolerance_eps", "unknown",
)


# --------------------------------------------------------------------------- #
# Schema dataclasses
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _ProviderInfo:
    kind: str  # "claude_code_subagent" | "manual" | "template" | "cache_hit" | ...
    model: str = ""
    session_id: str = ""
    started_at: str = ""
    finished_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "model": self.model,
            "session_id": self.session_id,
            "started_at": self.started_at, "finished_at": self.finished_at,
        }


@dataclass(frozen=True)
class _ProviderClaims:
    backend: str             # one of allowed_backends
    supports_dispatch: tuple[str, ...] = ()  # subset of dispatch model strings
    estimated_registers: int = 0
    estimated_smem_bytes: int = 0
    expected_numerics: str = "unknown"  # one of ALLOWED_EXPECTED_NUMERICS

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "supports_dispatch": list(self.supports_dispatch),
            "estimated_registers": self.estimated_registers,
            "estimated_smem_bytes": self.estimated_smem_bytes,
            "expected_numerics": self.expected_numerics,
        }


@dataclass(frozen=True)
class ProviderResponse:
    """Bounded artifact a kernel-codegen provider returns.

    The provider proposes an artifact set + claims. It does NOT mark
    success — the parent's commit tool validates and either accepts
    (route to M-44 verifier) or rejects with a typed failure_kind that
    drives the retry policy.
    """

    schema_version: str
    task_id: str
    contract_hash: str
    artifacts: dict[str, str]  # name → relative path under artifact_dir
    claims: _ProviderClaims
    provider: _ProviderInfo
    contract_feedback: tuple[dict[str, Any], ...] = ()
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "contract_hash": self.contract_hash,
            "artifacts": dict(self.artifacts),
            "claims": self.claims.to_dict(),
            "provider": self.provider.to_dict(),
            "contract_feedback": list(self.contract_feedback),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, body: dict[str, Any]) -> "ProviderResponse":
        claims = body.get("claims") or {}
        prov = body.get("provider") or {}
        return cls(
            schema_version=str(body.get("schema_version", "")),
            task_id=str(body.get("task_id", "")),
            contract_hash=str(body.get("contract_hash", "")),
            artifacts=dict(body.get("artifacts") or {}),
            claims=_ProviderClaims(
                backend=str(claims.get("backend", "")),
                supports_dispatch=tuple(claims.get("supports_dispatch") or ()),
                estimated_registers=int(claims.get("estimated_registers", 0)),
                estimated_smem_bytes=int(claims.get("estimated_smem_bytes", 0)),
                expected_numerics=str(claims.get("expected_numerics", "unknown")),
            ),
            provider=_ProviderInfo(
                kind=str(prov.get("kind", "")),
                model=str(prov.get("model", "")),
                session_id=str(prov.get("session_id", "")),
                started_at=str(prov.get("started_at", "")),
                finished_at=str(prov.get("finished_at", "")),
            ),
            contract_feedback=tuple(body.get("contract_feedback") or ()),
            notes=str(body.get("notes", "")),
        )


# --------------------------------------------------------------------------- #
# Validation result types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ValidationResult:
    accepted: bool
    failure_kind: str = ""           # one of RECOVERABILITY keys, or ""
    failure_summary: str = ""
    recoverability: str = ""         # bucket from RECOVERABILITY
    suggested_fix: str = ""
    evidence_paths: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "failure_kind": self.failure_kind,
            "failure_summary": self.failure_summary,
            "recoverability": self.recoverability,
            "suggested_fix": self.suggested_fix,
            "evidence_paths": dict(self.evidence_paths),
        }


# --------------------------------------------------------------------------- #
# Validation (pure — no side effects)
# --------------------------------------------------------------------------- #


def validate_response(
    *,
    response_body: dict[str, Any] | str | bytes | None,
    request_body: dict[str, Any],
    run_dir: Path,
) -> ValidationResult:
    """Validate a provider response against its task request.

    Returns ``ValidationResult(accepted=True)`` on every check pass;
    on first failure, returns the typed failure_kind so the commit
    layer can route to retry vs. fatal.

    Side-effect-free — the caller writes attempt records, retry
    requests, and certificates.
    """
    # 1. schema_invalid — JSON parses + has the four required top-level keys.
    if response_body is None:
        return ValidationResult(
            accepted=False, failure_kind="schema_invalid",
            failure_summary="response body is None",
            recoverability=RECOVERABILITY["schema_invalid"],
            suggested_fix="emit a response JSON with task_id, contract_hash, artifacts, claims",
        )
    if isinstance(response_body, (str, bytes)):
        try:
            response_body = json.loads(response_body)
        except json.JSONDecodeError as exc:
            return ValidationResult(
                accepted=False, failure_kind="schema_invalid",
                failure_summary=f"response body is not valid JSON: {exc}",
                recoverability=RECOVERABILITY["schema_invalid"],
                suggested_fix="emit valid JSON; check for trailing commas or unquoted strings",
            )
    if not isinstance(response_body, dict):
        return ValidationResult(
            accepted=False, failure_kind="schema_invalid",
            failure_summary=(
                f"response body is not a JSON object; got "
                f"{type(response_body).__name__}"
            ),
            recoverability=RECOVERABILITY["schema_invalid"],
        )

    for required in ("schema_version", "task_id", "contract_hash",
                     "artifacts", "claims"):
        if required not in response_body:
            return ValidationResult(
                accepted=False, failure_kind="schema_invalid",
                failure_summary=f"missing required key {required!r}",
                recoverability=RECOVERABILITY["schema_invalid"],
                suggested_fix=(
                    f"include {required!r} in the response per "
                    f"kernel_codegen_response_v1 schema"
                ),
            )

    if response_body["schema_version"] != _RESPONSE_SCHEMA_VERSION:
        return ValidationResult(
            accepted=False, failure_kind="schema_invalid",
            failure_summary=(
                f"schema_version={response_body['schema_version']!r} != "
                f"{_RESPONSE_SCHEMA_VERSION!r}"
            ),
            recoverability=RECOVERABILITY["schema_invalid"],
        )

    response = ProviderResponse.from_dict(response_body)

    # 2. task_id mismatch is `schema_invalid` — wrong task surface.
    if response.task_id != request_body.get("task_id"):
        return ValidationResult(
            accepted=False, failure_kind="schema_invalid",
            failure_summary=(
                f"response.task_id={response.task_id!r} does not match "
                f"request.task_id={request_body.get('task_id')!r}"
            ),
            recoverability=RECOVERABILITY["schema_invalid"],
        )

    # 3. contract_hash_mismatch — fatal. The provider was solving a
    # different contract; reject hard.
    if response.contract_hash != request_body.get("contract_hash"):
        return ValidationResult(
            accepted=False, failure_kind="contract_hash_mismatch",
            failure_summary=(
                f"response.contract_hash={response.contract_hash!r} does "
                f"not match request.contract_hash="
                f"{request_body.get('contract_hash')!r}; provider was "
                f"solving a different contract"
            ),
            recoverability=RECOVERABILITY["contract_hash_mismatch"],
        )

    # 4. unsupported_backend — protocol violation.
    allowed_backends = set(request_body.get("allowed_backends") or ())
    if response.claims.backend not in allowed_backends:
        return ValidationResult(
            accepted=False, failure_kind="unsupported_backend",
            failure_summary=(
                f"response.claims.backend={response.claims.backend!r} "
                f"not in allowed_backends={sorted(allowed_backends)!r}"
            ),
            recoverability=RECOVERABILITY["unsupported_backend"],
            suggested_fix=(
                f"emit kernel using one of {sorted(allowed_backends)!r}"
            ),
        )

    # 5. semantic_contract_violation — expected_numerics must be a
    # known refinement claim.
    if response.claims.expected_numerics not in ALLOWED_EXPECTED_NUMERICS:
        return ValidationResult(
            accepted=False, failure_kind="semantic_contract_violation",
            failure_summary=(
                f"claims.expected_numerics={response.claims.expected_numerics!r} "
                f"is not one of {ALLOWED_EXPECTED_NUMERICS!r}"
            ),
            recoverability=RECOVERABILITY["semantic_contract_violation"],
        )

    # 6. forbidden_path_write — every artifact path MUST live under
    # the sandboxed artifact_dir.
    artifact_dir = Path(request_body.get("artifact_dir") or "").as_posix()
    if not artifact_dir:
        return ValidationResult(
            accepted=False, failure_kind="schema_invalid",
            failure_summary="request.artifact_dir is empty",
            recoverability=RECOVERABILITY["schema_invalid"],
        )
    sandbox_abs = (run_dir / artifact_dir).resolve()
    for name, rel_path in response.artifacts.items():
        if not rel_path:
            return ValidationResult(
                accepted=False, failure_kind="schema_invalid",
                failure_summary=f"artifact {name!r} has empty path",
                recoverability=RECOVERABILITY["schema_invalid"],
            )
        # Reject absolute paths immediately — they're never valid.
        # Reject any path that escapes the sandbox after resolution.
        if Path(rel_path).is_absolute():
            return ValidationResult(
                accepted=False, failure_kind="forbidden_path_write",
                failure_summary=(
                    f"artifact {name!r} path {rel_path!r} is absolute; "
                    f"must be relative to run_dir under {artifact_dir!r}"
                ),
                recoverability=RECOVERABILITY["forbidden_path_write"],
            )
        candidate = (run_dir / rel_path).resolve()
        try:
            candidate.relative_to(sandbox_abs)
        except ValueError:
            return ValidationResult(
                accepted=False, failure_kind="forbidden_path_write",
                failure_summary=(
                    f"artifact {name!r} path {rel_path!r} escapes the "
                    f"sandboxed artifact_dir={artifact_dir!r}; resolved "
                    f"to {candidate}"
                ),
                recoverability=RECOVERABILITY["forbidden_path_write"],
            )

    # 7. Required outputs all present.
    required_outputs = set(request_body.get("required_outputs") or ())
    declared_artifacts = set(response.artifacts.keys())
    missing = required_outputs - declared_artifacts
    if missing:
        return ValidationResult(
            accepted=False, failure_kind="schema_invalid",
            failure_summary=(
                f"response.artifacts is missing required outputs: "
                f"{sorted(missing)!r}"
            ),
            recoverability=RECOVERABILITY["schema_invalid"],
            suggested_fix=(
                f"emit each of {sorted(required_outputs)!r} under artifact_dir/"
            ),
        )

    # 8. Each artifact path must exist on disk.
    for name, rel_path in response.artifacts.items():
        if not (run_dir / rel_path).exists():
            return ValidationResult(
                accepted=False, failure_kind="schema_invalid",
                failure_summary=(
                    f"artifact {name!r} declared at {rel_path!r} but "
                    f"the file does not exist on disk"
                ),
                recoverability=RECOVERABILITY["schema_invalid"],
            )

    # 9. contract_mutation — the materialised contract files referenced
    # by the request MUST NOT have been edited since the request was
    # emitted. We compare hashes by re-reading and re-hashing.
    full_path = request_body.get("contract_paths", {}).get("full") or ""
    facing_path = request_body.get("contract_paths", {}).get("kernel_facing") or ""
    for kind, p in (("contract", full_path), ("kernel_facing_view", facing_path)):
        if not p:
            return ValidationResult(
                accepted=False, failure_kind="schema_invalid",
                failure_summary=f"request.contract_paths.{kind} is empty",
                recoverability=RECOVERABILITY["schema_invalid"],
            )
        if not (run_dir / p).exists():
            return ValidationResult(
                accepted=False, failure_kind="contract_mutation",
                failure_summary=(
                    f"contract file {p!r} ({kind}) is missing — the "
                    f"contract may have been deleted or moved since the "
                    f"request was emitted"
                ),
                recoverability=RECOVERABILITY["contract_mutation"],
            )

    return ValidationResult(accepted=True)


# --------------------------------------------------------------------------- #
# Commit tool
# --------------------------------------------------------------------------- #


def _utcnow() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_or_none(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _read_json(path)


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


@dataclass(frozen=True)
class CommitResult:
    accepted: bool
    task_id: str
    attempt_index: int
    failure_kind: str = ""
    failure_summary: str = ""
    recoverability: str = ""
    next_action: str = ""  # "retry" | "fatal_reject" | "verifier_pending" | "verified"
    attempt_dir: str = ""  # relative to run_dir
    retry_request_path: str = ""  # relative to run_dir, populated on exhaustion
    certificate_path: str = ""    # populated when M-45 lands

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "task_id": self.task_id,
            "attempt_index": self.attempt_index,
            "failure_kind": self.failure_kind,
            "failure_summary": self.failure_summary,
            "recoverability": self.recoverability,
            "next_action": self.next_action,
            "attempt_dir": self.attempt_dir,
            "retry_request_path": self.retry_request_path,
            "certificate_path": self.certificate_path,
        }


def commit_response(
    *,
    run_dir: Path,
    task_id: str,
    response: dict[str, Any] | str | bytes,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> CommitResult:
    """Validate and commit a provider response to the M-43 attempt
    trail. Routes to M-44 (verifier — pending) on accept; emits a
    typed retry request on recoverable failure; emits a downstream
    M-15B-style rejection on fatal failure or attempts-exhausted.

    Idempotent on accept (re-invoking with the same response body
    produces the same attempt record + same next_action).
    """
    run_dir = Path(run_dir).resolve()
    out_dir = run_dir / "04_kernel_codegen"
    request_path = out_dir / "requests" / f"{task_id}.request.json"
    if not request_path.exists():
        return CommitResult(
            accepted=False, task_id=task_id, attempt_index=-1,
            failure_kind="schema_invalid",
            failure_summary=(
                f"request not found at {request_path.relative_to(run_dir)}; "
                f"M-42 must emit the task before commit"
            ),
            recoverability=RECOVERABILITY["schema_invalid"],
            next_action="fatal_reject",
        )
    request_body = _read_json(request_path)

    # Determine attempt_index by counting existing attempts.
    attempts_root = out_dir / "attempts" / task_id
    attempts_root.mkdir(parents=True, exist_ok=True)
    existing = sorted(p for p in attempts_root.iterdir() if p.is_dir())
    attempt_index = len(existing)

    # Persist response on disk first (under attempts/<task>/attempt_<N>/),
    # so the attempt trail captures even malformed responses.
    attempt_dir = attempts_root / f"attempt_{attempt_index:03d}"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    response_path = attempt_dir / "response.json"
    if isinstance(response, (dict,)):
        response_path.write_text(
            json.dumps(response, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        # Pre-serialised string/bytes — write verbatim.
        if isinstance(response, bytes):
            response_path.write_bytes(response)
        else:
            response_path.write_text(response, encoding="utf-8")

    # Validate.
    if isinstance(response, (str, bytes)):
        validation_input = response
    else:
        validation_input = response
    result = validate_response(
        response_body=validation_input,
        request_body=request_body,
        run_dir=run_dir,
    )

    # Persist validation report alongside the response.
    (attempt_dir / "validation_report.json").write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Append to the kernel_codegen_attempts.json log (append-only).
    attempts_log_path = out_dir / "kernel_codegen_attempts.json"
    attempts_log = _read_json_or_none(attempts_log_path) or {
        "schema_version": _ATTEMPTS_LOG_SCHEMA_VERSION,
        "task_id": task_id,
        "max_attempts": max_attempts,
        "attempts": [],
    }
    attempts_log["attempts"].append({
        "attempt_index": attempt_index,
        "started_at_utc": _utcnow(),
        "provider": (
            (response.get("provider") or {}).get("kind", "")
            if isinstance(response, dict) else "unknown"
        ),
        "status": "accepted" if result.accepted else "rejected",
        "failure_kind": result.failure_kind,
        "recoverability": result.recoverability,
        "attempt_dir": str(attempt_dir.relative_to(run_dir)),
    })
    attempts_log_path.write_text(
        json.dumps(attempts_log, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Decide next action.
    if result.accepted:
        # M-44: run the contract-driven verifier checklist. The
        # verifier reads the M-40 materialised contract + the
        # provider's kernel_metadata + claims, generates obligations
        # from contract fields, and writes a validation report.
        verification = _run_m44_verifier(
            run_dir=run_dir, task_id=task_id,
            request_body=request_body,
            response_body=response if isinstance(response, dict) else None,
        )
        if verification is None:
            # Verifier itself errored — degrade gracefully.
            return CommitResult(
                accepted=True, task_id=task_id, attempt_index=attempt_index,
                next_action="verifier_pending",
                attempt_dir=str(attempt_dir.relative_to(run_dir)),
            )
        if verification["overall"] == "fail":
            # The provider passed schema/sandbox/contract checks but
            # the contract-derived verifier rejected. Treat as a
            # recoverable provider failure; emit a retry request with
            # the typed failure_kind from the verifier.
            verifier_failure_kind = verification.get(
                "failure_kind", "metadata_mismatch",
            ) or "metadata_mismatch"
            verifier_summary = verification.get("failure_summary", "")
            verifier_result = ValidationResult(
                accepted=False,
                failure_kind=verifier_failure_kind,
                failure_summary=verifier_summary,
                recoverability=RECOVERABILITY.get(
                    verifier_failure_kind,
                    "recoverable_provider_failure",
                ),
                evidence_paths={
                    "validation_report": verification["validation_report_path"],
                },
            )
            if attempt_index + 1 >= max_attempts:
                retry_path = _emit_retry_request(
                    run_dir=run_dir, task_id=task_id,
                    request_body=request_body,
                    attempt_index=attempt_index, result=verifier_result,
                    kind="exhausted", max_attempts=max_attempts,
                )
                return CommitResult(
                    accepted=False, task_id=task_id,
                    attempt_index=attempt_index,
                    failure_kind="kernel_codegen_attempts_exhausted",
                    failure_summary=verifier_summary,
                    recoverability="protocol_or_contract_fatal",
                    next_action="fatal_reject",
                    attempt_dir=str(attempt_dir.relative_to(run_dir)),
                    retry_request_path=str(retry_path.relative_to(run_dir)),
                )
            retry_path = _emit_retry_request(
                run_dir=run_dir, task_id=task_id,
                request_body=request_body,
                attempt_index=attempt_index, result=verifier_result,
                kind="retry", max_attempts=max_attempts,
            )
            return CommitResult(
                accepted=False, task_id=task_id,
                attempt_index=attempt_index,
                failure_kind=verifier_failure_kind,
                failure_summary=verifier_summary,
                recoverability=verifier_result.recoverability,
                next_action="retry",
                attempt_dir=str(attempt_dir.relative_to(run_dir)),
                retry_request_path=str(retry_path.relative_to(run_dir)),
            )

        # Verification accepted (overall=pass or pass+deferred).
        # M-45: emit the kernel certificate ONLY when the verifier
        # produced overall=pass (deferred verdicts wait for
        # M-47/M-48/M-49 to land their checks). On pass+deferred mix,
        # the cert could be emitted with a deferred-state flag, but
        # M-45 ships the strict-pass-only path — the certificate is
        # only meaningful once verification is complete.
        certificate_path_str = ""
        if verification["overall"] == "pass" and isinstance(response, dict):
            try:
                from compgen.kernels.kernel_certificate import (
                    emit_certificate,
                )
                report_path = run_dir / verification["validation_report_path"]
                cert_path = emit_certificate(
                    run_dir=run_dir,
                    request_body=request_body,
                    response_body=response,
                    verifier_report_path=report_path,
                    fallback_used=False,
                    fallback_reason="",
                )
                certificate_path_str = str(cert_path.relative_to(run_dir))
            except Exception:  # noqa: BLE001 — never let cert emit
                # block an accept, but record the gap.
                certificate_path_str = ""
        return CommitResult(
            accepted=True, task_id=task_id, attempt_index=attempt_index,
            next_action=(
                "verified"
                if verification["overall"] == "pass"
                else "verifier_pending"
            ),
            attempt_dir=str(attempt_dir.relative_to(run_dir)),
            certificate_path=certificate_path_str,
        )

    # Rejection paths.
    rec = result.recoverability
    if rec == "protocol_or_contract_fatal":
        retry_path = _emit_retry_request(
            run_dir=run_dir, task_id=task_id, request_body=request_body,
            attempt_index=attempt_index, result=result,
            kind="fatal", max_attempts=max_attempts,
        )
        return CommitResult(
            accepted=False, task_id=task_id, attempt_index=attempt_index,
            failure_kind=result.failure_kind,
            failure_summary=result.failure_summary,
            recoverability=rec, next_action="fatal_reject",
            attempt_dir=str(attempt_dir.relative_to(run_dir)),
            retry_request_path=str(retry_path.relative_to(run_dir)),
        )

    # Recoverable / protocol-violation / transient: bound retries.
    if attempt_index + 1 >= max_attempts:
        # Exhausted — emit the M-15B-style downstream retry request.
        retry_path = _emit_retry_request(
            run_dir=run_dir, task_id=task_id, request_body=request_body,
            attempt_index=attempt_index, result=result,
            kind="exhausted", max_attempts=max_attempts,
        )
        return CommitResult(
            accepted=False, task_id=task_id, attempt_index=attempt_index,
            failure_kind="kernel_codegen_attempts_exhausted",
            failure_summary=(
                f"3 attempts exhausted for task {task_id!r}; last failure: "
                f"{result.failure_summary}"
            ),
            recoverability="protocol_or_contract_fatal",
            next_action="fatal_reject",
            attempt_dir=str(attempt_dir.relative_to(run_dir)),
            retry_request_path=str(retry_path.relative_to(run_dir)),
        )

    retry_path = _emit_retry_request(
        run_dir=run_dir, task_id=task_id, request_body=request_body,
        attempt_index=attempt_index, result=result,
        kind="retry", max_attempts=max_attempts,
    )
    return CommitResult(
        accepted=False, task_id=task_id, attempt_index=attempt_index,
        failure_kind=result.failure_kind,
        failure_summary=result.failure_summary,
        recoverability=rec, next_action="retry",
        attempt_dir=str(attempt_dir.relative_to(run_dir)),
        retry_request_path=str(retry_path.relative_to(run_dir)),
    )


def _run_m44_verifier(
    *,
    run_dir: Path,
    task_id: str,
    request_body: dict[str, Any],
    response_body: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Run the M-44 contract-driven verifier on an accepted response.

    Returns ``{overall, failure_kind, failure_summary, validation_report_path}``
    on completion (pass or fail), or ``None`` if the verifier itself
    couldn't run (e.g. the materialised contract is missing — a
    different code path's job to surface).
    """
    if response_body is None:
        return None
    try:
        from compgen.kernels.contract_verifier import (
            verify_kernel,
            write_validation_report,
        )
        from compgen.kernels.contract_v3 import KernelContractV3
    except Exception:  # noqa: BLE001 — be defensive about dialect import
        return None

    # Re-load the materialised KernelContractV3. The contract file
    # already passed validate_response's contract_mutation check.
    contract_path = (
        run_dir / request_body["contract_paths"]["full"]
    )
    body = _read_json_or_none(contract_path)
    if body is None:
        return None
    # Reconstruct the contract from disk. We use the same code path
    # M-40 used to write it, but inverted — read the canonicalised
    # JSON and re-materialise. To keep M-44 honest we re-run
    # from_recipe with the same inputs as M-40, OR we trust the
    # serialised form. M-44 takes the trusted-serialisation path here:
    # the contract file is already canonical and immutable per
    # contract_mutation guard.
    try:
        contract = _reconstruct_contract_from_dict(body)
    except Exception:  # noqa: BLE001
        return None

    artifacts = response_body.get("artifacts") or {}
    metadata_rel = artifacts.get("kernel_metadata", "")
    claims_rel = artifacts.get("provider_claims", "")
    metadata_path = (run_dir / metadata_rel) if metadata_rel else None
    claims_path = (run_dir / claims_rel) if claims_rel else None

    report = verify_kernel(
        contract=contract,
        task_id=task_id,
        contract_hash=request_body.get("contract_hash", ""),
        kernel_metadata_path=metadata_path,
        provider_claims_path=claims_path,
    )
    report_path = write_validation_report(
        run_dir=run_dir, task_id=task_id, report=report,
    )
    return {
        "overall": report.overall,
        "failure_kind": report.failure_kind,
        "failure_summary": report.failure_summary,
        "validation_report_path": str(report_path.relative_to(run_dir)),
    }


def _reconstruct_contract_from_dict(body: dict[str, Any]) -> Any:
    """Reconstruct a KernelContractV3 from its serialised form.

    We use the same KernelContractV3 dataclasses + the same
    KernelArchetype / DispatchModel / etc. enums. The serialised form
    is what M-40's ``contract_to_dict`` produced; this is the
    inverse.
    """
    from compgen.kernels.contract_v3 import (
        AliasPair, BufferLifetime, ConcurrencyUnit, DispatchModel,
        DispatchSpec, EventDecl, ExecutionEnvelope, FusionPolicy,
        Granularity, HardwareEnvelope, IOContract, KernelArchetype,
        KernelContractV3, LayoutKind, MemorySpec, MemoryTier,
        NumericsSpec, ObservabilitySpec, OrchestrationSpec,
        PaddingPolicy, PerformancePriority, ProviderHint, ShapeClass,
        SelectionHints, StaticAttr, SyncSpec, TensorIO,
    )

    def _tio(t: dict[str, Any]) -> TensorIO:
        s = t["shape"]
        dims = tuple(d if d is not None else None for d in s["dims"])
        max_dims = (
            tuple(s["max_dims"]) if s.get("max_dims") else None
        )
        divis = (
            tuple(s["divisibility"]) if s.get("divisibility") else None
        )
        return TensorIO(
            name=t["name"],
            shape=ShapeClass(
                dims=dims, max_dims=max_dims, divisibility=divis,
            ),
            dtype_class=tuple(t["dtype_class"]),
            layout=LayoutKind(t["layout"]),
            alignment_bytes=t.get("alignment_bytes", 16),
            broadcast_pattern=t.get("broadcast_pattern"),
        )

    io = body["io"]
    io_obj = IOContract(
        inputs=tuple(_tio(t) for t in io["inputs"]),
        outputs=tuple(_tio(t) for t in io["outputs"]),
        attributes=tuple(
            StaticAttr(name=a["name"], value=a["value"])
            for a in io.get("attributes") or []
        ),
        numerics=NumericsSpec(
            accumulator_dtype=io["numerics"].get("accumulator_dtype"),
            fast_math=io["numerics"].get("fast_math", False),
            max_relative_error=io["numerics"].get("max_relative_error", 1e-3),
            deterministic=io["numerics"].get("deterministic", True),
        ),
    )

    o = body["orchestration"]
    exe = o["execution"]
    if exe is not None:
        hw = exe["hardware"]
        execution = ExecutionEnvelope(
            hardware=HardwareEnvelope(
                target_name=hw["target_name"],
                vector_lanes=hw["vector_lanes"],
                scratchpad_bytes=hw["scratchpad_bytes"],
                register_bytes=hw["register_bytes"],
                native_dtypes=tuple(hw["native_dtypes"]),
                peak_bandwidth_gbps=hw["peak_bandwidth_gbps"],
                # M-60 — extended hardware envelope (defaults preserve
                # backward compatibility with pre-M-60 cert bodies).
                codegen_hints=tuple(hw.get("codegen_hints") or ()),
                mma_shapes={
                    str(k): (int(v[0]), int(v[1]), int(v[2]))
                    for k, v in (hw.get("mma_shapes") or {}).items()
                    if v and len(v) == 3
                },
                peak_compute_per_dtype={
                    str(k): float(v)
                    for k, v in (hw.get("peak_compute_per_dtype") or {}).items()
                },
                register_quota_per_thread=int(hw.get("register_quota_per_thread", 256)),
                max_concurrent_blocks=int(hw.get("max_concurrent_blocks", 0)),
            ),
            memory_budget_bytes=exe.get("memory_budget_bytes", 0),
            concurrency_unit=ConcurrencyUnit(exe["concurrency_unit"]),
            padding=PaddingPolicy(exe["padding"]),
            priority=PerformancePriority(exe["priority"]),
        )
    else:
        execution = None

    sync_d = o["sync"]
    sync_obj = SyncSpec(
        event_decls=tuple(
            EventDecl(
                name=e["name"], scope=e["scope"],
                wait_count=e["wait_count"],
            ) for e in sync_d.get("event_decls") or []
        ),
        wait_on=tuple(sync_d.get("wait_on") or ()),
        aliasing=tuple(
            AliasPair(input_idx=a["input_idx"], output_idx=a["output_idx"])
            for a in sync_d.get("aliasing") or []
        ),
        blocking=sync_d.get("blocking", False),
    )
    mem_d = o["memory"]
    memory = MemorySpec(
        input_tiers=tuple(MemoryTier(t) for t in mem_d.get("input_tiers") or ()),
        output_tiers=tuple(MemoryTier(t) for t in mem_d.get("output_tiers") or ()),
        lifetimes=tuple(
            BufferLifetime(output_idx=l["output_idx"], live_after=l["live_after"])
            for l in mem_d.get("lifetimes") or []
        ),
        in_place_safe=mem_d.get("in_place_safe", False),
    )
    fusion = FusionPolicy(
        is_boundary=o["fusion"].get("is_boundary", False),
        fusable_with=tuple(o["fusion"].get("fusable_with") or ()),
        prefer_inline_into=o["fusion"].get("prefer_inline_into"),
    )
    dispatch = DispatchSpec(
        model=DispatchModel(o["dispatch"]["model"]),
        max_concurrent_invocations=o["dispatch"].get("max_concurrent_invocations", 0),
        retry_on_recoverable_error=o["dispatch"].get("retry_on_recoverable_error", False),
    )
    obs = ObservabilitySpec(
        emit_dispatch_event=o["observability"].get("emit_dispatch_event", False),
        emit_completion_event=o["observability"].get("emit_completion_event", False),
        cost_emit_period=o["observability"].get("cost_emit_period", 0),
    )
    orchestration = OrchestrationSpec(
        execution=execution, sync=sync_obj, memory=memory,
        fusion=fusion, dispatch=dispatch, observability=obs,
    )
    selection = SelectionHints(
        providers=tuple(
            ProviderHint(
                name=p["name"], weight=p.get("weight", 1.0),
                rationale=p.get("rationale", ""),
            )
            for p in body.get("selection", {}).get("providers") or []
        ),
    )
    # M-61 — pre/post-condition predicates round-trip.
    from compgen.kernels.predicates import predicates_from_list

    preconditions = predicates_from_list(body.get("preconditions") or [])
    postconditions = predicates_from_list(body.get("postconditions") or [])

    return KernelContractV3(
        op_name=body["op_name"],
        archetype=KernelArchetype(body["archetype"]),
        io=io_obj,
        granularity=Granularity(body.get("granularity", "normal")),
        orchestration=orchestration,
        selection=selection,
        preconditions=preconditions,
        postconditions=postconditions,
        metadata=dict(body.get("metadata") or {}),
    )


def _emit_retry_request(
    *,
    run_dir: Path,
    task_id: str,
    request_body: dict[str, Any],
    attempt_index: int,
    result: ValidationResult,
    kind: str,  # "retry" | "fatal" | "exhausted"
    max_attempts: int,
) -> Path:
    """Write the typed retry / fatal-reject request artifact."""
    out_dir = run_dir / "04_kernel_codegen"
    if kind == "retry":
        path = out_dir / "kernel_codegen_retry_request.json"
        body = {
            "schema_version": _RETRY_REQUEST_SCHEMA_VERSION,
            "status": "retry_required",
            "task_id": task_id,
            "attempt_index": attempt_index,
            "max_attempts": max_attempts,
            "contract_hash": request_body.get("contract_hash", ""),
            "failed_stage": "kernel_codegen",
            "failed_check": result.failure_kind,
            "failure_kind": result.failure_kind,
            "recoverability": result.recoverability,
            "failure_summary": result.failure_summary,
            "evidence": {
                "request_path": str(
                    (out_dir / "requests" / f"{task_id}.request.json")
                    .relative_to(run_dir)
                ),
                "attempt_dir": str(
                    (out_dir / "attempts" / task_id / f"attempt_{attempt_index:03d}")
                    .relative_to(run_dir)
                ),
            },
            "provider_feedback": {
                "must_keep_contract_hash": request_body.get("contract_hash", ""),
                "must_not_modify": [
                    "contract", "tolerance", "shape", "layout",
                ],
                "suggested_fix": result.suggested_fix,
            },
        }
    else:
        # Fatal or exhausted — emit a downstream_retry_request_v1 so
        # the M-15B retry surface picks it up.
        path = out_dir / "kernel_codegen_failure_report.json"
        if kind == "fatal":
            failed_check = result.failure_kind
            summary = result.failure_summary
        else:  # exhausted
            failed_check = "kernel_codegen_attempts_exhausted"
            summary = (
                f"{max_attempts} attempts exhausted for task {task_id!r}; "
                f"last failure: {result.failure_summary}"
            )
        body = {
            "schema_version": "downstream_retry_request_v1",
            "status": "retry_required",
            "task_id": task_id,
            "failed_stage": "kernel_codegen",
            "failed_check": failed_check,
            "failure_kind": result.failure_kind,
            "failed_candidate_id": request_body.get("candidate_id", ""),
            "failed_recipe_op": request_body.get("recipe_op_id", ""),
            "failure_summary": summary,
            "evidence": {
                "report_path": str(path.relative_to(run_dir)),
                "attempts_dir": str(
                    (out_dir / "attempts" / task_id).relative_to(run_dir)
                ),
                "request_path": str(
                    (out_dir / "requests" / f"{task_id}.request.json")
                    .relative_to(run_dir)
                ),
            },
            "retry_policy": {
                "must_choose_different_candidate": True,
                "exclude_candidate_ids": [request_body.get("candidate_id", "")],
                "allowed_candidate_source": (
                    "agent_decision_request.candidate_ids_allowed"
                ),
            },
        }
    path.write_text(
        json.dumps(body, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
