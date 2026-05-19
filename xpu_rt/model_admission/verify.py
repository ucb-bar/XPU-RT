"""One-time HuggingFace source verification.

Given ``source_candidates.yaml`` (a tiny ``model_id -> candidate_hf_ref``
map), call ``HfApi().model_info()`` once per model and rewrite each
``configs/models/<id>.yaml`` with the canonical model_ref, the resolved
revision SHA, gated/private flags, and ``source_verified: true`` when
the lookup succeeds.

Verification status is captured into the YAML and committed to git, so
this command runs once per candidate (or whenever you intentionally
``--refresh``). The admission probe trusts the verified YAMLs.

Network policy: the verifier issues exactly one HTTP GET per candidate
(the read-only model_info endpoint -- it does NOT download weights). If
``HF_TOKEN`` is set in the environment, it is forwarded for gated repos.
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

import structlog
import yaml

log = structlog.get_logger(__name__)

CANDIDATES_SCHEMA = "model_admission_source_candidates_v1"


class VerifyStatus(StrEnum):
    """Outcome of one verification call. ``passed`` is the only good state."""

    PASSED = "passed"
    GATED = "gated"
    NOT_FOUND = "not_found"
    NETWORK_ERROR = "network_error"
    AUTH_REQUIRED = "auth_required"
    SKIPPED_NO_CANDIDATE = "skipped_no_candidate"
    UNKNOWN_MODEL_ID = "unknown_model_id"


@dataclass(frozen=True)
class VerifyResult:
    """One verification outcome (one model_id)."""

    model_id: str
    candidate_ref: str
    status: VerifyStatus
    canonical_ref: str = ""
    revision: str = ""
    gated: bool = False
    private: bool = False
    error: str = ""

    def as_summary_row(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "candidate_ref": self.candidate_ref,
            "canonical_ref": self.canonical_ref,
            "status": self.status.value,
            "revision": self.revision[:12] if self.revision else "",
            "gated": self.gated,
            "error": self.error,
        }


class _HfApiLike(Protocol):
    """Subset of ``huggingface_hub.HfApi`` we need; allows mocking in tests."""

    def model_info(self, repo_id: str, *, token: str | None = None, revision: str | None = None) -> Any: ...


# --------------------------------------------------------------------------- #
# Candidates file.
# --------------------------------------------------------------------------- #


def load_candidates(path: Path) -> dict[str, str]:
    """Load ``model_id -> candidate_hf_ref`` from a candidates YAML."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping")
    if raw.get("schema_version") != CANDIDATES_SCHEMA:
        raise ValueError(
            f"{path}: expected schema_version={CANDIDATES_SCHEMA!r}, got {raw.get('schema_version')!r}"
        )
    out: dict[str, str] = {}
    for row in raw.get("candidates", []) or []:
        if not isinstance(row, dict):
            raise ValueError(f"{path}: each candidate must be a mapping")
        mid = str(row["model_id"])
        ref = str(row.get("candidate_ref", "")).strip()
        if mid in out:
            raise ValueError(f"{path}: duplicate model_id={mid!r}")
        out[mid] = ref
    return out


# --------------------------------------------------------------------------- #
# Verifier core.
# --------------------------------------------------------------------------- #


def _make_hf_api() -> _HfApiLike:
    from huggingface_hub import HfApi

    return HfApi()


def _verifier_identity() -> str:
    user = os.environ.get("USER") or os.environ.get("USERNAME") or "unknown"
    try:
        host = socket.gethostname()
    except OSError:
        host = "unknown"
    return f"{user}@{host}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def verify_one(
    model_id: str,
    candidate_ref: str,
    api: _HfApiLike,
    token: str | None,
) -> VerifyResult:
    """Run one model_info call and classify the outcome.

    The probe never raises -- every failure mode collapses to a typed
    :class:`VerifyStatus` value.
    """

    if not candidate_ref or candidate_ref.upper().startswith("TO_BE_VERIFIED"):
        return VerifyResult(
            model_id=model_id,
            candidate_ref=candidate_ref,
            status=VerifyStatus.SKIPPED_NO_CANDIDATE,
        )

    try:
        from huggingface_hub.errors import (  # noqa: PLC0415
            GatedRepoError,
            HfHubHTTPError,
            RepositoryNotFoundError,
        )
    except ImportError:
        return VerifyResult(
            model_id=model_id,
            candidate_ref=candidate_ref,
            status=VerifyStatus.NETWORK_ERROR,
            error="huggingface_hub not installed",
        )

    try:
        info = api.model_info(candidate_ref, token=token)
    except GatedRepoError as exc:
        return VerifyResult(
            model_id=model_id,
            candidate_ref=candidate_ref,
            status=VerifyStatus.GATED,
            error=str(exc).splitlines()[0] if str(exc) else "gated repo",
            gated=True,
        )
    except RepositoryNotFoundError as exc:
        return VerifyResult(
            model_id=model_id,
            candidate_ref=candidate_ref,
            status=VerifyStatus.NOT_FOUND,
            error=str(exc).splitlines()[0] if str(exc) else "404",
        )
    except HfHubHTTPError as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", 0)
        if status_code in (401, 403):
            return VerifyResult(
                model_id=model_id,
                candidate_ref=candidate_ref,
                status=VerifyStatus.AUTH_REQUIRED,
                error=f"HTTP {status_code}",
            )
        return VerifyResult(
            model_id=model_id,
            candidate_ref=candidate_ref,
            status=VerifyStatus.NETWORK_ERROR,
            error=f"HfHubHTTPError: {exc}",
        )
    except (OSError, TimeoutError) as exc:
        return VerifyResult(
            model_id=model_id,
            candidate_ref=candidate_ref,
            status=VerifyStatus.NETWORK_ERROR,
            error=f"{type(exc).__name__}: {exc}",
        )
    except Exception as exc:
        return VerifyResult(
            model_id=model_id,
            candidate_ref=candidate_ref,
            status=VerifyStatus.NETWORK_ERROR,
            error=f"{type(exc).__name__}: {exc}",
        )

    canonical_ref = str(getattr(info, "id", candidate_ref)) or candidate_ref
    revision = str(getattr(info, "sha", "") or "")
    gated_attr = getattr(info, "gated", False)
    gated = bool(gated_attr) and gated_attr != "auto"
    private = bool(getattr(info, "private", False))

    return VerifyResult(
        model_id=model_id,
        candidate_ref=candidate_ref,
        canonical_ref=canonical_ref,
        revision=revision,
        gated=gated,
        private=private,
        status=VerifyStatus.PASSED,
    )


# --------------------------------------------------------------------------- #
# YAML rewriter (preserves shape; comments are minimal in our generated configs).
# --------------------------------------------------------------------------- #


def apply_to_model_yaml(yaml_path: Path, result: VerifyResult, *, verified_by: str, verified_at: str) -> bool:
    """Mutate one ``configs/models/<id>.yaml`` to record the verification result.

    Returns ``True`` if the file changed. ``False`` is benign (no-op for
    skipped / not-applicable cases).
    """

    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{yaml_path}: top-level YAML must be a mapping")
    if raw.get("schema_version") != "model_config_v1":
        return False

    src = dict(raw.get("source") or {})
    notes = list(raw.get("notes") or [])

    if result.status == VerifyStatus.PASSED:
        canonical = result.canonical_ref or result.candidate_ref
        src["provider"] = src.get("provider") or "huggingface"
        src["model_ref"] = canonical
        src["repo_url"] = f"https://huggingface.co/{canonical}"
        src["docs_url"] = f"https://huggingface.co/{canonical}"
        src["revision"] = result.revision or None
        src["verified_at"] = verified_at
        src["verified_by"] = verified_by
        src["source_verified"] = True
        # Strip prior placeholder reminder if present.
        notes = [
            n for n in notes
            if "must be verified online" not in n.lower()
            and "candidate model_ref not found" not in n.lower()
            and "gated, requires hf token" not in n.lower()
        ]
        if result.gated:
            notes.append("gated repo — admission probe still requires HF token + license acceptance.")
    elif result.status in (VerifyStatus.GATED, VerifyStatus.AUTH_REQUIRED):
        src["model_ref"] = result.candidate_ref
        src["repo_url"] = f"https://huggingface.co/{result.candidate_ref}"
        src["source_verified"] = False
        src["verified_at"] = verified_at
        src["verified_by"] = verified_by
        src["revision"] = None
        notes = [n for n in notes if "must be verified online" not in n.lower()]
        notes.append("gated, requires HF token agreement (set HF_TOKEN and re-run verify-sources).")
    elif result.status == VerifyStatus.NOT_FOUND:
        src["source_verified"] = False
        src["verified_at"] = verified_at
        src["verified_by"] = verified_by
        notes = [n for n in notes if "must be verified online" not in n.lower()]
        notes.append(
            f"candidate model_ref {result.candidate_ref!r} not found upstream "
            f"— update configs/model_admission/source_candidates.yaml."
        )
    else:
        # network errors / skipped — leave the YAML alone but still log.
        return False

    raw["source"] = src
    raw["notes"] = notes  # always set: empty list is meaningful (placeholder stripped).
    yaml_path.write_text(_dump_yaml(raw), encoding="utf-8")
    return True


def _dump_yaml(payload: dict[str, Any]) -> str:
    """Stable YAML emit: preserve top-level key order and avoid alphabetical sort."""

    return yaml.safe_dump(
        payload,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
        width=4096,
    )


# --------------------------------------------------------------------------- #
# Top-level entry point.
# --------------------------------------------------------------------------- #


@dataclass
class VerifyRun:
    results: list[VerifyResult]
    written: list[str]

    def by_status(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.results:
            out[r.status.value] = out.get(r.status.value, 0) + 1
        return out


def verify_sources(
    candidates_path: Path,
    models_dir: Path,
    *,
    api: _HfApiLike | None = None,
    token: str | None = None,
    refresh: bool = False,
    dry_run: bool = False,
    only_model_ids: list[str] | None = None,
) -> VerifyRun:
    """Run verification for every candidate and (unless ``dry_run``) update YAMLs.

    Args:
        candidates_path: Path to source_candidates.yaml.
        models_dir: Where to find configs/models/<id>.yaml files.
        api: Optional HfApi-like object for testing. Default: real HfApi().
        token: HF token. Default: ``$HF_TOKEN`` from env.
        refresh: If false (default), already-verified models are skipped.
            If true, re-run them (e.g. to pick up a new revision).
        dry_run: If true, do not write YAML changes; only return results.
        only_model_ids: If given, restrict to this subset.
    """

    if api is None:
        api = _make_hf_api()
    if token is None:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    candidates = load_candidates(candidates_path)
    verified_at = _now_iso()
    verified_by = _verifier_identity()

    results: list[VerifyResult] = []
    written: list[str] = []

    for model_id, candidate_ref in candidates.items():
        if only_model_ids is not None and model_id not in only_model_ids:
            continue

        yaml_path = models_dir / f"{model_id}.yaml"
        if not yaml_path.exists():
            results.append(
                VerifyResult(
                    model_id=model_id,
                    candidate_ref=candidate_ref,
                    status=VerifyStatus.UNKNOWN_MODEL_ID,
                    error=f"no config at {yaml_path}",
                )
            )
            continue

        # If already verified and not --refresh, skip the network call entirely.
        if not refresh:
            existing = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
            src = (existing.get("source") or {}) if isinstance(existing, dict) else {}
            if src.get("source_verified") is True and src.get("revision"):
                results.append(
                    VerifyResult(
                        model_id=model_id,
                        candidate_ref=candidate_ref,
                        canonical_ref=str(src.get("model_ref") or candidate_ref),
                        revision=str(src.get("revision") or ""),
                        gated=bool(src.get("gated", False)),
                        status=VerifyStatus.PASSED,
                    )
                )
                continue

        result = verify_one(model_id, candidate_ref, api, token)
        results.append(result)

        if dry_run:
            continue
        try:
            changed = apply_to_model_yaml(
                yaml_path, result, verified_by=verified_by, verified_at=verified_at
            )
        except Exception as exc:
            log.warning("verify_yaml_write_failed", model_id=model_id, error=str(exc))
            continue
        if changed:
            written.append(model_id)

    return VerifyRun(results=results, written=written)


__all__ = [
    "CANDIDATES_SCHEMA",
    "VerifyResult",
    "VerifyRun",
    "VerifyStatus",
    "apply_to_model_yaml",
    "load_candidates",
    "verify_one",
    "verify_sources",
]
