"""Unsupported-op / extension-task flow.

When graph analysis surfaces an unsupported op or provider gap,
the pipeline emits an :class:`compgen.extensions.manifest.ExtensionTask`
artifact set into ``.rcg-artifacts/tasks/<task_id>/``. A Claude
Code / Codex session (or a human operator) then writes a
sandboxed extension into ``.rcg-artifacts/extensions/<task_id>/``.

The flow has four typed entry points:

* :func:`emit_extension_task` — write the task artifact set to
  disk. Returns the task directory.
* :func:`commit_extension_response` — validate that the
  task-directory's declared ``output_dir`` now contains a valid
  ``compgen_extension.yaml``; build a per-run registry; report
  the typed outcome.
* :func:`resume_after_extension` — given a commit outcome, return
  ``proceeded`` or ``still_blocked`` with the typed reason.
* :func:`list_extension_tasks` — discover task directories under
  a root.

This module never invokes an LLM or makes a network call. It
operates purely on filesystem artifacts produced by the agent
loop one layer up.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from compgen.extensions.errors import (
    ExtensionError,
    ExtensionManifestError,
    ExtensionTaskError,
)
from compgen.extensions.manifest import (
    EXTENSION_TASK_SCHEMA_VERSION,
    ExtensionTask,
    load_extension_task,
)
from compgen.extensions.registry import (
    EXTENSION_MANIFEST_FILENAME,
    ExtensionRegistry,
    build_registry,
)

DEFAULT_TASKS_ROOT: Final[Path] = Path(".rcg-artifacts/tasks")
TASK_FILENAME: Final[str] = "extension_task.json"


# ---------------------------------------------------------------------------
# Outcomes — closed enums.
# ---------------------------------------------------------------------------

COMMIT_STATUSES: Final[tuple[str, ...]] = (
    "accepted",
    "missing_extension_dir",
    "missing_manifest",
    "rejected",
)

RESUME_STATUSES: Final[tuple[str, ...]] = (
    "proceeded",
    "still_blocked",
)


@dataclass(frozen=True)
class CommitOutcome:
    """Typed result of :func:`commit_extension_response`.

    ``status`` is one of :data:`COMMIT_STATUSES`. ``registry`` is
    only populated on ``accepted``; ``failed_check`` and ``detail``
    carry typed reasons for every other status.
    """

    status: str
    task_id: str
    extension_dir: Path | None
    registry: ExtensionRegistry | None
    failed_check: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "task_id": self.task_id,
            "extension_dir": str(self.extension_dir) if self.extension_dir else None,
            "failed_check": self.failed_check,
            "detail": self.detail,
            "registered_extension_ids": (
                list(self.registry.extension_ids()) if self.registry else []
            ),
            "rejected_extensions": (
                list(self.registry.rejected_summary()) if self.registry else []
            ),
        }


@dataclass(frozen=True)
class ResumeDecision:
    """Typed result of :func:`resume_after_extension`."""

    status: str
    task_id: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "task_id": self.task_id,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------


def emit_extension_task(
    task: ExtensionTask,
    *,
    tasks_root: Path | None = None,
    kernel_facing_contract: dict | None = None,
    region_dossier: dict | None = None,
    payload_ir_summary: dict | None = None,
) -> Path:
    """Write a task artifact set into ``tasks_root / task.task_id /``.

    The directory is created if missing. ``extension_task.json``
    plus the four optional companion documents are written when
    their corresponding argument is supplied. Returns the task
    directory.
    """

    if task.schema_version != EXTENSION_TASK_SCHEMA_VERSION:
        raise ExtensionTaskError(
            f"task.schema_version={task.schema_version!r} != "
            f"{EXTENSION_TASK_SCHEMA_VERSION!r}"
        )
    base = Path(tasks_root) if tasks_root is not None else DEFAULT_TASKS_ROOT
    task_dir = base / task.task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    task.write(task_dir / TASK_FILENAME)

    optional_docs = {
        "kernel_facing_contract.json": kernel_facing_contract,
        "region_dossier.json": region_dossier,
        "payload_ir_summary.json": payload_ir_summary,
    }
    for name, body in optional_docs.items():
        if body is None:
            continue
        (task_dir / name).write_text(
            json.dumps(body, indent=2, sort_keys=True)
        )

    # The ``allowed_outputs`` document is always emitted: it carries
    # the closed list of file extensions the response may write.
    (task_dir / "allowed_outputs.json").write_text(
        json.dumps(
            {
                "schema_version": "extension_task_allowed_outputs_v1",
                "task_id": task.task_id,
                "extensions": list(task.allowed_outputs),
                "forbidden_actions": list(task.forbidden),
                "verification_required": list(task.verification_required),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return task_dir


def list_extension_tasks(tasks_root: Path | None = None) -> Iterator[Path]:
    """Yield each task directory under ``tasks_root`` that carries
    an ``extension_task.json``."""

    base = Path(tasks_root) if tasks_root is not None else DEFAULT_TASKS_ROOT
    if not base.is_dir():
        return
    for sub in sorted(base.iterdir()):
        if sub.is_dir() and (sub / TASK_FILENAME).is_file():
            yield sub


# ---------------------------------------------------------------------------
# Commit + resume
# ---------------------------------------------------------------------------


def commit_extension_response(
    task_dir: Path,
    *,
    extensions_root: Path | None = None,
) -> CommitOutcome:
    """Validate the extension response declared by the task at
    ``task_dir`` and build a per-run registry containing it.

    The task's ``output_dir`` field names where the response
    should land relative to ``extensions_root`` (default
    ``.rcg-artifacts/extensions/``).

    Returns a typed :class:`CommitOutcome` — never raises on
    expected failure modes (missing dir, missing manifest, schema
    violation, sandbox escape, duplicate id). Bad task JSON raises
    :class:`ExtensionTaskError` from the underlying
    :func:`load_extension_task`.
    """

    task = load_extension_task(task_dir / TASK_FILENAME)

    if extensions_root is None:
        # Treat the task's output_dir as relative to the parent of the
        # extensions root used by the registry. If it's an absolute path
        # honour it; otherwise look under DEFAULT_EXTENSIONS_ROOT.
        out = Path(task.output_dir)
        if not out.is_absolute():
            from compgen.extensions.registry import DEFAULT_EXTENSIONS_ROOT
            # The output_dir field is typically
            # ``.rcg-artifacts/extensions/<task_id>``; we accept either
            # the full path or the trailing component.
            if out.parts[:2] == ("." + DEFAULT_EXTENSIONS_ROOT.parts[0], DEFAULT_EXTENSIONS_ROOT.parts[1]) or (
                len(out.parts) >= 2 and out.parts[-2] == DEFAULT_EXTENSIONS_ROOT.name
            ):
                extension_dir = out
            else:
                extension_dir = DEFAULT_EXTENSIONS_ROOT / out.name
        else:
            extension_dir = out
    else:
        extension_dir = Path(extensions_root) / Path(task.output_dir).name

    if not extension_dir.is_dir():
        return CommitOutcome(
            status="missing_extension_dir",
            task_id=task.task_id,
            extension_dir=extension_dir,
            registry=None,
            failed_check="missing_extension_dir",
            detail=str(extension_dir),
        )

    manifest_path = extension_dir / EXTENSION_MANIFEST_FILENAME
    if not manifest_path.is_file():
        return CommitOutcome(
            status="missing_manifest",
            task_id=task.task_id,
            extension_dir=extension_dir,
            registry=None,
            failed_check="missing_manifest",
            detail=str(manifest_path),
        )

    registry = build_registry(extension_dir.parent)
    relevant_rejected = [
        r for r in registry.rejected
        if Path(r.extension_dir).resolve() == extension_dir.resolve()
    ]
    if relevant_rejected:
        r = relevant_rejected[0]
        return CommitOutcome(
            status="rejected",
            task_id=task.task_id,
            extension_dir=extension_dir,
            registry=registry,
            failed_check=r.failed_check,
            detail=r.detail,
        )

    return CommitOutcome(
        status="accepted",
        task_id=task.task_id,
        extension_dir=extension_dir,
        registry=registry,
        failed_check="",
        detail="",
    )


def resume_after_extension(outcome: CommitOutcome) -> ResumeDecision:
    """Translate a commit outcome into a resume / block decision."""

    if outcome.status == "accepted":
        return ResumeDecision(
            status="proceeded",
            task_id=outcome.task_id,
            reason="extension_registered",
        )
    return ResumeDecision(
        status="still_blocked",
        task_id=outcome.task_id,
        reason=outcome.failed_check or outcome.status,
    )


def write_commit_log(
    outcome: CommitOutcome,
    *,
    task_dir: Path,
) -> Path:
    """Persist the commit outcome alongside the task artifact for
    audit replay."""

    log_path = Path(task_dir) / "commit_outcome.json"
    log_path.write_text(json.dumps(outcome.to_dict(), indent=2, sort_keys=True))
    return log_path
