"""Extension-authoring tool entrypoints.

Each function in this module is the Python entrypoint for one
:class:`compgen.tools.ToolCard` that wraps an existing
:mod:`compgen.extensions` flow step:

* :func:`emit_extension_task` — kick off the unsupported-op /
  extension-task flow by writing a typed task package under
  ``${run_dir}/<task_id>/``.
* :func:`validate_extension_manifest` — validate a single user-authored
  extension manifest against the schema + sandbox rules.

Both functions honour the ToolCard contract: request is a JSON-loaded
``dict``, output is a JSON-serialisable ``dict`` with a closed-enum
``status`` field. Errors that come from the underlying
:mod:`compgen.extensions` layer are translated into structured
``status=error`` payloads — the runner never sees an unmanaged
exception.

The forbidden-action constraints declared on the ToolCard YAMLs
(``mutate_payload_ir``, ``mutate_recipe_ir``, ``bypass_verifier``,
``write_outside_artifact_dir``) are enforced at the architecture
layer by :mod:`compgen.audit.extension_architecture` — the wrappers
here simply expose the flow.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from compgen.extensions.errors import (
    ExtensionError,
    ExtensionManifestError,
    ExtensionSandboxViolation,
    ExtensionTaskError,
)
from compgen.extensions.manifest import (
    EXTENSION_TASK_SCHEMA_VERSION,
    ExtensionTask,
    load_manifest,
)
from compgen.extensions.registry import build_registry


def emit_extension_task(
    request: dict[str, Any], *, out_dir: Path
) -> dict[str, Any]:
    """Emit a fresh extension task package into ``out_dir``.

    Request shape (mirrors the ``ExtensionTask`` dataclass closely
    enough for direct ``from_dict`` construction). All keys except
    ``task_id`` and ``reason`` have sensible defaults so a minimal
    request still produces a valid task package:

    ::

        {
          "task_id": "ext_001",
          "reason": "unsupported_op",
          "op": "torch.flash_attention",
          "region_id": "region_017",
          "contract_hash": "deadbeef",
          "allowed_extension_types": ["kernel_provider", "dialect_provider"],
          "allowed_outputs": [".py", ".yaml"],
          "verification_required": ["differential_test"],
          "kernel_facing_contract": {...},   # optional companion doc
          "region_dossier": {...},            # optional companion doc
          "payload_ir_summary": {...}         # optional companion doc
        }

    The task package is written to ``out_dir/<task_id>/`` (so multiple
    tasks can land in one run directory). The wrapper does *not* call
    :func:`compgen.extensions.task_flow.emit_extension_task` because
    that targets the global ``.rcg-artifacts/tasks/`` root; the
    ToolCard contract demands writes stay under ``out_dir``.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    body = dict(request)
    body.setdefault("schema_version", EXTENSION_TASK_SCHEMA_VERSION)
    body.setdefault("op", "unspecified")
    body.setdefault("region_id", "region_unknown")
    body.setdefault("contract_hash", "0" * 16)
    body.setdefault("allowed_extension_types", ["kernel_provider"])
    body.setdefault("allowed_outputs", [".py", ".yaml"])
    body.setdefault("forbidden", ["mutate_payload_ir", "mutate_recipe_ir"])
    body.setdefault("verification_required", ["differential_test"])

    # ``output_dir`` is bridge-managed: it tells a fresh-agent
    # responder where to place the extension's source files.
    task_id = str(body.get("task_id", "")) or "ext_unspecified"
    body.setdefault("output_dir", str((out_dir / task_id / "extension").resolve()))

    try:
        task = ExtensionTask.from_dict(body)
    except ExtensionTaskError as exc:
        return {
            "status": "error",
            "reason": "extension_task_schema_violation",
            "detail": str(exc),
            "task_id": body.get("task_id", ""),
            "artifacts": [],
        }

    task_dir = out_dir / task.task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    artifacts: list[str] = []

    # Required artifacts
    task_path = task_dir / "extension_task.json"
    task.write(task_path)
    artifacts.append(str(task_path.resolve()))

    allowed_outputs_path = task_dir / "allowed_outputs.json"
    allowed_outputs_path.write_text(
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
        ),
        encoding="utf-8",
    )
    artifacts.append(str(allowed_outputs_path.resolve()))

    # Optional companion documents
    for key, name in (
        ("kernel_facing_contract", "kernel_facing_contract.json"),
        ("region_dossier", "region_dossier.json"),
        ("payload_ir_summary", "payload_ir_summary.json"),
    ):
        companion = request.get(key)
        if companion is None:
            continue
        p = task_dir / name
        p.write_text(
            json.dumps(companion, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        artifacts.append(str(p.resolve()))

    return {
        "status": "ok",
        "task_id": task.task_id,
        "task_dir": str(task_dir.resolve()),
        "artifacts": artifacts,
        "reason": "extension task package emitted",
    }


def validate_extension_manifest(
    request: dict[str, Any], *, out_dir: Path
) -> dict[str, Any]:
    """Validate a single ``compgen_extension.yaml`` against the schema.

    Request shape:

    ::

        {
          "manifest_path": "/abs/path/to/compgen_extension.yaml"
        }

    Output (status enum):

    * ``ok`` — manifest parsed cleanly; sandbox + provides shape valid.
    * ``blocked`` — manifest references a real but unsupported extension
      type or carries a typed sandbox failure.
    * ``error`` — manifest body is malformed or missing.

    A typed report is also written to ``out_dir/validation_report.json``
    so callers downstream of the runner (audit, evidence
    pack) can re-read it without re-running the validation.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "validation_report.json"

    raw_path = request.get("manifest_path")
    if not raw_path:
        report = {
            "status": "error",
            "reason": "manifest_path_missing",
            "detail": "request must include 'manifest_path'",
            "artifacts": [],
        }
        _write_report(report_path, report)
        return {**report, "artifacts": [str(report_path.resolve())]}

    manifest_path = Path(raw_path)
    if not manifest_path.is_file():
        report = {
            "status": "error",
            "reason": "manifest_not_found",
            "detail": f"{manifest_path} is not a regular file",
            "manifest_path": str(manifest_path),
        }
        _write_report(report_path, report)
        return {**report, "artifacts": [str(report_path.resolve())]}

    try:
        manifest = load_manifest(manifest_path)
    except ExtensionManifestError as exc:
        report = {
            "status": "error",
            "reason": "manifest_schema_violation",
            "detail": str(exc),
            "manifest_path": str(manifest_path),
        }
        _write_report(report_path, report)
        return {**report, "artifacts": [str(report_path.resolve())]}
    except ExtensionSandboxViolation as exc:
        report = {
            "status": "blocked",
            "reason": "sandbox_violation",
            "detail": str(exc),
            "manifest_path": str(manifest_path),
        }
        _write_report(report_path, report)
        return {**report, "artifacts": [str(report_path.resolve())]}
    except ExtensionError as exc:
        report = {
            "status": "error",
            "reason": "extension_error",
            "detail": str(exc),
            "manifest_path": str(manifest_path),
        }
        _write_report(report_path, report)
        return {**report, "artifacts": [str(report_path.resolve())]}

    # Build a single-manifest registry to exercise the sandbox check.
    registry = build_registry(manifests=[manifest])
    if registry.rejected:
        rejected = registry.rejected[0]
        report = {
            "status": "blocked",
            "reason": rejected.kind,
            "detail": rejected.detail,
            "manifest_path": str(manifest_path),
            "extension_id": manifest.extension_id,
        }
        _write_report(report_path, report)
        return {**report, "artifacts": [str(report_path.resolve())]}

    report = {
        "status": "ok",
        "extension_id": manifest.extension_id,
        "version": manifest.version,
        "provides": _summarise_provides(manifest),
        "manifest_path": str(manifest_path),
    }
    _write_report(report_path, report)
    return {**report, "artifacts": [str(report_path.resolve())]}


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _summarise_provides(manifest: Any) -> dict[str, list[str]]:
    return {
        "kernel_providers": [c.provider_id for c in manifest.kernel_providers],
        "dialect_providers": [c.provider_id for c in manifest.dialect_providers],
        "targets": [c.target_id for c in manifest.targets],
        "pass_tools": [c.tool_id for c in manifest.pass_tools],
    }


__all__ = ["emit_extension_task", "validate_extension_manifest"]
