"""unsupported-op / extension-task flow end-to-end."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from compgen.extensions.manifest import (
    EXTENSION_TASK_SCHEMA_VERSION,
    MANIFEST_SCHEMA_VERSION,
    ExtensionTask,
)
from compgen.extensions.registry import EXTENSION_MANIFEST_FILENAME
from compgen.extensions.task_flow import (
    COMMIT_STATUSES,
    RESUME_STATUSES,
    TASK_FILENAME,
    commit_extension_response,
    emit_extension_task,
    list_extension_tasks,
    resume_after_extension,
    write_commit_log,
)


def _task(task_id: str = "ext_task_0001") -> ExtensionTask:
    return ExtensionTask.from_dict(
        {
            "schema_version": EXTENSION_TASK_SCHEMA_VERSION,
            "task_id": task_id,
            "reason": "unsupported_op",
            "op": "aten._scaled_dot_product_flash_attention",
            "region_id": "region_017",
            "contract_hash": "abc123",
            "allowed_extension_types": ["kernel_provider", "pass_tool"],
            "allowed_outputs": [".py", ".c", ".mlir"],
            "forbidden": [
                "modify_payload_ir_directly",
                "write_outside_extension_dir",
            ],
            "verification_required": ["probe", "contract_verification", "differential"],
            "output_dir": f".rcg-artifacts/extensions/{task_id}",
        }
    )


def _extension_manifest_body(extension_id: str) -> dict:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "extension": {
            "id": extension_id,
            "version": "0.1.0",
            "author": "claude_code",
            "description": "flash-attention provider response",
        },
        "provides": {
            "targets": [],
            "kernel_providers": [
                {
                    "schema_version": "provider_card_v1",
                    "provider_id": f"{extension_id}_flash",
                    "integration_level": "probe",
                    "target_families": ["cuda"],
                    "contract_kinds": ["attention"],
                    "emits": ["triton_source"],
                    "entrypoint": f"{extension_id}.provider:Provider",
                }
            ],
            "dialect_providers": [],
            "pass_tools": [],
        },
        "probes": {"required_env": [], "commands": [], "python_imports": []},
        "security": {"sandbox_required": True, "allowed_write_root": "."},
        "verification": {"required_checks": ["differential"]},
    }


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------


def test_emit_extension_task_writes_full_artifact_set(tmp_path: Path):
    task = _task()
    task_dir = emit_extension_task(
        task,
        tasks_root=tmp_path,
        kernel_facing_contract={"op": task.op, "shapes": {"q": [1, 16, 128, 64]}},
        region_dossier={"region_id": task.region_id, "estimated_flops": 8.4e9},
        payload_ir_summary={"ops": ["aten._scaled_dot_product_flash_attention"]},
    )
    assert task_dir == tmp_path / task.task_id
    assert (task_dir / TASK_FILENAME).is_file()
    assert (task_dir / "kernel_facing_contract.json").is_file()
    assert (task_dir / "region_dossier.json").is_file()
    assert (task_dir / "payload_ir_summary.json").is_file()
    assert (task_dir / "allowed_outputs.json").is_file()


def test_emitted_task_round_trips_through_loader(tmp_path: Path):
    task = _task()
    task_dir = emit_extension_task(task, tasks_root=tmp_path)
    body = json.loads((task_dir / TASK_FILENAME).read_text())
    assert body["task_id"] == task.task_id
    assert body["schema_version"] == EXTENSION_TASK_SCHEMA_VERSION


def test_allowed_outputs_document_carries_typed_fields(tmp_path: Path):
    task = _task()
    task_dir = emit_extension_task(task, tasks_root=tmp_path)
    body = json.loads((task_dir / "allowed_outputs.json").read_text())
    assert body["schema_version"] == "extension_task_allowed_outputs_v1"
    assert ".c" in body["extensions"]
    assert "modify_payload_ir_directly" in body["forbidden_actions"]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_list_extension_tasks_picks_up_emitted_tasks(tmp_path: Path):
    emit_extension_task(_task("ext_task_a"), tasks_root=tmp_path)
    emit_extension_task(_task("ext_task_b"), tasks_root=tmp_path)
    found = sorted(p.name for p in list_extension_tasks(tmp_path))
    assert found == ["ext_task_a", "ext_task_b"]


def test_list_missing_root_returns_nothing(tmp_path: Path):
    assert list(list_extension_tasks(tmp_path / "nope")) == []


# ---------------------------------------------------------------------------
# Commit
# ---------------------------------------------------------------------------


def _make_response(tmp_path: Path, task_id: str, body: dict | None = None) -> Path:
    ext_dir = tmp_path / "extensions" / task_id
    ext_dir.mkdir(parents=True)
    (ext_dir / EXTENSION_MANIFEST_FILENAME).write_text(
        yaml.safe_dump(body or _extension_manifest_body(task_id))
    )
    return ext_dir


def test_commit_accepted_returns_typed_outcome(tmp_path: Path):
    task = _task()
    task_dir = emit_extension_task(task, tasks_root=tmp_path / "tasks")
    _make_response(tmp_path, task.task_id)

    outcome = commit_extension_response(
        task_dir,
        extensions_root=tmp_path / "extensions",
    )
    assert outcome.status == "accepted"
    assert outcome.status in COMMIT_STATUSES
    assert outcome.failed_check == ""
    assert outcome.registry is not None
    assert task.task_id in outcome.registry.extension_ids()


def test_commit_missing_extension_dir(tmp_path: Path):
    task = _task()
    task_dir = emit_extension_task(task, tasks_root=tmp_path / "tasks")
    outcome = commit_extension_response(
        task_dir,
        extensions_root=tmp_path / "extensions",
    )
    assert outcome.status == "missing_extension_dir"
    assert outcome.failed_check == "missing_extension_dir"


def test_commit_missing_manifest_inside_extension_dir(tmp_path: Path):
    task = _task()
    task_dir = emit_extension_task(task, tasks_root=tmp_path / "tasks")
    ext_dir = tmp_path / "extensions" / task.task_id
    ext_dir.mkdir(parents=True)
    (ext_dir / "README.md").write_text("Hello")  # no compgen_extension.yaml

    outcome = commit_extension_response(
        task_dir,
        extensions_root=tmp_path / "extensions",
    )
    assert outcome.status == "missing_manifest"
    assert outcome.failed_check == "missing_manifest"


def test_commit_rejects_malformed_manifest(tmp_path: Path):
    task = _task()
    task_dir = emit_extension_task(task, tasks_root=tmp_path / "tasks")
    bad = _extension_manifest_body(task.task_id)
    bad["schema_version"] = "v999"
    _make_response(tmp_path, task.task_id, body=bad)

    outcome = commit_extension_response(
        task_dir,
        extensions_root=tmp_path / "extensions",
    )
    assert outcome.status == "rejected"
    assert outcome.failed_check == "manifest_schema"


def test_commit_rejects_sandbox_escape(tmp_path: Path):
    task = _task()
    task_dir = emit_extension_task(task, tasks_root=tmp_path / "tasks")
    body = _extension_manifest_body(task.task_id)
    body["security"]["allowed_write_root"] = "../../somewhere_else"
    _make_response(tmp_path, task.task_id, body=body)

    outcome = commit_extension_response(
        task_dir,
        extensions_root=tmp_path / "extensions",
    )
    assert outcome.status == "rejected"
    assert outcome.failed_check == "extension_sandbox_violation"


# ---------------------------------------------------------------------------
# Resume + log
# ---------------------------------------------------------------------------


def test_resume_after_accepted_returns_proceeded(tmp_path: Path):
    task = _task()
    task_dir = emit_extension_task(task, tasks_root=tmp_path / "tasks")
    _make_response(tmp_path, task.task_id)
    outcome = commit_extension_response(
        task_dir, extensions_root=tmp_path / "extensions"
    )
    decision = resume_after_extension(outcome)
    assert decision.status == "proceeded"
    assert decision.status in RESUME_STATUSES
    assert decision.reason == "extension_registered"


def test_resume_after_rejected_returns_still_blocked(tmp_path: Path):
    task = _task()
    task_dir = emit_extension_task(task, tasks_root=tmp_path / "tasks")
    outcome = commit_extension_response(
        task_dir, extensions_root=tmp_path / "extensions"
    )
    decision = resume_after_extension(outcome)
    assert decision.status == "still_blocked"
    assert decision.reason == "missing_extension_dir"


def test_write_commit_log_serializes_typed_outcome(tmp_path: Path):
    task = _task()
    task_dir = emit_extension_task(task, tasks_root=tmp_path / "tasks")
    _make_response(tmp_path, task.task_id)
    outcome = commit_extension_response(
        task_dir, extensions_root=tmp_path / "extensions"
    )
    log_path = write_commit_log(outcome, task_dir=task_dir)
    body = json.loads(log_path.read_text())
    assert body["status"] in COMMIT_STATUSES
    assert body["task_id"] == task.task_id
    assert task.task_id in body["registered_extension_ids"]


# ---------------------------------------------------------------------------
# Full loop
# ---------------------------------------------------------------------------


def test_end_to_end_unsupported_op_loop(tmp_path: Path):
    """unsupported op detected → task emitted → extension committed →
    probe + verify + register → compilation resumes."""

    task = _task("ext_e2e")
    task_dir = emit_extension_task(
        task,
        tasks_root=tmp_path / "tasks",
        kernel_facing_contract={"op": task.op, "region_id": task.region_id},
        region_dossier={"region_id": task.region_id},
        payload_ir_summary={"ops": [task.op]},
    )

    # Agent fulfills the task by dropping a sandboxed extension.
    _make_response(tmp_path, task.task_id)

    outcome = commit_extension_response(
        task_dir, extensions_root=tmp_path / "extensions"
    )
    assert outcome.status == "accepted"

    decision = resume_after_extension(outcome)
    assert decision.status == "proceeded"
    assert task.task_id in outcome.registry.extension_ids()
    assert f"{task.task_id}_flash" in outcome.registry.provider_ids()
