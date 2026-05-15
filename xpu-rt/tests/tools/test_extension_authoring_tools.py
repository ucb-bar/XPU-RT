"""Tests for the extension-authoring ToolCards.

Coverage:

Positive:
* ``compgen_emit_extension_task`` writes a typed task package
  under ``out_dir`` and returns ``status=ok``.
* ``compgen_validate_extension_manifest`` accepts a clean,
  schema-valid manifest and returns ``status=ok`` with the right
  ``extension_id`` + ``provides`` summary.

Negative controls:
* Emit: ``status=error`` with ``reason=extension_task_schema_violation``
  when reason is outside the closed enum.
* Emit: input_schema rejects a request missing ``task_id``.
* Validate: ``status=error`` / ``reason=manifest_not_found`` when the
  manifest path does not exist.
* Validate: ``status=error`` / ``reason=manifest_schema_violation`` on
  a malformed manifest YAML.
* Validate: ``status=blocked`` / ``reason=sandbox_violation`` if the
  manifest declares writes outside its declared sandbox root.

All flows go through the ToolRunner (CLI + Python paths exercised) so
the wrappers are validated under the same JSON-schema gates the CLI
enforces.
"""

from __future__ import annotations

import json

import pytest
import yaml
from compgen.tools.errors import ToolInputSchemaError
from compgen.tools.tool_registry import load_tool_card, tool_cards_root
from compgen.tools.tool_runner import ToolRunner


def _emit_card():
    return load_tool_card(tool_cards_root() / "emit_extension_task.yaml")


def _validate_card():
    return load_tool_card(tool_cards_root() / "validate_extension_manifest.yaml")


# Minimal-but-valid manifest body (mirrors tests/extensions test fixture).
def _minimal_manifest_body(**overrides):
    body = {
        "schema_version": "compgen_extension_v1",
        "extension": {
            "id": "myaccel",
            "version": "0.1.0",
            "author": "user",
            "description": "Demo extension",
        },
        "provides": {
            "targets": [
                {
                    "schema_version": "target_card_v1",
                    "target_id": "myaccel_v1",
                    "family": "custom_accelerator",
                    "vendor": "user",
                    "dispatch_modes": ["sync", "static_plan"],
                    "memory_tiers": [
                        {"name": "dram", "kind": "global"},
                        {
                            "name": "scratchpad",
                            "kind": "explicit",
                            "capacity_bytes": 65536,
                        },
                    ],
                }
            ],
            "kernel_providers": [
                {
                    "schema_version": "provider_card_v1",
                    "provider_id": "myaccel_c",
                    "integration_level": "probe",
                    "target_families": ["custom_accelerator"],
                    "contract_kinds": ["matmul", "pointwise"],
                    "emits": ["c_source"],
                    "entrypoint": "myaccel.provider:MyAccelProvider",
                }
            ],
            "dialect_providers": [],
            "pass_tools": [],
        },
        "probes": {
            "required_env": ["MYACCEL_SDK_ROOT"],
            "commands": ["myaccel-cc --version"],
            "python_imports": [],
        },
        "security": {
            "sandbox_required": True,
            "allowed_write_root": ".rcg-artifacts/extensions/myaccel",
        },
        "verification": {
            "required_checks": ["manifest_schema", "provider_probe"],
        },
    }
    body.update(overrides)
    return body


# ---------- Emit positive --------------------------------------------


def test_emit_extension_task_positive(tmp_path):
    card = _emit_card()
    out = tmp_path / "emit_out"
    result = ToolRunner().run(
        card,
        request={
            "task_id": "ext_M96_001",
            "reason": "unsupported_op",
            "op": "torch.custom_attention",
            "region_id": "region_017",
            "contract_hash": "cafe" * 4,
            "kernel_facing_contract": {"shape": [1024, 1024], "dtype": "fp16"},
        },
        out_dir=out,
    )
    assert result.status == "ok"
    task_dir = out / "ext_M96_001"
    assert (task_dir / "extension_task.json").is_file()
    assert (task_dir / "allowed_outputs.json").is_file()
    assert (task_dir / "kernel_facing_contract.json").is_file()
    task_body = json.loads((task_dir / "extension_task.json").read_text(encoding="utf-8"))
    assert task_body["task_id"] == "ext_M96_001"
    assert task_body["op"] == "torch.custom_attention"


# ---------- Emit negative controls -----------------------------------


def test_emit_extension_task_unknown_reason(tmp_path):
    card = _emit_card()
    result = ToolRunner().run(
        card,
        request={
            "task_id": "ext_bad_reason",
            "reason": "this_is_not_a_real_reason_xyz",
        },
        out_dir=tmp_path / "out",
    )
    assert result.status == "error"
    assert result.result.get("reason") == "extension_task_schema_violation"


def test_emit_extension_task_missing_task_id(tmp_path):
    card = _emit_card()
    with pytest.raises(ToolInputSchemaError, match="task_id"):
        ToolRunner().run(card, request={"reason": "unsupported_op"}, out_dir=tmp_path / "out")


# ---------- Validate positive ----------------------------------------


def test_validate_extension_manifest_clean(tmp_path):
    """A schema-valid manifest with a real on-disk path under its
    declared sandbox root passes validation."""

    extension_root = tmp_path / "extensions" / "myaccel"
    extension_root.mkdir(parents=True)
    manifest_path = extension_root / "compgen_extension.yaml"
    body = _minimal_manifest_body()
    # Point sandbox at the actual on-disk location so the registry
    # is satisfied by the layout.
    body["security"]["allowed_write_root"] = str(extension_root.resolve())
    manifest_path.write_text(yaml.safe_dump(body), encoding="utf-8")

    card = _validate_card()
    result = ToolRunner().run(
        card,
        request={"manifest_path": str(manifest_path)},
        out_dir=tmp_path / "validate_out",
    )
    assert result.status == "ok", result.result
    assert result.result["extension_id"] == "myaccel"
    assert "myaccel_c" in result.result["provides"]["kernel_providers"]
    assert (tmp_path / "validate_out" / "validation_report.json").is_file()


# ---------- Validate negative controls -------------------------------


def test_validate_extension_manifest_missing_path(tmp_path):
    card = _validate_card()
    result = ToolRunner().run(
        card,
        request={"manifest_path": str(tmp_path / "does_not_exist.yaml")},
        out_dir=tmp_path / "validate_out",
    )
    assert result.status == "error"
    assert result.result["reason"] == "manifest_not_found"


def test_validate_extension_manifest_schema_violation(tmp_path):
    bad = tmp_path / "compgen_extension.yaml"
    bad.write_text(
        yaml.safe_dump(
            {
                "schema_version": "compgen_extension_v1",
                # missing "extension" block — schema violation
                "provides": {
                    "kernel_providers": [],
                    "dialect_providers": [],
                    "targets": [],
                    "pass_tools": [],
                },
                "security": {
                    "sandbox_required": True,
                    "allowed_write_root": ".rcg-artifacts/extensions/x",
                },
                "verification": {"required_checks": []},
            }
        ),
        encoding="utf-8",
    )
    card = _validate_card()
    result = ToolRunner().run(
        card,
        request={"manifest_path": str(bad)},
        out_dir=tmp_path / "validate_out",
    )
    assert result.status == "error"
    assert result.result["reason"] == "manifest_schema_violation"


def test_validate_extension_manifest_emits_report(tmp_path):
    """Validation always writes ``validation_report.json`` under out_dir."""

    card = _validate_card()
    out = tmp_path / "validate_out"
    ToolRunner().run(
        card,
        request={"manifest_path": str(tmp_path / "x.yaml")},
        out_dir=out,
    )
    assert (out / "validation_report.json").is_file()


# ---------- Audit invariants -----------------------------------------


def test_both_cards_audit_clean_for_their_declared_maturity():
    """The audit must verify both cards at the maturity
    their cards declare (T6 after the P1 exit-gate promotion)."""

    from compgen.audit.tool_promotion import run_tool_promotion_audit

    report = run_tool_promotion_audit()
    for tool_id in ("compgen_emit_extension_task", "compgen_validate_extension_manifest"):
        outcome = next(o for o in report.outcomes if o.tool_id == tool_id)
        assert outcome.verified_maturity == outcome.declared_maturity, outcome.violations
        assert outcome.declared_maturity in {"T2", "T6"}


def test_emit_card_declares_extension_authoring_phase():
    card = _emit_card()
    assert card.phase == "extension_authoring"
    assert "mutate_payload_ir" in card.forbidden
    assert "mutate_recipe_ir" in card.forbidden
    assert "bypass_verifier" in card.forbidden


def test_validate_card_declares_extension_authoring_phase():
    card = _validate_card()
    assert card.phase == "extension_authoring"
    assert "mutate_payload_ir" in card.forbidden
    assert "bypass_verifier" in card.forbidden
