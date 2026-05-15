"""ExtensionManifest + ExtensionTask schema tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from compgen.extensions.errors import (
    ExtensionManifestError,
    ExtensionTaskError,
)
from compgen.extensions.manifest import (
    ALLOWED_EXTENSION_TASK_TYPES,
    EXTENSION_TASK_REASONS,
    EXTENSION_TASK_SCHEMA_VERSION,
    MANIFEST_SCHEMA_VERSION,
    ExtensionManifest,
    ExtensionTask,
    load_extension_task,
    load_manifest,
)


def _minimal_manifest_body(**overrides):
    body = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
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


def test_manifest_round_trips_through_yaml(tmp_path: Path):
    body = _minimal_manifest_body()
    p = tmp_path / "compgen_extension.yaml"
    p.write_text(yaml.safe_dump(body))
    manifest = load_manifest(p)
    assert manifest.extension_id == "myaccel"
    assert manifest.kernel_providers[0].provider_id == "myaccel_c"
    assert manifest.targets[0].target_id == "myaccel_v1"
    assert manifest.security.allowed_write_root.endswith("extensions/myaccel")
    # to_dict → from_dict stable
    reparsed = ExtensionManifest.from_dict(manifest.to_dict())
    assert reparsed.extension_id == manifest.extension_id


def test_manifest_wrong_schema_version_rejected():
    with pytest.raises(ExtensionManifestError, match="schema_version"):
        ExtensionManifest.from_dict(_minimal_manifest_body(schema_version="v999"))


def test_manifest_missing_extension_id_rejected():
    body = _minimal_manifest_body()
    body["extension"].pop("id")
    with pytest.raises(ExtensionManifestError, match="extension.id"):
        ExtensionManifest.from_dict(body)


def test_manifest_sandbox_required_without_root_rejected():
    body = _minimal_manifest_body()
    body["security"]["allowed_write_root"] = ""
    with pytest.raises(ExtensionManifestError, match="allowed_write_root"):
        ExtensionManifest.from_dict(body)


def test_manifest_paper_claimable_card_with_probe_level_rejected():
    body = _minimal_manifest_body()
    body["provides"]["kernel_providers"][0]["paper_claimable"] = True
    body["provides"]["kernel_providers"][0]["integration_level"] = "probe"
    with pytest.raises(Exception, match="paper_claimable"):
        ExtensionManifest.from_dict(body)


def _minimal_task_body(**overrides):
    body = {
        "schema_version": EXTENSION_TASK_SCHEMA_VERSION,
        "task_id": "ext_task_0001",
        "reason": "unsupported_op",
        "op": "aten._scaled_dot_product_flash_attention",
        "region_id": "region_017",
        "contract_hash": "abc123",
        "allowed_extension_types": ["kernel_provider", "pass_tool"],
        "allowed_outputs": [".py", ".mlir", ".c"],
        "forbidden": [
            "modify_payload_ir_directly",
            "write_outside_extension_dir",
        ],
        "verification_required": ["probe", "contract_verification", "differential"],
        "output_dir": ".rcg-artifacts/extensions/ext_task_0001",
    }
    body.update(overrides)
    return body


def test_extension_task_round_trips(tmp_path: Path):
    body = _minimal_task_body()
    p = tmp_path / "extension_task.json"
    p.write_text(json.dumps(body))
    task = load_extension_task(p)
    assert task.task_id == "ext_task_0001"
    assert task.reason in EXTENSION_TASK_REASONS
    written = task.write(tmp_path / "round_trip.json")
    again = load_extension_task(written)
    assert again == task


def test_extension_task_wrong_schema_version_rejected():
    with pytest.raises(ExtensionTaskError, match="schema_version"):
        ExtensionTask.from_dict(_minimal_task_body(schema_version="v0"))


def test_extension_task_unknown_reason_rejected():
    with pytest.raises(ExtensionTaskError, match="reason"):
        ExtensionTask.from_dict(_minimal_task_body(reason="totally_made_up"))


def test_extension_task_untyped_allowed_extension_type_rejected():
    with pytest.raises(ExtensionTaskError, match="allowed_extension_types"):
        ExtensionTask.from_dict(
            _minimal_task_body(allowed_extension_types=["wave_hands"])
        )


def test_extension_task_required_fields_enforced():
    body = _minimal_task_body()
    body.pop("contract_hash")
    with pytest.raises(ExtensionTaskError, match="contract_hash"):
        ExtensionTask.from_dict(body)


def test_known_reasons_and_types_are_typed_enums():
    assert "unsupported_op" in EXTENSION_TASK_REASONS
    assert "provider_gap" in EXTENSION_TASK_REASONS
    assert "kernel_provider" in ALLOWED_EXTENSION_TASK_TYPES
    assert "pass_tool" in ALLOWED_EXTENSION_TASK_TYPES
    assert "totally_made_up" not in ALLOWED_EXTENSION_TASK_TYPES
