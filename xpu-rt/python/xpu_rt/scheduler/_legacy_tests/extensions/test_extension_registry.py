"""extension discovery + per-run registration."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from compgen.extensions.errors import ExtensionSandboxViolation
from compgen.extensions.manifest import (
    ExtensionManifest,
    MANIFEST_SCHEMA_VERSION,
)
from compgen.extensions.registry import (
    EXTENSION_MANIFEST_FILENAME,
    ExtensionRegistry,
    assert_artifact_write_allowed,
    build_registry,
    discover_manifests,
    is_artifact_write_allowed,
)


def _manifest_body(extension_id: str = "myaccel", **overrides) -> dict:
    body = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "extension": {
            "id": extension_id,
            "version": "0.1.0",
            "author": "tester",
            "description": "test ext",
        },
        "provides": {
            "targets": [
                {
                    "schema_version": "target_card_v1",
                    "target_id": f"{extension_id}_v1",
                    "family": "custom_accelerator",
                    "vendor": "tester",
                    "dispatch_modes": ["sync"],
                    "memory_tiers": [{"name": "dram", "kind": "global"}],
                }
            ],
            "kernel_providers": [
                {
                    "schema_version": "provider_card_v1",
                    "provider_id": f"{extension_id}_c",
                    "integration_level": "probe",
                    "target_families": ["custom_accelerator"],
                    "contract_kinds": ["matmul"],
                    "emits": ["c_source"],
                    "entrypoint": f"{extension_id}.provider:Provider",
                }
            ],
            "dialect_providers": [],
            "pass_tools": [],
        },
        "probes": {
            "required_env": [],
            "commands": [],
            "python_imports": [],
        },
        "security": {
            "sandbox_required": True,
            "allowed_write_root": ".",
        },
        "verification": {"required_checks": []},
    }
    body.update(overrides)
    return body


def _drop(tmp_path: Path, ext_id: str, body: dict | None = None) -> Path:
    """Write a manifest at ``tmp_path / ext_id / compgen_extension.yaml``."""

    body = body or _manifest_body(ext_id)
    ext_dir = tmp_path / ext_id
    ext_dir.mkdir(parents=True, exist_ok=True)
    p = ext_dir / EXTENSION_MANIFEST_FILENAME
    p.write_text(yaml.safe_dump(body))
    return ext_dir


def test_discover_manifests_yields_each_extension(tmp_path: Path):
    _drop(tmp_path, "alpha")
    _drop(tmp_path, "beta")
    _drop(tmp_path, "gamma")
    paths = list(discover_manifests(tmp_path))
    assert len(paths) == 3
    assert all(p.name == EXTENSION_MANIFEST_FILENAME for p in paths)


def test_discover_missing_root_returns_no_results(tmp_path: Path):
    paths = list(discover_manifests(tmp_path / "does_not_exist"))
    assert paths == []


def test_discover_ignores_directories_without_manifest(tmp_path: Path):
    (tmp_path / "no_manifest").mkdir()
    (tmp_path / "no_manifest" / "README.md").write_text("hello")
    _drop(tmp_path, "real_ext")
    paths = list(discover_manifests(tmp_path))
    assert len(paths) == 1
    assert paths[0].parent.name == "real_ext"


def test_build_registry_accepts_valid_manifest(tmp_path: Path):
    _drop(tmp_path, "alpha")
    registry = build_registry(tmp_path)
    assert isinstance(registry, ExtensionRegistry)
    assert registry.extension_ids() == ("alpha",)
    assert "alpha_c" in registry.provider_ids()
    assert "alpha_v1" in registry.target_ids()
    assert registry.rejected == []


def test_build_registry_rejects_malformed_manifest(tmp_path: Path):
    bad = _manifest_body("bad")
    bad["schema_version"] = "v999"
    _drop(tmp_path, "bad_ext", bad)
    registry = build_registry(tmp_path)
    assert registry.accepted == []
    assert len(registry.rejected) == 1
    assert registry.rejected[0].failed_check == "manifest_schema"


def test_build_registry_rejects_sandbox_escape(tmp_path: Path):
    body = _manifest_body("escapee")
    body["security"]["allowed_write_root"] = "../../somewhere_else"
    _drop(tmp_path, "escapee", body)
    registry = build_registry(tmp_path)
    assert registry.accepted == []
    assert len(registry.rejected) == 1
    r = registry.rejected[0]
    assert r.failed_check == "extension_sandbox_violation"


def test_build_registry_rejects_duplicate_extension_id(tmp_path: Path):
    _drop(tmp_path, "dup_a", _manifest_body("dup"))
    _drop(tmp_path, "dup_b", _manifest_body("dup"))
    registry = build_registry(tmp_path)
    assert registry.extension_ids() == ("dup",)
    assert len(registry.rejected) == 1
    assert registry.rejected[0].failed_check == "duplicate_extension_id"


def test_per_run_registry_is_fresh_each_call(tmp_path: Path):
    _drop(tmp_path, "alpha")
    r1 = build_registry(tmp_path)
    r2 = build_registry(tmp_path)
    assert r1 is not r2
    assert r1.accepted is not r2.accepted


def test_artifact_write_allowed_within_extension_dir(tmp_path: Path):
    ext_dir = _drop(tmp_path, "alpha")
    registry = build_registry(tmp_path)
    [manifest] = registry.accepted
    p = ext_dir / "kernel.c"
    resolved = assert_artifact_write_allowed(p, manifest)
    assert resolved == p.resolve()


def test_artifact_write_outside_extension_dir_rejected(tmp_path: Path):
    _drop(tmp_path, "alpha")
    registry = build_registry(tmp_path)
    [manifest] = registry.accepted
    target = tmp_path / "sibling" / "kernel.c"
    with pytest.raises(ExtensionSandboxViolation) as exc:
        assert_artifact_write_allowed(target, manifest)
    assert exc.value.reason == "escapes_allowed_root"


def test_artifact_write_forbidden_filename_rejected(tmp_path: Path):
    ext_dir = _drop(tmp_path, "alpha")
    registry = build_registry(tmp_path)
    [manifest] = registry.accepted
    with pytest.raises(ExtensionSandboxViolation) as exc:
        assert_artifact_write_allowed(ext_dir / "payload.mlir", manifest)
    assert exc.value.reason.startswith("forbidden_filename:")


def test_is_artifact_write_allowed_non_raising(tmp_path: Path):
    ext_dir = _drop(tmp_path, "alpha")
    registry = build_registry(tmp_path)
    [manifest] = registry.accepted
    assert is_artifact_write_allowed(ext_dir / "kernel.c", manifest) is True
    assert is_artifact_write_allowed(tmp_path / "outside.c", manifest) is False


def test_rejected_summary_is_typed(tmp_path: Path):
    bad = _manifest_body("bad")
    bad["schema_version"] = "v999"
    _drop(tmp_path, "bad_ext", bad)
    registry = build_registry(tmp_path)
    summary = registry.rejected_summary()
    assert len(summary) == 1
    entry = summary[0]
    assert set(entry.keys()) == {"extension_dir", "failed_check", "detail"}
    assert entry["failed_check"] == "manifest_schema"


def test_build_registry_accepts_explicit_manifests_list(tmp_path: Path):
    body = _manifest_body("programmatic")
    body["security"]["sandbox_required"] = False
    manifest = ExtensionManifest.from_dict(
        body, source=tmp_path / "programmatic" / EXTENSION_MANIFEST_FILENAME
    )
    registry = build_registry(manifests=[manifest])
    assert registry.extension_ids() == ("programmatic",)
