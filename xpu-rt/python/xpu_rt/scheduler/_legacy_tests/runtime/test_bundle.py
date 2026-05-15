"""Tests for artifact bundling."""

from __future__ import annotations

from pathlib import Path

import pytest

xdsl_builtin = pytest.importorskip("xdsl.dialects.builtin")
ModuleOp = xdsl_builtin.ModuleOp

from compgen.runtime.bundle import BundleBuilder, BundleManifest, create_bundle
from compgen.runtime.planner import ExecutionPlan, MemoryPlan


def test_bundle_manifest_defaults() -> None:
    """BundleManifest should have sensible defaults."""
    manifest = BundleManifest()
    assert manifest.version == "1.0"
    assert manifest.artifacts == {}


def test_bundle_builder_creates_manifest(tmp_path: Path) -> None:
    """BundleBuilder should create a manifest.json with the core artifact contract."""
    builder = BundleBuilder(output_dir=tmp_path / "bundle")
    manifest = builder.build(
        module=ModuleOp([]),
        execution_plan=ExecutionPlan(memory_plans=[MemoryPlan(device_index=0, peak_bytes=1024)]),
        target_name="cuda-a100",
        exported_program_text="graph()",
        recipe_mlir_text="module { }",
        recipe_yaml_text="- _op: recipe.region",
        kernel_contracts=[{"index": 0, "type": "Contract"}],
        verification_report={"passed": True},
    )

    assert (tmp_path / "bundle" / "manifest.json").exists()
    assert "payload" in manifest.artifacts
    assert "execution_plan" in manifest.artifacts
    assert "memory_plan" in manifest.artifacts
    assert "recipe_mlir" in manifest.artifacts
    assert "verification_report" in manifest.artifacts


def test_bundle_integrity(tmp_path: Path) -> None:
    """Bundle should contain all referenced artifacts."""
    manifest = create_bundle(
        output_dir=tmp_path / "bundle",
        module=ModuleOp([]),
        execution_plan=ExecutionPlan(memory_plans=[MemoryPlan(device_index=0, peak_bytes=1024)]),
        target_name="test-target",
        exported_program_text="graph()",
        recipe_mlir_text="module { }",
        recipe_yaml_text="- _op: recipe.region",
        kernel_contracts=[{"index": 0, "type": "Contract"}],
        verification_report={"passed": True},
    )

    bundle_root = tmp_path / "bundle"
    for relative in manifest.artifacts.values():
        artifact_path = bundle_root / relative
        if relative.endswith("/"):
            assert artifact_path.is_dir()
        else:
            assert artifact_path.exists(), relative


def test_bundle_carries_lowered_recipe_metadata_for_promotion() -> None:
    """The agent loop's promotion path stuffs recipe-lowering outputs
    (transform_scripts / kernel_jobs / plan_fragments) into the bundle's
    ``metadata`` dict — those are JSON-safe and not fields of
    ``BundleManifest`` itself. Pre-fix this raised
    ``TypeError: BundleManifest.__init__() got an unexpected keyword
    argument 'transform_scripts'`` which the surrounding try/except in
    ``compgen.agent.loop.core`` silently swallowed, so the promotion
    path was effectively dead.
    """
    import json

    bundle = BundleManifest(
        target_profile="test-target",
        model_hash="abc123",
        objective="latency",
        metadata={
            "transform_scripts": ["%0 = transform.match ..."],
            "kernel_jobs": [{"name": "matmul_kernel"}],
            "plan_fragments": [{"placement": "device:0"}],
        },
    )
    serialized = json.dumps(bundle.to_dict())
    parsed = json.loads(serialized)
    assert parsed["metadata"]["transform_scripts"] == ["%0 = transform.match ..."]
    assert parsed["metadata"]["kernel_jobs"][0]["name"] == "matmul_kernel"
