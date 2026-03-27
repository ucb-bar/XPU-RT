"""Tests for the benchmark registry and workspace config."""

from __future__ import annotations

from pathlib import Path

from benchmarks.registry import build_default_registry
from benchmarks.spec import WorkspaceConfig


def test_workspace_default_external_resolution() -> None:
    ws = WorkspaceConfig.default("/tmp/compgen")
    resolved = ws.resolve_external("iree")
    assert resolved == Path("/tmp/iree")


def test_workspace_explicit_external_resolution(tmp_path: Path) -> None:
    ws = WorkspaceConfig(repo_root=tmp_path / "CompGen", external_roots={"xla": tmp_path / "custom-xla"})
    assert ws.resolve_external("xla") == tmp_path / "custom-xla"


def test_default_registry_contains_paper_subset() -> None:
    registry = build_default_registry()
    assert "paper_subset" in registry.studies
    assert "cuda_a100" in registry.targets
    assert "riscv_soc" in registry.targets
    assert "multi_device" in registry.targets
    assert "BundleT" in registry.bundles
    assert "BundleM" in registry.bundles
    assert "simple_mlp" in registry.workloads
    assert "transformer_block" in registry.workloads
    assert "quantized_mlp" in registry.workloads


def test_verification_red_team_defects_registered() -> None:
    registry = build_default_registry()
    assert "wrong_tile_sizes" in registry.defects
    assert "numerically_wrong_kernel" in registry.defects
