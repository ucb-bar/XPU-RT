"""Tests for extension-pack discovery, loading, and branch planning."""

from __future__ import annotations

from pathlib import Path

from benchmarks.spec import WorkspaceConfig
from compgen.packs import default_pack_root, discover_pack_paths, load_builtin_packs, load_pack


def test_discover_builtin_pack_paths() -> None:
    names = {path.name for path in discover_pack_paths()}
    assert "cuda_tile" in names
    assert "snax_mlir" in names
    assert "gemmini_mx" in names


def test_load_builtin_pack_manifest() -> None:
    pack = load_pack(default_pack_root() / "cuda_tile")
    assert pack.manifest.name == "cuda_tile"
    assert "DialectPack" in pack.manifest.kinds
    assert "tile_dialect_semantics" in pack.manifest.sealed_surfaces


def test_probe_uses_workspace_pack_root(tmp_path: Path) -> None:
    source = tmp_path / "cuda-tile"
    source.mkdir()
    (source / "README.md").write_text("cuda tile")
    workspace = WorkspaceConfig(
        repo_root=tmp_path / "CompGen",
        pack_roots={"cuda_tile": source},
        integration_worktrees_root=tmp_path / "worktrees",
    )

    pack = load_pack(default_pack_root() / "cuda_tile")
    probe = pack.pack.probe(workspace)
    branch = pack.pack.branch_plan(workspace, run_id="smoke")

    assert probe.available
    assert probe.source_root == source.resolve()
    assert branch.branch_name == "compgen/integration/cuda_tile/smoke"
    assert branch.worktree_path == (tmp_path / "worktrees" / "cuda_tile" / "smoke")


def test_branch_plan_uses_llvm_fork_when_declared(tmp_path: Path) -> None:
    source = tmp_path / "gemmini"
    source.mkdir()
    (source / "README.md").write_text("gemmini")
    llvm_fork = tmp_path / "llvm-gemmini"
    llvm_fork.mkdir()
    workspace = WorkspaceConfig(
        repo_root=tmp_path / "CompGen",
        pack_roots={"gemmini_mx": source},
        llvm_forks={"gemmini": llvm_fork},
        integration_worktrees_root=tmp_path / "worktrees",
    )

    pack = load_pack(default_pack_root() / "gemmini_mx")
    branch = pack.pack.branch_plan(workspace, run_id="r1")
    assert branch.llvm_fork_path == llvm_fork.resolve()


def test_load_builtin_packs_includes_profiler_packs() -> None:
    names = {pack.manifest.name for pack in load_builtin_packs()}
    assert "iree_cpu_events" in names
    assert "iree_tracy" in names
