"""Tests for compgen.audit.fresh_agent task pack builder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compgen.audit.errors import TaskPackContaminated, TaskPackIncomplete
from compgen.audit.fresh_agent import (
    ALLOWLISTED_PATHS,
    FORBIDDEN_PATHS,
    REQUIRED_PATHS,
    build_task_pack,
    verify_task_pack,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_build_task_pack_smoke(tmp_path: Path) -> None:
    """Building a pack produces a manifest + the required paths."""
    out = tmp_path / "pack"
    pack = build_task_pack(
        out_dir=out, commit="testcommit",
        repo_root=REPO_ROOT, skip_python_package=True,
    )
    assert pack.manifest_path.exists()
    assert pack.task_prompt_path is not None
    assert pack.task_prompt_path.exists()
    # Required paths land in the pack
    for required in REQUIRED_PATHS:
        if required.startswith("python/compgen"):
            continue  # skipped by --skip-python-package in this test
        assert (out / required).exists(), f"{required} missing from pack"


def test_pack_contains_no_forbidden_paths(tmp_path: Path) -> None:
    """A built pack must not include any FORBIDDEN_PATHS."""
    out = tmp_path / "pack"
    build_task_pack(
        out_dir=out, commit="testcommit",
        repo_root=REPO_ROOT, skip_python_package=True,
    )
    # Walk and verify nothing matches forbidden globs
    import fnmatch as _fn

    for path in out.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(out).as_posix()
        for forbidden in FORBIDDEN_PATHS:
            assert not _fn.fnmatch(rel, forbidden), (
                f"forbidden file {rel} matched {forbidden}"
            )


def test_pack_does_not_include_claude_projects(tmp_path: Path) -> None:
    """The single most important exclusion: ``.claude/projects/``."""
    out = tmp_path / "pack"
    build_task_pack(
        out_dir=out, commit="testcommit",
        repo_root=REPO_ROOT, skip_python_package=True,
    )
    # Walk for any file under .claude/projects (which holds private
    # session memory / chat transcripts)
    leaks = list((out / ".claude" / "projects").rglob("*")) if (out / ".claude").exists() else []
    leaks = [p for p in leaks if p.is_file()]
    assert not leaks, f"task pack leaked private chat memory: {leaks}"


def test_pack_excludes_results_and_caches(tmp_path: Path) -> None:
    out = tmp_path / "pack"
    build_task_pack(
        out_dir=out, commit="testcommit",
        repo_root=REPO_ROOT, skip_python_package=True,
    )
    forbidden_dirs = (".compgen_cache", ".crg-artifacts", "results", "tmp")
    for d in forbidden_dirs:
        assert not (out / d).exists(), f"task pack includes forbidden dir {d}"


def test_verify_task_pack_round_trip(tmp_path: Path) -> None:
    out = tmp_path / "pack"
    pack = build_task_pack(
        out_dir=out, commit="abc123",
        repo_root=REPO_ROOT, skip_python_package=True,
    )
    re = verify_task_pack(out)
    assert re.commit == pack.commit
    assert re.files_copied == pack.files_copied


def test_missing_required_file_raises(tmp_path: Path) -> None:
    """If a required file gets deleted from the pack, verify raises."""
    out = tmp_path / "pack"
    build_task_pack(
        out_dir=out, commit="abc",
        repo_root=REPO_ROOT, skip_python_package=True,
    )
    # Delete a required file
    (out / "CLAUDE.md").unlink()
    with pytest.raises(TaskPackIncomplete, match="CLAUDE.md"):
        verify_task_pack(out)


def test_contaminated_pack_raises(tmp_path: Path) -> None:
    """If a forbidden file gets injected, verify raises."""
    out = tmp_path / "pack"
    build_task_pack(
        out_dir=out, commit="abc",
        repo_root=REPO_ROOT, skip_python_package=True,
    )
    # Inject a forbidden file
    bad_dir = out / ".claude" / "projects" / "fake-session"
    bad_dir.mkdir(parents=True, exist_ok=True)
    (bad_dir / "memory.md").write_text("# private chat memory")
    with pytest.raises(TaskPackContaminated, match="forbidden"):
        verify_task_pack(out)


def test_realness_contracts_present_in_pack(tmp_path: Path) -> None:
    """The pack must include the seed realness contracts."""
    out = tmp_path / "pack"
    build_task_pack(
        out_dir=out, commit="abc",
        repo_root=REPO_ROOT, skip_python_package=True,
    )
    realness = out / "docs" / "realness"
    assert realness.is_dir()
    contracts = list(realness.glob("*.yaml"))
    assert len(contracts) >= 6, f"expected ≥6 realness contracts, got {len(contracts)}"


def test_holdout_configs_present_in_pack(tmp_path: Path) -> None:
    """The pack must include the holdout YAML configs (for the task)."""
    out = tmp_path / "pack"
    build_task_pack(
        out_dir=out, commit="abc",
        repo_root=REPO_ROOT, skip_python_package=True,
    )
    holdout_yaml = out / "configs" / "models" / "holdout_mlp_odd_shapes.yaml"
    assert holdout_yaml.exists()


def test_pack_manifest_records_metadata(tmp_path: Path) -> None:
    out = tmp_path / "pack"
    pack = build_task_pack(
        out_dir=out, commit="abc1234",
        repo_root=REPO_ROOT, skip_python_package=True,
    )
    raw = json.loads(pack.manifest_path.read_text())
    assert raw["commit"] == "abc1234"
    assert raw["files_copied"] > 0
    assert raw["bytes_copied"] > 0
    assert raw["task_prompt_path"]
