"""Tests for ``compgen scaffold-pack`` and :func:`scaffold_pack`."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner
from compgen.cli import main
from compgen.packs import load_pack
from compgen.packs.scaffolding import SUPPORTED_KINDS, scaffold_pack


@pytest.mark.parametrize("kind", SUPPORTED_KINDS)
def test_scaffold_pack_writes_complete_skeleton(tmp_path: Path, kind: str) -> None:
    result = scaffold_pack(kind=kind, name="my_pack", out_dir=tmp_path)

    assert result.pack_root == tmp_path / "my_pack"
    assert (result.pack_root / "pyproject.toml").exists()
    assert (result.pack_root / "README.md").exists()
    assert (result.package_root / "__init__.py").exists()
    assert (result.package_root / "manifest.yaml").exists()
    assert result.scheme_path.exists()
    assert result.scheme_path.parent == result.package_root


def test_scaffolded_manifest_is_valid_yaml(tmp_path: Path) -> None:
    result = scaffold_pack(kind="quantization", name="my_fp4", out_dir=tmp_path)
    data = yaml.safe_load(result.manifest_path.read_text())
    assert data["name"] == "my_fp4"
    assert data["kinds"] == ["KernelPack"]
    assert data["source_root"] == "src/my_fp4"


def test_scaffolded_pyproject_declares_entry_point(tmp_path: Path) -> None:
    result = scaffold_pack(kind="provider", name="my_provider", out_dir=tmp_path)
    content = result.pyproject_path.read_text()
    assert '[project.entry-points."compgen.packs"]' in content
    assert 'my_provider = "my_provider"' in content


def test_scaffolded_package_exposes_pack_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = scaffold_pack(kind="dialect", name="my_dial", out_dir=tmp_path)
    # Make the scaffolded src directory importable
    monkeypatch.syspath_prepend(str(result.pack_root / "src"))
    module = __import__("my_dial")
    try:
        assert isinstance(module.PACK_ROOT, Path)
        assert (module.PACK_ROOT / "manifest.yaml").exists()
    finally:
        sys.modules.pop("my_dial", None)


def test_scaffold_then_load_pack_by_entry_point_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scaffold_pack(kind="quantization", name="my_fp8", out_dir=tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path / "my_fp8" / "src"))

    # Entry-point value form: resolves module → __file__.parent → pack root
    loaded = load_pack("my_fp8")
    try:
        assert loaded.manifest.name == "my_fp8"
        assert "KernelPack" in loaded.manifest.kinds
    finally:
        sys.modules.pop("my_fp8", None)


def test_scaffold_rejects_invalid_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        scaffold_pack(kind="quantization", name="1bad", out_dir=tmp_path)
    with pytest.raises(ValueError):
        scaffold_pack(kind="quantization", name="my-pack", out_dir=tmp_path)
    with pytest.raises(ValueError):
        scaffold_pack(kind="quantization", name="class", out_dir=tmp_path)


def test_scaffold_rejects_unknown_kind(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        scaffold_pack(kind="bogus", name="mypack", out_dir=tmp_path)


def test_scaffold_existing_directory_without_overwrite_errors(tmp_path: Path) -> None:
    scaffold_pack(kind="quantization", name="my_fp8", out_dir=tmp_path)
    with pytest.raises(FileExistsError):
        scaffold_pack(kind="quantization", name="my_fp8", out_dir=tmp_path)


def test_scaffold_overwrite_replaces_existing(tmp_path: Path) -> None:
    first = scaffold_pack(kind="quantization", name="my_fp8", out_dir=tmp_path)
    # Drop a sentinel file that should be erased by overwrite
    sentinel = first.pack_root / "LEGACY.txt"
    sentinel.write_text("old")
    second = scaffold_pack(kind="quantization", name="my_fp8", out_dir=tmp_path, overwrite=True)
    assert not sentinel.exists()
    assert second.pack_root == first.pack_root


def test_cli_scaffold_pack_happy_path(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["scaffold-pack", "--kind", "provider", "--name", "my_prov", "--out", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "my_prov" / "pyproject.toml").exists()
    assert (tmp_path / "my_prov" / "src" / "my_prov" / "manifest.yaml").exists()


def test_cli_scaffold_pack_bad_kind_errors(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["scaffold-pack", "--kind", "bogus", "--name", "mypack", "--out", str(tmp_path)],
    )
    assert result.exit_code != 0
