"""Tests for sealed-surface aperture enforcement via validate_pack."""

from __future__ import annotations

from pathlib import Path

from compgen.packs.base import LoadedPack
from compgen.packs.compose import ManifestExtensionPack
from compgen.packs.schema import ExtensionPackManifest
from compgen.packs.validate import validate_pack
from compgen.packs.verify import check_surface_allowed


def _make_pack(
    tmp_path: Path,
    name: str,
    *,
    owned_surfaces: tuple[str, ...] = (),
    sealed_surfaces: tuple[str, ...] = (),
) -> LoadedPack:
    """Helper to create a minimal LoadedPack rooted at *tmp_path*."""
    root = tmp_path / name
    root.mkdir(exist_ok=True)
    manifest = ExtensionPackManifest(
        name=name,
        version="0.1.0",
        kinds=("TargetPack",),
        owned_surfaces=owned_surfaces,
        sealed_surfaces=sealed_surfaces,
        source_root=str(root),
    )
    pack = ManifestExtensionPack(root=root, manifest=manifest)
    return LoadedPack(root=root, manifest=manifest, pack=pack)


def test_sealed_surface_blocks_owner(tmp_path: Path) -> None:
    """A pack that owns a surface sealed by another pack triggers a violation."""
    sealer = _make_pack(tmp_path, "sealer", sealed_surfaces=("tile_dialect_semantics",))
    owner = _make_pack(tmp_path, "owner", owned_surfaces=("tile_dialect_semantics",))

    result = validate_pack(owner, all_packs=[sealer, owner])
    assert result.ok is False
    assert len(result.aperture_violations) == 1
    assert result.aperture_violations[0].surface == "tile_dialect_semantics"
    assert result.aperture_violations[0].pack_name == "sealer"


def test_allowed_surface_passes(tmp_path: Path) -> None:
    """A pack that owns a surface no one seals passes validation."""
    other = _make_pack(tmp_path, "other", sealed_surfaces=("unrelated_surface",))
    owner = _make_pack(tmp_path, "owner", owned_surfaces=("my_surface",))

    result = validate_pack(owner, all_packs=[other, owner])
    assert result.ok is True
    assert result.aperture_violations == []


def test_check_surface_allowed_directly(tmp_path: Path) -> None:
    """Low-level check_surface_allowed detects a sealed surface."""
    sealer = _make_pack(tmp_path, "sealer", sealed_surfaces=("kernel_gen",))
    other = _make_pack(tmp_path, "other")

    violation = check_surface_allowed([sealer, other], requested_surface="kernel_gen")
    assert violation is not None
    assert violation.reason == "sealed_surface"

    no_violation = check_surface_allowed([sealer, other], requested_surface="open_surface")
    assert no_violation is None


def test_multiple_violations(tmp_path: Path) -> None:
    """A pack owning two surfaces sealed by different packs collects both violations."""
    sealer_a = _make_pack(tmp_path, "sealer_a", sealed_surfaces=("surface_x",))
    sealer_b = _make_pack(tmp_path, "sealer_b", sealed_surfaces=("surface_y",))
    owner = _make_pack(tmp_path, "owner", owned_surfaces=("surface_x", "surface_y"))

    result = validate_pack(owner, all_packs=[sealer_a, sealer_b, owner])
    assert result.ok is False
    assert len(result.aperture_violations) == 2
    surfaces = {v.surface for v in result.aperture_violations}
    assert surfaces == {"surface_x", "surface_y"}


def test_validation_combines_probe_and_env(tmp_path: Path) -> None:
    """validate_pack populates both probe and env_check fields."""
    pack = _make_pack(tmp_path, "combo")
    result = validate_pack(pack, required_tools=["python"])
    assert result.probe.pack_name == "combo"
    assert result.env_check.ok is True
    assert result.ok is True


def test_env_failure_fails_validation(tmp_path: Path) -> None:
    """A missing required tool causes the overall validation to fail."""
    pack = _make_pack(tmp_path, "envfail")
    result = validate_pack(pack, required_tools=["nonexistent_tool_xyz"])
    assert result.ok is False
    assert result.env_check.ok is False
    assert "nonexistent_tool_xyz" in result.env_check.missing_tools
