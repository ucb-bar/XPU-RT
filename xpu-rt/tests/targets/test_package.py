"""Tests for target package generation."""

from __future__ import annotations

from pathlib import Path

from compgen.targets.capability import TargetClass
from compgen.targets.maturity import TargetMaturity
from compgen.targets.package import (
    TargetPackageManifest,
    generate_target_package,
    load_target_package,
    validate_target_package,
)
from compgen.targets.schema import load_profile

PROFILES_DIR = Path(__file__).parent.parent.parent / "examples" / "target_profiles"


def test_manifest_defaults() -> None:
    m = TargetPackageManifest()
    assert m.package_version == "1.0"
    assert m.existing_backend is None


def test_manifest_existing_backend() -> None:
    m = TargetPackageManifest(target_name="merlin-target", existing_backend="merlin")
    assert m.existing_backend == "merlin"


def test_generate_triton_target_package(tmp_path: Path) -> None:
    """Generating a Triton-friendly target should produce a valid L0 package."""
    profile = load_profile(PROFILES_DIR / "cuda_a100.yaml")
    pkg = generate_target_package(profile, tmp_path / "cuda_pkg")

    assert pkg.maturity == TargetMaturity.L0_RECOGNIZED
    assert pkg.manifest.target_class == TargetClass.TRITON_FRIENDLY
    assert (pkg.root / "manifest.json").exists()
    assert (pkg.root / "target_profile.yaml").exists()
    assert (pkg.root / "capabilities.yaml").exists()
    assert (pkg.root / "recipes").is_dir()
    assert (pkg.root / "kernels").is_dir()
    assert (pkg.root / "verification").is_dir()

    errors = validate_target_package(pkg)
    assert errors == []


def test_generate_accel_target_package(tmp_path: Path) -> None:
    """Accelerator target should get an ir/ subdirectory."""
    profile = load_profile(PROFILES_DIR / "trainium1.yaml")
    pkg = generate_target_package(profile, tmp_path / "trn_pkg")

    assert pkg.manifest.target_class == TargetClass.ACCEL_NATIVE
    assert (pkg.root / "ir").is_dir()


def test_generate_with_existing_backend(tmp_path: Path) -> None:
    """Generating with existing_backend should note it in the manifest."""
    profile = load_profile(PROFILES_DIR / "cuda_a100.yaml")
    pkg = generate_target_package(profile, tmp_path / "iree_pkg", existing_backend="iree")

    assert pkg.manifest.existing_backend == "iree"


def test_load_target_package(tmp_path: Path) -> None:
    """Loading a generated package should reconstruct TargetPackage."""
    profile = load_profile(PROFILES_DIR / "cuda_a100.yaml")
    pkg_dir = tmp_path / "load_test"
    generate_target_package(profile, pkg_dir)

    loaded = load_target_package(pkg_dir)
    assert loaded.manifest.target_name == "cuda-a100"
    assert loaded.profile is not None
    assert loaded.capabilities is not None
