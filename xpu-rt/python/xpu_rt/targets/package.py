"""Target package -- the first-class unit CompGen generates per target.

A target package is NOT a full compiler. It is a **target enablement package**:
a verified, target-specific compilation stack configuration plus backend hooks
plus runtime plan templates.

Components of a target package:
    1. Target profile (hardware description)
    2. Capability spec (op-to-backend-lane mapping)
    3. Recipe library (transform families, tiling rules, placement policies)
    4. Target-specific IR (accel dialect skeleton if needed)
    5. Kernel implementation paths (Triton, accel, ukernel, vendor, fallback)
    6. Plan/runtime integration (placement, copy/sync, memory, bundle format)
    7. Verification package (structural checks, differential tests, CHECK files)

The target package progresses through maturity levels:
    L0 -> L1 -> L2 -> L3 (recognized -> correct -> optimized -> promoted)

For targets with existing Merlin/IREE/XLA backends, CompGen generates
the recipe/control layer and plugs it into the existing backend, NOT
a second copy of that backend.

Invariants:
    - A target package is a directory with a manifest.
    - Packages are self-contained (all paths relative to package root).
    - Generation is incremental (can re-generate parts without losing others).

TODO: Implement generate_target_package() with full scaffolding.
TODO: Implement load_target_package() from directory.
TODO: Implement validate_target_package() for completeness checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from compgen.targets.capability import CapabilitySpec, TargetClass
from compgen.targets.maturity import TargetMaturity
from compgen.targets.schema import TargetProfile


@dataclass(frozen=True)
class TargetPackageManifest:
    """Manifest for a target package directory.

    Attributes:
        package_version: Package format version.
        target_name: Target profile name.
        target_class: Target classification.
        maturity: Current maturity level.
        components: Dict mapping component name to relative path.
        existing_backend: If this target plugs into an existing backend (e.g., "merlin", "iree").
    """

    package_version: str = "1.0"
    target_name: str = ""
    target_class: TargetClass = TargetClass.TRITON_FRIENDLY
    maturity: TargetMaturity = TargetMaturity.L0_RECOGNIZED
    components: dict[str, str] = field(default_factory=dict)
    existing_backend: str | None = None


@dataclass
class TargetPackage:
    """A target enablement package.

    Attributes:
        root: Package root directory.
        manifest: Package manifest.
        profile: Target hardware profile.
        capabilities: Capability specification.
        maturity: Current maturity level.
    """

    root: Path
    manifest: TargetPackageManifest = field(default_factory=TargetPackageManifest)
    profile: TargetProfile | None = None
    capabilities: CapabilitySpec | None = None
    maturity: TargetMaturity = TargetMaturity.L0_RECOGNIZED


def generate_target_package(
    profile: TargetProfile,
    output_dir: str | Path,
    docs_dir: str | Path | None = None,
    existing_backend: str | None = None,
) -> TargetPackage:
    """Generate a target enablement package.

    Steps:
        1. Parse profile and infer capabilities
        2. Classify target (Triton-friendly / accel / ukernel / hybrid)
        3. Generate directory structure:
           - target_profile.yaml
           - capabilities.yaml
           - constraints.yaml
           - recipes/ (transform templates for this target class)
           - kernels/ (kernel search configs per backend lane)
           - ir/ (accel dialect skeleton if ACCEL_NATIVE or HYBRID)
           - verification/ (test corpus, CHECK files, golden harness)
           - runtime/ (driver config, planner constraints, bundle format)
        4. If existing_backend is set, generate integration layer instead of full backend
        5. Write manifest.json
        6. Assess maturity (should be L0)

    Args:
        profile: Target hardware profile.
        output_dir: Where to generate the package.
        docs_dir: Optional hardware documentation for agent builder integration.
        existing_backend: If set ("merlin", "iree", "xla"), generate plugin layer only.

    Returns:
        TargetPackage at L0 maturity.

    """
    import json

    import yaml

    from compgen.targets.capability import classify_target, infer_capabilities

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Classify and infer capabilities
    target_class = classify_target(profile)
    capabilities = infer_capabilities(profile)

    # Create directory structure
    subdirs = ["recipes", "kernels", "verification", "runtime"]
    if target_class in (TargetClass.ACCEL_NATIVE, TargetClass.HYBRID):
        subdirs.append("ir")
    for subdir in subdirs:
        (output / subdir).mkdir(exist_ok=True)

    # Write target_profile.yaml
    profile_data = {
        "name": profile.name,
        "schema_version": profile.schema_version,
        "devices": [
            {
                "device_type": d.device_type,
                "name": d.name,
                "vendor": d.vendor,
                "supported_ops": d.supported_ops,
                "features": d.features,
                "kernel_backends": d.kernel_backends,
            }
            for d in profile.devices
        ],
        "constraints": profile.constraints,
        "metadata": profile.metadata,
    }
    with open(output / "target_profile.yaml", "w") as f:
        yaml.dump(profile_data, f, default_flow_style=False)

    # Write capabilities.yaml
    caps_data = {
        "target_class": target_class.value,
        "default_lane": capabilities.default_lane.value,
        "ops": {
            name: {"preferred_lane": cap.preferred_lane.value, "supported_dtypes": sorted(cap.supported_dtypes)}
            for name, cap in capabilities.op_capabilities.items()
        },
    }
    with open(output / "capabilities.yaml", "w") as f:
        yaml.dump(caps_data, f, default_flow_style=False)

    # Build manifest
    components = {
        "target_profile": "target_profile.yaml",
        "capabilities": "capabilities.yaml",
        "recipes": "recipes/",
        "kernels": "kernels/",
        "verification": "verification/",
        "runtime": "runtime/",
    }
    if "ir" in subdirs:
        components["ir"] = "ir/"

    manifest = TargetPackageManifest(
        target_name=profile.name,
        target_class=target_class,
        maturity=TargetMaturity.L0_RECOGNIZED,
        components=components,
        existing_backend=existing_backend,
    )

    # Write manifest.json
    manifest_data = {
        "package_version": manifest.package_version,
        "target_name": manifest.target_name,
        "target_class": manifest.target_class.value,
        "maturity": manifest.maturity.name,
        "components": manifest.components,
        "existing_backend": manifest.existing_backend,
    }
    with open(output / "manifest.json", "w") as f:
        json.dump(manifest_data, f, indent=2)

    package = TargetPackage(
        root=output,
        manifest=manifest,
        profile=profile,
        capabilities=capabilities,
        maturity=TargetMaturity.L0_RECOGNIZED,
    )

    return package


def load_target_package(package_dir: str | Path) -> TargetPackage:
    """Load an existing target package from a directory."""
    import json

    from compgen.targets.capability import infer_capabilities
    from compgen.targets.schema import load_profile as _load_profile

    root = Path(package_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json in {root}")

    with open(manifest_path) as f:
        manifest_data = json.load(f)

    # Load profile
    profile_path = root / manifest_data.get("components", {}).get("target_profile", "target_profile.yaml")
    profile = _load_profile(profile_path) if profile_path.exists() else None

    # Infer capabilities
    capabilities = infer_capabilities(profile) if profile else None

    manifest = TargetPackageManifest(
        package_version=manifest_data.get("package_version", "1.0"),
        target_name=manifest_data.get("target_name", ""),
        target_class=TargetClass(manifest_data.get("target_class", "triton_friendly")),
        maturity=TargetMaturity[manifest_data.get("maturity", "L0_RECOGNIZED")],
        components=manifest_data.get("components", {}),
        existing_backend=manifest_data.get("existing_backend"),
    )

    return TargetPackage(
        root=root,
        manifest=manifest,
        profile=profile,
        capabilities=capabilities,
        maturity=manifest.maturity,
    )


def validate_target_package(package: TargetPackage) -> list[str]:
    """Validate a target package for completeness. Returns list of errors."""
    errors: list[str] = []

    if package.profile is None:
        errors.append("Profile is missing")
    if package.capabilities is None:
        errors.append("Capabilities are missing")

    # Check manifest components exist on disk
    for name, rel_path in package.manifest.components.items():
        full_path = package.root / rel_path
        if not full_path.exists():
            errors.append(f"Component '{name}' not found at {rel_path}")

    return errors


__all__ = [
    "TargetPackage",
    "TargetPackageManifest",
    "generate_target_package",
    "load_target_package",
    "validate_target_package",
]
