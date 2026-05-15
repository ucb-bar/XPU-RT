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
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from compgen.packs import LoadedPack, PackContextSummary, default_pack_root, load_pack
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
    composed_from_packs: tuple[str, ...] = ()
    sealed_surfaces: tuple[str, ...] = ()
    generation_apertures: tuple[str, ...] = ()
    integration_artifacts: dict[str, dict[str, str]] = field(default_factory=dict)


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
    extension_packs: tuple[LoadedPack, ...] = ()
    owned_surfaces: tuple[str, ...] = ()
    sealed_surfaces: tuple[str, ...] = ()
    generation_apertures: tuple[str, ...] = ()
    integration_artifacts: dict[str, dict[str, str]] = field(default_factory=dict)

    def pack_context(self) -> PackContextSummary:
        """Summarize the active pack ownership context."""

        available_profilers = sorted(
            {profiler for pack in self.extension_packs for profiler in pack.manifest.available_profilers}
        )
        benchmark_targets = sorted(
            {target for pack in self.extension_packs for target in pack.manifest.benchmark_targets}
        )
        integration_branches = [pack.pack.branch_plan(run_id="package").branch_name for pack in self.extension_packs]
        return PackContextSummary(
            active_packs=tuple(pack.manifest.name for pack in self.extension_packs),
            sealed_surfaces=self.sealed_surfaces,
            generation_apertures=self.generation_apertures,
            available_profilers=tuple(available_profilers),
            benchmark_targets=tuple(benchmark_targets),
            integration_branch=",".join(integration_branches),
        )


def _coerce_loaded_pack(spec: str | Path | LoadedPack) -> LoadedPack:
    if isinstance(spec, LoadedPack):
        return spec
    candidate = Path(spec)
    if candidate.exists():
        return load_pack(candidate)
    # Try the repo-local builtin pack directory first, then fall back to
    # entry-point / import resolution (``load_pack`` handles both).
    builtin_root = default_pack_root() / str(spec)
    if (builtin_root / "manifest.yaml").exists():
        return load_pack(builtin_root)
    return load_pack(str(spec))


def _coerce_loaded_packs(extension_packs: Iterable[str | Path | LoadedPack] | None) -> tuple[LoadedPack, ...]:
    if not extension_packs:
        return ()
    return tuple(_coerce_loaded_pack(spec) for spec in extension_packs)


def generate_target_package(
    profile: TargetProfile,
    output_dir: str | Path,
    docs_dir: str | Path | None = None,
    existing_backend: str | None = None,
    extension_packs: Iterable[str | Path | LoadedPack] | None = None,
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
    loaded_packs = _coerce_loaded_packs(extension_packs)

    # Classify and infer capabilities
    target_class = classify_target(profile)
    capabilities = infer_capabilities(profile)

    # Create directory structure
    subdirs = ["recipes", "kernels", "verification", "runtime"]
    if target_class in (TargetClass.ACCEL_NATIVE, TargetClass.HYBRID):
        subdirs.append("ir")
    for subdir in subdirs:
        (output / subdir).mkdir(exist_ok=True)
    if loaded_packs:
        (output / "packs").mkdir(exist_ok=True)

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

    pack_artifacts: dict[str, dict[str, str]] = {}
    owned_surfaces = sorted({surface for pack in loaded_packs for surface in pack.manifest.owned_surfaces})
    sealed_surfaces = sorted({surface for pack in loaded_packs for surface in pack.manifest.sealed_surfaces})
    generation_apertures = sorted(
        {aperture for pack in loaded_packs for aperture in pack.manifest.generation_apertures}
    )

    for pack in loaded_packs:
        pack_dir = output / "packs" / pack.manifest.name
        pack_dir.mkdir(parents=True, exist_ok=True)
        manifest_copy = pack_dir / "manifest.yaml"
        manifest_copy.write_text((pack.root / "manifest.yaml").read_text())
        components[f"pack:{pack.manifest.name}"] = str(Path("packs") / pack.manifest.name / "manifest.yaml")
        pack_artifacts[pack.manifest.name] = {
            "manifest": str(Path("packs") / pack.manifest.name / "manifest.yaml"),
            "integration_mode": pack.manifest.integration_mode,
            "reference_runner": pack.manifest.reference_runner,
        }
        if pack.manifest.llvm_fork_key:
            pack_artifacts[pack.manifest.name]["llvm_fork_key"] = pack.manifest.llvm_fork_key

    manifest = TargetPackageManifest(
        target_name=profile.name,
        target_class=target_class,
        maturity=TargetMaturity.L0_RECOGNIZED,
        components=components,
        existing_backend=existing_backend,
        composed_from_packs=tuple(pack.manifest.name for pack in loaded_packs),
        sealed_surfaces=tuple(sealed_surfaces),
        generation_apertures=tuple(generation_apertures),
        integration_artifacts=pack_artifacts,
    )

    # Write manifest.json
    manifest_data = {
        "package_version": manifest.package_version,
        "target_name": manifest.target_name,
        "target_class": manifest.target_class.value,
        "maturity": manifest.maturity.name,
        "components": manifest.components,
        "existing_backend": manifest.existing_backend,
        "composed_from_packs": list(manifest.composed_from_packs),
        "sealed_surfaces": list(manifest.sealed_surfaces),
        "generation_apertures": list(manifest.generation_apertures),
        "integration_artifacts": manifest.integration_artifacts,
    }
    with open(output / "manifest.json", "w") as f:
        json.dump(manifest_data, f, indent=2)

    package = TargetPackage(
        root=output,
        manifest=manifest,
        profile=profile,
        capabilities=capabilities,
        maturity=TargetMaturity.L0_RECOGNIZED,
        extension_packs=loaded_packs,
        owned_surfaces=tuple(owned_surfaces),
        sealed_surfaces=tuple(sealed_surfaces),
        generation_apertures=tuple(generation_apertures),
        integration_artifacts=pack_artifacts,
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

    pack_names = tuple(manifest_data.get("composed_from_packs", ()))
    loaded_packs = tuple(
        load_pack(root / "packs" / pack_name)
        for pack_name in pack_names
        if (root / "packs" / pack_name / "manifest.yaml").exists()
    )

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
        composed_from_packs=pack_names,
        sealed_surfaces=tuple(manifest_data.get("sealed_surfaces", ())),
        generation_apertures=tuple(manifest_data.get("generation_apertures", ())),
        integration_artifacts=dict(manifest_data.get("integration_artifacts", {})),
    )

    return TargetPackage(
        root=root,
        manifest=manifest,
        profile=profile,
        capabilities=capabilities,
        maturity=manifest.maturity,
        extension_packs=loaded_packs,
        owned_surfaces=tuple(sorted({surface for pack in loaded_packs for surface in pack.manifest.owned_surfaces})),
        sealed_surfaces=manifest.sealed_surfaces,
        generation_apertures=manifest.generation_apertures,
        integration_artifacts=manifest.integration_artifacts,
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

    if package.manifest.composed_from_packs and not package.extension_packs:
        errors.append("Manifest declares composed packs but none were loaded")

    return errors


__all__ = [
    "TargetPackage",
    "TargetPackageManifest",
    "generate_target_package",
    "load_target_package",
    "validate_target_package",
]
