"""Extension-pack discovery, loading, and ownership helpers."""

from __future__ import annotations

from compgen.packs.base import ExtensionPack, LoadedPack
from compgen.packs.compose import ManifestExtensionPack
from compgen.packs.envcheck import EnvCheckResult, check_pack_environment
from compgen.packs.loader import load_manifest, load_pack
from compgen.packs.registry import PackRegistry, default_pack_root, discover_pack_paths, load_builtin_packs
from compgen.packs.schema import (
    BranchPlan,
    ExtensionPackManifest,
    PackContextSummary,
    PackContribution,
    PackProbeResult,
)
from compgen.packs.validate import PackValidationResult, validate_pack
from compgen.packs.verify import OwnershipViolation, check_surface_allowed

__all__ = [
    "BranchPlan",
    "EnvCheckResult",
    "ExtensionPack",
    "ExtensionPackManifest",
    "LoadedPack",
    "ManifestExtensionPack",
    "OwnershipViolation",
    "PackContextSummary",
    "PackContribution",
    "PackProbeResult",
    "PackRegistry",
    "PackValidationResult",
    "check_pack_environment",
    "check_surface_allowed",
    "default_pack_root",
    "discover_pack_paths",
    "load_builtin_packs",
    "load_manifest",
    "load_pack",
    "validate_pack",
]
