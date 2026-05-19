"""Extension-pack discovery, loading, and ownership helpers."""

from __future__ import annotations

from xpu_rt.packs.base import ExtensionPack, LoadedPack
from xpu_rt.packs.compose import ManifestExtensionPack
from xpu_rt.packs.envcheck import EnvCheckResult, check_pack_environment
from xpu_rt.packs.loader import load_manifest, load_pack, resolve_entry_point_target
from xpu_rt.packs.registry import (
    ENTRY_POINT_GROUP,
    ENV_VAR,
    PackRegistry,
    default_pack_root,
    discover_pack_paths,
    discover_packs,
    load_builtin_packs,
    load_discovered_packs,
)
from xpu_rt.packs.schema import (
    BranchPlan,
    ExtensionPackManifest,
    PackContextSummary,
    PackContribution,
    PackProbeResult,
)
from xpu_rt.packs.validate import PackValidationResult, validate_pack
from xpu_rt.packs.verify import OwnershipViolation, check_surface_allowed

__all__ = [
    "BranchPlan",
    "ENTRY_POINT_GROUP",
    "ENV_VAR",
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
    "discover_packs",
    "load_builtin_packs",
    "load_discovered_packs",
    "load_manifest",
    "load_pack",
    "resolve_entry_point_target",
    "validate_pack",
]
