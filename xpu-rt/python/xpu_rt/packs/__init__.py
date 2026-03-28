"""Extension-pack discovery, loading, and ownership helpers."""

from __future__ import annotations

from compgen.packs.base import ExtensionPack, LoadedPack
from compgen.packs.compose import ManifestExtensionPack
from compgen.packs.loader import load_manifest, load_pack
from compgen.packs.registry import PackRegistry, default_pack_root, discover_pack_paths, load_builtin_packs
from compgen.packs.schema import (
    BranchPlan,
    ExtensionPackManifest,
    PackContextSummary,
    PackContribution,
    PackProbeResult,
)
from compgen.packs.verify import OwnershipViolation, check_surface_allowed

__all__ = [
    "BranchPlan",
    "ExtensionPack",
    "ExtensionPackManifest",
    "LoadedPack",
    "ManifestExtensionPack",
    "OwnershipViolation",
    "PackContextSummary",
    "PackContribution",
    "PackProbeResult",
    "PackRegistry",
    "check_surface_allowed",
    "default_pack_root",
    "discover_pack_paths",
    "load_builtin_packs",
    "load_manifest",
    "load_pack",
]
