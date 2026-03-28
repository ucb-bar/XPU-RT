"""Manifest loading and declarative pack resolution."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import yaml

from compgen.packs.base import ExtensionPack, LoadedPack
from compgen.packs.compose import ManifestExtensionPack
from compgen.packs.schema import ExtensionPackManifest


def _tuple_strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    return (str(value),)


def _manifest_from_data(data: dict[str, Any]) -> ExtensionPackManifest:
    return ExtensionPackManifest(
        name=str(data.get("name", "")),
        version=str(data.get("version", "0.1.0")),
        kinds=tuple(data.get("kinds", ())),
        owned_surfaces=_tuple_strings(data.get("owned_surfaces")),
        sealed_surfaces=_tuple_strings(data.get("sealed_surfaces")),
        generation_apertures=_tuple_strings(data.get("generation_apertures")),
        integration_mode=str(data.get("integration_mode", "readonly")),
        benchmark_suite=str(data.get("benchmark_suite", "pack_integrations")),
        benchmark_targets=_tuple_strings(data.get("benchmark_targets")),
        reference_runner=str(data.get("reference_runner", "")),
        source_root=str(data.get("source_root", "")),
        workspace_keys=_tuple_strings(data.get("workspace_keys")),
        third_party_names=_tuple_strings(data.get("third_party_names")),
        expected_files=_tuple_strings(data.get("expected_files")),
        available_profilers=_tuple_strings(data.get("available_profilers")),
        llvm_fork_key=str(data.get("llvm_fork_key", "")),
        entry_module=str(data.get("entry_module", "")),
        metadata=dict(data.get("metadata", {})),
    )


def load_manifest(path: str | Path) -> ExtensionPackManifest:
    """Load only the manifest data from a YAML file."""

    payload = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Pack manifest must be a mapping: {path}")
    return _manifest_from_data(payload)


def _load_declared_pack(root: Path, manifest: ExtensionPackManifest) -> ExtensionPack:
    if not manifest.entry_module:
        return ManifestExtensionPack(root=root, manifest=manifest)

    module = importlib.import_module(manifest.entry_module)
    if hasattr(module, "build_pack"):
        return module.build_pack(root=root, manifest=manifest)
    if hasattr(module, "PACK"):
        pack = getattr(module, "PACK")
        return pack(root=root, manifest=manifest) if isinstance(pack, type) else pack
    if hasattr(module, "Pack"):
        return getattr(module, "Pack")(root=root, manifest=manifest)
    raise ValueError(f"Pack module {manifest.entry_module} does not expose build_pack/PACK/Pack")


def load_pack(root: str | Path) -> LoadedPack:
    """Load an extension pack rooted at a manifest directory."""

    root = Path(root)
    manifest_path = root / "manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.yaml in {root}")
    manifest = load_manifest(manifest_path)
    pack = _load_declared_pack(root, manifest)
    return LoadedPack(root=root, manifest=manifest, pack=pack)

