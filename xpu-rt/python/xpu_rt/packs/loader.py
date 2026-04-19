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


def resolve_entry_point_target(value: str) -> Path:
    """Resolve a ``pkg.mod[:attr]`` entry-point value to a pack-root Path.

    Rules:
      - No ``:attr`` → use ``Path(module.__file__).parent`` as the pack root.
      - ``:attr`` is a ``Path``/``str`` → treat as the pack root path.
      - ``:attr`` is callable → call with no args; result must be a Path or str.
    """

    module_name, _, attr_name = value.partition(":")
    module = importlib.import_module(module_name)
    if not attr_name:
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            raise ValueError(f"entry-point module {module_name!r} has no __file__")
        return Path(module_file).parent

    target = getattr(module, attr_name)
    if isinstance(target, Path):
        return target
    if isinstance(target, str):
        return Path(target)
    if callable(target):
        result = target()
        if isinstance(result, Path):
            return result
        if isinstance(result, str):
            return Path(result)
        raise TypeError(
            f"entry-point {value!r} callable returned {type(result).__name__}; expected Path|str"
        )
    raise TypeError(
        f"entry-point {value!r} resolved to {type(target).__name__}; expected Path|str|callable"
    )


def _resolve_pack_source(source: str | Path) -> Path:
    """Accept a Path, a path-like string, or an entry-point value; return a Path."""

    if isinstance(source, Path):
        return source
    candidate = Path(source)
    if candidate.exists():
        return candidate
    return resolve_entry_point_target(source)


def load_pack(root: str | Path) -> LoadedPack:
    """Load an extension pack rooted at a manifest directory or entry point."""

    resolved_root = _resolve_pack_source(root)
    manifest_path = resolved_root / "manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.yaml in {resolved_root}")
    manifest = load_manifest(manifest_path)
    pack = _load_declared_pack(resolved_root, manifest)
    return LoadedPack(root=resolved_root, manifest=manifest, pack=pack)

