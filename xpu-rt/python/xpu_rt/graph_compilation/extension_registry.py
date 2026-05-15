"""User-space extension registry.

A single YAML file at ``.crg-artifacts/extensions/registry.yaml`` lists
every verified extension. Gap Discovery consults this file (when called
with ``--extension-registry``) to skip targets that already have a
verified implementation, producing a smaller ``gap_action_queue.json``.

The registry is content-addressed: each entry carries the
``extension_id`` (which is a sha8 of the gap record), the on-disk
``extension_path``, plus the verify-time ``max_abs_error`` /
``max_rel_error`` so a downstream consumer can decide whether the
verified tolerance is acceptable for its use case.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RegistryEntry:
    gap_kind: str
    fx_target: str
    extension_id: str
    extension_path: str  # absolute or repo-relative
    verification_status: str  # "pass" | "fail"
    verified_at_utc: str
    max_abs_error: float
    max_rel_error: float
    rtol: float
    atol: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "gap_kind": self.gap_kind,
            "fx_target": self.fx_target,
            "extension_id": self.extension_id,
            "extension_path": self.extension_path,
            "verification_status": self.verification_status,
            "verified_at_utc": self.verified_at_utc,
            "max_abs_error": self.max_abs_error,
            "max_rel_error": self.max_rel_error,
            "rtol": self.rtol,
            "atol": self.atol,
        }


@dataclass
class ExtensionRegistry:
    schema_version: str = "extension_registry_v1"
    entries: list[RegistryEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "entries": [e.to_dict() for e in self.entries],
        }

    def has(self, gap_kind: str, fx_target: str) -> bool:
        return any(
            e.gap_kind == gap_kind
            and e.fx_target == fx_target
            and e.verification_status == "pass"
            for e in self.entries
        )

    def lookup(self, gap_kind: str, fx_target: str) -> RegistryEntry | None:
        for e in self.entries:
            if (
                e.gap_kind == gap_kind
                and e.fx_target == fx_target
                and e.verification_status == "pass"
            ):
                return e
        return None

    def upsert(self, entry: RegistryEntry) -> None:
        """Insert or replace by ``(gap_kind, fx_target, extension_id)``."""
        key = (entry.gap_kind, entry.fx_target, entry.extension_id)
        for i, existing in enumerate(self.entries):
            if (existing.gap_kind, existing.fx_target, existing.extension_id) == key:
                self.entries[i] = entry
                return
        self.entries.append(entry)


def load_registry(path: Path) -> ExtensionRegistry:
    """Load ``registry.yaml``; return empty registry if missing."""
    p = Path(path)
    if not p.exists():
        return ExtensionRegistry()
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if raw.get("schema_version") != "extension_registry_v1":
        # Treat as empty rather than blow up — tolerant load policy for
        # forward compatibility with future schema bumps.
        return ExtensionRegistry()
    entries = [
        RegistryEntry(
            gap_kind=e["gap_kind"],
            fx_target=e["fx_target"],
            extension_id=e["extension_id"],
            extension_path=e["extension_path"],
            verification_status=e["verification_status"],
            verified_at_utc=e["verified_at_utc"],
            max_abs_error=float(e["max_abs_error"]),
            max_rel_error=float(e["max_rel_error"]),
            rtol=float(e.get("rtol", 0.0)),
            atol=float(e.get("atol", 0.0)),
        )
        for e in raw.get("entries", [])
    ]
    return ExtensionRegistry(entries=entries)


def save_registry(registry: ExtensionRegistry, path: Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(registry.to_dict(), sort_keys=True), encoding="utf-8")


def register_extension(
    *,
    workspace: Path,
    verification_result: Any,  # extension_verify.VerifyResult
    registry_path: Path,
) -> RegistryEntry:
    """Add or update a registry entry from a verified extension workspace."""
    import json

    contract = json.loads((workspace / "extension_contract.json").read_text(encoding="utf-8"))
    entry = RegistryEntry(
        gap_kind=contract["gap_kind"],
        fx_target=contract["fx_target"],
        extension_id=contract["extension_id"],
        extension_path=str(workspace.resolve()),
        verification_status=verification_result.status,
        verified_at_utc=datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        max_abs_error=float(verification_result.max_abs_error),
        max_rel_error=float(verification_result.max_rel_error),
        rtol=float(contract["verification"]["rtol"]),
        atol=float(contract["verification"]["atol"]),
    )
    registry = load_registry(registry_path)
    registry.upsert(entry)
    save_registry(registry, registry_path)
    _update_manifest_post_register(workspace, entry)
    return entry


def _update_manifest_post_register(workspace: Path, entry: RegistryEntry) -> None:
    """Patch ``manifest.yaml`` to record registration."""
    manifest_path = workspace / "manifest.yaml"
    if not manifest_path.exists():
        return
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    raw["registered_at_utc"] = entry.verified_at_utc
    raw["status"] = "registered"
    manifest_path.write_text(yaml.safe_dump(raw, sort_keys=True), encoding="utf-8")
