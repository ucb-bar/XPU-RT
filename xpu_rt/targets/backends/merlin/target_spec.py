"""Merlin target spec loader.

Each merlin target lives at
``$MERLIN_ROOT/target_specs/examples/<name>/capability.yaml``. The YAML
follows merlin's own ``schema_version`` contract (currently ``1``).
This module exposes a lightweight typed view; we deliberately do NOT
copy every field — only the bits XPU-RT actually consumes to build
its own ``target_card_v1`` / ``graphcomp_target_config_v1`` artifacts.
Callers wanting the raw structure read ``MerlinTargetSpec.raw``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _default_merlin_root() -> Path:
    return Path(os.environ.get("MERLIN_ROOT", "/scratch2/agustin/merlin"))


def _target_dir(merlin_root: Path, name: str) -> Path:
    return merlin_root / "target_specs" / "examples" / name


@dataclass(frozen=True)
class MerlinTargetSpec:
    """Typed projection of a merlin ``capability.yaml``."""

    name: str
    display_name: str
    vendor: str
    maturity: str
    host_isa: str
    environments: tuple[str, ...]
    execution_kind: str
    isa_features: tuple[str, ...]
    runtime_executable_format: str
    has_simulator: bool
    simulator_kind: str | None
    raw: dict[str, Any] = field(repr=False)

    @property
    def spec_dir(self) -> Path:
        """Resolved on-disk directory (set via :func:`load_target_spec`)."""
        return Path(self.raw.get("_spec_dir", ""))


def list_targets(merlin_root: Path | None = None) -> list[str]:
    """Enumerate target spec directories under
    ``$MERLIN_ROOT/target_specs/examples/``.

    Returns names sorted alphabetically. Skips dot-dirs and any entry
    that does not contain a ``capability.yaml`` (some merlin folders
    hold prompt-only assets).
    """
    root = (merlin_root or _default_merlin_root()) / "target_specs" / "examples"
    if not root.is_dir():
        return []
    out: list[str] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        if (entry / "capability.yaml").is_file():
            out.append(entry.name)
    return out


def load_target_spec(
    name: str,
    merlin_root: Path | None = None,
) -> MerlinTargetSpec:
    """Parse ``capability.yaml`` for one target.

    Raises ``FileNotFoundError`` if the target dir or its
    ``capability.yaml`` doesn't exist.
    """
    root = merlin_root or _default_merlin_root()
    spec_dir = _target_dir(root, name)
    yaml_path = spec_dir / "capability.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(
            f"capability.yaml not found for merlin target {name!r}: {yaml_path}"
        )
    raw: dict[str, Any] = yaml.safe_load(yaml_path.read_text()) or {}
    raw["_spec_dir"] = str(spec_dir)

    target = raw.get("target") or {}
    platform = raw.get("platform") or {}
    execution = raw.get("execution_model") or {}
    isa = raw.get("isa") or {}
    runtime = raw.get("runtime") or {}
    verification = raw.get("verification") or {}
    simulator = (verification.get("simulator") or {}) if isinstance(verification, dict) else {}
    return MerlinTargetSpec(
        name=str(target.get("name") or name),
        display_name=str(target.get("display_name") or name),
        vendor=str(target.get("vendor") or "unknown"),
        maturity=str(target.get("maturity") or "experimental"),
        host_isa=str(platform.get("host_isa") or "unknown"),
        environments=tuple(platform.get("environments") or ()),
        execution_kind=str(execution.get("kind") or "unknown"),
        isa_features=tuple(isa.get("features") or ()),
        runtime_executable_format=str(
            runtime.get("executable_format") or "unspecified"
        ),
        has_simulator=bool(simulator.get("available") or False),
        simulator_kind=(
            str(simulator.get("kind")) if simulator.get("kind") else None
        ),
        raw=raw,
    )
