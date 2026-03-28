"""Registry and discovery helpers for extension packs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from compgen.packs.base import LoadedPack
from compgen.packs.loader import load_pack
from compgen.packs.schema import PackContextSummary


def default_pack_root(repo_root: str | Path | None = None) -> Path:
    """Return the repo-local directory containing declarative user packs."""

    if repo_root is None:
        return Path(__file__).resolve().parents[3] / "userpacks"
    return Path(repo_root).resolve() / "userpacks"


def discover_pack_paths(repo_root: str | Path | None = None) -> list[Path]:
    """Discover declarative pack manifests under ``userpacks/``."""

    root = default_pack_root(repo_root)
    if not root.exists():
        return []
    return sorted(path.parent for path in root.glob("*/manifest.yaml"))


def load_builtin_packs(repo_root: str | Path | None = None) -> list[LoadedPack]:
    """Load all repo-local declarative packs."""

    return [load_pack(path) for path in discover_pack_paths(repo_root)]


@dataclass
class PackRegistry:
    """Mutable registry of loaded extension packs."""

    packs: dict[str, LoadedPack] = field(default_factory=dict)

    def register(self, pack: LoadedPack) -> None:
        self.packs[pack.manifest.name] = pack

    def register_many(self, packs: Iterable[LoadedPack]) -> None:
        for pack in packs:
            self.register(pack)

    def get(self, name: str) -> LoadedPack:
        return self.packs[name]

    def names(self) -> list[str]:
        return sorted(self.packs.keys())

    def summarize(self, names: Iterable[str]) -> PackContextSummary:
        loaded = [self.packs[name] for name in names if name in self.packs]
        sealed = sorted({surface for pack in loaded for surface in pack.manifest.sealed_surfaces})
        apertures = sorted({surface for pack in loaded for surface in pack.manifest.generation_apertures})
        profilers = sorted({prof for pack in loaded for prof in pack.manifest.available_profilers})
        benchmarks = sorted({target for pack in loaded for target in pack.manifest.benchmark_targets})
        return PackContextSummary(
            active_packs=tuple(pack.manifest.name for pack in loaded),
            sealed_surfaces=tuple(sealed),
            generation_apertures=tuple(apertures),
            available_profilers=tuple(profilers),
            benchmark_targets=tuple(benchmarks),
        )

