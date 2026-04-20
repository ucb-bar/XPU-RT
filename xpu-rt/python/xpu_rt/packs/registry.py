"""Registry and discovery helpers for extension packs."""

from __future__ import annotations

import importlib.metadata
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from compgen.packs.base import LoadedPack
from compgen.packs.loader import load_pack, resolve_entry_point_target
from compgen.packs.schema import PackContextSummary

ENTRY_POINT_GROUP = "compgen.packs"
ENV_VAR = "COMPGEN_PACKS_PATH"


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


def _entry_point_pack_paths() -> list[Path]:
    """Resolve each ``compgen.packs`` entry point to a pack-root Path."""

    out: list[Path] = []
    try:
        eps = importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)
    except TypeError:  # pragma: no cover - <3.10 fallback
        eps = importlib.metadata.entry_points().get(ENTRY_POINT_GROUP, [])  # type: ignore[assignment]
    for ep in eps:
        try:
            out.append(resolve_entry_point_target(ep.value))
        except Exception:
            continue
    return out


def _env_var_pack_paths(env_var: str = ENV_VAR) -> list[Path]:
    """Expand ``$COMPGEN_PACKS_PATH`` into a list of pack-root Paths."""

    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return []
    out: list[Path] = []
    for segment in raw.split(os.pathsep):
        p = Path(segment).expanduser()
        if not p.exists():
            continue
        if (p / "manifest.yaml").exists():
            out.append(p)
            continue
        # Treat as a directory that contains multiple pack roots.
        out.extend(sorted(m.parent for m in p.glob("*/manifest.yaml")))
    return out


def discover_packs(
    repo_root: str | Path | None = None,
    *,
    include_repo: bool = True,
    include_entry_points: bool = True,
    include_env: bool = True,
    env_var: str = ENV_VAR,
) -> list[Path]:
    """Discover pack roots across all enabled sources, deduped in order.

    Sources (in priority order, earlier wins on duplicates):
      1. Repo-local ``userpacks/`` directory (if ``include_repo``).
      2. ``compgen.packs`` entry points (if ``include_entry_points``).
      3. ``$COMPGEN_PACKS_PATH`` colon-separated list (if ``include_env``).
    """

    candidates: list[Path] = []
    if include_repo:
        candidates.extend(discover_pack_paths(repo_root))
    if include_entry_points:
        candidates.extend(_entry_point_pack_paths())
    if include_env:
        candidates.extend(_env_var_pack_paths(env_var))

    seen: set[Path] = set()
    unique: list[Path] = []
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(candidate)
    return unique


def load_builtin_packs(repo_root: str | Path | None = None) -> list[LoadedPack]:
    """Load all repo-local declarative packs."""

    return [load_pack(path) for path in discover_pack_paths(repo_root)]


def load_discovered_packs(**discover_kwargs) -> list[LoadedPack]:
    """Load every pack returned by :func:`discover_packs`."""

    return [load_pack(path) for path in discover_packs(**discover_kwargs)]


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
