"""Card discovery + loading for provider / target / dialect cards.

Cards live as YAML files under three roots:

* ``python/compgen/providers/cards/*.yaml``
* ``python/compgen/targets/cards/*.yaml``
* ``python/compgen/dialects/cards/*.yaml``

User extensions contribute cards through the manifest's ``provides``
section; those flow through ``ExtensionManifest.from_dict``
rather than this loader.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import yaml

from compgen.dialects.dialect_provider_types import DialectProviderCard
from compgen.providers.provider_types import ProviderCard
from compgen.targets.target_types import TargetCard


def _provider_cards_root() -> Path:
    return Path(__file__).resolve().parent / "cards"


def _target_cards_root() -> Path:
    from compgen.targets import target_types as _tt
    return Path(_tt.__file__).resolve().parent / "cards"


def _dialect_cards_root() -> Path:
    from compgen.dialects import dialect_provider_types as _dt
    return Path(_dt.__file__).resolve().parent / "cards"


def _load_yaml(path: Path) -> dict:
    body = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(body, dict):
        raise ValueError(
            f"card {path} must be a YAML mapping; got {type(body).__name__}"
        )
    return body


def iter_provider_cards(root: Path | None = None) -> Iterator[ProviderCard]:
    base = root or _provider_cards_root()
    if not base.is_dir():
        return
    for path in sorted(base.glob("*.yaml")):
        yield ProviderCard.from_dict(_load_yaml(path), source=path)


def iter_target_cards(root: Path | None = None) -> Iterator[TargetCard]:
    base = root or _target_cards_root()
    if not base.is_dir():
        return
    for path in sorted(base.glob("*.yaml")):
        yield TargetCard.from_dict(_load_yaml(path), source=path)


def iter_dialect_cards(root: Path | None = None) -> Iterator[DialectProviderCard]:
    base = root or _dialect_cards_root()
    if not base.is_dir():
        return
    for path in sorted(base.glob("*.yaml")):
        yield DialectProviderCard.from_dict(_load_yaml(path), source=path)


def load_all_cards(
    provider_root: Path | None = None,
    target_root: Path | None = None,
    dialect_root: Path | None = None,
) -> tuple[
    tuple[ProviderCard, ...],
    tuple[TargetCard, ...],
    tuple[DialectProviderCard, ...],
]:
    return (
        tuple(iter_provider_cards(provider_root)),
        tuple(iter_target_cards(target_root)),
        tuple(iter_dialect_cards(dialect_root)),
    )
