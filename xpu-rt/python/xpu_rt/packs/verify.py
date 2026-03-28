"""Ownership and aperture verification helpers for packs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from compgen.packs.base import LoadedPack


@dataclass(frozen=True)
class OwnershipViolation:
    """A forbidden attempt to mutate a sealed or unmanaged surface."""

    pack_name: str
    surface: str
    reason: str


def check_surface_allowed(
    packs: Iterable[LoadedPack],
    *,
    requested_surface: str,
) -> OwnershipViolation | None:
    """Return a violation if a requested surface is sealed by any active pack."""

    for pack in packs:
        if requested_surface in pack.manifest.sealed_surfaces:
            return OwnershipViolation(
                pack_name=pack.manifest.name,
                surface=requested_surface,
                reason="sealed_surface",
            )
    return None

