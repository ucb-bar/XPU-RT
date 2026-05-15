"""TargetCard schema.

A ``TargetCard`` is the static, declarative shape of a deployment
target — family, vendor, dispatch modes, memory tiers — separate from
the richer :class:`compgen.targets.schema.TargetSchema` /
``CapabilitySpec`` records used by the capture+lower pipeline.
Cards drive the provider/target matrix and the probe.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

DISPATCH_MODES: Final[tuple[str, ...]] = (
    "sync",
    "async",
    "static_plan",
    "megakernel",
    "event_tensor",
)

MEMORY_TIER_KINDS: Final[tuple[str, ...]] = (
    "global",
    "shared",
    "explicit",
    "host",
    "registers",
)


class TargetCardError(ValueError):
    """A TargetCard YAML body violated the schema."""


@dataclass(frozen=True)
class MemoryTier:
    name: str
    kind: str
    capacity_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "capacity_bytes": self.capacity_bytes,
        }


@dataclass(frozen=True)
class TargetCard:
    schema_version: str
    target_id: str
    family: str
    vendor: str
    dispatch_modes: tuple[str, ...]
    memory_tiers: tuple[MemoryTier, ...]
    description: str = ""

    @classmethod
    def from_dict(cls, body: dict[str, Any], *, source: Path | None = None) -> "TargetCard":
        try:
            target_id = str(body["target_id"])
            family = str(body["family"])
            vendor = str(body["vendor"])
            schema_version = str(body["schema_version"])
        except KeyError as exc:
            raise TargetCardError(
                f"target card missing required field {exc.args[0]!r} (source={source})"
            ) from exc
        modes_raw = tuple(body.get("dispatch_modes", ()))
        if not modes_raw:
            raise TargetCardError(
                f"target {target_id!r} declares no dispatch_modes (source={source})"
            )
        for mode in modes_raw:
            if mode not in DISPATCH_MODES:
                raise TargetCardError(
                    f"target {target_id!r} dispatch_mode={mode!r} must be one of "
                    f"{DISPATCH_MODES} (source={source})"
                )
        tiers_raw = body.get("memory_tiers", [])
        tiers: list[MemoryTier] = []
        for entry in tiers_raw:
            name = str(entry["name"])
            kind = str(entry["kind"])
            if kind not in MEMORY_TIER_KINDS:
                raise TargetCardError(
                    f"target {target_id!r} memory_tier {name!r} kind={kind!r} "
                    f"must be one of {MEMORY_TIER_KINDS} (source={source})"
                )
            cap = entry.get("capacity_bytes")
            tiers.append(
                MemoryTier(name=name, kind=kind, capacity_bytes=int(cap) if cap is not None else None)
            )
        return cls(
            schema_version=schema_version,
            target_id=target_id,
            family=family,
            vendor=vendor,
            dispatch_modes=modes_raw,
            memory_tiers=tuple(tiers),
            description=str(body.get("description", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "target_id": self.target_id,
            "family": self.family,
            "vendor": self.vendor,
            "dispatch_modes": list(self.dispatch_modes),
            "memory_tiers": [t.to_dict() for t in self.memory_tiers],
            "description": self.description,
        }
