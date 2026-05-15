"""Multi-level IR analysis snapshots.

Emits per-level summaries the agent reads instead of raw IR. Each
snapshot is a typed :class:`IRAnalysisSnapshot` carrying the level
identifier, the source artifact path, and the regions discovered at
that level. Levels with no producer at run time emit a
``not_available`` snapshot with the typed reason — never a silent
omission.

Levels (closed set):

* ``fx_graph``                — torch FX graph (Stage 0)
* ``payload_ir``              — canonical xDSL/MLIR (Stage 1)
* ``recipe_ir``               — Recipe-IR decisions (Stage 2)
* ``tile_ir``                 — compgen.tile dialect (Stage 3)
* ``dialect_ir``              — vendor / accelerator dialect (Stage 3+)
* ``kernel_artifact``         — compiled kernel sources (Stage 4)
* ``execution_plan``          — runtime plan (Stage 5)
* ``runtime_profile``         — measured runtime evidence

Schema: ``ir_analysis_snapshot_v1``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

SCHEMA_VERSION: Final[str] = "ir_analysis_snapshot_v1"

IR_LEVELS: Final[tuple[str, ...]] = (
    "fx_graph",
    "payload_ir",
    "recipe_ir",
    "tile_ir",
    "dialect_ir",
    "kernel_artifact",
    "execution_plan",
    "runtime_profile",
)

SNAPSHOT_FILENAMES: Final[dict[str, str]] = {
    level: f"{level}_analysis.json" for level in IR_LEVELS
}

NOT_AVAILABLE_REASONS: Final[tuple[str, ...]] = (
    "stage_not_run",
    "artifact_missing",
    "level_unsupported_for_run",
    "extraction_failed",
)


class IRSnapshotError(ValueError):
    """A snapshot body violated the schema."""


@dataclass(frozen=True)
class UnsupportedProvider:
    """Reason a provider can't serve a region at this IR level."""

    provider_id: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"provider_id": self.provider_id, "reason": self.reason}


@dataclass(frozen=True)
class RegionSummary:
    """One region's analysis at a given IR level.

    Mandatory: ``region_id`` and ``ops``. Optional: provider /
    fusion / lowering info. Missing data is represented as an empty
    sequence — never a None sentinel.
    """

    region_id: str
    ops: tuple[str, ...]
    supported_providers: tuple[str, ...] = ()
    unsupported_providers: tuple[UnsupportedProvider, ...] = ()
    fusion_candidates: tuple[str, ...] = ()
    lowering_gaps: tuple[str, ...] = ()
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "region_id": self.region_id,
            "ops": list(self.ops),
            "supported_providers": list(self.supported_providers),
            "unsupported_providers": [u.to_dict() for u in self.unsupported_providers],
            "fusion_candidates": list(self.fusion_candidates),
            "lowering_gaps": list(self.lowering_gaps),
        }
        if self.extras:
            body["extras"] = dict(self.extras)
        return body

    @classmethod
    def from_dict(cls, body: dict[str, Any]) -> "RegionSummary":
        ups = tuple(
            UnsupportedProvider(
                provider_id=str(u["provider_id"]),
                reason=str(u["reason"]),
            )
            for u in body.get("unsupported_providers", [])
        )
        return cls(
            region_id=str(body["region_id"]),
            ops=tuple(body.get("ops", ())),
            supported_providers=tuple(body.get("supported_providers", ())),
            unsupported_providers=ups,
            fusion_candidates=tuple(body.get("fusion_candidates", ())),
            lowering_gaps=tuple(body.get("lowering_gaps", ())),
            extras=dict(body.get("extras", {}) or {}),
        )


@dataclass(frozen=True)
class IRAnalysisSnapshot:
    """Per-IR-level analysis summary.

    ``status="available"`` snapshots carry one or more
    :class:`RegionSummary` entries. ``status="not_available"``
    snapshots carry a typed reason and an empty ``regions`` list —
    the agent must surface those rather than silently skip.
    """

    schema_version: str
    level: str
    status: str  # "available" | "not_available"
    source_artifact: str = ""
    regions: tuple[RegionSummary, ...] = ()
    not_available_reason: str = ""
    detail: str = ""

    def __post_init__(self) -> None:
        if self.level not in IR_LEVELS:
            raise IRSnapshotError(
                f"unknown IR level {self.level!r}; must be one of {IR_LEVELS}"
            )
        if self.status not in ("available", "not_available"):
            raise IRSnapshotError(
                f"status={self.status!r} must be 'available' or 'not_available'"
            )
        if self.status == "not_available":
            if not self.not_available_reason:
                raise IRSnapshotError(
                    f"status=not_available requires a typed reason"
                )
            if self.not_available_reason not in NOT_AVAILABLE_REASONS:
                raise IRSnapshotError(
                    f"not_available_reason={self.not_available_reason!r} must be "
                    f"one of {NOT_AVAILABLE_REASONS}"
                )
            if self.regions:
                raise IRSnapshotError(
                    f"status=not_available must not carry regions"
                )
        if self.status == "available" and self.not_available_reason:
            raise IRSnapshotError(
                f"status=available must not carry not_available_reason"
            )

    def to_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "schema_version": self.schema_version,
            "level": self.level,
            "status": self.status,
            "source_artifact": self.source_artifact,
            "regions": [r.to_dict() for r in self.regions],
        }
        if self.not_available_reason:
            body["not_available_reason"] = self.not_available_reason
        if self.detail:
            body["detail"] = self.detail
        return body

    @classmethod
    def from_dict(cls, body: dict[str, Any]) -> "IRAnalysisSnapshot":
        regions = tuple(
            RegionSummary.from_dict(r) for r in body.get("regions", [])
        )
        return cls(
            schema_version=str(body.get("schema_version", SCHEMA_VERSION)),
            level=str(body["level"]),
            status=str(body["status"]),
            source_artifact=str(body.get("source_artifact", "")),
            regions=regions,
            not_available_reason=str(body.get("not_available_reason", "")),
            detail=str(body.get("detail", "")),
        )

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return p


def make_available(
    *,
    level: str,
    source_artifact: str,
    regions: list[RegionSummary] | tuple[RegionSummary, ...],
    detail: str = "",
) -> IRAnalysisSnapshot:
    return IRAnalysisSnapshot(
        schema_version=SCHEMA_VERSION,
        level=level,
        status="available",
        source_artifact=source_artifact,
        regions=tuple(regions),
        detail=detail,
    )


def make_not_available(
    *,
    level: str,
    reason: str,
    source_artifact: str = "",
    detail: str = "",
) -> IRAnalysisSnapshot:
    return IRAnalysisSnapshot(
        schema_version=SCHEMA_VERSION,
        level=level,
        status="not_available",
        source_artifact=source_artifact,
        regions=(),
        not_available_reason=reason,
        detail=detail,
    )


# ---------------------------------------------------------------------------
# Bulk writer
# ---------------------------------------------------------------------------


def write_snapshots(
    snapshots: dict[str, IRAnalysisSnapshot],
    out_dir: str | Path,
) -> dict[str, Path]:
    """Write one snapshot per level under ``out_dir``.

    Missing levels in ``snapshots`` are automatically filled with
    ``not_available`` entries carrying ``stage_not_run`` so the
    agent surface is complete and every level is represented.

    Returns a ``{level: written_path}`` mapping.
    """

    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    for level in IR_LEVELS:
        snap = snapshots.get(level)
        if snap is None:
            snap = make_not_available(
                level=level,
                reason="stage_not_run",
                detail="no snapshot supplied for this level",
            )
        if snap.level != level:
            raise IRSnapshotError(
                f"snapshot for level={level!r} mislabeled as {snap.level!r}"
            )
        written[level] = snap.write(base / SNAPSHOT_FILENAMES[level])

    return written


def load_snapshot(path: str | Path) -> IRAnalysisSnapshot:
    body = json.loads(Path(path).read_text(encoding="utf-8"))
    return IRAnalysisSnapshot.from_dict(body)


def discover_snapshots(out_dir: str | Path) -> dict[str, IRAnalysisSnapshot]:
    """Load every snapshot file present under ``out_dir``."""

    base = Path(out_dir)
    out: dict[str, IRAnalysisSnapshot] = {}
    if not base.is_dir():
        return out
    for level, filename in SNAPSHOT_FILENAMES.items():
        p = base / filename
        if p.is_file():
            out[level] = load_snapshot(p)
    return out
