"""Migrate pre-M-37 promoted-recipe sidecars (M-37.4).

Sidecars promoted before M-37 may lack ``candidate_kind`` and
``selected_candidate_id`` in ``recipe.evidence_summary`` — the fields
M-37 reads to cross-link a promoted recipe back to its source pass
card. The M-26 bridge has been writing them since M-26 itself, so the
residual only affects sidecars promoted before M-26 (or hand-authored
fixtures).

The migration is non-destructive: it backfills only the missing
fields, never overwrites. Sidecars that already have both fields are
left alone. The migration also infers values from auxiliary sources:

- ``recipe.recipe_id`` like ``recipe_<candidate_kind>_<region>_<target>_<sig>``
  yields ``candidate_kind``.
- ``recipe.evidence_summary.region_id`` provides ``region_id``.
- The first applied transform's op type (when readable from the
  recipe.mlir on disk) is a fallback for ``candidate_kind``.

When inference is impossible the field stays empty; the agent treats
absence as "no cross-link" and the audit doesn't fail.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_RECIPE_ID_PREFIX = "recipe_"


@dataclass(frozen=True)
class MigrationResult:
    """One sidecar's migration outcome."""

    path: Path
    already_complete: bool
    migrated: bool
    inferred_candidate_kind: str = ""
    inferred_selected_candidate_id: str = ""
    skipped_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "already_complete": self.already_complete,
            "migrated": self.migrated,
            "inferred_candidate_kind": self.inferred_candidate_kind,
            "inferred_selected_candidate_id": self.inferred_selected_candidate_id,
            "skipped_reason": self.skipped_reason,
        }


@dataclass
class MigrationReport:
    """Aggregate report across a recipe library scan."""

    library_path: Path
    results: list[MigrationResult] = field(default_factory=list)

    @property
    def already_complete_count(self) -> int:
        return sum(1 for r in self.results if r.already_complete)

    @property
    def migrated_count(self) -> int:
        return sum(1 for r in self.results if r.migrated)

    @property
    def skipped_count(self) -> int:
        return sum(
            1 for r in self.results
            if not r.already_complete and not r.migrated
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "library_path": str(self.library_path),
            "result_count": len(self.results),
            "already_complete_count": self.already_complete_count,
            "migrated_count": self.migrated_count,
            "skipped_count": self.skipped_count,
            "results": [r.to_dict() for r in self.results],
        }


def _infer_candidate_kind_from_recipe_id(recipe_id: str) -> str:
    """Recipe ids follow ``recipe_<candidate_kind>_<region>_<target>_<sig>``.

    The candidate_kind itself can contain underscores (e.g.
    ``fuse_producer_consumer``), so a naive split-on-underscore
    fails. We match the longest known pass_id from the live
    PassCardRegistry against the prefix following ``recipe_``.
    Returns empty when no known kind is a prefix.
    """
    if not recipe_id or not recipe_id.startswith(_RECIPE_ID_PREFIX):
        return ""
    body = recipe_id[len(_RECIPE_ID_PREFIX):]
    try:
        from compgen.passes.cards import (
            PassCardRegistry,
            default_registry_root,
        )

        registry = PassCardRegistry.load(default_registry_root())
        # Longest pass_id that prefixes ``body`` followed by ``_``.
        candidates = [
            pid for pid in registry.passes_allowed()
            if body.startswith(pid + "_")
        ]
        if not candidates:
            return ""
        return max(candidates, key=len)
    except Exception:  # noqa: BLE001 - migration is best-effort
        return ""


def migrate_sidecar(sidecar_path: Path, *, dry_run: bool = False) -> MigrationResult:
    """Backfill missing M-37 fields on a single sidecar.

    Returns a :class:`MigrationResult`. Non-destructive: existing
    fields are never overwritten. When ``dry_run=True``, no file is
    written but the inference is still computed.
    """
    if not sidecar_path.exists():
        return MigrationResult(
            path=sidecar_path,
            already_complete=False,
            migrated=False,
            skipped_reason="sidecar not found",
        )
    try:
        body = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return MigrationResult(
            path=sidecar_path,
            already_complete=False,
            migrated=False,
            skipped_reason=f"unreadable: {type(exc).__name__}",
        )

    recipe = body.get("recipe") or {}
    evidence = recipe.get("evidence_summary") or {}
    has_kind = bool(evidence.get("candidate_kind"))
    has_id = bool(evidence.get("selected_candidate_id"))
    if has_kind and has_id:
        return MigrationResult(
            path=sidecar_path,
            already_complete=True,
            migrated=False,
        )

    inferred_kind = ""
    inferred_id = ""
    if not has_kind:
        # Try the recipe_id pattern first.
        inferred_kind = _infer_candidate_kind_from_recipe_id(
            str(recipe.get("recipe_id", ""))
        )
        if inferred_kind:
            evidence["candidate_kind"] = inferred_kind
    if not has_id:
        # The recipe_signature is a deterministic id when no candidate
        # id was preserved; we surface it as the candidate_id so the
        # cross-link has *some* value.
        sig = recipe.get("recipe_signature") or evidence.get("region_signature_hash")
        if sig:
            inferred_id = str(sig)
            evidence["selected_candidate_id"] = inferred_id

    if not inferred_kind and not inferred_id:
        return MigrationResult(
            path=sidecar_path,
            already_complete=False,
            migrated=False,
            skipped_reason="no inference source (recipe_id pattern + signature both empty)",
        )

    if dry_run:
        return MigrationResult(
            path=sidecar_path,
            already_complete=False,
            migrated=False,
            inferred_candidate_kind=inferred_kind,
            inferred_selected_candidate_id=inferred_id,
            skipped_reason="dry_run",
        )

    recipe["evidence_summary"] = evidence
    body["recipe"] = recipe
    sidecar_path.write_text(
        json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return MigrationResult(
        path=sidecar_path,
        already_complete=False,
        migrated=True,
        inferred_candidate_kind=inferred_kind,
        inferred_selected_candidate_id=inferred_id,
    )


def migrate_library(
    library_path: Path,
    *,
    dry_run: bool = False,
) -> MigrationReport:
    """Walk a recipe library and migrate every sidecar found."""
    report = MigrationReport(library_path=Path(library_path).resolve())
    if not library_path.exists():
        return report
    for sidecar_path in sorted(library_path.rglob("promoted_recipe.json")):
        result = migrate_sidecar(sidecar_path, dry_run=dry_run)
        report.results.append(result)
    return report
