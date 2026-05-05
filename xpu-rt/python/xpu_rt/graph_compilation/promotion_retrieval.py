"""Phase B promotion retrieval (M-28, read side).

Before Phase B emits an ``agent_decision_request.json``, query the
promotion cache + memory.promotions index and surface promoted recipes
that match the region pattern as preferred candidates. The agent (or
deterministic resolver) ranks them ahead of fresh candidates so a
recipe proven on one model can be reused on another with the same
region signature without re-running the full search.

This is the *read side* of Section 19. The bridge in M-26 writes
promoted recipes to ``.compgen_cache/recipes/<key>/`` with a
``promoted_recipe.json`` sidecar carrying the two-tier cache key
(``contract_hash``, ``region_signature``). M-28 scans those sidecars
plus the SQLite ``memory.promotions`` table.

Two-tier lookup, in priority order:

1. **Exact contract match** — sidecars whose ``key.contract_hash``
   equals the requested ``contract_hash``. These are the *strongest*
   matches: same kernel codegen-identical inputs.

2. **Region pattern match** — sidecars whose ``key.region_signature``
   equals the requested ``region_signature`` *and* whose
   ``validity.target_class`` matches the target. These cross models —
   a tile decision proven on ``merlin_mlp_wide`` surfaces on
   ``proxy_vla`` when both regions hash to the same signature.

The function is best-effort: missing library, malformed sidecars, or
SQLite errors degrade to an empty result, never raise.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)


# Default library path mirrors the M-26 bridge default.
_DEFAULT_LIBRARY_PATH = Path(".compgen_cache") / "recipes"


@dataclass(frozen=True)
class PromotedCandidate:
    """A promoted recipe that matches a region's pattern.

    The fields mirror the M-26 ``promoted_recipe.json`` sidecar so the
    agent can rank candidates by gate level, inspect evidence, and
    inspect ``applies_when`` predicates without opening the bundle.

    Attributes:
        recipe_id: Stable human-readable id from
            :class:`compgen.promotion.promote.PromotedRecipe`.
        recipe_key: Full directory name in the recipe library
            (``target_hash_model_hash_objective_hash_vN``).
        region_signature: Two-tier cache key, region tier.
        contract_hash: Two-tier cache key, kernel-contract tier.
        target_class: Canonical target-class string the recipe is
            valid for (matched against the requesting region).
        recipe_path: Absolute path to the recipe directory on disk.
        match_kind: ``"exact_contract"`` or ``"region_pattern"`` —
            tells the agent how strong the match is.
        gate_level: M-29 promotion-gate level if recorded
            (``observed`` / ``verified_fx`` / ``verified_kernel`` /
            ``characterized`` / ``promoted`` / ``portable``).
        evidence_summary: M-26 evidence projection (cost preview,
            differential outcomes, etc.).
        applies_when: Fact predicates that must hold; empty when the
            recipe was promoted before M-27 wired this up.
        fallback_chain: Alternative candidate ids to try.
    """

    recipe_id: str
    recipe_key: str
    region_signature: str
    contract_hash: str
    target_class: str
    recipe_path: str
    match_kind: str
    gate_level: str = ""
    evidence_summary: dict[str, Any] = field(default_factory=dict)
    applies_when: tuple[str, ...] = ()
    fallback_chain: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe_id": self.recipe_id,
            "recipe_key": self.recipe_key,
            "region_signature": self.region_signature,
            "contract_hash": self.contract_hash,
            "target_class": self.target_class,
            "recipe_path": self.recipe_path,
            "match_kind": self.match_kind,
            "gate_level": self.gate_level,
            "evidence_summary": dict(self.evidence_summary),
            "applies_when": list(self.applies_when),
            "fallback_chain": list(self.fallback_chain),
        }


def _read_sidecar(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _candidate_from_sidecar(
    *, sidecar: dict[str, Any], recipe_dir: Path, match_kind: str
) -> PromotedCandidate | None:
    """Decode a ``promoted_recipe.json`` sidecar into a PromotedCandidate."""
    if not isinstance(sidecar, dict):
        return None
    key = sidecar.get("key") or {}
    recipe = sidecar.get("recipe") or {}
    if not isinstance(key, dict) or not isinstance(recipe, dict):
        return None

    target_hash = str(key.get("target_hash", ""))
    model_hash = str(key.get("model_hash", ""))
    objective_hash = str(key.get("objective_hash", ""))
    version = int(key.get("version") or 1)
    recipe_key_str = (
        f"{target_hash}_{model_hash}_{objective_hash}_v{version}"
        if target_hash and model_hash and objective_hash
        else recipe_dir.name
    )

    return PromotedCandidate(
        recipe_id=str(recipe.get("recipe_id", "") or recipe_dir.name),
        recipe_key=recipe_key_str,
        region_signature=str(key.get("region_signature", "")),
        contract_hash=str(key.get("contract_hash", "")),
        target_class=str((recipe.get("validity") or {}).get("target_class", "")),
        recipe_path=str(recipe_dir),
        match_kind=match_kind,
        gate_level=str(recipe.get("gate_level", "")),
        evidence_summary=dict(recipe.get("evidence_summary") or {}),
        applies_when=tuple(recipe.get("applies_when") or ()),
        fallback_chain=tuple(recipe.get("fallback_chain") or ()),
    )


def _scan_library_for_sidecars(library_path: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Walk the recipe library and return (recipe_dir, sidecar_body) pairs.

    Skips ``.invalid`` directories and any recipe missing the M-26
    sidecar (legacy bundles still in the library).
    """
    if not library_path.exists() or not library_path.is_dir():
        return []
    out: list[tuple[Path, dict[str, Any]]] = []
    for recipe_dir in sorted(library_path.iterdir()):
        if not recipe_dir.is_dir() or recipe_dir.name.endswith(".invalid"):
            continue
        sidecar = _read_sidecar(recipe_dir / "promoted_recipe.json")
        if sidecar is None:
            continue
        out.append((recipe_dir, sidecar))
    return out


def retrieve_for_region(
    *,
    region_signature: str,
    contract_hash: str = "",
    target_class: str = "",
    library_path: Path | None = None,
    memory: Any = None,
) -> list[PromotedCandidate]:
    """Find promoted recipes matching this region's pattern.

    Args:
        region_signature: 16-char hex hash from
            :func:`compgen.promotion.region_signature.hash_region_signature`.
            Required: an empty string yields no matches (nothing to
            match against — the M-28 agent_decision write side calls
            this per region with the region's freshly-derived signature).
        contract_hash: Optional kernel-contract hash. When non-empty,
            exact-contract matches are returned with
            ``match_kind="exact_contract"`` and ranked first.
        target_class: Canonical target-class string. Region-pattern
            matches are filtered to this target_class so a recipe
            proven on host_cpu doesn't surface for cuda_sm75.
        library_path: Recipe library root. Defaults to
            ``.compgen_cache/recipes/`` rooted at CWD.
        memory: Optional :class:`compgen.memory.store.CompilerMemory`
            for SQLite-indexed lookups (faster on large libraries; the
            on-disk scan above is the fallback when memory is None).

    Returns:
        A list of :class:`PromotedCandidate` ordered by match strength
        (exact_contract first, then region_pattern). Empty list when
        nothing matches or the library does not exist.
    """
    # M-31A.2: COMPGEN_DISABLE_RECIPE_MEMORY=1 forces a cold run by
    # short-circuiting the retrieval. The agent_decision_request writer
    # records this in the request's `disabled_by_env` field so the audit
    # trail explains why no promoted candidates surfaced.
    if os.environ.get("COMPGEN_DISABLE_RECIPE_MEMORY") == "1":
        return []
    if not region_signature and not contract_hash:
        return []

    library = Path(library_path) if library_path else _DEFAULT_LIBRARY_PATH

    exact_contract: list[PromotedCandidate] = []
    region_pattern: list[PromotedCandidate] = []

    try:
        sidecars = _scan_library_for_sidecars(library)
    except Exception as exc:  # noqa: BLE001 - retrieval is best-effort
        log.warning(
            "promotion_retrieval_library_scan_failed",
            library=str(library),
            error=type(exc).__name__,
            message=str(exc),
        )
        sidecars = []

    seen_keys: set[str] = set()
    for recipe_dir, sidecar in sidecars:
        key = sidecar.get("key") or {}
        if not isinstance(key, dict):
            continue
        side_contract = str(key.get("contract_hash", ""))
        side_region = str(key.get("region_signature", ""))
        recipe = sidecar.get("recipe") or {}
        side_target = str((recipe.get("validity") or {}).get("target_class", ""))

        # Tier 1: exact contract match.
        if contract_hash and side_contract and side_contract == contract_hash:
            cand = _candidate_from_sidecar(
                sidecar=sidecar, recipe_dir=recipe_dir, match_kind="exact_contract"
            )
            if cand and cand.recipe_key not in seen_keys:
                exact_contract.append(cand)
                seen_keys.add(cand.recipe_key)
            continue

        # Tier 2: region pattern match.
        if (
            region_signature
            and side_region == region_signature
            and (not target_class or not side_target or side_target == target_class)
        ):
            cand = _candidate_from_sidecar(
                sidecar=sidecar, recipe_dir=recipe_dir, match_kind="region_pattern"
            )
            if cand and cand.recipe_key not in seen_keys:
                region_pattern.append(cand)
                seen_keys.add(cand.recipe_key)

    # Memory-indexed lookups are additive — they catch promotions
    # whose on-disk sidecars are missing or unreadable, and they
    # tend to be faster on large libraries. Best-effort: any
    # exception falls through to whatever the on-disk scan produced.
    if memory is not None:
        try:
            for row in memory.db.fetchall(
                "SELECT promotion_key FROM promotions "
                "WHERE region_signature = ? OR contract_hash = ?",
                (region_signature, contract_hash),
            ):
                pk = row["promotion_key"]
                if pk in seen_keys or not pk:
                    continue
                # The library scan owns full PromotedCandidate construction;
                # we only use memory to prove a sidecar exists. If the
                # sidecar isn't on disk (legacy memory rows), skip it —
                # the agent can't apply a recipe whose IR is gone.
                recipe_dir = library / pk
                sidecar = _read_sidecar(recipe_dir / "promoted_recipe.json")
                if sidecar is None:
                    continue
                cand = _candidate_from_sidecar(
                    sidecar=sidecar,
                    recipe_dir=recipe_dir,
                    match_kind="region_pattern",
                )
                if cand and cand.recipe_key not in seen_keys:
                    region_pattern.append(cand)
                    seen_keys.add(cand.recipe_key)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "promotion_retrieval_memory_query_failed",
                error=type(exc).__name__,
                message=str(exc),
            )

    return [*exact_contract, *region_pattern]


__all__ = ["PromotedCandidate", "retrieve_for_region"]
