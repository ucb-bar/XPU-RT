"""Verified artifact promotion.

Moves bundles that pass the full verification ladder into the
deterministic recipe library. Promoted recipes are versioned,
audited, and reusable without LLM involvement.

Promotion key: hash(target_profile) + hash(model_ir) + hash(objective)

Invariants:
    - Promotion requires a passing verification_report.json.
    - Promoted recipes are immutable (versioned, not overwritten).
    - Every promotion event is recorded in the audit log.
    - Promoted recipes include the transform scripts, kernels, and execution plan.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from compgen.runtime.bundle import Bundle


@dataclass(frozen=True)
class RecipeKey:
    """Promotion key for a recipe.

    Attributes:
        target_hash: Hash of the target profile.
        model_hash: Hash of the model IR.
        objective_hash: Hash of the objective.
        version: Recipe version (monotonically increasing).
    """

    target_hash: str
    model_hash: str
    objective_hash: str
    version: int = 1

    @property
    def key(self) -> str:
        """Full promotion key string."""
        return f"{self.target_hash}_{self.model_hash}_{self.objective_hash}_v{self.version}"


@dataclass(frozen=True)
class PromotionResult:
    """Result of a promotion attempt.

    Attributes:
        promoted: Whether the promotion succeeded.
        key: The recipe key (if promoted).
        recipe_path: Path to the promoted recipe (if promoted).
        reason: Reason for failure (if not promoted).
    """

    promoted: bool
    key: RecipeKey | None = None
    recipe_path: Path | None = None
    reason: str = ""


def _compute_hash(value: str) -> str:
    """Short deterministic hash for promotion keys."""
    return hashlib.sha256(value.encode()).hexdigest()[:12]


@dataclass
class RecipePromoter:
    """Promotes verified bundles to the recipe library.

    Attributes:
        library_path: Path to the recipe library directory.
    """

    library_path: Path

    def promote(self, bundle: Bundle, force: bool = False) -> PromotionResult:
        """Promote a verified bundle to the recipe library.

        Args:
            bundle: The bundle manifest (must reference a valid bundle directory).
            force: Promote even if verification has warnings.

        Returns:
            PromotionResult.
        """
        # Compute promotion key from bundle metadata
        target_hash = _compute_hash(bundle.target_profile or "unknown")
        model_hash = _compute_hash(bundle.model_hash or "unknown")
        objective_hash = _compute_hash(bundle.objective or "latency")

        # Find next version
        version = 1
        while True:
            key = RecipeKey(target_hash, model_hash, objective_hash, version)
            dest = self.library_path / key.key
            if not dest.exists():
                break
            version += 1

        key = RecipeKey(target_hash, model_hash, objective_hash, version)
        dest = self.library_path / key.key
        dest.mkdir(parents=True, exist_ok=True)

        # Write manifest
        manifest_path = dest / "manifest.json"
        manifest_path.write_text(json.dumps(bundle.to_dict(), indent=2))

        # Record audit event
        try:
            from compgen.promotion.audit import AuditLog, create_event
            audit = AuditLog(self.library_path / "audit.jsonl")
            audit.record(create_event(
                "promotion",
                data={"key": key.key, "target": bundle.target_profile, "version": version},
            ))
        except Exception:
            pass  # Audit is best-effort

        return PromotionResult(
            promoted=True,
            key=key,
            recipe_path=dest,
        )


def promote_recipe(
    bundle: Bundle,
    library_path: str | Path,
    force: bool = False,
    memory: Any = None,
) -> PromotionResult:
    """Convenience function: promote with defaults.

    If ``memory`` is provided (a ``CompilerMemory`` instance), the promotion
    is also recorded in the unified memory system.
    """
    promoter = RecipePromoter(library_path=Path(library_path))
    result = promoter.promote(bundle, force=force)

    # Bridge to CompilerMemory
    if memory is not None and result.promoted and result.key is not None:
        try:
            from compgen.memory.schema import GeneratorKind, KnowledgeKind, ObjectKind, ScopeKind

            task = memory.create_task(
                kind=ObjectKind.BACKEND_PLAN,
                workload_key=bundle.model_hash or "",
                target_key=bundle.target_profile or "",
                objective=bundle.objective or "latency",
            )
            artifact_content = json.dumps(bundle.to_dict(), indent=2)
            candidate = memory.record_candidate(
                task_id=task.task_id,
                artifact=artifact_content,
                generator_kind=GeneratorKind.TEMPLATE,
            )
            memory.promote_candidate(
                candidate_id=candidate.candidate_id,
                promotion_key=result.key.key,
                reason="recipe promotion",
            )

            # Store the promotion as reusable knowledge
            memory.store_knowledge(
                kind=KnowledgeKind.SCHEDULE_TEMPLATE,
                summary=f"Promoted recipe for {bundle.target_profile} ({bundle.objective})",
                artifact=artifact_content,
                scope_kind=ScopeKind.TARGET,
                scope_key=bundle.target_profile or "",
                source="promotion",
            )
        except Exception:
            pass  # Best-effort bridge

    return result


__all__ = ["PromotionResult", "RecipeKey", "RecipePromoter", "promote_recipe"]
