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
from typing import Any

from compgen.promotion.errors import (
    PromotionBlockedError,
    PromotionBlockReason,
    VerificationGateResult,
)
from compgen.runtime.bundle import Bundle

# Verification levels that must be present + passing before a bundle
# can be promoted. Levels not in this set are tolerated as SKIPPED.
# Keep this set aligned with ``compgen.transforms.verify.VerificationLevel``.
_REQUIRED_VERIFY_LEVELS: frozenset[str] = frozenset({"structural", "differential"})


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


def _bundle_root(bundle: Bundle) -> Path | None:
    """Return the on-disk root of a bundle, if recorded."""
    root = bundle.metadata.get("bundle_root") if bundle.metadata else None
    return Path(root) if root else None


def _inspect_verification(bundle: Bundle) -> VerificationGateResult:
    """Read the bundle's ``verification_report.json`` and decide.

    Gate semantics:

    1. The bundle must record ``verification_report`` in its manifest
       artifacts AND the referenced file must exist.
    2. The file must parse as JSON with the schema written by
       :func:`compgen.api._run_inline_verification`:
       ``{passed: bool, levels_run: [str], levels_passed: [str],
         max_abs_error: float|null, details: {...}}``.
    3. ``passed`` must be ``True``.
    4. Every level in :data:`_REQUIRED_VERIFY_LEVELS` must appear in
       both ``levels_run`` and ``levels_passed``. If a required level
       is present in ``levels_run`` but NOT in ``levels_passed``, that
       is a failure. If it isn't run at all, the gate also fails — a
       production bundle must have its core ladder exercised.

    Returns:
        A :class:`VerificationGateResult` with ``passed`` and, when
        failing, a list of :class:`PromotionBlockReason` entries.
    """
    reasons: list[PromotionBlockReason] = []
    artifacts = bundle.artifacts if bundle.artifacts else {}
    verify_rel = artifacts.get("verification_report")
    root = _bundle_root(bundle)

    if not verify_rel:
        reasons.append(
            PromotionBlockReason(
                code="missing_verification_report",
                detail="bundle manifest has no 'verification_report' entry under artifacts",
            )
        )
        return VerificationGateResult(passed=False, reasons=reasons)

    if root is None:
        reasons.append(
            PromotionBlockReason(
                code="missing_verification_report",
                detail="bundle has no metadata.bundle_root; can't locate verification_report.json",
            )
        )
        return VerificationGateResult(passed=False, reasons=reasons)

    report_path = root / verify_rel
    if not report_path.is_file():
        reasons.append(
            PromotionBlockReason(
                code="missing_verification_report",
                detail=f"verification_report.json does not exist at {report_path}",
                path=str(report_path),
            )
        )
        return VerificationGateResult(passed=False, reasons=reasons)

    try:
        report = json.loads(report_path.read_text())
    except Exception as exc:
        reasons.append(
            PromotionBlockReason(
                code="verification_report_unreadable",
                detail=f"{report_path}: {exc!r}",
                path=str(report_path),
            )
        )
        return VerificationGateResult(passed=False, reasons=reasons)

    passed_top = bool(report.get("passed", False))
    levels_run = set(report.get("levels_run") or [])
    levels_passed = set(report.get("levels_passed") or [])
    details = report.get("details") or {}

    if not passed_top:
        reasons.append(
            PromotionBlockReason(
                code="verification_failed",
                detail=(
                    "verification_report.passed is False; "
                    f"max_abs_error={report.get('max_abs_error')!r}; "
                    f"details={details!r}"
                ),
                path=str(report_path),
            )
        )

    missing_required = _REQUIRED_VERIFY_LEVELS - levels_run
    if missing_required:
        reasons.append(
            PromotionBlockReason(
                code="level_skipped",
                detail=f"required verification level(s) not run: {sorted(missing_required)}",
                path=str(report_path),
            )
        )

    failing_required = _REQUIRED_VERIFY_LEVELS & levels_run - levels_passed
    if failing_required:
        reasons.append(
            PromotionBlockReason(
                code="level_failed_strict",
                detail=f"required verification level(s) failed: {sorted(failing_required)}",
                path=str(report_path),
            )
        )

    # An explicit "SKIPPED" substring in the details surfaces required
    # levels that claim PASS but actually did no work.
    for lvl in _REQUIRED_VERIFY_LEVELS:
        msg = str(details.get(lvl, ""))
        if "SKIPPED" in msg.upper():
            reasons.append(
                PromotionBlockReason(
                    code="level_skipped",
                    detail=f"required level '{lvl}' reported SKIPPED in details: {msg!r}",
                    path=str(report_path),
                )
            )

    return VerificationGateResult(
        passed=(len(reasons) == 0),
        reasons=reasons,
        report=report,
    )


@dataclass
class RecipePromoter:
    """Promotes verified bundles to the recipe library.

    Attributes:
        library_path: Path to the recipe library directory.
    """

    library_path: Path

    def promote(self, bundle: Bundle, force: bool = False) -> PromotionResult:
        """Promote a verified bundle to the recipe library.

        Enforces the verification gate: the bundle's
        ``verification_report.json`` must exist, be parseable, and
        report ``passed=True`` with every required level in
        :data:`_REQUIRED_VERIFY_LEVELS` actually run (not SKIPPED).

        Args:
            bundle: The bundle manifest (must reference a valid bundle directory).
            force: Skip the verification gate. Use only when you're
                knowingly promoting an unverified bundle (e.g. a
                baseline capture) — the gate is there to prevent
                promoting a broken compile.

        Returns:
            :class:`PromotionResult`.

        Raises:
            PromotionBlockedError: ``force=False`` and the bundle
                failed the verification gate.
        """
        # Verification gate — production-grade, not advisory. This used
        # to be absent; the docstring on this module promised
        # "Promotion requires a passing verification_report.json" but
        # the code happily promoted anything. Fixed here.
        if not force:
            gate = _inspect_verification(bundle)
            if not gate.passed:
                raise PromotionBlockedError(
                    reasons=gate.reasons,
                    bundle_root=_bundle_root(bundle),
                )

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

        # Copy full bundle artifacts into promoted directory
        bundle_root = bundle.metadata.get("bundle_root") if bundle.metadata else None
        if bundle_root:
            import shutil

            src = Path(bundle_root)
            if src.is_dir():
                for artifact_name, rel_path in bundle.artifacts.items():
                    artifact_src = src / rel_path
                    if artifact_src.exists():
                        artifact_dest = dest / rel_path
                        artifact_dest.parent.mkdir(parents=True, exist_ok=True)
                        if artifact_src.is_dir():
                            shutil.copytree(artifact_src, artifact_dest, dirs_exist_ok=True)
                        else:
                            shutil.copy2(artifact_src, artifact_dest)

        # Write manifest (always, even if bundle_root was unavailable)
        manifest_path = dest / "manifest.json"
        manifest_path.write_text(json.dumps(bundle.to_dict(), indent=2))

        # Record audit event
        try:
            from compgen.promotion.audit import AuditLog, create_event

            audit = AuditLog(self.library_path / "audit.jsonl")
            audit.record(
                create_event(
                    "promotion",
                    data={"key": key.key, "target": bundle.target_profile, "version": version},
                )
            )
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
