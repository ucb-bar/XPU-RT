"""Post-search promotion and knowledge extraction.

After a search completes, this module:
1. Promotes the best candidate to L3 (immutable library)
2. Extracts reusable knowledge from the search trajectory
3. Updates quality scores for retrieved knowledge that was used
"""

from __future__ import annotations

import structlog

from compgen.memory.schema import (
    CandidateStatus,
    KnowledgeItem,
    KnowledgeKind,
    Promotion,
    ScopeKind,
)
from compgen.memory.store import CompilerMemory

log = structlog.get_logger()


class SearchPromoter:
    """Post-search promotion and knowledge extraction.

    Attributes:
        memory: The unified CompilerMemory.
    """

    def __init__(self, memory: CompilerMemory) -> None:
        self.memory = memory

    def promote_best(
        self,
        task_id: str,
        promotion_key: str = "",
        min_score: float = 0.0,
    ) -> Promotion | None:
        """Promote the best verified candidate for a task.

        Args:
            task_id: The task to promote from.
            promotion_key: Key for the promotion library.
            min_score: Minimum score threshold for promotion.

        Returns:
            Promotion record, or None if no candidate qualifies.
        """
        candidates = self.memory.get_candidates(task_id, CandidateStatus.VERIFIED)
        if not candidates:
            candidates = self.memory.get_candidates(task_id)

        if not candidates:
            return None

        # Find the best candidate by evaluation score
        best_candidate = None
        best_score = min_score

        for candidate in candidates:
            evals = self.memory.get_evaluations(candidate.candidate_id)
            for eval_ in evals:
                if eval_.correctness_ok and eval_.score > best_score:
                    best_score = eval_.score
                    best_candidate = candidate

        if best_candidate is None:
            return None

        promo = self.memory.promote_candidate(
            candidate_id=best_candidate.candidate_id,
            promotion_key=promotion_key or f"{task_id}_best",
            measured_gain=best_score,
            reason="search_promotion",
        )

        log.info(
            "search.promote",
            task_id=task_id,
            candidate_id=best_candidate.candidate_id,
            score=best_score,
        )

        return promo

    def extract_knowledge(
        self,
        task_id: str,
        task_kind: str = "",
        op_family: str = "",
    ) -> list[KnowledgeItem]:
        """Extract reusable knowledge from a completed search.

        Analyzes the search trajectory and promoted candidates to
        extract tactics, templates, and repair patterns that can
        help future searches.

        Args:
            task_id: The completed task.
            task_kind: Object kind for scoping.
            op_family: Operator family for scoping.

        Returns:
            List of newly created knowledge items.
        """
        items: list[KnowledgeItem] = []

        # Get promoted candidates
        promoted = self.memory.get_candidates(task_id, CandidateStatus.PROMOTED)
        for candidate in promoted:
            artifact = self.memory.blobs.load(candidate.artifact_hash)
            if not artifact:
                continue

            # Extract the plan/tactic as a knowledge item
            evals = self.memory.get_evaluations(candidate.candidate_id)
            best_eval = max(evals, key=lambda e: e.score) if evals else None

            summary = (
                f"{task_kind} tactic for {op_family}: score={best_eval.score:.2f}, latency={best_eval.latency_us:.1f}us"
                if best_eval
                else f"{task_kind} tactic for {op_family}"
            )

            item = self.memory.store_knowledge(
                kind=KnowledgeKind.OPTIMIZATION_TACTIC,
                summary=summary,
                artifact=artifact[:500],  # first 500 chars as template
                scope_kind=ScopeKind.OPERATOR_FAMILY if op_family else ScopeKind.GLOBAL,
                scope_key=op_family,
                source=f"search:{task_id}",
            )
            items.append(item)

        log.info("search.extract_knowledge", task_id=task_id, items=len(items))
        return items

    def update_retrieval_stats(
        self,
        used_knowledge_ids: list[str],
        task_succeeded: bool,
    ) -> None:
        """Update quality scores for knowledge items used in search.

        Args:
            used_knowledge_ids: IDs of knowledge items that were retrieved and used.
            task_succeeded: Whether the overall search succeeded.
        """
        for kid in used_knowledge_ids:
            self.memory.record_knowledge_use(kid, won=task_succeeded)


__all__ = ["SearchPromoter"]
