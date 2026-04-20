"""Retrieval-augmented search seeding.

Before generating a new candidate, retrieve prior knowledge from
the unified memory to seed the search. This implements the
KernelBlaster-style profiling-guided retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

from compgen.memory.schema import KnowledgeItem
from compgen.memory.search.task import SearchTask
from compgen.memory.store import CompilerMemory

log = structlog.get_logger()


@dataclass(frozen=True)
class RetrievalResult:
    """What retrieval found to seed search.

    Attributes:
        schedule_templates: Reusable schedule/code templates.
        tactics: Optimization tactics (hardware rules, strategies).
        error_repairs: Known failure→fix patterns.
        similar_candidates: Prior candidates from similar tasks.
    """

    schedule_templates: list[KnowledgeItem] = field(default_factory=list)
    tactics: list[KnowledgeItem] = field(default_factory=list)
    error_repairs: list[KnowledgeItem] = field(default_factory=list)
    similar_candidates: list[KnowledgeItem] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.schedule_templates) + len(self.tactics) + len(self.error_repairs) + len(self.similar_candidates)

    @property
    def is_empty(self) -> bool:
        return self.total == 0


class SearchRetriever:
    """Retrieve prior knowledge to seed search.

    Uses the task's state signature to find relevant knowledge from
    the unified memory. Retrieval is by op_family, hardware, and
    bottleneck — not just exact workload hash.

    Attributes:
        memory: The unified CompilerMemory.
        top_k: Maximum items per category.
    """

    def __init__(self, memory: CompilerMemory, top_k: int = 5) -> None:
        self.memory = memory
        self.top_k = top_k

    def retrieve_for_task(self, task: SearchTask) -> RetrievalResult:
        """Retrieve all relevant knowledge for a search task.

        Args:
            task: The search task with its state signature.

        Returns:
            RetrievalResult with categorized knowledge items.
        """
        state = task.state

        # Retrieve by category
        from compgen.memory.schema import KnowledgeKind

        templates = self.memory.retrieve_knowledge(
            kind=KnowledgeKind.SCHEDULE_TEMPLATE,
            top_k=self.top_k,
        )

        tactics = self.memory.retrieve_similar(
            op_family=state.op_family,
            hardware_signature=state.hardware_signature,
            bottleneck_signature=state.bottleneck_signature,
            top_k=self.top_k,
        )

        repairs = self.memory.retrieve_knowledge(
            kind=KnowledgeKind.ERROR_REPAIR,
            top_k=self.top_k,
        )

        # Filter tactics by kind
        actual_tactics = [t for t in tactics if t.knowledge_kind != KnowledgeKind.SCHEDULE_TEMPLATE]
        actual_templates = templates + [t for t in tactics if t.knowledge_kind == KnowledgeKind.SCHEDULE_TEMPLATE]

        # Deduplicate
        seen: set[str] = set()
        unique_templates: list[KnowledgeItem] = []
        for item in actual_templates:
            if item.knowledge_id not in seen:
                seen.add(item.knowledge_id)
                unique_templates.append(item)

        result = RetrievalResult(
            schedule_templates=unique_templates[: self.top_k],
            tactics=actual_tactics[: self.top_k],
            error_repairs=repairs[: self.top_k],
        )

        log.info(
            "search.retrieve",
            task=task.task_id,
            op_family=state.op_family,
            templates=len(result.schedule_templates),
            tactics=len(result.tactics),
            repairs=len(result.error_repairs),
        )

        return result


__all__ = ["RetrievalResult", "SearchRetriever"]
