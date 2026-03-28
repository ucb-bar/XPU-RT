"""Search task definition.

A SearchTask encapsulates one optimization problem (kernel, pass, guard,
etc.) with its state signature and retrieval priors. It's the input to
both local (beam search) and global (frontier) search loops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from compgen.memory.schema import KnowledgeItem, ObjectKind, StateSignature


@dataclass(frozen=True)
class SearchTask:
    """One optimization search problem.

    Attributes:
        task_id: Unique identifier (from CompilerMemory).
        kind: What kind of object we're searching for.
        state: State signature for retrieval.
        budget_iterations: Max search iterations.
        budget_time_ms: Max wall-clock time.
        retrieval_priors: Pre-fetched knowledge items to seed search.
    """

    task_id: str
    kind: ObjectKind
    state: StateSignature
    budget_iterations: int = 10
    budget_time_ms: int = 60_000
    retrieval_priors: list[KnowledgeItem] = field(default_factory=list)

    @classmethod
    def for_kernel(
        cls,
        task_id: str,
        op_family: str,
        shapes: str = "",
        dtype: str = "f32",
        hardware: str = "",
        budget_iterations: int = 10,
        priors: list[KnowledgeItem] | None = None,
    ) -> SearchTask:
        """Create a kernel search task."""
        state = StateSignature(
            state_id=task_id,
            task_id=task_id,
            op_family=op_family,
            shape_signature=shapes,
            dtype_signature=dtype,
            hardware_signature=hardware,
        )
        return cls(
            task_id=task_id,
            kind=ObjectKind.KERNEL,
            state=state,
            budget_iterations=budget_iterations,
            retrieval_priors=priors or [],
        )

    @classmethod
    def for_pass(
        cls,
        task_id: str,
        op_family: str,
        bottleneck: str = "",
        hardware: str = "",
        budget_iterations: int = 10,
        priors: list[KnowledgeItem] | None = None,
    ) -> SearchTask:
        """Create a pass search task."""
        state = StateSignature(
            state_id=task_id,
            task_id=task_id,
            op_family=op_family,
            bottleneck_signature=bottleneck,
            hardware_signature=hardware,
        )
        return cls(
            task_id=task_id,
            kind=ObjectKind.PASS,
            state=state,
            budget_iterations=budget_iterations,
            retrieval_priors=priors or [],
        )


__all__ = ["SearchTask"]
