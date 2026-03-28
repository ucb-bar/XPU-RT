"""Candidate lineage graph querying.

Walks ``parent_candidate_id`` chains in the candidates table and
correlates with promotion records to build a full lineage graph
for any candidate.  Useful for understanding how a promoted recipe
evolved through mutation / refinement rounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog

from compgen.memory.schema import Candidate, Promotion
from compgen.memory.store import CompilerMemory

log = structlog.get_logger()


@dataclass(frozen=True)
class LineageNode:
    """Single node in a candidate lineage graph.

    Attributes:
        candidate_id: Unique ID of this candidate.
        parent_id: ID of the parent candidate, or ``None`` for root.
        status: Lifecycle status string (from ``CandidateStatus``).
        generation_round: Generation round in the search loop.
        generator_kind: How this candidate was produced (from ``GeneratorKind``).
        promotion_key: Promotion key if the candidate was promoted, else ``None``.
    """

    candidate_id: str
    parent_id: str | None
    status: str
    generation_round: int
    generator_kind: str
    promotion_key: str | None


@dataclass(frozen=True)
class LineageGraph:
    """Ordered lineage from root ancestor to the queried candidate.

    Attributes:
        nodes: Tuple of ``LineageNode`` ordered from root (oldest ancestor) to leaf.
        root_id: Candidate ID of the oldest ancestor.
    """

    nodes: tuple[LineageNode, ...]
    root_id: str


def _row_to_promotion(row: Any) -> Promotion:
    """Convert a SQLite Row to a Promotion dataclass."""
    return Promotion(
        promotion_id=row["promotion_id"],
        candidate_id=row["candidate_id"],
        promotion_key=row["promotion_key"],
        version=row["version"],
        reason=row["reason"],
        measured_gain=row["measured_gain"],
        verified_by=row["verified_by"],
        created_at=row["created_at"],
    )


def _get_candidate_by_id(memory: CompilerMemory, candidate_id: str) -> Candidate | None:
    """Fetch a single candidate by its ID."""
    row = memory.db.fetchone(
        "SELECT * FROM candidates WHERE candidate_id = ?",
        (candidate_id,),
    )
    if row is None:
        return None
    return CompilerMemory._row_to_candidate(row)


def _get_promotion_for_candidate(memory: CompilerMemory, candidate_id: str) -> Promotion | None:
    """Fetch the promotion record for a candidate, if any."""
    row = memory.db.fetchone(
        "SELECT * FROM promotions WHERE candidate_id = ?",
        (candidate_id,),
    )
    if row is None:
        return None
    return _row_to_promotion(row)


def build_lineage_graph(memory: CompilerMemory, candidate_id: str) -> LineageGraph:
    """Walk the ``parent_candidate_id`` chain upward to the root ancestor.

    For each node in the chain the corresponding promotion record (if any)
    is fetched so callers can see which ancestors were promoted.

    Args:
        memory: Active ``CompilerMemory`` instance.
        candidate_id: Starting candidate (leaf of the chain).

    Returns:
        A ``LineageGraph`` ordered root-to-leaf.  If the candidate does not
        exist the graph has an empty ``nodes`` tuple and ``root_id`` is ``""``.
    """
    nodes: list[LineageNode] = []
    visited: set[str] = set()
    current_id: str | None = candidate_id

    while current_id and current_id not in visited:
        visited.add(current_id)
        candidate = _get_candidate_by_id(memory, current_id)
        if candidate is None:
            break

        promotion = _get_promotion_for_candidate(memory, candidate.candidate_id)
        node = LineageNode(
            candidate_id=candidate.candidate_id,
            parent_id=candidate.parent_candidate_id or None,
            status=candidate.status.value,
            generation_round=candidate.generation_round,
            generator_kind=candidate.generator_kind.value,
            promotion_key=promotion.promotion_key if promotion else None,
        )
        nodes.append(node)

        current_id = candidate.parent_candidate_id or None

    # Reverse so root is first (we collected leaf-to-root).
    nodes.reverse()

    if not nodes:
        log.debug("lineage.empty", candidate_id=candidate_id)
        return LineageGraph(nodes=(), root_id="")

    return LineageGraph(nodes=tuple(nodes), root_id=nodes[0].candidate_id)


def get_promotion_history(memory: CompilerMemory, promotion_key: str) -> list[Promotion]:
    """Return all promotion versions for a given key, ordered by version.

    Args:
        memory: Active ``CompilerMemory`` instance.
        promotion_key: The key to look up (e.g. ``"matmul_h100_latency"``).

    Returns:
        List of ``Promotion`` records ordered ascending by version.
    """
    rows = memory.db.fetchall(
        "SELECT * FROM promotions WHERE promotion_key = ? ORDER BY version",
        (promotion_key,),
    )
    return [_row_to_promotion(r) for r in rows]


def find_lineage_siblings(memory: CompilerMemory, candidate_id: str) -> list[Candidate]:
    """Find other candidates sharing the same parent as *candidate_id*.

    The returned list does **not** include *candidate_id* itself.

    Args:
        memory: Active ``CompilerMemory`` instance.
        candidate_id: The candidate whose siblings we want.

    Returns:
        List of sibling ``Candidate`` objects (may be empty).
    """
    candidate = _get_candidate_by_id(memory, candidate_id)
    if candidate is None or not candidate.parent_candidate_id:
        return []

    rows = memory.db.fetchall(
        "SELECT * FROM candidates WHERE parent_candidate_id = ? AND candidate_id != ? ORDER BY generation_round",
        (candidate.parent_candidate_id, candidate_id),
    )
    return [CompilerMemory._row_to_candidate(r) for r in rows]


__all__ = [
    "LineageGraph",
    "LineageNode",
    "build_lineage_graph",
    "find_lineage_siblings",
    "get_promotion_history",
]
