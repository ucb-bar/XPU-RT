"""Prepack opportunity identification.

Scans a NetworkAnalysis to find operands that benefit from prepacking --
converting their memory layout ahead of time so that every consumer reads
an already-optimal buffer.  Constant weights and high-reuse operands are
the primary targets.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass

from compgen.agent.analyzer import NetworkAnalysis

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Heuristic constants
# ---------------------------------------------------------------------------

# Assumed bytes-per-element for rough benefit estimation when actual size is
# unavailable.  4 bytes = float32.
_DEFAULT_BYTES_PER_ELEMENT = 4

# Cost factor (microseconds per byte) modelling the overhead of an
# un-prepacked materialisation at runtime.
_MATERIALIZATION_COST_FACTOR_US = 1e-4

# Prefixes / substrings that indicate a constant (parameter) operand.
_CONSTANT_PREFIXES = ("p_", "weight", "bias", "embed")


def _is_constant_name(name: str) -> bool:
    """Heuristic: return True if *name* looks like a model parameter."""
    lower = name.lower()
    return any(lower.startswith(pfx) or pfx in lower for pfx in _CONSTANT_PREFIXES)


# ---------------------------------------------------------------------------
# PrepackCandidate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrepackCandidate:
    """A single operand that is a candidate for prepacking.

    Attributes:
        region_id: Region that consumes this operand.
        operand_name: FX node name of the operand.
        operand_index: Positional index of the operand within the region.
        is_constant: Whether the operand is a constant (weight/bias).
        reuse_count: Number of consuming regions for this operand.
        estimated_benefit_us: Estimated time saved by prepacking (microseconds).
    """

    region_id: str
    operand_name: str
    operand_index: int
    is_constant: bool
    reuse_count: int
    estimated_benefit_us: float


# ---------------------------------------------------------------------------
# PrepackPlanner
# ---------------------------------------------------------------------------


class PrepackPlanner:
    """Identifies operands worth prepacking across the computation graph.

    The planner scans clusters for constant weights and operands consumed
    by multiple regions, estimates the benefit of prepacking each, and
    returns a sorted list of candidates (best first).
    """

    def identify_prepack_opportunities(
        self,
        analysis: NetworkAnalysis,
    ) -> list[PrepackCandidate]:
        """Scan the analysis and return prepack candidates sorted by benefit.

        Args:
            analysis: A ``NetworkAnalysis`` produced by ``NetworkAnalyzer``.

        Returns:
            List of ``PrepackCandidate`` sorted by ``estimated_benefit_us``
            descending.
        """
        # Step 1: Build a cross-region operand reuse map.
        # Count how many clusters each node name appears in.
        node_region_count: Counter[str] = Counter()
        node_to_regions: dict[str, list[str]] = {}
        for cluster in analysis.clusters:
            for name in cluster.node_names:
                node_region_count[name] += 1
                node_to_regions.setdefault(name, []).append(cluster.cluster_id)

        # Step 2: Collect candidates from every cluster.
        candidates: list[PrepackCandidate] = []

        for cluster in analysis.clusters:
            for idx, name in enumerate(cluster.node_names):
                is_const = _is_constant_name(name)
                reuse = node_region_count.get(name, 1)

                # Only consider constants or operands with cross-region reuse.
                if not is_const and reuse <= 1:
                    continue

                # Rough byte estimate: use cluster-level total_bytes divided
                # evenly across nodes as a proxy when per-operand sizes are
                # unavailable.
                node_count = max(len(cluster.node_names), 1)
                approx_bytes = cluster.total_bytes / node_count

                benefit = reuse * approx_bytes * _MATERIALIZATION_COST_FACTOR_US

                candidates.append(
                    PrepackCandidate(
                        region_id=cluster.cluster_id,
                        operand_name=name,
                        operand_index=idx,
                        is_constant=is_const,
                        reuse_count=reuse,
                        estimated_benefit_us=benefit,
                    )
                )

        # Sort best-first.
        candidates.sort(key=lambda c: c.estimated_benefit_us, reverse=True)

        logger.debug("PrepackPlanner found %d candidates", len(candidates))
        return candidates


__all__ = ["PrepackCandidate", "PrepackPlanner"]
