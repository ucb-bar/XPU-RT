"""Transpose profitability analysis.

Classifies every transpose-like operation in a network analysis as
absorbable, propagatable, eliminable, or a hard boundary that must
materialise.  The classification drives later layout optimisation passes.
"""

from __future__ import annotations

import logging
from enum import Enum

from compgen.agent.analyzer import NetworkAnalysis, PatternCluster
from compgen.ir.payload.contracts import KernelContract

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pattern keywords
# ---------------------------------------------------------------------------
_TRANSPOSE_KEYWORDS = ("transpose", "permute", "t(")
_ELEMENTWISE_KEYWORDS = ("elementwise", "relu", "gelu", "add", "mul", "sigmoid", "tanh", "silu")


def _is_transpose_like(pattern_type: str) -> bool:
    """Return True if *pattern_type* looks like a transpose operation."""
    lower = pattern_type.lower()
    return any(kw in lower for kw in _TRANSPOSE_KEYWORDS)


def _is_elementwise_like(pattern_type: str) -> bool:
    """Return True if *pattern_type* is a pure elementwise operation."""
    lower = pattern_type.lower()
    return any(kw in lower for kw in _ELEMENTWISE_KEYWORDS)


# ---------------------------------------------------------------------------
# TransposeClassification
# ---------------------------------------------------------------------------


class TransposeClassification(Enum):
    """Classification of a transpose operation for layout optimisation.

    Attributes:
        ABSORBABLE: Downstream kernel can handle the transposed layout directly.
        PROPAGATABLE: Can be pushed through elementwise ops at zero cost.
        BOUNDARY: Must materialise because the consumer requires a specific layout.
        ELIMINABLE: Double-transpose or identity -- can be removed entirely.
    """

    ABSORBABLE = "absorbable"
    PROPAGATABLE = "propagatable"
    BOUNDARY = "boundary"
    ELIMINABLE = "eliminable"


# ---------------------------------------------------------------------------
# TransposeProfitabilityAnalyzer
# ---------------------------------------------------------------------------


class TransposeProfitabilityAnalyzer:
    """Classifies transpose operations in a network analysis.

    For each cluster that involves a transpose-like pattern the analyser
    determines whether it can be absorbed by a consumer kernel, propagated
    through elementwise neighbours, eliminated as a redundant pair, or must
    remain as a materialised layout boundary.
    """

    def classify_transposes(
        self,
        analysis: NetworkAnalysis,
        contracts: list[KernelContract],
    ) -> dict[str, TransposeClassification]:
        """Classify every transpose-like node in *analysis*.

        Args:
            analysis: A ``NetworkAnalysis`` produced by ``NetworkAnalyzer``.
            contracts: Kernel contracts extracted from the payload IR.  Used
                to determine whether a consumer kernel can absorb transposes.

        Returns:
            Dict mapping FX node name to its ``TransposeClassification``.
        """
        contract_by_name = self._build_contract_index(contracts)
        cluster_by_id = {c.cluster_id: c for c in analysis.clusters}

        # Build a cluster adjacency map from the data flow edges.
        consumer_map: dict[str, list[str]] = {}
        for edge in analysis.data_flow:
            consumer_map.setdefault(edge.src, []).append(edge.dst)

        # Track which cluster_ids are transpose-like so we can detect
        # back-to-back transposes.
        transpose_cluster_ids: set[str] = set()
        for cluster in analysis.clusters:
            if _is_transpose_like(cluster.pattern_type):
                transpose_cluster_ids.add(cluster.cluster_id)

        classifications: dict[str, TransposeClassification] = {}

        for cluster in analysis.clusters:
            if not _is_transpose_like(cluster.pattern_type):
                continue

            classification = self._classify_cluster(
                cluster=cluster,
                consumer_ids=consumer_map.get(cluster.cluster_id, []),
                cluster_by_id=cluster_by_id,
                contract_by_name=contract_by_name,
                transpose_cluster_ids=transpose_cluster_ids,
            )

            for name in cluster.node_names:
                classifications[name] = classification

        logger.debug(
            "TransposeProfitabilityAnalyzer classified %d nodes",
            len(classifications),
        )
        return classifications

    # -- internal helpers ----------------------------------------------------

    @staticmethod
    def _build_contract_index(
        contracts: list[KernelContract],
    ) -> dict[str, KernelContract]:
        """Index contracts by op_name for O(1) lookup."""
        index: dict[str, KernelContract] = {}
        for contract in contracts:
            index[contract.op_name] = contract
        return index

    def _classify_cluster(
        self,
        *,
        cluster: PatternCluster,
        consumer_ids: list[str],
        cluster_by_id: dict[str, PatternCluster],
        contract_by_name: dict[str, KernelContract],
        transpose_cluster_ids: set[str],
    ) -> TransposeClassification:
        """Classify a single transpose cluster."""
        # 1) Eliminable: consumer is also a transpose (double-transpose).
        for cid in consumer_ids:
            if cid in transpose_cluster_ids:
                return TransposeClassification.ELIMINABLE

        # 2) Absorbable: consumer's kernel contract declares absorption.
        for cid in consumer_ids:
            consumer = cluster_by_id.get(cid)
            if consumer is None:
                continue
            contract = contract_by_name.get(consumer.pattern_type)
            if contract is not None and contract.metadata.get("can_absorb_transpose", False):
                return TransposeClassification.ABSORBABLE
            # Also check per-node contracts within the consumer.
            for node_name in consumer.node_names:
                node_contract = contract_by_name.get(node_name)
                if node_contract is not None and node_contract.metadata.get(
                    "can_absorb_transpose", False,
                ):
                    return TransposeClassification.ABSORBABLE

        # 3) Propagatable: all consumers are elementwise.
        if consumer_ids and all(
            _is_elementwise_like(cluster_by_id[cid].pattern_type)
            for cid in consumer_ids
            if cid in cluster_by_id
        ):
            return TransposeClassification.PROPAGATABLE

        # 4) Boundary: must materialise.
        return TransposeClassification.BOUNDARY


__all__ = ["TransposeClassification", "TransposeProfitabilityAnalyzer"]
