"""Learned cost model weights — per-target storage and retrieval.

Tracks which cost model weight tuples (fusion_weight, transfer_weight,
backend_match_weight) work best for each target family. Persists in
CompilerMemory so future compilations for the same target start with
better defaults.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from compgen.memory.store import CompilerMemory

log = structlog.get_logger()


def store_cost_weights(
    memory: CompilerMemory,
    target_key: str,
    weights: dict[str, float],
    measured_gain: float,
) -> str:
    """Store a successful cost weight configuration.

    Args:
        memory: CompilerMemory instance.
        target_key: Target profile name.
        weights: Dict with fusion_weight, transfer_weight, backend_match_weight.
        measured_gain: Improvement percentage achieved with these weights.

    Returns:
        Knowledge item ID.
    """
    from compgen.memory.schema import KnowledgeKind, ScopeKind

    summary = (
        f"cost_weights target={target_key} "
        f"fusion={weights.get('fusion_weight', 1.0):.3f} "
        f"transfer={weights.get('transfer_weight', 1.0):.3f} "
        f"backend={weights.get('backend_match_weight', 1.0):.3f} "
        f"gain={measured_gain:+.1f}%"
    )
    artifact = json.dumps({
        "weights": weights,
        "measured_gain": measured_gain,
        "target_key": target_key,
    })

    item = memory.store_knowledge(
        kind=KnowledgeKind.HARDWARE_RULE,
        summary=summary,
        artifact=artifact,
        scope_kind=ScopeKind.TARGET,
        scope_key=target_key,
        source="learned_weights",
    )
    log.info("learned_weights.stored", target=target_key, gain=measured_gain)
    return item.knowledge_id


def retrieve_best_weights(
    memory: CompilerMemory,
    target_key: str,
) -> dict[str, float] | None:
    """Retrieve the best-performing cost weights for a target.

    Args:
        memory: CompilerMemory instance.
        target_key: Target profile name.

    Returns:
        Dict with weight keys, or None if no prior data.
    """
    from compgen.memory.schema import KnowledgeKind, ScopeKind

    items = memory.retrieve_knowledge(
        kind=KnowledgeKind.HARDWARE_RULE,
        scope_kind=ScopeKind.TARGET,
        scope_key=target_key,
        top_k=5,
    )

    # Filter to learned_weights source and pick highest quality
    weight_items = [i for i in items if i.source == "learned_weights"]
    if not weight_items:
        return None

    best = weight_items[0]  # Already sorted by quality_score DESC
    try:
        blob = memory.blobs.load(best.artifact_hash)
        data = json.loads(blob)
        weights = data.get("weights", {})
        if weights:
            log.info("learned_weights.retrieved", target=target_key, weights=weights)
            return weights
    except Exception:
        pass

    return None


__all__ = ["retrieve_best_weights", "store_cost_weights"]
