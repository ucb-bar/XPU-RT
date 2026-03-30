"""Performance feedback calibration — estimated vs measured cost tracking.

Records pairs of (estimated_latency, measured_latency) per target and
op_family. Computes calibration factors to correct future cost estimates.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from compgen.memory.store import CompilerMemory

log = structlog.get_logger()


def record_calibration(
    memory: CompilerMemory,
    target_key: str,
    op_family: str,
    estimated_us: float,
    measured_us: float,
) -> str:
    """Record an estimated-vs-measured calibration pair.

    Args:
        memory: CompilerMemory instance.
        target_key: Target profile name.
        op_family: Operation family (e.g., "matmul").
        estimated_us: Estimated latency in microseconds.
        measured_us: Measured latency in microseconds.

    Returns:
        Knowledge item ID.
    """
    from compgen.memory.schema import KnowledgeKind, ScopeKind

    ratio = measured_us / max(estimated_us, 1e-9)
    summary = (
        f"calibration {op_family}@{target_key}: "
        f"est={estimated_us:.1f}us meas={measured_us:.1f}us ratio={ratio:.2f}"
    )
    artifact = json.dumps({
        "target_key": target_key,
        "op_family": op_family,
        "estimated_us": estimated_us,
        "measured_us": measured_us,
        "ratio": ratio,
    })

    item = memory.store_knowledge(
        kind=KnowledgeKind.HARDWARE_RULE,
        summary=summary,
        artifact=artifact,
        scope_kind=ScopeKind.OPERATOR_FAMILY,
        scope_key=op_family,
        source="calibration",
    )
    log.info(
        "calibration.recorded",
        target=target_key,
        op=op_family,
        ratio=f"{ratio:.2f}",
    )
    return item.knowledge_id


def get_calibration_factor(
    memory: CompilerMemory,
    target_key: str,
    op_family: str,
) -> float:
    """Get the average calibration factor for a target + op_family.

    The factor is `measured / estimated`. Multiply an estimate by this
    factor to get a corrected prediction.

    Args:
        memory: CompilerMemory instance.
        target_key: Target profile name.
        op_family: Operation family.

    Returns:
        Average calibration factor, or 1.0 if no data.
    """
    from compgen.memory.schema import KnowledgeKind, ScopeKind

    items = memory.retrieve_knowledge(
        kind=KnowledgeKind.HARDWARE_RULE,
        scope_kind=ScopeKind.OPERATOR_FAMILY,
        scope_key=op_family,
        top_k=20,
    )

    ratios: list[float] = []
    for item in items:
        if item.source != "calibration":
            continue
        try:
            blob = memory.blobs.load(item.artifact_hash)
            data = json.loads(blob)
            if data.get("target_key") == target_key:
                ratios.append(data["ratio"])
        except Exception:
            continue

    if not ratios:
        return 1.0

    avg = sum(ratios) / len(ratios)
    log.info(
        "calibration.factor",
        target=target_key,
        op=op_family,
        factor=f"{avg:.2f}",
        samples=len(ratios),
    )
    return avg


def calibrate_cost(estimated_us: float, factor: float) -> float:
    """Apply calibration factor to an estimated cost.

    Args:
        estimated_us: Original estimate in microseconds.
        factor: Calibration factor (measured/estimated ratio).

    Returns:
        Corrected estimate.
    """
    return estimated_us * factor


__all__ = ["calibrate_cost", "get_calibration_factor", "record_calibration"]
