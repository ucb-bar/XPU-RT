"""Contiguous segmentation for equality saturation.

Segments a module's functions into tractable pieces for e-graph
exploration. Uses contiguous instruction-order partitioning with
a threshold on non-blackboxed ops per segment (from Constable).

Key properties:
  - Preserves instruction order (temporal/spatial locality)
  - Respects dataflow edges across segments
  - Only counts profitable ops against the threshold τ
  - Blackboxed ops pass through as boundaries
"""

from __future__ import annotations

from dataclasses import dataclass, field

from xdsl.dialects import func
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation

from compgen.eqsat.blackbox import OpClass, classify_op


@dataclass
class Segment:
    """A contiguous segment of ops for e-graph exploration.

    Attributes:
        segment_id: Unique identifier.
        ops: Ordered list of operations in this segment.
        profitable_count: Number of profitable (non-blackboxed) ops.
        blackbox_count: Number of blackboxed ops.
        has_dataflow_in: Whether this segment consumes values from earlier segments.
        has_dataflow_out: Whether this segment produces values used by later segments.
    """

    segment_id: int
    ops: list[Operation] = field(default_factory=list)
    profitable_count: int = 0
    blackbox_count: int = 0
    has_dataflow_in: bool = False
    has_dataflow_out: bool = False


def segment_function(
    func_op: func.FuncOp,
    threshold: int = 200,
) -> list[Segment]:
    """Segment a function's body into contiguous pieces.

    Walks the function body in instruction order.  Each segment holds
    up to ``threshold`` profitable (non-blackboxed) ops.  When the
    threshold is reached, a new segment starts.

    Args:
        func_op: The function to segment.
        threshold: Max non-blackboxed ops per segment (τ).

    Returns:
        List of Segment objects in instruction order.
    """
    segments: list[Segment] = []
    current = Segment(segment_id=0)

    # Walk every op in every block. Real captures (e.g. HF Llama with
    # attention masking) emit func bodies whose Region has multiple
    # blocks — using ``.block`` would raise on those.
    for block in func_op.body.blocks:
        for op in block.ops:
            # Skip structural ops
            if isinstance(op, func.ReturnOp):
                continue

            if not op.results:
                continue

            classification = classify_op(op)

            if classification == OpClass.PROFITABLE:
                # Check if adding this op exceeds threshold
                if current.profitable_count >= threshold and current.profitable_count > 0:
                    segments.append(current)
                    current = Segment(segment_id=len(segments))

                current.ops.append(op)
                current.profitable_count += 1
            else:
                # Blackboxed ops join the current segment
                current.ops.append(op)
                current.blackbox_count += 1

    # Don't forget the last segment
    if current.ops:
        segments.append(current)

    # Mark dataflow edges
    _mark_dataflow(segments)

    return segments


def segment_module(
    module: ModuleOp,
    threshold: int = 200,
) -> list[Segment]:
    """Segment all functions in a module.

    Args:
        module: The module to segment.
        threshold: Max non-blackboxed ops per segment (τ).

    Returns:
        List of all segments across all functions.
    """
    all_segments: list[Segment] = []

    for op in module.body.block.ops:
        if isinstance(op, func.FuncOp):
            segments = segment_function(op, threshold)
            # Re-number globally
            for seg in segments:
                seg.segment_id = len(all_segments)
                all_segments.append(seg)

    return all_segments


def _mark_dataflow(segments: list[Segment]) -> None:
    """Mark which segments have cross-segment dataflow.

    A segment has dataflow_in if any of its ops use values defined
    in a previous segment. It has dataflow_out if any of its results
    are used in a later segment.
    """
    # Build op → segment mapping
    op_to_segment: dict[Operation, int] = {}
    for seg in segments:
        for op in seg.ops:
            op_to_segment[op] = seg.segment_id

    for seg in segments:
        for op in seg.ops:
            # Check operands: are they from a different segment?
            for operand in op.operands:
                if hasattr(operand, "owner"):
                    owner = operand.owner
                    if isinstance(owner, Operation):
                        owner_seg = op_to_segment.get(owner)
                        if owner_seg is not None and owner_seg != seg.segment_id:
                            seg.has_dataflow_in = True
                            segments[owner_seg].has_dataflow_out = True
