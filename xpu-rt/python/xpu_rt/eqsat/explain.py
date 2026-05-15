"""E-graph summaries for LLM consumption.

Produces compact, structured summaries of e-graph state that are
designed for agent/LLM prompts, not human reading.
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.dialects import equivalence
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import OpResult


@dataclass(frozen=True)
class EClassSummary:
    """Summary of one e-class."""

    eclass_id: int
    num_alternatives: int
    op_types: tuple[str, ...]
    has_constant: bool


@dataclass(frozen=True)
class EGraphSummary:
    """Complete summary of an e-graph for agent consumption."""

    num_eclasses: int
    num_enodes: int
    ambiguous_eclasses: int
    eclass_summaries: tuple[EClassSummary, ...]
    op_type_counts: dict[str, int]


def summarize_egraph(module: ModuleOp) -> EGraphSummary:
    """Produce a compact summary of the e-graph in a module.

    The module should have equivalence.class ops (call create_egraph first).

    Args:
        module: Module with equivalence.class ops.

    Returns:
        EGraphSummary with stats about the e-graph.
    """
    eclass_summaries: list[EClassSummary] = []
    op_type_counts: dict[str, int] = {}
    eclass_id = 0
    total_enodes = 0

    for op in module.walk():
        if isinstance(op, equivalence.AnyClassOp):
            # Collect op types of alternatives in this eclass
            op_types: list[str] = []
            has_constant = isinstance(op, equivalence.ConstantClassOp)
            for operand in op.operands:
                if isinstance(operand, OpResult):
                    op_name = operand.owner.name
                    op_types.append(op_name)
                    op_type_counts[op_name] = op_type_counts.get(op_name, 0) + 1
                    total_enodes += 1

            eclass_summaries.append(
                EClassSummary(
                    eclass_id=eclass_id,
                    num_alternatives=len(op.operands),
                    op_types=tuple(op_types),
                    has_constant=has_constant,
                )
            )
            eclass_id += 1

    ambiguous = sum(1 for s in eclass_summaries if s.num_alternatives > 1)

    return EGraphSummary(
        num_eclasses=len(eclass_summaries),
        num_enodes=total_enodes,
        ambiguous_eclasses=ambiguous,
        eclass_summaries=tuple(eclass_summaries),
        op_type_counts=op_type_counts,
    )


def summary_to_prompt(summary: EGraphSummary) -> str:
    """Convert an EGraphSummary to a compact text prompt for the LLM.

    Args:
        summary: E-graph summary.

    Returns:
        Compact text string for LLM context.
    """
    lines: list[str] = [
        f"E-graph: {summary.num_eclasses} e-classes, {summary.num_enodes} e-nodes, "
        f"{summary.ambiguous_eclasses} ambiguous",
    ]

    if summary.op_type_counts:
        top_ops = sorted(summary.op_type_counts.items(), key=lambda x: -x[1])[:8]
        ops_str = ", ".join(f"{name}:{count}" for name, count in top_ops)
        lines.append(f"Ops: {ops_str}")

    if summary.ambiguous_eclasses > 0:
        ambiguous = [s for s in summary.eclass_summaries if s.num_alternatives > 1]
        for s in ambiguous[:5]:
            lines.append(f"  eclass_{s.eclass_id}: {s.num_alternatives} alternatives [{', '.join(s.op_types)}]")

    return "\n".join(lines)
