"""Semantic dialect definitions.

Defines the semantic primitives used to encode dialect semantics for
verification. These are the building blocks that translation validation,
peephole verification, and dataflow analysis verification are built on.

Invariants:
    - Semantic ops have precise mathematical meaning.
    - Every semantic op can lower to an SMT query.
    - Semantic dialects are dialect-agnostic (work for any payload dialect).

TODO: Implement core semantic ops (bitvector, integer, refinement).
TODO: Implement SMT lowering for each semantic op.
TODO: Define how payload dialect ops lower to semantic ops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SemanticType:
    """A type in the semantic domain.

    Attributes:
        kind: Type kind ("bitvector", "integer", "boolean", "real", "array").
        width: Bit width (for bitvector types).
        params: Additional type parameters.
    """

    kind: str
    width: int | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PredicateOp:
    """A semantic predicate (boolean-valued assertion).

    Attributes:
        name: Predicate name (e.g., "eq", "slt", "ule", "no_overflow").
        operands: Operand names or expressions.
        semantic_type: Type of the operands.
    """

    name: str
    operands: list[str] = field(default_factory=list)
    semantic_type: SemanticType = field(default_factory=lambda: SemanticType(kind="integer"))


@dataclass(frozen=True)
class RefinementRelation:
    """A refinement relation between two program states.

    Used for translation validation: the target program refines the source
    if every behavior of the target is a legal behavior of the source.

    Attributes:
        source_expr: Source program expression (symbolic).
        target_expr: Target program expression (symbolic).
        conditions: Conditions under which refinement holds.
    """

    source_expr: str
    target_expr: str
    conditions: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class InvariantOp:
    """A loop or region invariant.

    Attributes:
        region_id: Region the invariant applies to.
        predicate: The invariant predicate.
    """

    region_id: str
    predicate: PredicateOp = field(default_factory=lambda: PredicateOp(name="true"))


__all__ = ["InvariantOp", "PredicateOp", "RefinementRelation", "SemanticType"]
