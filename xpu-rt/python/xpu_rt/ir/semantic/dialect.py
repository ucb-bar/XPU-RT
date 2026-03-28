"""Semantic dialect definitions.

Defines the semantic primitives used to encode dialect semantics for
verification. These are the building blocks that translation validation,
peephole verification, and dataflow analysis verification are built on.

Invariants:
    - Semantic ops have precise mathematical meaning.
    - Every semantic op can lower to an SMT query.
    - Semantic dialects are dialect-agnostic (work for any payload dialect).
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

    def to_z3_sort(self) -> Any:
        """Convert to a Z3 sort.

        Returns:
            A Z3 sort corresponding to this semantic type.

        Raises:
            RuntimeError: If z3 is not installed.
            ValueError: If the type kind is not supported.
        """
        try:
            import z3
        except ImportError as e:
            raise RuntimeError("z3-solver is required: pip install z3-solver") from e

        if self.kind == "bitvector":
            width = self.width or 32
            return z3.BitVecSort(width)
        elif self.kind == "integer":
            return z3.IntSort()
        elif self.kind == "boolean":
            return z3.BoolSort()
        elif self.kind == "real":
            return z3.RealSort()
        elif self.kind == "array":
            key_sort = z3.IntSort()
            val_sort = z3.BitVecSort(self.width or 32)
            return z3.ArraySort(key_sort, val_sort)
        else:
            raise ValueError(f"Unsupported semantic type kind: {self.kind}")


# Z3 predicate builders keyed by predicate name
def _build_predicate_map() -> dict[str, Any]:
    """Build the predicate name → Z3 builder mapping (lazy)."""
    try:
        import z3
    except ImportError:
        return {}

    def _eq(a: Any, b: Any) -> Any:
        return a == b

    def _ne(a: Any, b: Any) -> Any:
        return a != b

    def _slt(a: Any, b: Any) -> Any:
        return a < b

    def _sle(a: Any, b: Any) -> Any:
        return a <= b

    def _sgt(a: Any, b: Any) -> Any:
        return a > b

    def _sge(a: Any, b: Any) -> Any:
        return a >= b

    def _ult(a: Any, b: Any) -> Any:
        return z3.ULT(a, b)

    def _ule(a: Any, b: Any) -> Any:
        return z3.ULE(a, b)

    def _ugt(a: Any, b: Any) -> Any:
        return z3.UGT(a, b)

    def _uge(a: Any, b: Any) -> Any:
        return z3.UGE(a, b)

    return {
        "eq": _eq, "ne": _ne,
        "slt": _slt, "sle": _sle, "sgt": _sgt, "sge": _sge,
        "ult": _ult, "ule": _ule, "ugt": _ugt, "uge": _uge,
    }


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

    def to_z3(self, var_map: dict[str, Any]) -> Any:
        """Convert to a Z3 boolean expression.

        Args:
            var_map: Mapping from operand names to Z3 expressions.

        Returns:
            A Z3 boolean expression representing this predicate.
        """
        try:
            import z3
        except ImportError as e:
            raise RuntimeError("z3-solver is required: pip install z3-solver") from e

        if self.name == "true":
            return z3.BoolVal(True)
        if self.name == "false":
            return z3.BoolVal(False)

        pred_map = _build_predicate_map()
        builder = pred_map.get(self.name)
        if builder is None:
            raise ValueError(f"Unknown predicate: {self.name}")

        if len(self.operands) != 2:
            raise ValueError(f"Predicate '{self.name}' requires exactly 2 operands, got {len(self.operands)}")

        a = var_map.get(self.operands[0])
        b = var_map.get(self.operands[1])
        if a is None or b is None:
            missing = [o for o in self.operands if o not in var_map]
            raise ValueError(f"Missing variables in var_map: {missing}")

        return builder(a, b)


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

    def to_z3(self, var_map: dict[str, Any]) -> Any:
        """Convert to a Z3 refinement formula.

        The refinement check asserts: ForAll inputs,
        conditions => target_expr == source_expr.

        If the negation is UNSAT, refinement holds.

        Args:
            var_map: Mapping from expression names to Z3 expressions.

        Returns:
            A Z3 formula asserting the refinement does NOT hold
            (check UNSAT to prove refinement).
        """
        try:
            import z3
        except ImportError as e:
            raise RuntimeError("z3-solver is required: pip install z3-solver") from e

        src = var_map.get(self.source_expr)
        tgt = var_map.get(self.target_expr)
        if src is None or tgt is None:
            raise ValueError(f"Missing expressions in var_map: source={self.source_expr}, target={self.target_expr}")

        # The negation of refinement: source != target
        # If UNSAT, refinement holds
        mismatch = src != tgt

        # Add conditions as implications
        if self.conditions:
            cond_exprs = []
            for c in self.conditions:
                cond = var_map.get(c)
                if cond is not None:
                    cond_exprs.append(cond)
            if cond_exprs:
                combined_cond = z3.And(*cond_exprs) if len(cond_exprs) > 1 else cond_exprs[0]
                return z3.And(combined_cond, mismatch)

        return mismatch


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
