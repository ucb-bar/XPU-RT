"""Fusion-enabling rewrite rules for equality saturation.

These rules don't change semantics but create alternative forms
that downstream fusion passes can exploit:

- Matmul + elementwise → mark fusible boundary
- Distribution: mul(a, add(b, c)) → add(mul(a,b), mul(a,c))
  (may enable partial fusion with one branch)
"""

from __future__ import annotations

from xdsl.dialects import arith, equivalence
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import OpResult
from xdsl.rewriter import InsertPoint, Rewriter

from compgen.eqsat.rules.python_rules import (
    EqSatRewriteRule,
    add_alternative_to_eclass,
    get_eclass_for_result,
)


class DistributeMuliOverAddiRule(EqSatRewriteRule):
    """mul(a, add(b, c)) → add(mul(a, b), mul(a, c)): distribution.

    This can enable partial fusion when one of mul(a,b) or mul(a,c)
    can be fused with a downstream consumer.
    """

    @property
    def name(self) -> str:
        return "distribute_muli_addi"

    def match_and_add(self, module: ModuleOp) -> int:
        count = 0
        muli_ops = [op for op in module.walk() if isinstance(op, arith.MuliOp)]

        for mul_op in muli_ops:
            mul_lhs = mul_op.lhs  # 'a'
            mul_rhs = mul_op.rhs  # should be add(b, c) via eclass

            # Check if rhs comes through eclass from an addi
            if not isinstance(mul_rhs, OpResult):
                continue
            rhs_owner = mul_rhs.owner
            if not isinstance(rhs_owner, equivalence.AnyClassOp):
                continue

            for operand in rhs_owner.operands:
                if not isinstance(operand, OpResult):
                    continue
                if not isinstance(operand.owner, arith.AddiOp):
                    continue

                add_op = operand.owner
                val_b = add_op.lhs
                val_c = add_op.rhs

                eclass = get_eclass_for_result(mul_op.result)
                if eclass is None:
                    continue

                # Create: mul(a, b)
                mul_ab = arith.MuliOp(mul_lhs, val_b)
                Rewriter.insert_op(mul_ab, InsertPoint.before(eclass))
                mul_ab_eclass = equivalence.ClassOp(mul_ab.result)
                Rewriter.insert_op(mul_ab_eclass, InsertPoint.after(mul_ab))

                # Create: mul(a, c)
                mul_ac = arith.MuliOp(mul_lhs, val_c)
                Rewriter.insert_op(mul_ac, InsertPoint.before(eclass))
                mul_ac_eclass = equivalence.ClassOp(mul_ac.result)
                Rewriter.insert_op(mul_ac_eclass, InsertPoint.after(mul_ac))

                # Create: add(mul(a,b), mul(a,c))
                distributed = arith.AddiOp(mul_ab_eclass.result, mul_ac_eclass.result)
                Rewriter.insert_op(distributed, InsertPoint.before(eclass))
                add_alternative_to_eclass(eclass, distributed.result)
                count += 1
                break  # One alternative per mul_op per iteration

        return count


class FactorAddiIntoMuliRule(EqSatRewriteRule):
    """add(a, a) → mul(a, 2): strength reduction / fusion enabling.

    When a value is added to itself, replacing with mul by 2 may
    enable better vectorization or fusion with other multiplies.
    """

    @property
    def name(self) -> str:
        return "factor_addi_self"

    def match_and_add(self, module: ModuleOp) -> int:
        count = 0
        addi_ops = [op for op in module.walk() if isinstance(op, arith.AddiOp)]

        for add_op in addi_ops:
            # Check if both operands are the same eclass
            if add_op.lhs != add_op.rhs:
                continue

            eclass = get_eclass_for_result(add_op.result)
            if eclass is None:
                continue

            # Check not already factored
            already_exists = any(
                isinstance(operand, OpResult)
                and isinstance(operand.owner, arith.MuliOp)
                for operand in eclass.operands
            )
            if already_exists:
                continue

            # Create: constant 2
            from xdsl.dialects.builtin import IntegerAttr
            const_2 = arith.ConstantOp(IntegerAttr.from_index_int_value(2))
            Rewriter.insert_op(const_2, InsertPoint.before(eclass))
            const_2_eclass = equivalence.ClassOp(const_2.result)
            Rewriter.insert_op(const_2_eclass, InsertPoint.after(const_2))

            # Create: mul(a, 2)
            mul_op = arith.MuliOp(add_op.lhs, const_2_eclass.result)
            Rewriter.insert_op(mul_op, InsertPoint.before(eclass))
            add_alternative_to_eclass(eclass, mul_op.result)
            count += 1

        return count


def get_default_fusion_rules() -> list[EqSatRewriteRule]:
    """Return the default set of fusion-enabling rules."""
    return [
        DistributeMuliOverAddiRule(),
        FactorAddiIntoMuliRule(),
    ]
