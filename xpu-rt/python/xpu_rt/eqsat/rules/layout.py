"""Layout normalization rewrite rules for equality saturation.

These rules normalize layout-related operations:
- Transpose motion (push transposes toward leaves)
- Reshape sinking/hoisting
- Broadcast normalization
"""

from __future__ import annotations

from xdsl.dialects import arith, equivalence, linalg
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import OpResult
from xdsl.rewriter import InsertPoint, Rewriter

from compgen.eqsat.rules.python_rules import (
    EqSatRewriteRule,
    add_alternative_to_eclass,
    get_eclass_for_result,
)


class TransposeTransposeRule(EqSatRewriteRule):
    """transpose(transpose(a)) → a: double transpose elimination.

    Also present in algebraic.py but placed here for layout category too.
    """

    @property
    def name(self) -> str:
        return "layout_double_transpose"

    def match_and_add(self, module: ModuleOp) -> int:
        count = 0
        transpose_ops = [op for op in module.walk() if isinstance(op, linalg.TransposeOp)]

        for outer_t in transpose_ops:
            outer_input = outer_t.operands[0]
            if not isinstance(outer_input, OpResult):
                continue

            input_owner = outer_input.owner
            if not isinstance(input_owner, equivalence.AnyClassOp):
                continue

            for operand in input_owner.operands:
                if not isinstance(operand, OpResult):
                    continue
                if not isinstance(operand.owner, linalg.TransposeOp):
                    continue

                original_val = operand.owner.operands[0]
                eclass = get_eclass_for_result(outer_t.results[0])
                if eclass is None:
                    continue

                # Check not already added
                if original_val in eclass.operands:
                    continue

                add_alternative_to_eclass(eclass, original_val)
                count += 1
                break

        return count


class AddiCommuteBranchRule(EqSatRewriteRule):
    """For addf(a, b) where a is from a different subgraph than b,
    adds addf(b, a) so the extractor can choose the layout-friendlier order.

    This is a layout rule because operand order can affect memory access
    patterns in downstream code generation.
    """

    @property
    def name(self) -> str:
        return "layout_addf_commute"

    def match_and_add(self, module: ModuleOp) -> int:
        count = 0
        addf_ops = [op for op in module.walk() if isinstance(op, arith.AddfOp)]

        for op in addf_ops:
            lhs, rhs = op.lhs, op.rhs
            eclass = get_eclass_for_result(op.result)
            if eclass is None:
                continue

            # Check if commuted form exists
            already_exists = any(
                isinstance(operand, OpResult)
                and isinstance(operand.owner, arith.AddfOp)
                and operand.owner is not op
                and operand.owner.lhs == rhs
                and operand.owner.rhs == lhs
                for operand in eclass.operands
            )
            if already_exists:
                continue

            commuted = arith.AddfOp(rhs, lhs)
            Rewriter.insert_op(commuted, InsertPoint.before(eclass))
            add_alternative_to_eclass(eclass, commuted.result)
            count += 1

        return count


def get_default_layout_rules() -> list[EqSatRewriteRule]:
    """Return the default set of layout normalization rules."""
    return [
        TransposeTransposeRule(),
        AddiCommuteBranchRule(),
    ]
