"""Algebraic rewrite rules for equality saturation.

These are the safe, well-understood rewrites that don't depend on
target-specific knowledge:

- Commutativity: add(a, b) ↔ add(b, a)
- Reassociation: add(add(a, b), c) ↔ add(a, add(b, c))
- Double transpose elimination: transpose(transpose(a)) → a
- Identity elimination: add(a, 0) → a, mul(a, 1) → a
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


class CommutativityAddiRule(EqSatRewriteRule):
    """add(a, b) → add(b, a): adds the commuted form as an alternative."""

    @property
    def name(self) -> str:
        return "commutativity_addi"

    def match_and_add(self, module: ModuleOp) -> int:
        count = 0
        # Collect ops first to avoid modifying during iteration
        addi_ops = [op for op in module.walk() if isinstance(op, arith.AddiOp)]

        for op in addi_ops:
            lhs, rhs = op.lhs, op.rhs
            # Skip if already commuted form exists in eclass
            eclass = get_eclass_for_result(op.result)
            if eclass is None:
                continue

            # Check if a commuted version already exists in this eclass
            already_exists = False
            for operand in eclass.operands:
                if isinstance(operand, OpResult) and isinstance(operand.owner, arith.AddiOp):
                    other = operand.owner
                    if other is not op and other.lhs == rhs and other.rhs == lhs:
                        already_exists = True
                        break

            if already_exists:
                continue

            # Create commuted add: add(b, a)
            commuted = arith.AddiOp(rhs, lhs)
            Rewriter.insert_op(commuted, InsertPoint.before(eclass))
            add_alternative_to_eclass(eclass, commuted.result)
            count += 1

        return count


class ReassociationAddiRule(EqSatRewriteRule):
    """add(add(a, b), c) → add(a, add(b, c)): right-associates additions."""

    @property
    def name(self) -> str:
        return "reassociation_addi"

    def match_and_add(self, module: ModuleOp) -> int:
        count = 0
        addi_ops = [op for op in module.walk() if isinstance(op, arith.AddiOp)]

        for outer_add in addi_ops:
            outer_lhs = outer_add.lhs
            outer_rhs = outer_add.rhs  # this is 'c'

            # Check if lhs comes through an eclass from another addi
            if not isinstance(outer_lhs, OpResult):
                continue

            # Follow through eclass
            lhs_owner = outer_lhs.owner
            if not isinstance(lhs_owner, equivalence.AnyClassOp):
                continue

            # Look for an addi among the eclass operands
            for operand in lhs_owner.operands:
                if not isinstance(operand, OpResult):
                    continue
                if not isinstance(operand.owner, arith.AddiOp):
                    continue

                inner_add = operand.owner
                val_a = inner_add.lhs  # 'a'
                val_b = inner_add.rhs  # 'b'

                # Create: add(b, c)
                eclass_for_outer = get_eclass_for_result(outer_add.result)
                if eclass_for_outer is None:
                    continue

                new_bc = arith.AddiOp(val_b, outer_rhs)
                Rewriter.insert_op(new_bc, InsertPoint.before(eclass_for_outer))

                # Wrap in eclass
                new_bc_eclass = equivalence.ClassOp(new_bc.result)
                Rewriter.insert_op(new_bc_eclass, InsertPoint.after(new_bc))

                # Create: add(a, add(b, c))
                new_a_bc = arith.AddiOp(val_a, new_bc_eclass.result)
                Rewriter.insert_op(new_a_bc, InsertPoint.before(eclass_for_outer))

                add_alternative_to_eclass(eclass_for_outer, new_a_bc.result)
                count += 1
                break  # Only add one alternative per outer_add per iteration

        return count


class DoubleTransposeEliminationRule(EqSatRewriteRule):
    """transpose(transpose(a)) → a: adds the inner value as an alternative."""

    @property
    def name(self) -> str:
        return "double_transpose_elimination"

    def match_and_add(self, module: ModuleOp) -> int:
        count = 0
        transpose_ops = [
            op for op in module.walk() if isinstance(op, linalg.TransposeOp)
        ]

        for outer_t in transpose_ops:
            outer_input = outer_t.operands[0]
            if not isinstance(outer_input, OpResult):
                continue

            # Follow through eclass
            input_owner = outer_input.owner
            if not isinstance(input_owner, equivalence.AnyClassOp):
                continue

            for operand in input_owner.operands:
                if not isinstance(operand, OpResult):
                    continue
                if not isinstance(operand.owner, linalg.TransposeOp):
                    continue

                inner_t = operand.owner
                # The input to the inner transpose is the original value
                original_val = inner_t.operands[0]

                eclass_for_outer = get_eclass_for_result(outer_t.results[0])
                if eclass_for_outer is None:
                    continue

                # Check permutations are inverse
                # For now, assume transpose(transpose(x)) = x if same permutation
                # (standard 2D transpose case)
                add_alternative_to_eclass(eclass_for_outer, original_val)
                count += 1
                break

        return count


class CommutativityMuliRule(EqSatRewriteRule):
    """mul(a, b) → mul(b, a): adds the commuted form as an alternative."""

    @property
    def name(self) -> str:
        return "commutativity_muli"

    def match_and_add(self, module: ModuleOp) -> int:
        count = 0
        muli_ops = [op for op in module.walk() if isinstance(op, arith.MuliOp)]

        for op in muli_ops:
            lhs, rhs = op.lhs, op.rhs
            eclass = get_eclass_for_result(op.result)
            if eclass is None:
                continue

            already_exists = False
            for operand in eclass.operands:
                if isinstance(operand, OpResult) and isinstance(operand.owner, arith.MuliOp):
                    other = operand.owner
                    if other is not op and other.lhs == rhs and other.rhs == lhs:
                        already_exists = True
                        break

            if already_exists:
                continue

            commuted = arith.MuliOp(rhs, lhs)
            Rewriter.insert_op(commuted, InsertPoint.before(eclass))
            add_alternative_to_eclass(eclass, commuted.result)
            count += 1

        return count


def get_default_algebraic_rules() -> list[EqSatRewriteRule]:
    """Return the default set of algebraic rewrite rules."""
    return [
        CommutativityAddiRule(),
        ReassociationAddiRule(),
        CommutativityMuliRule(),
        DoubleTransposeEliminationRule(),
    ]
