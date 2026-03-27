"""Tests for LLM ↔ e-graph interaction layer."""

from __future__ import annotations

from compgen.eqsat.explain import EClassSummary, EGraphSummary, summarize_egraph, summary_to_prompt
from compgen.eqsat.llm_interface import (
    format_extraction_objective_prompt,
    format_rule_proposal_prompt,
    format_search_state_prompt,
    validate_rule_code,
)
from compgen.eqsat.pipeline import _print_ir, create_egraph, run_eqsat_pass
from xdsl.dialects import arith, func
from xdsl.dialects.builtin import IndexType, ModuleOp
from xdsl.ir import Block, Region


def _make_add_module() -> ModuleOp:
    idx = IndexType()
    block = Block(arg_types=[idx, idx])
    a, b = block.args
    add = arith.AddiOp(a, b)
    block.add_op(add)
    block.add_op(func.ReturnOp(add.result))
    return ModuleOp([func.FuncOp("test", ([idx, idx], [idx]), Region([block]))])


# ============================================================================
# E-graph explain tests
# ============================================================================


class TestExplain:
    def test_summarize_basic_egraph(self) -> None:
        module = _make_add_module()
        create_egraph(module)

        summary = summarize_egraph(module)
        assert summary.num_eclasses > 0
        assert summary.num_enodes > 0
        assert summary.ambiguous_eclasses == 0  # no alternatives yet

    def test_summarize_with_alternatives(self) -> None:
        module = _make_add_module()
        create_egraph(module)

        # Add a commuted alternative
        from compgen.eqsat.rules.algebraic import CommutativityAddiRule
        CommutativityAddiRule().apply(module)

        summary = summarize_egraph(module)
        assert summary.ambiguous_eclasses >= 1  # at least one eclass has >1 alt

    def test_summary_to_prompt_format(self) -> None:
        summary = EGraphSummary(
            num_eclasses=10,
            num_enodes=15,
            ambiguous_eclasses=3,
            eclass_summaries=(
                EClassSummary(0, 2, ("arith.addi", "arith.muli"), False),
                EClassSummary(1, 1, ("arith.addi",), False),
            ),
            op_type_counts={"arith.addi": 5, "arith.muli": 2},
        )
        prompt = summary_to_prompt(summary)
        assert "10 e-classes" in prompt
        assert "15 e-nodes" in prompt
        assert "3 ambiguous" in prompt
        assert "arith.addi" in prompt

    def test_summary_to_prompt_no_ambiguous(self) -> None:
        summary = EGraphSummary(
            num_eclasses=3,
            num_enodes=3,
            ambiguous_eclasses=0,
            eclass_summaries=(),
            op_type_counts={"arith.addi": 2},
        )
        prompt = summary_to_prompt(summary)
        assert "0 ambiguous" in prompt


# ============================================================================
# Rule validation tests
# ============================================================================


class TestRuleValidation:
    def test_valid_rule_code(self) -> None:
        code = '''
class TestCommute(EqSatRewriteRule):
    @property
    def name(self):
        return "test_commute"

    def match_and_add(self, module):
        count = 0
        for op in module.walk():
            if isinstance(op, arith.AddiOp):
                eclass = get_eclass_for_result(op.result)
                if eclass is None:
                    continue
                commuted = arith.AddiOp(op.rhs, op.lhs)
                Rewriter.insert_op(commuted, InsertPoint.before(eclass))
                add_alternative_to_eclass(eclass, commuted.result)
                count += 1
        return count
'''
        result = validate_rule_code(code)
        assert result.valid
        assert result.rule is not None
        assert result.rule.name == "test_commute"

    def test_invalid_syntax(self) -> None:
        code = "class Broken(EqSatRewriteRule:\n    pass"
        result = validate_rule_code(code)
        assert not result.valid
        assert "Syntax error" in result.error

    def test_no_rule_class(self) -> None:
        code = "x = 42"
        result = validate_rule_code(code)
        assert not result.valid
        assert "No EqSatRewriteRule subclass found" in result.error

    def test_missing_name_property(self) -> None:
        code = '''
class BadRule(EqSatRewriteRule):
    def match_and_add(self, module):
        return 0
'''
        result = validate_rule_code(code)
        # This should fail because `name` is an abstract property
        assert not result.valid

    def test_validated_rule_actually_works(self) -> None:
        """A validated rule should actually work on a real module."""
        code = '''
class NopRule(EqSatRewriteRule):
    @property
    def name(self):
        return "nop_rule"

    def match_and_add(self, module):
        return 0
'''
        result = validate_rule_code(code)
        assert result.valid

        module = _make_add_module()
        create_egraph(module)
        count = result.rule.apply(module)
        assert count == 0


# ============================================================================
# Prompt template tests
# ============================================================================


class TestPromptTemplates:
    def _make_summary(self) -> EGraphSummary:
        return EGraphSummary(
            num_eclasses=10,
            num_enodes=15,
            ambiguous_eclasses=2,
            eclass_summaries=(),
            op_type_counts={"arith.addi": 5},
        )

    def test_rule_proposal_prompt(self) -> None:
        prompt = format_rule_proposal_prompt(
            self._make_summary(),
            target_description="NVIDIA A100 GPU",
            objective="minimize latency",
        )
        assert "e-graph" in prompt.lower() or "E-graph" in prompt
        assert "NVIDIA A100" in prompt
        assert "EqSatRewriteRule" in prompt

    def test_search_state_prompt(self) -> None:
        prompt = format_search_state_prompt(
            self._make_summary(),
            rule_stats={"commutativity_addi": 10, "reassociation": 5},
            best_cost=42.0,
        )
        assert "commutativity_addi" in prompt
        assert "42.0" in prompt
        assert "PROPOSE_RULE" in prompt

    def test_extraction_objective_prompt(self) -> None:
        prompt = format_extraction_objective_prompt(
            self._make_summary(),
            target_description="NVIDIA A100 GPU",
            current_weights={"fusion": 1.0, "transfer": 1.0, "backend_match": 1.0},
        )
        assert "fusion" in prompt
        assert "transfer" in prompt
        assert "JSON" in prompt


# ============================================================================
# Integration: validated rule applied to e-graph
# ============================================================================


class TestRuleIntegration:
    def test_llm_generated_rule_applied_to_egraph(self) -> None:
        """Simulate the full LLM rule proposal pipeline."""
        # Step 1: LLM generates rule code
        llm_rule_code = '''
class MulCommute(EqSatRewriteRule):
    @property
    def name(self):
        return "llm_mul_commute"

    def match_and_add(self, module):
        count = 0
        ops = [op for op in module.walk() if isinstance(op, arith.MuliOp)]
        for op in ops:
            eclass = get_eclass_for_result(op.result)
            if eclass is None:
                continue
            already = any(
                isinstance(o, OpResult) and isinstance(o.owner, arith.MuliOp)
                and o.owner is not op and o.owner.lhs == op.rhs and o.owner.rhs == op.lhs
                for o in eclass.operands
            )
            if already:
                continue
            commuted = arith.MuliOp(op.rhs, op.lhs)
            Rewriter.insert_op(commuted, InsertPoint.before(eclass))
            add_alternative_to_eclass(eclass, commuted.result)
            count += 1
        return count
'''
        # Step 2: Validate
        validation = validate_rule_code(llm_rule_code)
        assert validation.valid
        assert validation.rule.name == "llm_mul_commute"

        # Step 3: Apply to a module with muli
        idx = IndexType()
        block = Block(arg_types=[idx, idx])
        a, b = block.args
        mul = arith.MuliOp(a, b)
        block.add_op(mul)
        block.add_op(func.ReturnOp(mul.result))
        module = ModuleOp([func.FuncOp("t", ([idx, idx], [idx]), Region([block]))])

        # Step 4: Run eqsat with the LLM-generated rule
        result = run_eqsat_pass(module, rules=[validation.rule])

        assert result.rule_stats.get("llm_mul_commute", 0) >= 1
        ir = _print_ir(module)
        assert "equivalence" not in ir  # extracted
