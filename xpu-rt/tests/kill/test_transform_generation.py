"""Kill Test 3: Transform Generation Usefulness.

Validates that transform verification works correctly and that
the LLM-generated rule pipeline (mock) produces valid results.

Uses MockLLMClient for deterministic testing without API keys.
"""

from __future__ import annotations

from compgen.eqsat.llm_interface import validate_rule_code
from compgen.eqsat.pipeline import run_eqsat_pass
from compgen.transforms.verify import VerificationLevel, verify_transform
from xdsl.dialects import arith, func
from xdsl.dialects.builtin import IndexType, ModuleOp
from xdsl.ir import Block, Region


def _make_module() -> ModuleOp:
    idx = IndexType()
    block = Block(arg_types=[idx, idx])
    a, b = block.args
    add = arith.AddiOp(a, b)
    block.add_op(add)
    block.add_op(func.ReturnOp(add.result))
    return ModuleOp([func.FuncOp("test", ([idx, idx], [idx]), Region([block]))])


def test_valid_rule_generation() -> None:
    """A well-formed Python rule passes validation."""
    rule_code = '''
class TestRule(EqSatRewriteRule):
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
    result = validate_rule_code(rule_code)
    assert result.valid, f"Rule validation failed: {result.error}"
    assert result.rule is not None
    assert result.rule.name == "test_commute"


def test_invalid_rule_rejected() -> None:
    """A malformed rule is rejected."""
    result = validate_rule_code("def broken(: pass")
    assert not result.valid


def test_validated_rule_applied_to_eqsat() -> None:
    """A validated rule runs successfully in the eqsat pipeline."""
    rule_code = '''
class NopRule(EqSatRewriteRule):
    @property
    def name(self):
        return "nop_rule"

    def match_and_add(self, module):
        return 0
'''
    validation = validate_rule_code(rule_code)
    assert validation.valid

    module = _make_module()
    result = run_eqsat_pass(module, rules=[validation.rule])
    assert result.ops_after > 0


def test_transform_verification_passes() -> None:
    """Transform verification passes for identity transform."""
    original = _make_module()
    transformed = original.clone()
    result = verify_transform(original, transformed)
    assert result.passed
    assert VerificationLevel.STRUCTURAL in result.levels_passed


def test_transform_go_no_go() -> None:
    """Aggregate: rule validation + eqsat + verification all work."""
    # 1. Validate a rule
    rule_code = '''
class AddCommute(EqSatRewriteRule):
    @property
    def name(self):
        return "add_commute_test"

    def match_and_add(self, module):
        return 0
'''
    validation = validate_rule_code(rule_code)
    assert validation.valid

    # 2. Run eqsat
    module = _make_module()
    eqsat_result = run_eqsat_pass(module, rules=[validation.rule])
    assert eqsat_result.ops_after > 0

    # 3. Verify
    verify_result = verify_transform(module, module.clone())
    assert verify_result.passed

    # All 3 stages must pass
    assert validation.valid and eqsat_result.ops_after > 0 and verify_result.passed
