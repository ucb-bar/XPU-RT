"""Foundation tests for the equality saturation pipeline.

Tests the end-to-end flow: create IR → eqsat pass → verify result.
"""

from __future__ import annotations

import pytest
from compgen.eqsat.config import EqSatConfig
from compgen.eqsat.pipeline import (
    EqSatResult,
    _count_eclasses,
    _count_enodes,
    _print_ir,
    assign_costs_and_extract,
    create_egraph,
    run_eqsat_pass,
)
from compgen.eqsat.rules.algebraic import (
    CommutativityAddiRule,
    ReassociationAddiRule,
)
from compgen.eqsat.rules.python_rules import (
    add_alternative_to_eclass,
)
from xdsl.dialects import arith, equivalence, func
from xdsl.dialects.builtin import IndexType, ModuleOp
from xdsl.ir import Block, Region


def _make_add_module() -> ModuleOp:
    """Create: func @test(%a, %b: index) { return add(a, b) }"""
    idx = IndexType()
    block = Block(arg_types=[idx, idx])
    a, b = block.args
    add_op = arith.AddiOp(a, b)
    block.add_op(add_op)
    block.add_op(func.ReturnOp(add_op.result))
    return ModuleOp([func.FuncOp("test", ([idx, idx], [idx]), Region([block]))])


def _make_chain_add_module() -> ModuleOp:
    """Create: func @test(%a, %b, %c: index) { return add(add(a, b), c) }"""
    idx = IndexType()
    block = Block(arg_types=[idx, idx, idx])
    a, b, c = block.args
    add_ab = arith.AddiOp(a, b)
    block.add_op(add_ab)
    add_abc = arith.AddiOp(add_ab.result, c)
    block.add_op(add_abc)
    block.add_op(func.ReturnOp(add_abc.result))
    return ModuleOp([func.FuncOp("test", ([idx, idx, idx], [idx]), Region([block]))])


def _make_matmul_add_module() -> ModuleOp:
    """Create a module with matmul + addi (simplified ML-like pattern)."""
    idx = IndexType()
    block = Block(arg_types=[idx, idx, idx])
    a, b, c = block.args
    # add(a, b) then add(result, c)
    add_ab = arith.AddiOp(a, b)
    block.add_op(add_ab)
    add_abc = arith.AddiOp(add_ab.result, c)
    block.add_op(add_abc)
    block.add_op(func.ReturnOp(add_abc.result))
    return ModuleOp([func.FuncOp("mlp", ([idx, idx, idx], [idx]), Region([block]))])


# ============================================================================
# Test: e-graph creation
# ============================================================================


class TestCreateEGraph:
    def test_create_eclasses_wraps_ops(self) -> None:
        module = _make_add_module()
        assert _count_eclasses(module) == 0

        create_egraph(module)
        assert _count_eclasses(module) > 0

    def test_create_eclasses_preserves_semantics(self) -> None:
        module = _make_add_module()
        _print_ir(module)  # ensure printable

        create_egraph(module)
        config = EqSatConfig(default_cost=1)
        assign_costs_and_extract(module, config)

        ir_after = _print_ir(module)
        # After create + extract with no rewrites, should be identical
        assert "arith.addi" in ir_after
        assert "equivalence" not in ir_after

    def test_eclass_count_matches_values(self) -> None:
        module = _make_chain_add_module()
        create_egraph(module)
        # 3 args + 2 addi results = 5 eclasses
        assert _count_eclasses(module) == 5


# ============================================================================
# Test: commutativity rule
# ============================================================================


class TestCommutativityRule:
    def test_commutativity_adds_alternative(self) -> None:
        module = _make_add_module()
        create_egraph(module)

        rule = CommutativityAddiRule()
        count = rule.apply(module)
        assert count == 1

        # The eclass for the addi result should now have 2 operands
        for op in module.walk():
            if isinstance(op, equivalence.ClassOp):
                for operand in op.operands:
                    if hasattr(operand, "owner") and isinstance(
                        operand.owner, arith.AddiOp
                    ):
                        # This eclass should have the commuted alternative
                        if len(op.operands) == 2:
                            return
        pytest.fail("Expected an eclass with 2 alternatives")

    def test_commutativity_idempotent(self) -> None:
        module = _make_add_module()
        create_egraph(module)

        rule = CommutativityAddiRule()
        rule.apply(module)
        count2 = rule.apply(module)
        assert count2 == 0  # Should not add duplicate


# ============================================================================
# Test: reassociation rule
# ============================================================================


class TestReassociationRule:
    def test_reassociation_creates_alternative(self) -> None:
        module = _make_chain_add_module()
        create_egraph(module)

        rule = ReassociationAddiRule()
        count = rule.apply(module)
        assert count >= 1

    def test_reassociation_e2e(self) -> None:
        """End-to-end: add(add(a,b),c) with reassociation + cost-guided extraction."""
        module = _make_chain_add_module()
        create_egraph(module)

        rule = ReassociationAddiRule()
        rule.apply(module)

        enodes = _count_enodes(module)
        assert enodes > 5  # More than initial (alternatives added)

        # Extract with uniform costs
        config = EqSatConfig(default_cost=1)
        assign_costs_and_extract(module, config)

        ir = _print_ir(module)
        assert "arith.addi" in ir
        assert "equivalence" not in ir


# ============================================================================
# Test: full pipeline
# ============================================================================


class TestFullPipeline:
    def test_run_eqsat_pass_returns_result(self) -> None:
        module = _make_add_module()
        result = run_eqsat_pass(module)

        assert isinstance(result, EqSatResult)
        assert result.ops_before > 0
        assert result.ops_after > 0
        assert result.eclasses_initial > 0

    def test_run_eqsat_pass_with_config(self) -> None:
        module = _make_chain_add_module()
        config = EqSatConfig(
            max_iterations=5,
            default_cost=1,
            rule_categories=("algebraic",),
        )
        result = run_eqsat_pass(module, config=config)
        assert result.eclasses_after_rewrite >= result.eclasses_initial

    def test_run_eqsat_pass_no_eclasses_remain(self) -> None:
        """After eqsat pass, no equivalence ops should remain in the IR."""
        module = _make_chain_add_module()
        run_eqsat_pass(module)

        ir = _print_ir(module)
        assert "equivalence.class" not in ir
        assert "equivalence.graph" not in ir

    def test_run_eqsat_pass_valid_ir(self) -> None:
        """After eqsat pass, the IR should be valid (has func, return, ops)."""
        module = _make_chain_add_module()
        run_eqsat_pass(module)

        ir = _print_ir(module)
        assert "func.func" in ir
        assert "func.return" in ir
        assert "arith.addi" in ir

    def test_run_eqsat_pass_with_custom_rules(self) -> None:
        module = _make_add_module()
        rules = [CommutativityAddiRule()]
        result = run_eqsat_pass(module, rules=rules)
        assert "commutativity_addi" in result.rule_stats

    def test_run_eqsat_pass_with_cost_dict(self) -> None:
        """Custom cost dict should influence extraction."""
        module = _make_chain_add_module()
        cost_dict = {"arith.addi": 5}
        result = run_eqsat_pass(module, cost_dict=cost_dict)
        assert result.ops_after > 0

    def test_commutativity_detected_as_change(self) -> None:
        """Commutativity may or may not change the IR (depends on extraction),
        but the rule should fire."""
        module = _make_add_module()
        rules = [CommutativityAddiRule()]
        result = run_eqsat_pass(module, rules=rules)
        assert result.rule_stats.get("commutativity_addi", 0) > 0

    def test_ml_like_pattern(self) -> None:
        """Test on a pattern resembling an ML computation graph."""
        module = _make_matmul_add_module()
        result = run_eqsat_pass(module)
        assert result.ops_after > 0
        ir = _print_ir(module)
        assert "equivalence" not in ir


# ============================================================================
# Test: cost-influenced extraction
# ============================================================================


class TestCostExtraction:
    def test_cheaper_alternative_wins(self) -> None:
        """When one alternative is cheaper, extraction picks it."""
        from xdsl.rewriter import InsertPoint, Rewriter

        idx = IndexType()
        block = Block(arg_types=[idx, idx])
        a, b = block.args
        add_op = arith.AddiOp(a, b)
        block.add_op(add_op)
        block.add_op(func.ReturnOp(add_op.result))
        module = ModuleOp(
            [func.FuncOp("test", ([idx, idx], [idx]), Region([block]))]
        )

        # Create e-graph manually
        create_egraph(module)

        # Find the eclass for the addi result and eclass results for args
        add_eclass = None
        a_eclass_result = None
        b_eclass_result = None
        for op in module.walk():
            if isinstance(op, equivalence.ClassOp):
                if len(op.operands) == 1:
                    operand = op.operands[0]
                    if hasattr(operand, "owner") and isinstance(operand.owner, arith.AddiOp):
                        add_eclass = op
                    elif operand is a:
                        a_eclass_result = op.result
                    elif operand is b:
                        b_eclass_result = op.result

        assert add_eclass is not None
        assert a_eclass_result is not None
        assert b_eclass_result is not None

        # Add a "cheaper" alternative: muli(a, b)
        mul_op = arith.MuliOp(a_eclass_result, b_eclass_result)
        Rewriter.insert_op(mul_op, InsertPoint.before(add_eclass))
        add_alternative_to_eclass(add_eclass, mul_op.result)

        # Assign costs directly and extract (skip run_eqsat_pass which re-creates eclasses)
        from xdsl.context import Context
        ctx = Context()
        ctx.allow_unregistered = True
        # addi=10, muli=1 → muli should win
        import json
        import tempfile

        from xdsl.transforms.eqsat_add_costs import EqsatAddCostsPass
        from xdsl.transforms.eqsat_extract import EqsatExtractPass
        cost_dict = {"arith.addi": 10, "arith.muli": 1}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(cost_dict, f)
            cost_file = f.name

        EqsatAddCostsPass(cost_file=cost_file, default=5).apply(ctx, module)
        EqsatExtractPass().apply(ctx, module)

        ir = _print_ir(module)
        # The cheaper muli should win
        assert "arith.muli" in ir
        assert "equivalence" not in ir
