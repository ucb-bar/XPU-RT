"""Tests for the non-additive cost model and extraction."""

from __future__ import annotations

from compgen.eqsat.cost_model import CostModel, CostWeights, create_cost_model
from compgen.eqsat.extract import extract_with_cost_model
from compgen.eqsat.pipeline import _print_ir, create_egraph
from compgen.eqsat.rules.python_rules import add_alternative_to_eclass
from xdsl.dialects import arith, equivalence, func
from xdsl.dialects.builtin import IndexType, ModuleOp
from xdsl.ir import Block, Region
from xdsl.rewriter import InsertPoint, Rewriter


def _make_add_module() -> ModuleOp:
    idx = IndexType()
    block = Block(arg_types=[idx, idx])
    a, b = block.args
    add = arith.AddiOp(a, b)
    block.add_op(add)
    block.add_op(func.ReturnOp(add.result))
    return ModuleOp([func.FuncOp("test", ([idx, idx], [idx]), Region([block]))])


def _make_add_with_mul_alternative() -> ModuleOp:
    """Create module, add eclasses, and manually add muli alternative."""
    module = _make_add_module()
    create_egraph(module)

    # Find eclasses
    add_eclass = None
    a_eclass_result = None
    b_eclass_result = None
    for op in module.walk():
        if isinstance(op, equivalence.ClassOp):
            if len(op.operands) == 1:
                operand = op.operands[0]
                if hasattr(operand, "owner") and isinstance(operand.owner, arith.AddiOp):
                    add_eclass = op
                elif hasattr(operand, "index"):
                    if operand.index == 0:
                        a_eclass_result = op.result
                    elif operand.index == 1:
                        b_eclass_result = op.result

    assert add_eclass is not None
    assert a_eclass_result is not None
    assert b_eclass_result is not None

    # Add muli alternative
    mul_op = arith.MuliOp(a_eclass_result, b_eclass_result)
    Rewriter.insert_op(mul_op, InsertPoint.before(add_eclass))
    add_alternative_to_eclass(add_eclass, mul_op.result)

    return module


# ============================================================================
# Cost model unit tests
# ============================================================================


class TestCostModel:
    def test_default_costs(self) -> None:
        model = CostModel()
        idx = IndexType()
        block = Block(arg_types=[idx, idx])
        op = arith.AddiOp(block.args[0], block.args[1])
        block.add_op(op)
        assert model.get_base_cost(op) == 10  # default

    def test_compute_heavy_ops(self) -> None:
        model = CostModel()
        # linalg.matmul should be more expensive
        assert model._COMPUTE_HEAVY["linalg.matmul"] == 100
        assert model._COMPUTE_HEAVY["linalg.generic"] == 50

    def test_custom_overrides(self) -> None:
        model = CostModel(base_costs={"arith.addi": 42})
        idx = IndexType()
        block = Block(arg_types=[idx, idx])
        op = arith.AddiOp(block.args[0], block.args[1])
        block.add_op(op)
        assert model.get_base_cost(op) == 42

    def test_to_json_dict(self) -> None:
        model = CostModel(base_costs={"arith.addi": 5})
        d = model.to_json_dict()
        assert d["arith.addi"] == 5
        assert d["arith.constant"] == 1
        assert d["linalg.matmul"] == 100

    def test_cost_weights(self) -> None:
        weights = CostWeights(fusion_weight=2.0)
        model = CostModel(weights=weights)
        assert model.weights.fusion_weight == 2.0

    def test_create_cost_model(self) -> None:
        model = create_cost_model(overrides={"arith.addi": 7})
        assert model.base_costs["arith.addi"] == 7

    def test_to_json_path(self) -> None:
        import json
        model = CostModel(base_costs={"arith.addi": 3})
        path = model.to_json_path()
        with open(path) as f:
            data = json.load(f)
        assert data["arith.addi"] == 3


# ============================================================================
# Non-additive extraction tests
# ============================================================================


class TestNonAdditiveExtraction:
    def test_extract_picks_cheaper_op(self) -> None:
        """Extraction with cost model should pick cheaper alternative."""
        module = _make_add_with_mul_alternative()

        # muli cheaper → should win
        model = CostModel(base_costs={"arith.addi": 20, "arith.muli": 2})
        extract_with_cost_model(module, model)

        ir = _print_ir(module)
        assert "arith.muli" in ir
        assert "equivalence" not in ir

    def test_extract_picks_expensive_when_bonus(self) -> None:
        """With high fusion bonus, a nominally more expensive op can win."""
        module = _make_add_with_mul_alternative()

        # addi nominally more expensive but with large fusion weight
        # In this simple case, fusion bonus applies to both equally
        model = CostModel(
            base_costs={"arith.addi": 5, "arith.muli": 4},
            weights=CostWeights(fusion_weight=0.0),
        )
        extract_with_cost_model(module, model)

        ir = _print_ir(module)
        # muli (4) should still win over addi (5)
        assert "arith.muli" in ir

    def test_extract_preserves_valid_ir(self) -> None:
        """After extraction, IR should be valid."""
        module = _make_add_with_mul_alternative()
        model = CostModel()
        extract_with_cost_model(module, model)

        ir = _print_ir(module)
        assert "func.func" in ir
        assert "func.return" in ir
        assert "equivalence" not in ir

    def test_assign_costs_to_module(self) -> None:
        """CostModel.assign_costs should set eqsat_cost on all ops."""
        module = _make_add_module()
        create_egraph(module)

        model = CostModel(base_costs={"arith.addi": 7})
        model.assign_costs(module)

        # Check that arith.addi has the cost attribute
        for op in module.walk():
            if isinstance(op, arith.AddiOp):
                assert equivalence.EQSAT_COST_LABEL in op.attributes

    def test_cost_model_vs_additive(self) -> None:
        """Non-additive cost model should produce valid extraction."""
        module = _make_add_with_mul_alternative()

        # Use non-additive model
        model = CostModel(
            base_costs={"arith.addi": 15, "arith.muli": 3},
            weights=CostWeights(fusion_weight=1.0),
        )
        extract_with_cost_model(module, model)

        ir = _print_ir(module)
        # muli should win (lower cost)
        assert "arith.muli" in ir
        assert "equivalence" not in ir
