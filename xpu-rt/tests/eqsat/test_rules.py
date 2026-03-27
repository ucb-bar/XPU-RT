"""Tests for rewrite rules library and registry."""

from __future__ import annotations

from compgen.eqsat.config import EqSatConfig
from compgen.eqsat.pipeline import _print_ir, assign_costs_and_extract, create_egraph
from compgen.eqsat.rules.algebraic import (
    CommutativityAddiRule,
    get_default_algebraic_rules,
)
from compgen.eqsat.rules.fusion import (
    DistributeMuliOverAddiRule,
    FactorAddiIntoMuliRule,
    get_default_fusion_rules,
)
from compgen.eqsat.rules.layout import get_default_layout_rules
from compgen.eqsat.rules.registry import RuleRegistry, create_default_registry
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


def _make_mul_add_module() -> ModuleOp:
    """mul(a, add(b, c)) — for distribution rule."""
    idx = IndexType()
    block = Block(arg_types=[idx, idx, idx])
    a, b, c = block.args
    add_bc = arith.AddiOp(b, c)
    block.add_op(add_bc)
    mul_a_bc = arith.MuliOp(a, add_bc.result)
    block.add_op(mul_a_bc)
    block.add_op(func.ReturnOp(mul_a_bc.result))
    return ModuleOp([func.FuncOp("test", ([idx, idx, idx], [idx]), Region([block]))])


def _make_self_add_module() -> ModuleOp:
    """add(a, a) — for factor rule."""
    idx = IndexType()
    block = Block(arg_types=[idx])
    a = block.args[0]
    add_aa = arith.AddiOp(a, a)
    block.add_op(add_aa)
    block.add_op(func.ReturnOp(add_aa.result))
    return ModuleOp([func.FuncOp("test", ([idx], [idx]), Region([block]))])


# ============================================================================
# Registry tests
# ============================================================================


class TestRegistry:
    def test_create_default_registry(self) -> None:
        registry = create_default_registry()
        assert registry.count() > 0
        assert "algebraic" in registry.categories()
        assert "layout" in registry.categories()
        assert "fusion" in registry.categories()

    def test_register_and_get(self) -> None:
        registry = RuleRegistry()
        rule = CommutativityAddiRule()
        registry.register("test_cat", rule)
        assert registry.count("test_cat") == 1
        rules = registry.get_rules(("test_cat",))
        assert len(rules) == 1
        assert rules[0].name == "commutativity_addi"

    def test_no_duplicates(self) -> None:
        registry = RuleRegistry()
        rule = CommutativityAddiRule()
        registry.register("algebraic", rule)
        registry.register("algebraic", rule)
        assert registry.count("algebraic") == 1

    def test_filter_by_category(self) -> None:
        registry = create_default_registry()
        algebraic = registry.get_rules(("algebraic",))
        fusion = registry.get_rules(("fusion",))
        all_rules = registry.get_rules()
        assert len(algebraic) > 0
        assert len(fusion) > 0
        assert len(all_rules) >= len(algebraic) + len(fusion)

    def test_remove_rule(self) -> None:
        registry = RuleRegistry()
        registry.register("test", CommutativityAddiRule())
        assert registry.remove("commutativity_addi")
        assert registry.count("test") == 0
        assert not registry.remove("nonexistent")

    def test_get_nonexistent_category(self) -> None:
        registry = RuleRegistry()
        rules = registry.get_rules(("nonexistent",))
        assert rules == []


# ============================================================================
# Fusion rule tests
# ============================================================================


class TestFusionRules:
    def test_distribution_rule(self) -> None:
        module = _make_mul_add_module()
        create_egraph(module)

        rule = DistributeMuliOverAddiRule()
        count = rule.apply(module)
        assert count >= 1

        # After extraction, the distributed form should be available
        config = EqSatConfig(default_cost=1)
        assign_costs_and_extract(module, config)
        ir = _print_ir(module)
        assert "equivalence" not in ir

    def test_factor_self_add(self) -> None:
        module = _make_self_add_module()
        create_egraph(module)

        rule = FactorAddiIntoMuliRule()
        count = rule.apply(module)
        assert count == 1

        # The eclass should now have both add(a,a) and mul(a,2)
        has_muli = False
        for op in module.walk():
            if isinstance(op, arith.MuliOp):
                has_muli = True
        assert has_muli

    def test_factor_idempotent(self) -> None:
        module = _make_self_add_module()
        create_egraph(module)

        rule = FactorAddiIntoMuliRule()
        rule.apply(module)
        count2 = rule.apply(module)
        assert count2 == 0

    def test_factor_extraction_picks_cheaper(self) -> None:
        """If muli is cheaper, extraction should pick it over addi."""
        module = _make_self_add_module()
        create_egraph(module)
        FactorAddiIntoMuliRule().apply(module)

        # Make muli cheaper
        import json
        import tempfile
        cost_dict = {"arith.addi": 10, "arith.muli": 2, "arith.constant": 1}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(cost_dict, f)
            cost_file = f.name

        from xdsl.context import Context
        from xdsl.transforms.eqsat_add_costs import EqsatAddCostsPass
        from xdsl.transforms.eqsat_extract import EqsatExtractPass

        ctx = Context()
        ctx.allow_unregistered = True
        EqsatAddCostsPass(cost_file=cost_file, default=5).apply(ctx, module)
        EqsatExtractPass().apply(ctx, module)

        ir = _print_ir(module)
        assert "arith.muli" in ir


# ============================================================================
# Combined rules tests
# ============================================================================


class TestCombinedRules:
    def test_all_default_rules_have_names(self) -> None:
        for rule in get_default_algebraic_rules():
            assert rule.name
        for rule in get_default_layout_rules():
            assert rule.name
        for rule in get_default_fusion_rules():
            assert rule.name

    def test_all_default_rules_are_unique(self) -> None:
        all_rules = (
            get_default_algebraic_rules()
            + get_default_layout_rules()
            + get_default_fusion_rules()
        )
        names = [r.name for r in all_rules]
        assert len(names) == len(set(names))
