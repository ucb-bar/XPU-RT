"""Tests for the ukernel constraint evaluator.

Validates that declarative constraint strings are evaluated correctly
against a ConstraintContext for shape, dtype, feature, device, and
layout predicates -- all without eval() or code execution.
"""

from __future__ import annotations

from compgen.ir.ukernel.constraints import (
    ConstraintContext,
    evaluate_all_constraints,
    evaluate_constraint,
)


class TestShapeModuloConstraint:
    """Shape modulo predicate: M%16==0."""

    def test_mod_passes_when_divisible(self) -> None:
        ctx = ConstraintContext(shapes={"M": 128})
        assert evaluate_constraint("M%16==0", ctx) is True

    def test_mod_fails_when_not_divisible(self) -> None:
        ctx = ConstraintContext(shapes={"M": 17})
        assert evaluate_constraint("M%16==0", ctx) is False


class TestShapeComparisonConstraint:
    """Shape comparison predicates: >=, <=, ==, !=, >, <."""

    def test_gte_passes_at_boundary(self) -> None:
        ctx = ConstraintContext(shapes={"K": 32})
        assert evaluate_constraint("K>=32", ctx) is True

    def test_gte_fails_below_boundary(self) -> None:
        ctx = ConstraintContext(shapes={"K": 16})
        assert evaluate_constraint("K>=32", ctx) is False

    def test_lte_passes_at_boundary(self) -> None:
        ctx = ConstraintContext(shapes={"N": 4096})
        assert evaluate_constraint("N<=4096", ctx) is True

    def test_lte_fails_above_boundary(self) -> None:
        ctx = ConstraintContext(shapes={"N": 8192})
        assert evaluate_constraint("N<=4096", ctx) is False

    def test_eq_passes(self) -> None:
        ctx = ConstraintContext(shapes={"M": 128})
        assert evaluate_constraint("M==128", ctx) is True

    def test_eq_fails(self) -> None:
        ctx = ConstraintContext(shapes={"M": 64})
        assert evaluate_constraint("M==128", ctx) is False

    def test_neq_passes(self) -> None:
        ctx = ConstraintContext(shapes={"K": 64})
        assert evaluate_constraint("K!=0", ctx) is True

    def test_neq_fails(self) -> None:
        ctx = ConstraintContext(shapes={"K": 0})
        assert evaluate_constraint("K!=0", ctx) is False


class TestFeaturePredicateConstraint:
    """Feature predicates: has_tensor_core, has_rvv, etc."""

    def test_feature_present(self) -> None:
        ctx = ConstraintContext(target_features=frozenset({"has_tensor_core"}))
        assert evaluate_constraint("has_tensor_core", ctx) is True

    def test_feature_absent(self) -> None:
        ctx = ConstraintContext(target_features=frozenset())
        assert evaluate_constraint("has_tensor_core", ctx) is False


class TestDeviceTypeConstraint:
    """Device type predicates: device_type==gpu."""

    def test_device_matches(self) -> None:
        ctx = ConstraintContext(device_type="gpu")
        assert evaluate_constraint("device_type==gpu", ctx) is True

    def test_device_mismatch(self) -> None:
        ctx = ConstraintContext(device_type="gpu")
        assert evaluate_constraint("device_type==npu", ctx) is False


class TestDtypeConstraint:
    """Dtype predicates: dtype==float32, dtype_in(float16,bfloat16)."""

    def test_dtype_eq_passes(self) -> None:
        ctx = ConstraintContext(dtypes=("float32",))
        assert evaluate_constraint("dtype==float32", ctx) is True

    def test_dtype_eq_fails(self) -> None:
        ctx = ConstraintContext(dtypes=("int8",))
        assert evaluate_constraint("dtype==float32", ctx) is False

    def test_dtype_in_passes(self) -> None:
        ctx = ConstraintContext(dtypes=("float16",))
        assert evaluate_constraint("dtype_in(float16,bfloat16)", ctx) is True

    def test_dtype_in_fails(self) -> None:
        ctx = ConstraintContext(dtypes=("int8",))
        assert evaluate_constraint("dtype_in(float16,bfloat16)", ctx) is False


class TestLayoutConstraint:
    """Layout predicates: lhs_rowmajor, rhs_prepacked."""

    def test_layout_matches(self) -> None:
        ctx = ConstraintContext(layouts={"lhs": "rowmajor"})
        assert evaluate_constraint("lhs_rowmajor", ctx) is True

    def test_layout_mismatch(self) -> None:
        ctx = ConstraintContext(layouts={"lhs": "colmajor"})
        assert evaluate_constraint("lhs_rowmajor", ctx) is False


class TestEdgeCases:
    """Empty, unknown, and missing-dimension constraints."""

    def test_empty_constraint_always_passes(self) -> None:
        ctx = ConstraintContext()
        assert evaluate_constraint("", ctx) is True
        assert evaluate_constraint("   ", ctx) is True

    def test_unknown_constraint_returns_false(self) -> None:
        ctx = ConstraintContext()
        assert evaluate_constraint("totally_unknown_syntax!!!", ctx) is False

    def test_missing_shape_dimension_returns_false(self) -> None:
        ctx = ConstraintContext(shapes={"M": 128})
        assert evaluate_constraint("Z%16==0", ctx) is False
        assert evaluate_constraint("Z>=32", ctx) is False


class TestEvaluateAllConstraints:
    """evaluate_all_constraints: conjunction of all predicates."""

    def test_all_pass(self) -> None:
        ctx = ConstraintContext(
            shapes={"M": 128, "K": 64},
            dtypes=("float32",),
            target_features=frozenset({"has_tensor_core"}),
        )
        constraints = ["M%16==0", "K>=32", "dtype==float32", "has_tensor_core"]
        assert evaluate_all_constraints(constraints, ctx) is True

    def test_one_fails(self) -> None:
        ctx = ConstraintContext(shapes={"M": 17, "K": 64})
        constraints = ["M%16==0", "K>=32"]
        assert evaluate_all_constraints(constraints, ctx) is False

    def test_empty_list_passes(self) -> None:
        ctx = ConstraintContext()
        assert evaluate_all_constraints([], ctx) is True
