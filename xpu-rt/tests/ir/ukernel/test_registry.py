"""Tests for the ukernel registry and selection engine.

Validates registration, constraint-based selection, priority ordering,
target-family body selection, and introspection APIs.
"""

from __future__ import annotations

from compgen.ir.ukernel.constraints import ConstraintContext
from compgen.ir.ukernel.ops import UkernelBodyOp, UkernelDeclOp, UkernelMatchOp
from compgen.ir.ukernel.registry import UkernelRegistry


def _make_decl(name: str, **kwargs) -> UkernelDeclOp:
    return UkernelDeclOp(kernel_name=name, **kwargs)


def _make_match(name: str, op_family: str = "matmul", priority: int = 0, **kwargs) -> UkernelMatchOp:
    return UkernelMatchOp(kernel_name=name, op_family=op_family, priority=priority, **kwargs)


def _make_body(name: str, target_family: str = "any", **kwargs) -> UkernelBodyOp:
    return UkernelBodyOp(kernel_name=name, target_family=target_family, **kwargs)


class TestRegistration:
    """Register ukernels and verify registry state."""

    def test_register_ukernel_increments_len(self) -> None:
        reg = UkernelRegistry()
        decl = _make_decl("k1")
        reg.register_ukernel(decl)
        assert len(reg) == 1

    def test_register_ukernel_with_matches_and_bodies(self) -> None:
        reg = UkernelRegistry()
        decl = _make_decl("k1")
        matches = [_make_match("k1", priority=5)]
        bodies = [_make_body("k1")]
        reg.register_ukernel(decl, matches, bodies)

        assert len(reg) == 1
        assert len(reg.matches_for("k1")) == 1
        assert len(reg.bodies_for("k1")) == 1

    def test_register_decl_then_match_then_body(self) -> None:
        reg = UkernelRegistry()
        decl = _make_decl("k1")
        reg.register_decl(decl)
        reg.register_match(_make_match("k1", priority=3))
        reg.register_body(_make_body("k1", target_family="cuda"))

        assert len(reg) == 1
        assert len(reg.matches_for("k1")) == 1
        assert reg.bodies_for("k1")[0].target_family == "cuda"


class TestSelectUkernel:
    """select_ukernel: constraint evaluation + priority ordering."""

    def test_single_match_returns_decl(self) -> None:
        reg = UkernelRegistry()
        decl = _make_decl("k1")
        match = _make_match("k1", op_family="matmul", priority=10, dtype_constraints=("dtype==float32",))
        reg.register_ukernel(decl, [match])

        ctx = ConstraintContext(dtypes=("float32",))
        result = reg.select_ukernel("matmul", ctx)
        assert result is not None
        assert result.kernel_name == "k1"

    def test_no_match_returns_none(self) -> None:
        reg = UkernelRegistry()
        decl = _make_decl("k1")
        match = _make_match("k1", op_family="matmul", priority=5, target_constraints=("has_tensor_core",))
        reg.register_ukernel(decl, [match])

        ctx = ConstraintContext(target_features=frozenset())
        result = reg.select_ukernel("matmul", ctx)
        assert result is None

    def test_highest_priority_wins(self) -> None:
        reg = UkernelRegistry()

        decl_a = _make_decl("k_low")
        match_a = _make_match("k_low", op_family="matmul", priority=1, dtype_constraints=("dtype==float32",))
        reg.register_ukernel(decl_a, [match_a])

        decl_b = _make_decl("k_high")
        match_b = _make_match("k_high", op_family="matmul", priority=20, dtype_constraints=("dtype==float32",))
        reg.register_ukernel(decl_b, [match_b])

        ctx = ConstraintContext(dtypes=("float32",))
        result = reg.select_ukernel("matmul", ctx)
        assert result is not None
        assert result.kernel_name == "k_high"

    def test_wrong_op_family_not_selected(self) -> None:
        reg = UkernelRegistry()
        decl = _make_decl("k1")
        match = _make_match("k1", op_family="matmul", priority=10)
        reg.register_ukernel(decl, [match])

        ctx = ConstraintContext()
        result = reg.select_ukernel("attention", ctx)
        assert result is None


class TestSelectBody:
    """select_body: target-family preference + fallback to 'any'."""

    def test_exact_target_preferred_over_any(self) -> None:
        reg = UkernelRegistry()
        decl = _make_decl("k1")
        body_any = _make_body("k1", target_family="any", inline_body="generic_impl")
        body_cuda = _make_body("k1", target_family="cuda", inline_body="cuda_impl")
        reg.register_ukernel(decl, bodies=[body_any, body_cuda])

        result = reg.select_body("k1", target_family="cuda")
        assert result is not None
        assert result.target_family == "cuda"
        assert result.inline_body == "cuda_impl"

    def test_fallback_to_any(self) -> None:
        reg = UkernelRegistry()
        decl = _make_decl("k1")
        body_any = _make_body("k1", target_family="any", inline_body="generic_impl")
        reg.register_ukernel(decl, bodies=[body_any])

        result = reg.select_body("k1", target_family="rvv")
        assert result is not None
        assert result.target_family == "any"

    def test_returns_none_for_unknown_kernel(self) -> None:
        reg = UkernelRegistry()
        result = reg.select_body("nonexistent_kernel", target_family="any")
        assert result is None


class TestIntrospection:
    """all_decls(), matches_for(), bodies_for() introspection."""

    def test_all_decls(self) -> None:
        reg = UkernelRegistry()
        reg.register_ukernel(_make_decl("k1"))
        reg.register_ukernel(_make_decl("k2"))
        decls = reg.all_decls()
        names = {d.kernel_name for d in decls}
        assert names == {"k1", "k2"}

    def test_matches_for_known_kernel(self) -> None:
        reg = UkernelRegistry()
        decl = _make_decl("k1")
        m1 = _make_match("k1", priority=1)
        m2 = _make_match("k1", priority=2)
        reg.register_ukernel(decl, [m1, m2])
        assert len(reg.matches_for("k1")) == 2

    def test_matches_for_unknown_kernel(self) -> None:
        reg = UkernelRegistry()
        assert reg.matches_for("nope") == []

    def test_bodies_for_known_kernel(self) -> None:
        reg = UkernelRegistry()
        decl = _make_decl("k1")
        b1 = _make_body("k1", target_family="cuda")
        b2 = _make_body("k1", target_family="any")
        reg.register_ukernel(decl, bodies=[b1, b2])
        assert len(reg.bodies_for("k1")) == 2

    def test_bodies_for_unknown_kernel(self) -> None:
        reg = UkernelRegistry()
        assert reg.bodies_for("nope") == []
