"""Tests for promoted guard artifacts, registry, and runtime evaluation."""

from __future__ import annotations

from compgen.semantic.synthesis.guard_lang import Cmp, CmpOp, Const, Var
from compgen.semantic.synthesis.promote import GuardArtifact
from compgen.semantic.synthesis.registry import GuardRegistry
from compgen.semantic.synthesis.runtime import GuardRuntime


def test_guard_runtime_accepts_matching_env() -> None:
    registry = GuardRegistry()
    registry.register(
        GuardArtifact(
            guard_key="guard.fusion.legality.generic.1",
            transform_family="fusion",
            guard_kind="legality",
            fragments=(Cmp(CmpOp.EQ, Var("fusible"), Const(True)),),
        )
    )
    runtime = GuardRuntime(registry)
    verdict = runtime.evaluate("guard.fusion.legality.generic.1", {"fusible": True})
    assert verdict.allow is True
    assert verdict.reason == "guard_matched"


def test_guard_runtime_rejects_non_matching_env() -> None:
    registry = GuardRegistry()
    registry.register(
        GuardArtifact(
            guard_key="guard.local_mem.placement.generic.1",
            transform_family="local_mem",
            guard_kind="placement",
            fragments=(Cmp(CmpOp.EQ, Var("local_mem_fit"), Const(True)),),
        )
    )
    runtime = GuardRuntime(registry)
    verdict = runtime.evaluate("guard.local_mem.placement.generic.1", {"local_mem_fit": False})
    assert verdict.allow is False
    assert verdict.failed_fragment_index == 0
