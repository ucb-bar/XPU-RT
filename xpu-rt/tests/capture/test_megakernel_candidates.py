"""Tests for megakernel-candidate detection (Phase A.9)."""

from __future__ import annotations

from compgen.capture.inductor_harvest import (
    MegakernelCandidate,
    estimate_megakernel_candidates,
)


def test_returns_empty_when_no_pattern_matches() -> None:
    hist = {"aten.cat.default": 1, "aten.view.default": 3}
    assert estimate_megakernel_candidates(hist, []) == []


def test_detects_matmul_collective_pattern() -> None:
    hist = {
        "aten.mm.default": 4,
        "aten.reduce_scatter.default": 1,
    }
    candidates = estimate_megakernel_candidates(hist, [])
    patterns = {c.pattern for c in candidates}
    assert "matmul_collective" in patterns
    mc = next(c for c in candidates if c.pattern == "matmul_collective")
    assert "aten.mm.default" in mc.ops
    assert "aten.reduce_scatter.default" in mc.ops
    assert mc.confidence >= 0.8


def test_detects_attention_pipeline_pattern() -> None:
    hist = {
        "aten._scaled_dot_product_flash_attention.default": 1,
        "aten.mm.default": 6,
    }
    candidates = estimate_megakernel_candidates(hist, [])
    assert any(c.pattern == "attention_pipeline" for c in candidates)


def test_detects_moe_routing_pattern() -> None:
    hist = {
        "aten.topk.default": 1,
        "aten.scatter.default": 1,
        "aten.mm.default": 8,
    }
    candidates = estimate_megakernel_candidates(hist, [])
    assert any(c.pattern == "moe_routing" for c in candidates)


def test_detects_unfused_chain_when_long_groups_present() -> None:
    long = ("aten.add.Tensor",) * 4
    candidates = estimate_megakernel_candidates({}, [long])
    assert any(c.pattern == "unfused_chain" for c in candidates)


def test_short_chain_does_not_trigger_unfused_chain() -> None:
    short = ("aten.add.Tensor", "aten.relu.default")
    candidates = estimate_megakernel_candidates({}, [short])
    assert not any(c.pattern == "unfused_chain" for c in candidates)


def test_returns_list_of_megakernel_candidate_dataclasses() -> None:
    hist = {"aten.mm.default": 1, "aten.all_gather.default": 1}
    candidates = estimate_megakernel_candidates(hist, [])
    assert candidates
    assert all(isinstance(c, MegakernelCandidate) for c in candidates)
    for c in candidates:
        assert 0.0 <= c.confidence <= 1.0
        assert c.rationale
        assert c.ops


def test_multiple_patterns_can_coexist() -> None:
    hist = {
        "aten.mm.default": 4,
        "aten.reduce_scatter.default": 1,
        "aten.topk.default": 1,
        "aten.gather.default": 1,
    }
    long = ("aten.relu.default",) * 5
    patterns = {c.pattern for c in estimate_megakernel_candidates(hist, [long])}
    assert {"matmul_collective", "moe_routing", "unfused_chain"} <= patterns
