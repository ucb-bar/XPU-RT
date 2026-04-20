"""Tests for DeterministicDefaultPolicy."""

from __future__ import annotations

from compgen.agent.policies import DeterministicDefaultPolicy
from compgen.llm.registry import (
    InventSlot,
    Registry,
    Tool,
    ToolArg,
    ToolResult,
)


def _make_tool(name: str, phase: int, stub: bool, kind: str = "tool") -> Tool:
    # Tool.is_stub returns True when EITHER stub=True OR impl is None.
    # Provide a minimal impl so stub=False tools are recognised as real.
    def _noop_impl(**kw):
        return {"status": "ok"}

    return Tool(
        name=name,
        phase=phase,
        kind=kind,
        wraps_pass="stub",
        autocomp_cost_impact="low",
        args=(ToolArg("region", "region_ref", "region", required=False, default=""),),
        result=ToolResult("ok", "ok"),
        description=name,
        impl=_noop_impl,
        stub=stub,
    )


def _make_slot(name: str, phase: int, stub: bool) -> InventSlot:
    # InventSlot.is_stub is True when gate_impl is None too; provide one.
    def _accept(_proposal, **_ctx):
        return {"status": "accepted"}

    return InventSlot(
        name=name,
        phase=phase,
        input_schema="",
        output_op=f"recipe.{name}",
        gate="",
        autocomp_cost_impact="high",
        description=name,
        gate_impl=_accept,
        stub=stub,
    )


def test_empty_phase_returns_no_steps() -> None:
    r = Registry()
    p = DeterministicDefaultPolicy()
    assert p(2, r, {}) == []


def test_skips_stubs_by_default() -> None:
    r = Registry()
    r.register_tool(_make_tool("real", 2, stub=False))
    r.register_tool(_make_tool("stubbed", 2, stub=True))
    p = DeterministicDefaultPolicy()
    steps = p(2, r, {})
    names = [s[0] for s in steps]
    assert "real" in names
    assert "stubbed" not in names


def test_include_stubs_true_adds_stubs() -> None:
    r = Registry()
    r.register_tool(_make_tool("stubbed", 2, stub=True))
    p = DeterministicDefaultPolicy(include_stubs=True)
    steps = p(2, r, {})
    assert [s[0] for s in steps] == ["stubbed"]


def test_observability_and_verification_tools_excluded() -> None:
    r = Registry()
    r.register_tool(_make_tool("obs", 2, stub=False, kind="observability"))
    r.register_tool(_make_tool("ver", 2, stub=False, kind="verification"))
    p = DeterministicDefaultPolicy()
    assert p(2, r, {}) == []


def test_invent_slots_with_baseline() -> None:
    r = Registry()
    r.register_invent_slot(_make_slot("propose_x", 3, stub=False))
    p = DeterministicDefaultPolicy()
    steps = p(3, r, {})
    assert steps == [("propose_x", {"use_baseline_seed": True})]


def test_invent_strategy_skip() -> None:
    r = Registry()
    r.register_invent_slot(_make_slot("propose_x", 3, stub=False))
    p = DeterministicDefaultPolicy(invent_strategy="skip")
    assert p(3, r, {}) == []


def test_default_args_merge_with_tool_defaults() -> None:
    r = Registry()
    r.register_tool(_make_tool("t", 2, stub=False))
    p = DeterministicDefaultPolicy(default_args={"t": {"region": "r0"}})
    steps = p(2, r, {})
    assert steps[0] == ("t", {"region": "r0"})


def test_policy_against_real_registry() -> None:
    """Sanity-check against the live registry populated on import."""
    import compgen.agent.invent_slots  # noqa: F401
    import compgen.llm.tools  # noqa: F401
    from compgen.llm import get_registry

    r = get_registry()
    p = DeterministicDefaultPolicy()
    # Phase 2 has real tools + real invent-slots
    steps_2 = p(2, r, {})
    names_2 = [s[0] for s in steps_2]
    assert "decompose_concat" in names_2
    assert any(n.startswith("propose_") for n in names_2)

    # Phase 4 now hosts the ETC megakernel invent-slots
    # (propose_megakernel_synthesis, propose_scheduling_policy) alongside
    # the existing Phase-4 Tools.  Both flavours should appear.
    steps_4 = p(4, r, {})
    names_4 = [s[0] for s in steps_4]
    assert "propose_megakernel_synthesis" in names_4
    assert "propose_scheduling_policy" in names_4
