"""Tests for compgen.llm.registry."""

from __future__ import annotations

import pytest

from compgen.llm import (
    InventSlot,
    Registry,
    Tool,
    ToolArg,
    ToolResult,
    get_registry,
)


def _make_tool(name: str, phase: int = 2, impl=None, stub: bool = True) -> Tool:
    return Tool(
        name=name,
        phase=phase,
        kind="tool",
        wraps_pass=f"stub:{name}",
        autocomp_cost_impact="medium",
        args=(ToolArg("region", "region_ref", "region"),),
        result=ToolResult("diff", "diff"),
        description=f"test tool {name}",
        impl=impl,
        stub=stub,
    )


def _make_invent_slot(
    name: str, phase: int = 3, gate_impl=None, stub: bool = True
) -> InventSlot:
    return InventSlot(
        name=name,
        phase=phase,
        input_schema="input",
        output_op=f"recipe.{name}",
        gate="stub",
        autocomp_cost_impact="high",
        description=f"test slot {name}",
        gate_impl=gate_impl,
        stub=stub,
    )


@pytest.fixture
def fresh_registry() -> Registry:
    r = Registry()
    return r


def test_empty_counts(fresh_registry: Registry) -> None:
    assert fresh_registry.counts() == {
        2: {"tools": 0, "invent_slots": 0},
        3: {"tools": 0, "invent_slots": 0},
        4: {"tools": 0, "invent_slots": 0},
        5: {"tools": 0, "invent_slots": 0},
    }


def test_register_and_lookup_tool(fresh_registry: Registry) -> None:
    t = _make_tool("alpha", phase=2)
    fresh_registry.register_tool(t)
    assert fresh_registry.counts()[2]["tools"] == 1
    looked = fresh_registry.lookup_tool("alpha")
    assert looked is t
    assert fresh_registry.lookup_tool("alpha", phase=2) is t
    assert fresh_registry.lookup_tool("alpha", phase=3) is None


def test_duplicate_tool_rejected(fresh_registry: Registry) -> None:
    fresh_registry.register_tool(_make_tool("alpha", phase=2))
    with pytest.raises(ValueError, match="already registered"):
        fresh_registry.register_tool(_make_tool("alpha", phase=2))


def test_phase_mismatch_rejected(fresh_registry: Registry) -> None:
    # Tool says phase=9 which isn't a valid LLM phase
    t = Tool(
        name="bad",
        phase=9,
        kind="tool",
        wraps_pass="stub",
        autocomp_cost_impact="low",
        args=(),
        result=ToolResult("diff", ""),
        description="",
    )
    with pytest.raises(ValueError, match="is not one of"):
        fresh_registry.register_tool(t)


def test_register_invent_slot(fresh_registry: Registry) -> None:
    s = _make_invent_slot("propose_test", phase=3)
    fresh_registry.register_invent_slot(s)
    assert fresh_registry.counts()[3]["invent_slots"] == 1
    assert fresh_registry.lookup_invent_slot("propose_test") is s


def test_list_scoped_to_phase(fresh_registry: Registry) -> None:
    fresh_registry.register_tool(_make_tool("alpha", phase=2))
    fresh_registry.register_tool(_make_tool("beta", phase=3))
    assert {t.name for t in fresh_registry.list_tools(phase=2)} == {"alpha"}
    assert {t.name for t in fresh_registry.list_tools(phase=3)} == {"beta"}
    assert {t.name for t in fresh_registry.list_tools()} == {"alpha", "beta"}


def test_stub_invoke_returns_no_impl() -> None:
    t = _make_tool("stubbed", phase=2, impl=None, stub=True)
    result = t.invoke(region="r0")
    assert result["status"] == "no_impl"
    assert result["tool_name"] == "stubbed"
    assert result["echoed_args"] == {"region": "r0"}


def test_real_tool_invoke_calls_impl() -> None:
    def _real(*, region: str) -> dict:
        return {"status": "ok", "region": region}

    t = _make_tool("real", phase=2, impl=_real, stub=False)
    assert t.is_stub is False
    assert t.invoke(region="r1") == {"status": "ok", "region": "r1"}


def test_invent_slot_gate_stub_deferred() -> None:
    s = _make_invent_slot("pslot", phase=3, gate_impl=None, stub=True)
    assert s.verify({"chosen": {}}) == {
        "status": "deferred",
        "details": {"reason": "no_gate_impl"},
    }


def test_invent_slot_gate_real() -> None:
    def _gate(proposal, **ctx):
        return {
            "status": "accepted" if proposal.get("chosen") else "rejected",
            "details": {},
        }

    s = _make_invent_slot("pslot", phase=3, gate_impl=_gate, stub=False)
    assert s.verify({"chosen": {"x": 1}})["status"] == "accepted"
    assert s.verify({})["status"] == "rejected"


def test_global_registry_is_singleton() -> None:
    a = get_registry()
    b = get_registry()
    assert a is b


def test_clear_resets() -> None:
    r = Registry()
    r.register_tool(_make_tool("alpha", phase=2))
    assert r.counts()[2]["tools"] == 1
    r.clear()
    assert r.counts()[2]["tools"] == 0
