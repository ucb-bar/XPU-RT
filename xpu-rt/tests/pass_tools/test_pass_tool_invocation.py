"""pass tools as agent-callable: registry + invocation."""

from __future__ import annotations

import pytest

from compgen.pass_tools.pass_tool_registry import (
    PassToolRegistry,
    PassToolRegistryError,
    apply_pass_tool,
    build_pass_tool_registry,
    iter_pass_tool_cards,
    resolve_entrypoint,
)
from compgen.pass_tools.pass_tool_result import (
    PassToolResult,
    PassToolResultError,
    RESULT_STATUSES,
    make_blocked,
    make_no_op,
    make_proposal,
)
from compgen.pass_tools.pass_tool_types import PassToolCard


# ---------------------------------------------------------------------------
# PassToolResult shape
# ---------------------------------------------------------------------------


def test_proposal_requires_recipe_delta():
    with pytest.raises(PassToolResultError, match="non-empty recipe_delta"):
        PassToolResult(
            schema_version="pass_tool_result_v1",
            tool_id="x",
            status="proposal",
            recipe_delta=(),
        )


def test_non_proposal_status_rejects_recipe_delta():
    with pytest.raises(PassToolResultError, match="emits a recipe_delta"):
        PassToolResult(
            schema_version="pass_tool_result_v1",
            tool_id="x",
            status="no_op",
            recipe_delta=({"op": "FuseElementwise"},),
        )


def test_recipe_delta_entry_must_have_op_field():
    with pytest.raises(PassToolResultError, match="missing 'op' field"):
        PassToolResult(
            schema_version="pass_tool_result_v1",
            tool_id="x",
            status="proposal",
            recipe_delta=({"region": "r"},),
        )


def test_unknown_status_rejected():
    with pytest.raises(PassToolResultError, match="status"):
        PassToolResult(
            schema_version="pass_tool_result_v1",
            tool_id="x",
            status="wave_hands",
        )


def test_result_round_trips_through_dict():
    r = make_proposal(
        tool_id="x",
        recipe_delta=[{"op": "FuseElementwise", "region": "r"}],
        refinement_claim="tolerance_eps",
    )
    body = r.to_dict()
    assert body["status"] == "proposal"
    assert body["schema_version"] == "pass_tool_result_v1"
    restored = PassToolResult.from_dict(body)
    assert restored.tool_id == r.tool_id
    assert restored.recipe_delta == r.recipe_delta


def test_helpers_produce_typed_statuses():
    assert make_proposal(tool_id="x", recipe_delta=[{"op": "A"}], refinement_claim="").status == "proposal"
    assert make_no_op(tool_id="x").status == "no_op"
    assert make_blocked(tool_id="x", detail="why").status == "blocked"


def test_all_three_status_helpers_round_trip():
    for r in (
        make_proposal(tool_id="x", recipe_delta=[{"op": "A"}], refinement_claim=""),
        make_no_op(tool_id="x", detail="nothing matched"),
        make_blocked(tool_id="x", detail="missing precondition"),
    ):
        body = r.to_dict()
        restored = PassToolResult.from_dict(body)
        assert restored.status == r.status
        assert restored.status in RESULT_STATUSES


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_discovers_shipped_cards():
    reg = build_pass_tool_registry()
    assert "fuse_matmul_bias_relu" in reg.tool_ids()
    card = reg.card_for("fuse_matmul_bias_relu")
    assert isinstance(card, PassToolCard)
    assert card.phase == "recipe_authoring"
    assert card.refinement_kind == "tolerance_eps"


def test_registry_unknown_tool_id_raises_typed():
    reg = build_pass_tool_registry()
    with pytest.raises(PassToolRegistryError) as exc:
        reg.card_for("no_such_tool")
    assert exc.value.reason == "unknown_tool_id"


def test_registry_rejects_duplicate_tool_id():
    extras = (
        PassToolCard(
            schema_version="pass_tool_card_v1",
            tool_id="fuse_matmul_bias_relu",  # already shipped
            phase="recipe_authoring",
            reads=(),
            writes=("recipe_delta",),
            allowed_recipe_ops=(),
            refinement_kind="tolerance_eps",
            verifier="differential_then_z3_if_promoted",
            entrypoint="x:Y",
        ),
    )
    with pytest.raises(PassToolRegistryError) as exc:
        build_pass_tool_registry(extra_cards=extras)
    assert exc.value.reason == "duplicate_tool_id"


def test_resolve_entrypoint_typed_failures():
    bad_card = PassToolCard(
        schema_version="pass_tool_card_v1",
        tool_id="t",
        phase="recipe_authoring",
        reads=(),
        writes=("recipe_delta",),
        allowed_recipe_ops=(),
        refinement_kind="none",
        verifier="",
        entrypoint="no_colon_here",
    )
    with pytest.raises(PassToolRegistryError) as exc:
        resolve_entrypoint(bad_card)
    assert exc.value.reason == "bad_entrypoint_syntax"


def test_resolve_entrypoint_missing_module():
    bad_card = PassToolCard(
        schema_version="pass_tool_card_v1",
        tool_id="t",
        phase="recipe_authoring",
        reads=(),
        writes=("recipe_delta",),
        allowed_recipe_ops=(),
        refinement_kind="none",
        verifier="",
        entrypoint="this.module.does.not.exist:run",
    )
    with pytest.raises(PassToolRegistryError) as exc:
        resolve_entrypoint(bad_card)
    assert exc.value.reason == "module_not_importable"


def test_resolve_entrypoint_missing_symbol():
    bad_card = PassToolCard(
        schema_version="pass_tool_card_v1",
        tool_id="t",
        phase="recipe_authoring",
        reads=(),
        writes=("recipe_delta",),
        allowed_recipe_ops=(),
        refinement_kind="none",
        verifier="",
        entrypoint="compgen.pass_tools.builtin.fuse_matmul_bias_relu:does_not_exist",
    )
    with pytest.raises(PassToolRegistryError) as exc:
        resolve_entrypoint(bad_card)
    assert exc.value.reason == "symbol_not_in_module"


# ---------------------------------------------------------------------------
# Invocation
# ---------------------------------------------------------------------------


def test_apply_pass_tool_matches_pattern_and_returns_proposal():
    reg = build_pass_tool_registry()
    r = apply_pass_tool(
        reg,
        "fuse_matmul_bias_relu",
        region_id="region_017",
        ops=("matmul", "bias_add", "relu"),
        single_consumer=True,
    )
    assert r.status == "proposal"
    assert {op["op"] for op in r.recipe_delta} == {
        "FuseElementwise",
        "SetAccumulator",
    }
    assert r.refinement_claim == "tolerance_eps"
    assert r.evidence["matched_pattern"] == "matmul_bias_relu"


def test_apply_pass_tool_returns_no_op_on_pattern_miss():
    reg = build_pass_tool_registry()
    r = apply_pass_tool(
        reg,
        "fuse_matmul_bias_relu",
        region_id="region_017",
        ops=("conv", "relu"),
    )
    assert r.status == "no_op"
    assert "do not match" in r.detail


def test_apply_pass_tool_no_op_when_multiple_consumers():
    reg = build_pass_tool_registry()
    r = apply_pass_tool(
        reg,
        "fuse_matmul_bias_relu",
        region_id="region_017",
        ops=("matmul", "bias_add", "relu"),
        single_consumer=False,
    )
    assert r.status == "no_op"


def test_apply_pass_tool_rejects_unknown_recipe_op():
    """Hard rule 4 enforcement: a pass tool emitting an op not in
    its card's allowed_recipe_ops set is rejected."""

    bad_card = PassToolCard(
        schema_version="pass_tool_card_v1",
        tool_id="rogue_tool",
        phase="recipe_authoring",
        reads=(),
        writes=("recipe_delta",),
        allowed_recipe_ops=("FuseElementwise",),
        refinement_kind="tolerance_eps",
        verifier="",
        entrypoint="tests.pass_tools._rogue_pass:run",
    )
    extras = (bad_card,)
    registry = build_pass_tool_registry(extra_cards=extras)

    # Inject the rogue function via importable module
    import sys
    import types

    mod = types.ModuleType("tests.pass_tools._rogue_pass")
    def run(**kwargs):
        return make_proposal(
            tool_id="rogue_tool",
            recipe_delta=[{"op": "TotallyMadeUpOp"}],  # outside allowed_recipe_ops
            refinement_claim="tolerance_eps",
        )
    mod.run = run
    sys.modules["tests.pass_tools._rogue_pass"] = mod

    with pytest.raises(PassToolResultError, match="allowed_recipe_ops"):
        apply_pass_tool(
            registry,
            "rogue_tool",
            region_id="r",
            ops=("matmul",),
        )


def test_apply_pass_tool_rejects_wrong_tool_id_in_result():
    """A pass tool that returns a result claiming a different
    tool_id must be rejected."""

    bad_card = PassToolCard(
        schema_version="pass_tool_card_v1",
        tool_id="mismatch_tool",
        phase="recipe_authoring",
        reads=(),
        writes=("recipe_delta",),
        allowed_recipe_ops=("FuseElementwise",),
        refinement_kind="tolerance_eps",
        verifier="",
        entrypoint="tests.pass_tools._mismatch_pass:run",
    )
    registry = build_pass_tool_registry(extra_cards=(bad_card,))

    import sys
    import types

    mod = types.ModuleType("tests.pass_tools._mismatch_pass")
    def run(**kwargs):
        return make_proposal(
            tool_id="not_mismatch_tool",  # WRONG
            recipe_delta=[{"op": "FuseElementwise"}],
            refinement_claim="tolerance_eps",
        )
    mod.run = run
    sys.modules["tests.pass_tools._mismatch_pass"] = mod

    with pytest.raises(PassToolResultError, match="tool_id mismatch"):
        apply_pass_tool(registry, "mismatch_tool")


def test_apply_pass_tool_rejects_non_result_return():
    """A pass tool that returns a non-PassToolResult is rejected."""

    bad_card = PassToolCard(
        schema_version="pass_tool_card_v1",
        tool_id="bad_return_tool",
        phase="recipe_authoring",
        reads=(),
        writes=("recipe_delta",),
        allowed_recipe_ops=(),
        refinement_kind="none",
        verifier="",
        entrypoint="tests.pass_tools._bad_return_pass:run",
    )
    registry = build_pass_tool_registry(extra_cards=(bad_card,))

    import sys
    import types

    mod = types.ModuleType("tests.pass_tools._bad_return_pass")
    def run(**kwargs):
        return {"this_is_just": "a_dict"}
    mod.run = run
    sys.modules["tests.pass_tools._bad_return_pass"] = mod

    with pytest.raises(PassToolResultError, match="expected PassToolResult"):
        apply_pass_tool(registry, "bad_return_tool")


def test_card_inventory_covers_all_shipped_pass_tools():
    """Every YAML under python/compgen/pass_tools/cards/ must load
    via iter_pass_tool_cards()."""
    cards = tuple(iter_pass_tool_cards())
    assert cards
    for c in cards:
        assert c.tool_id
        assert c.phase
