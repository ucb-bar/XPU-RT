"""Tests for the megakernel persistent-kernel gate."""

from __future__ import annotations

from compgen.agent.gates.megakernel import megakernel_persistent_kernel_gate


def _mk_proposal(**chosen_overrides: object) -> dict[str, object]:
    chosen = {
        "megakernel_name": "mk",
        "fused_region_refs": ["r0", "r1"],
        "event_tensor_decls": [
            {"name": "E", "shape": [4], "wait_count": 1, "scope": "device"}
        ],
        "task_partition": {"r0": [4], "r1": [4]},
    }
    chosen.update(chosen_overrides)
    return {
        "chosen": chosen,
        "select_vs_invent": "invent",
    }


# ---------------------------------------------------------------------------
# Acceptance path
# ---------------------------------------------------------------------------


def test_well_formed_proposal_is_accepted() -> None:
    out = megakernel_persistent_kernel_gate(_mk_proposal())
    assert out["status"] == "accepted"
    assert out["details"]["fused_region_count"] == 2
    assert out["details"]["event_decl_count"] == 1


def test_target_features_satisfied_is_accepted() -> None:
    out = megakernel_persistent_kernel_gate(
        _mk_proposal(),
        target_features={"persistent_kernels", "semaphore_atomics", "sm_count"},
    )
    assert out["status"] == "accepted"


def test_policy_proposal_with_static_choice_is_accepted() -> None:
    proposal = {
        "chosen": {"policy": "static", "sm_count": 108, "early_push": False},
        "select_vs_invent": "select",
    }
    out = megakernel_persistent_kernel_gate(proposal)
    assert out["status"] == "accepted"
    assert out["details"]["policy"] == "static"


def test_policy_proposal_with_dynamic_choice_is_accepted() -> None:
    proposal = {
        "chosen": {"policy": "dynamic", "sm_count": 108, "early_push": True},
        "select_vs_invent": "invent",
    }
    out = megakernel_persistent_kernel_gate(proposal)
    assert out["status"] == "accepted"


# ---------------------------------------------------------------------------
# Rejection paths
# ---------------------------------------------------------------------------


def test_missing_chosen_is_rejected() -> None:
    out = megakernel_persistent_kernel_gate({"select_vs_invent": "invent"})
    assert out["status"] == "rejected"
    assert "missing_or_invalid_chosen" in out["details"]["reason"]


def test_empty_fused_region_refs_is_rejected() -> None:
    out = megakernel_persistent_kernel_gate(_mk_proposal(fused_region_refs=[]))
    assert out["status"] == "rejected"
    assert "fused_region_refs" in out["details"]["reason"]


def test_empty_event_shape_is_rejected() -> None:
    out = megakernel_persistent_kernel_gate(
        _mk_proposal(
            event_tensor_decls=[{"name": "E", "shape": [], "wait_count": 1}],
        )
    )
    assert out["status"] == "rejected"
    assert "event_decl shape empty" in out["details"]["reason"]


def test_negative_wait_count_is_rejected() -> None:
    out = megakernel_persistent_kernel_gate(
        _mk_proposal(
            event_tensor_decls=[{"name": "E", "shape": [4], "wait_count": -1}],
        )
    )
    assert out["status"] == "rejected"
    assert "wait_count" in out["details"]["reason"]


def test_invalid_policy_is_rejected() -> None:
    proposal = {
        "chosen": {"policy": "greedy", "sm_count": 108},
        "select_vs_invent": "select",
    }
    out = megakernel_persistent_kernel_gate(proposal)
    assert out["status"] == "rejected"
    assert "invalid scheduling policy" in out["details"]["reason"]


def test_target_lacking_required_capabilities_is_rejected() -> None:
    out = megakernel_persistent_kernel_gate(
        _mk_proposal(),
        target_features={"sm_count"},  # missing both persistent_kernels and semaphore_atomics
    )
    assert out["status"] == "rejected"
    assert set(out["details"]["missing"]) == {
        "persistent_kernels",
        "semaphore_atomics",
    }


def test_target_lacking_one_capability_is_rejected() -> None:
    out = megakernel_persistent_kernel_gate(
        _mk_proposal(),
        target_features={"persistent_kernels", "sm_count"},
    )
    assert out["status"] == "rejected"
    assert out["details"]["missing"] == ["semaphore_atomics"]


# ---------------------------------------------------------------------------
# Ukernel-call exclusion
# ---------------------------------------------------------------------------


def test_event_graph_with_no_ukernel_call_is_accepted() -> None:
    from xdsl.dialects.builtin import IntegerAttr, IntegerType, StringAttr
    from xdsl.ir import Block, Region

    from compgen.ir.event.attrs import EventTensorTypeAttr
    from compgen.ir.event.ops import EventTensorOp, GraphOp

    block = Block()
    block.add_op(
        EventTensorOp.create(
            properties={
                "sym_name": StringAttr("E"),
                "event_type": EventTensorTypeAttr([1]),
                "wait_count": IntegerAttr(0, IntegerType(64)),
            },
        ),
    )
    graph = GraphOp(sym_name="g", policy="static", body=Region([block]))
    out = megakernel_persistent_kernel_gate(_mk_proposal(), event_graph=graph)
    assert out["status"] == "accepted"
