"""Tests for :mod:`compgen.ir.recipe.payload_mutators`.

Asserts that the direct-mutation pass actually stamps payload op
attributes when the recipe contains FuseOp / ProposeFusionOp / TileOp
/ PlaceOnDeviceOp / ProposeMegakernelSynthesisOp.
"""

from __future__ import annotations

from compgen.agent.recipe_bridge_invent import proposal_to_recipe_op
from compgen.ir.recipe.attrs import DeviceRefAttr, ProvenanceAttr
from compgen.ir.recipe.ops_candidate import FuseOp, PlaceOnDeviceOp, TileOp
from compgen.ir.recipe.payload_mutators import (
    PayloadMutationReport,
    apply_recipe_to_payload,
)
from xdsl.dialects.builtin import (
    ArrayAttr,
    Float32Type,
    IntegerAttr,
    IntegerType,
    ModuleOp,
    StringAttr,
    SymbolRefAttr,
    TensorType,
)
from xdsl.dialects.func import FuncOp
from xdsl.ir import Block, Region


def _payload_with_regions(regions: list[str]) -> ModuleOp:
    """Build a tiny payload module: one func.func with a few func.call ops
    each tagged with ``compgen.region_id``."""
    from xdsl.dialects.func import CallOp, ReturnOp

    module = ModuleOp(Region([Block()]))
    inputs = [TensorType(Float32Type(), [4, 4]) for _ in regions]
    outputs = [TensorType(Float32Type(), [4, 4]) for _ in regions]
    func = FuncOp(name="forward", function_type=(inputs, outputs))
    block = func.body.blocks[0]
    rets = []
    for i, region in enumerate(regions):
        # Body-less private callee declaration.
        module.body.block.add_op(
            FuncOp.external(
                f"helper_{i}",
                [inputs[i]],
                [outputs[i]],
            )
        )
        call = CallOp(
            callee=f"helper_{i}",
            arguments=[block.args[i]],
            return_types=[outputs[i]],
        )
        call.attributes["compgen.region_id"] = StringAttr(region)
        block.add_op(call)
        rets.append(call.results[0])
    block.add_op(ReturnOp(*rets))
    module.body.block.add_op(func)
    return module


def _empty_recipe() -> ModuleOp:
    return ModuleOp(Region([Block()]))


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------


def test_empty_recipe_is_noop() -> None:
    payload = _payload_with_regions(["r_0", "r_1"])
    report = apply_recipe_to_payload(_empty_recipe(), payload)
    assert isinstance(report, PayloadMutationReport)
    assert report.total() == 0


# ---------------------------------------------------------------------------
# ProposeFusionOp + FuseOp
# ---------------------------------------------------------------------------


def test_propose_fusion_stamps_fused_into_on_matching_ops() -> None:
    payload = _payload_with_regions(["r_a", "r_b", "r_c"])
    recipe = _empty_recipe()
    op = proposal_to_recipe_op(
        "propose_fusion",
        {
            "chosen": {"grouped_regions": ["r_a", "r_b"]},
            "select_vs_invent": "invent",
        },
    )
    recipe.body.block.add_op(op)

    report = apply_recipe_to_payload(recipe, payload)
    assert report.fusions_applied == 1
    assert report.payload_ops_touched == 2

    # Walk payload, check the two matching ops got new region_id +
    # fused_into; the third (r_c) stays unchanged.
    region_ids = [o.attributes["compgen.region_id"].data for o in payload.walk() if "compgen.region_id" in o.attributes]
    fused_ids = [o.attributes.get("compgen.fused_into") for o in payload.walk() if "compgen.fused_into" in o.attributes]
    # Two of the three ops were rewritten to a shared fused_<digest>.
    fused_set = {r for r in region_ids if r.startswith("fused_")}
    assert len(fused_set) == 1
    assert len(fused_ids) == 2
    # The untouched op kept r_c.
    assert "r_c" in region_ids


def test_fuseop_directly_also_stamps() -> None:
    payload = _payload_with_regions(["r_x", "r_y"])
    recipe = _empty_recipe()
    op = FuseOp.build(
        properties={
            "sym_name": StringAttr("fuse_xy"),
            "fuse_regions": ArrayAttr(
                [
                    SymbolRefAttr("r_x"),
                    SymbolRefAttr("r_y"),
                ]
            ),
            "provenance": ProvenanceAttr("agent", 0),
        }
    )
    recipe.body.block.add_op(op)
    report = apply_recipe_to_payload(recipe, payload)
    assert report.fusions_applied == 1
    assert report.payload_ops_touched == 2


def test_fusion_idempotent_on_repeat() -> None:
    payload = _payload_with_regions(["r_p", "r_q"])
    recipe = _empty_recipe()
    op = proposal_to_recipe_op(
        "propose_fusion",
        {
            "chosen": {"grouped_regions": ["r_p", "r_q"]},
            "select_vs_invent": "invent",
        },
    )
    recipe.body.block.add_op(op)
    apply_recipe_to_payload(recipe, payload)

    # Second pass — region_ids are now `fused_*`, so the propose_fusion
    # which still references r_p/r_q will not match any ops. Touch
    # count should be 0 the second time.
    report2 = apply_recipe_to_payload(recipe, payload)
    assert report2.payload_ops_touched == 0


# ---------------------------------------------------------------------------
# TileOp
# ---------------------------------------------------------------------------


def test_tileop_stamps_tile_sizes() -> None:
    payload = _payload_with_regions(["r_t"])
    recipe = _empty_recipe()
    tile = TileOp.build(
        properties={
            "sym_name": StringAttr("tile_t"),
            "region_ref": SymbolRefAttr("r_t"),
            "tile_sizes": ArrayAttr(
                [
                    IntegerAttr(32, IntegerType(64)),
                    IntegerAttr(64, IntegerType(64)),
                ]
            ),
            "provenance": ProvenanceAttr("agent", 0),
        }
    )
    recipe.body.block.add_op(tile)
    report = apply_recipe_to_payload(recipe, payload)
    assert report.tiles_applied == 1
    touched = [o for o in payload.walk() if "compgen.tile_sizes_str" in o.attributes]
    assert len(touched) == 1
    assert touched[0].attributes["compgen.tile_sizes_str"].data == "32,64"


# ---------------------------------------------------------------------------
# PlaceOnDeviceOp
# ---------------------------------------------------------------------------


def test_placeop_stamps_device_attribute() -> None:
    payload = _payload_with_regions(["r_d"])
    recipe = _empty_recipe()
    place = PlaceOnDeviceOp.build(
        properties={
            "sym_name": StringAttr("place_d"),
            "region_ref": SymbolRefAttr("r_d"),
            "device": DeviceRefAttr(2, "device"),
            "provenance": ProvenanceAttr("agent", 0),
        }
    )
    recipe.body.block.add_op(place)
    report = apply_recipe_to_payload(recipe, payload)
    assert report.placements_applied == 1
    touched = [
        o
        for o in payload.walk()
        if o.attributes.get("compgen.device") is not None and "device_2" in o.attributes["compgen.device"].data
    ]
    assert len(touched) == 1


# ---------------------------------------------------------------------------
# ProposeMegakernelSynthesisOp
# ---------------------------------------------------------------------------


def test_propose_megakernel_stamps_megakernel_name() -> None:
    payload = _payload_with_regions(["r_mk_0", "r_mk_1"])
    recipe = _empty_recipe()
    op = proposal_to_recipe_op(
        "propose_megakernel_synthesis",
        {
            "chosen": {
                "megakernel_name": "gemma_block_mk",
                "fused_region_refs": ["r_mk_0", "r_mk_1"],
            },
            "select_vs_invent": "invent",
            "target_feature_justification": "persistent_kernels",
        },
    )
    recipe.body.block.add_op(op)
    report = apply_recipe_to_payload(recipe, payload)
    assert report.megakernels_applied == 1
    touched = [o for o in payload.walk() if o.attributes.get("compgen.megakernel") is not None]
    assert len(touched) == 2
    assert all(o.attributes["compgen.megakernel"].data == "gemma_block_mk" for o in touched)


def test_report_to_dict_is_serialisable() -> None:
    payload = _payload_with_regions(["r_0", "r_1"])
    recipe = _empty_recipe()
    op = proposal_to_recipe_op(
        "propose_fusion",
        {"chosen": {"grouped_regions": ["r_0", "r_1"]}, "select_vs_invent": "invent"},
    )
    recipe.body.block.add_op(op)
    report = apply_recipe_to_payload(recipe, payload)
    d = report.to_dict()
    import json

    json.dumps(d)
    assert d["fusions_applied"] == 1
    assert d["payload_ops_touched"] == 2


# ---------------------------------------------------------------------------
# Structural fusion (collapse N CallOps -> 1)
# ---------------------------------------------------------------------------


def _payload_with_chain(regions: list[str]) -> ModuleOp:
    """Build a payload with a real producer-consumer chain.

    forward(in_0):
        t0 = call helper_0(in_0)        # region "<regions[0]>"
        t1 = call helper_1(t0)          # region "<regions[1]>"
        ...
        return t_last
    """
    from xdsl.dialects.func import CallOp, ReturnOp

    n = len(regions)
    module = ModuleOp(Region([Block()]))
    in_t = TensorType(Float32Type(), [4, 4])
    out_t = TensorType(Float32Type(), [4, 4])
    func = FuncOp(name="forward", function_type=([in_t], [out_t]))
    block = func.body.blocks[0]
    cur = block.args[0]
    last_call = None
    # Add helper declarations FIRST. ``FuncOp.external`` builds a
    # body-less private declaration that passes xDSL's verifier (the
    # plain ctor would leave an empty single-block region).
    for i in range(n):
        module.body.block.add_op(
            FuncOp.external(
                f"helper_{i}",
                [in_t],
                [out_t],
            )
        )
    for i in range(n):
        call = CallOp(
            callee=f"helper_{i}",
            arguments=[cur],
            return_types=[out_t],
        )
        call.attributes["compgen.region_id"] = StringAttr(regions[i])
        block.add_op(call)
        cur = call.results[0]
        last_call = call
    block.add_op(ReturnOp(cur))
    module.body.block.add_op(func)
    return module


def _count_calls_in_forward(module: ModuleOp) -> int:
    from xdsl.dialects.func import CallOp as _CallOp

    count = 0
    for op in module.walk():
        if isinstance(op, _CallOp):
            count += 1
    return count


def _has_func_named(module: ModuleOp, name: str) -> bool:
    from xdsl.dialects.func import FuncOp as _FuncOp

    return any(isinstance(o, _FuncOp) and o.sym_name.data == name for o in module.body.block.ops)


def test_structural_fusion_collapses_two_call_chain() -> None:
    """fuse(r_a, r_b) on a producer-consumer chain → 1 CallOp + new fused decl."""
    module = _payload_with_chain(["r_a", "r_b"])
    calls_before = _count_calls_in_forward(module)
    assert calls_before == 2

    recipe = _empty_recipe()
    op = proposal_to_recipe_op(
        "propose_fusion",
        {"chosen": {"grouped_regions": ["r_a", "r_b"]}, "select_vs_invent": "invent"},
    )
    recipe.body.block.add_op(op)
    report = apply_recipe_to_payload(recipe, module)

    assert report.structural_fusions == 1
    assert report.structural_callees_added == 1
    calls_after = _count_calls_in_forward(module)
    assert calls_after == 1, "expected the two CallOps to collapse into one fused call"
    # The fused callee declaration must exist as a private FuncOp.
    assert _has_func_named(module, "fused_helper_0__helper_1")


def test_structural_fusion_three_op_chain() -> None:
    module = _payload_with_chain(["r_a", "r_b", "r_c"])
    assert _count_calls_in_forward(module) == 3

    recipe = _empty_recipe()
    recipe.body.block.add_op(
        proposal_to_recipe_op(
            "propose_fusion",
            {"chosen": {"grouped_regions": ["r_a", "r_b", "r_c"]}, "select_vs_invent": "invent"},
        )
    )
    report = apply_recipe_to_payload(recipe, module)

    assert report.structural_fusions == 1
    assert _count_calls_in_forward(module) == 1
    assert _has_func_named(module, "fused_helper_0__helper_1__helper_2")


def test_non_chain_falls_back_to_attribute_stamping() -> None:
    """Two parallel CallOps (no producer-consumer link) → no structural fuse."""
    payload = _payload_with_regions(["r_par_0", "r_par_1"])
    calls_before = _count_calls_in_forward(payload)

    recipe = _empty_recipe()
    recipe.body.block.add_op(
        proposal_to_recipe_op(
            "propose_fusion",
            {"chosen": {"grouped_regions": ["r_par_0", "r_par_1"]}, "select_vs_invent": "invent"},
        )
    )
    report = apply_recipe_to_payload(recipe, payload)

    assert report.structural_fusions == 0
    assert report.fusions_applied == 1  # attribute-only stamp succeeded
    # Op count unchanged.
    assert _count_calls_in_forward(payload) == calls_before


def test_structural_fusion_module_remains_verifiable() -> None:
    """Running verify() on the post-fusion module must not raise."""
    module = _payload_with_chain(["r_a", "r_b"])
    recipe = _empty_recipe()
    recipe.body.block.add_op(
        proposal_to_recipe_op(
            "propose_fusion",
            {"chosen": {"grouped_regions": ["r_a", "r_b"]}, "select_vs_invent": "invent"},
        )
    )
    apply_recipe_to_payload(recipe, module)
    module.verify()  # must not raise


def test_structural_fusion_does_not_break_other_consumers() -> None:
    """When an intermediate value has multiple uses, fall back to stamping."""
    from xdsl.dialects.func import CallOp, ReturnOp

    module = ModuleOp(Region([Block()]))
    in_t = TensorType(Float32Type(), [4, 4])
    for i in range(3):
        module.body.block.add_op(
            FuncOp.external(
                f"helper_{i}",
                [in_t],
                [in_t],
            )
        )
    func = FuncOp(name="forward", function_type=([in_t], [in_t]))
    block = func.body.blocks[0]
    c0 = CallOp(callee="helper_0", arguments=[block.args[0]], return_types=[in_t])
    c0.attributes["compgen.region_id"] = StringAttr("rA")
    block.add_op(c0)
    c1 = CallOp(callee="helper_1", arguments=[c0.results[0]], return_types=[in_t])
    c1.attributes["compgen.region_id"] = StringAttr("rB")
    block.add_op(c1)
    # Second consumer of c0's result — outside the chain — blocks structural fusion.
    c2 = CallOp(callee="helper_2", arguments=[c0.results[0]], return_types=[in_t])
    c2.attributes["compgen.region_id"] = StringAttr("rC_external")
    block.add_op(c2)
    block.add_op(ReturnOp(c1.results[0]))
    module.body.block.add_op(func)

    recipe = _empty_recipe()
    recipe.body.block.add_op(
        proposal_to_recipe_op(
            "propose_fusion",
            {"chosen": {"grouped_regions": ["rA", "rB"]}, "select_vs_invent": "invent"},
        )
    )
    report = apply_recipe_to_payload(recipe, module)
    # Structural fusion must be refused — c0's output also feeds c2.
    assert report.structural_fusions == 0
    # But attribute-stamping should still have stamped both ops.
    assert report.fusions_applied == 1
    assert _count_calls_in_forward(module) == 3  # nothing erased
