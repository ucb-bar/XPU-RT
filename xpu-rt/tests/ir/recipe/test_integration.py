"""Integration tests for the Recipe IR dialect.

Tests the full pipeline: seed generation → validation → lowering,
and the agent bridge for action ↔ recipe op conversion.
"""

from __future__ import annotations

from compgen.ir.recipe.attrs import DeviceRefAttr, ProvenanceAttr
from compgen.ir.recipe.lower import LoweringOutput, lower_recipe
from compgen.ir.recipe.ops_candidate import PlaceOnDeviceOp, TileOp
from compgen.ir.recipe.ops_choice import RequireEqsatOp, RequireSolverOp
from compgen.ir.recipe.ops_fact import BackendAvailableOp
from compgen.ir.recipe.ops_provenance import FromTemplateOp
from compgen.ir.recipe.ops_scope import RecipeRegionOp
from compgen.ir.recipe.ops_verify import RequireDiffTestOp, RequireMemoryBoundOp
from compgen.ir.recipe.validate import validate_recipe_module
from xdsl.dialects.builtin import (
    ArrayAttr,
    IntegerAttr,
    IntegerType,
    ModuleOp,
    StringAttr,
    SymbolRefAttr,
)
from xdsl.ir import Block, Region


def _build_simple_recipe() -> ModuleOp:
    """Build a minimal valid recipe module for testing."""
    block = Block()

    # Provenance
    block.add_op(FromTemplateOp.build(properties={
        "template_name": StringAttr("test"),
        "template_version": IntegerAttr(1, IntegerType(64)),
    }))

    # Region
    block.add_op(RecipeRegionOp.build(properties={
        "sym_name": StringAttr("r0"),
        "payload_region_id": StringAttr("matmul_0"),
    }))

    # Facts
    block.add_op(BackendAvailableOp.build(properties={
        "region_ref": SymbolRefAttr("r0"),
        "backend": StringAttr("triton"),
    }))

    # Candidate actions
    block.add_op(TileOp.build(properties={
        "region_ref": SymbolRefAttr("r0"),
        "tile_sizes": ArrayAttr([
            IntegerAttr(64, IntegerType(64)),
            IntegerAttr(32, IntegerType(64)),
        ]),
        "provenance": ProvenanceAttr("seed", 0),
    }))
    block.add_op(PlaceOnDeviceOp.build(properties={
        "region_ref": SymbolRefAttr("r0"),
        "device": DeviceRefAttr(0, "gpu0"),
    }))

    # Verification
    block.add_op(RequireDiffTestOp.build(properties={
        "region_ref": SymbolRefAttr("r0"),
    }))

    return ModuleOp(Region(block))


# ---- Validation tests ----


def test_valid_recipe_passes_validation() -> None:
    module = _build_simple_recipe()
    result = validate_recipe_module(module)
    assert result.valid


def test_unresolved_symbol_fails_validation() -> None:
    """Reference to undefined region should fail."""
    block = Block()
    # No RegionOp defined, but TileOp references "r_missing"
    block.add_op(TileOp.build(properties={
        "region_ref": SymbolRefAttr("r_missing"),
        "tile_sizes": ArrayAttr([IntegerAttr(64, IntegerType(64))]),
    }))
    module = ModuleOp(Region(block))
    result = validate_recipe_module(module)
    assert not result.valid
    assert any("r_missing" in e.message for e in result.errors)


def test_conflicting_placement_fails_validation() -> None:
    """Two PlaceOnDeviceOp for same region with different devices."""
    block = Block()
    block.add_op(RecipeRegionOp.build(properties={
        "sym_name": StringAttr("r0"),
        "payload_region_id": StringAttr("op_0"),
    }))
    block.add_op(PlaceOnDeviceOp.build(properties={
        "region_ref": SymbolRefAttr("r0"),
        "device": DeviceRefAttr(0, "gpu0"),
    }))
    block.add_op(PlaceOnDeviceOp.build(properties={
        "region_ref": SymbolRefAttr("r0"),
        "device": DeviceRefAttr(1, "gpu1"),
    }))
    module = ModuleOp(Region(block))
    result = validate_recipe_module(module)
    assert not result.valid
    assert any("conflict" in e.message.lower() for e in result.errors)


# ---- Lowering tests ----


def test_lower_produces_transform_scripts() -> None:
    module = _build_simple_recipe()
    output = lower_recipe(module)
    assert isinstance(output, LoweringOutput)
    # TileOp should produce a transform script
    assert len(output.transform_scripts) >= 1
    assert "tile" in output.transform_scripts[0].lower()


def test_lower_produces_plan_fragments() -> None:
    module = _build_simple_recipe()
    output = lower_recipe(module)
    # PlaceOnDeviceOp should produce a plan fragment
    assert len(output.plan_fragments) >= 1
    assert output.plan_fragments[0]["type"] == "placement"


def test_lower_produces_verification_obligations() -> None:
    module = _build_simple_recipe()
    output = lower_recipe(module)
    assert len(output.verification_obligations) >= 1
    assert output.verification_obligations[0]["type"] == "differential"


def test_lower_solver_job() -> None:
    block = Block()
    block.add_op(RequireSolverOp.build(properties={
        "solve_type": StringAttr("placement"),
        "timeout_ms": IntegerAttr(5000, IntegerType(64)),
    }))
    module = ModuleOp(Region(block))
    output = lower_recipe(module)
    assert len(output.plan_fragments) == 1
    assert output.plan_fragments[0]["type"] == "solver"
    assert output.plan_fragments[0]["solve_type"] == "placement"


def test_lower_eqsat_job() -> None:
    block = Block()
    block.add_op(RecipeRegionOp.build(properties={
        "sym_name": StringAttr("r0"),
        "payload_region_id": StringAttr("matmul_0"),
    }))
    block.add_op(RequireEqsatOp.build(properties={
        "region_ref": SymbolRefAttr("r0"),
        "rule_categories": ArrayAttr([StringAttr("algebraic"), StringAttr("fusion")]),
    }))
    module = ModuleOp(Region(block))
    output = lower_recipe(module)
    assert len(output.eqsat_jobs) == 1
    assert output.eqsat_jobs[0]["type"] == "eqsat"
    assert "algebraic" in output.eqsat_jobs[0]["rule_categories"]


def test_lower_memory_bound() -> None:
    block = Block()
    block.add_op(RecipeRegionOp.build(properties={
        "sym_name": StringAttr("r0"),
        "payload_region_id": StringAttr("op_0"),
    }))
    block.add_op(RequireMemoryBoundOp.build(properties={
        "region_ref": SymbolRefAttr("r0"),
        "max_bytes": IntegerAttr(1048576, IntegerType(64)),
        "device": DeviceRefAttr(0, "gpu0"),
    }))
    module = ModuleOp(Region(block))
    output = lower_recipe(module)
    assert len(output.verification_obligations) == 1
    assert output.verification_obligations[0]["max_bytes"] == 1048576


# ---- Agent bridge tests ----


def test_action_to_recipe_op_tile() -> None:
    from compgen.agent.env import TileAction
    from compgen.agent.recipe_bridge import action_to_recipe_op

    action = TileAction(region_id="r0", tile_sizes=(128, 64))
    op = action_to_recipe_op(action, iteration=3)
    assert op is not None
    assert isinstance(op, TileOp)
    assert op.region_ref.root_reference.data == "r0"


def test_action_to_recipe_op_place() -> None:
    from compgen.agent.env import AssignDeviceAction
    from compgen.agent.recipe_bridge import action_to_recipe_op

    action = AssignDeviceAction(region_id="r0", device_index=1)
    op = action_to_recipe_op(action, iteration=0)
    assert op is not None
    assert isinstance(op, PlaceOnDeviceOp)
    assert op.device.index.value.data == 1


def test_action_to_recipe_op_noop_returns_none() -> None:
    from compgen.agent.env import NoopAction
    from compgen.agent.recipe_bridge import action_to_recipe_op

    action = NoopAction()
    op = action_to_recipe_op(action)
    assert op is None


def test_recipe_op_to_action_tile() -> None:
    from compgen.agent.env import TileAction
    from compgen.agent.recipe_bridge import recipe_op_to_action

    op = TileOp.build(properties={
        "region_ref": SymbolRefAttr("r0"),
        "tile_sizes": ArrayAttr([IntegerAttr(64, IntegerType(64))]),
    })
    action = recipe_op_to_action(op)
    assert isinstance(action, TileAction)
    assert action.tile_sizes == (64,)


# ---- Seed generation tests ----


def test_seed_generation_basic() -> None:
    """Seed generation from an empty payload module."""
    from compgen.ir.recipe.seed import generate_seed_recipe

    payload = ModuleOp(Region(Block()))
    seed = generate_seed_recipe(payload)
    assert seed is not None
    # Should have at least a provenance op
    ops = list(seed.body.block.ops)
    assert len(ops) >= 1


# ---- Compat shim tests ----


def test_compat_dataclass_to_xdsl_tile() -> None:
    from compgen.ir.recipe.compat import dataclass_to_xdsl
    from compgen.ir.recipe.ops import SetTileParams

    old_op = SetTileParams(region_id="r0", tile_sizes=(128, 64, 32))
    new_op = dataclass_to_xdsl(old_op)
    assert new_op is not None
    assert isinstance(new_op, TileOp)


def test_compat_dataclass_to_xdsl_assign_device() -> None:
    from compgen.ir.recipe.compat import dataclass_to_xdsl
    from compgen.ir.recipe.ops import AssignDevice

    old_op = AssignDevice(region_id="r0", device_index=1, reason="gpu")
    new_op = dataclass_to_xdsl(old_op)
    assert new_op is not None
    assert isinstance(new_op, PlaceOnDeviceOp)
    assert new_op.device.index.value.data == 1


def test_compat_round_trip() -> None:
    from compgen.ir.recipe.compat import (
        module_to_recipe_list,
        recipe_list_to_module,
    )
    from compgen.ir.recipe.ops import MatchRegion, SetTileParams

    old_ops = [
        MatchRegion(region_id="r0"),
        SetTileParams(region_id="r0", tile_sizes=(64, 64)),
    ]
    module = recipe_list_to_module(old_ops)
    assert module is not None

    recovered = module_to_recipe_list(module)
    assert len(recovered) >= 1  # lossy: MatchRegion maps to RecipeRegionOp → back
