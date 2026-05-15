"""Tests for Recipe IR print/parse round-trip.

Verifies that representative ops from each family survive a
print -> parse -> print cycle with identical textual output.
"""

from __future__ import annotations

import io

from compgen.ir.recipe.attrs import CostAttr, DeviceRefAttr, ProvenanceAttr
from compgen.ir.recipe.dialect import Recipe
from compgen.ir.recipe.ops_candidate import (
    BlackboxOp,
    FuseOp,
    PlaceOnDeviceOp,
    TileOp,
    VectorizeOp,
)
from compgen.ir.recipe.ops_choice import (
    AlternativesOp,
    RankOp,
    SearchBudgetOp,
)
from compgen.ir.recipe.ops_fact import (
    BackendAvailableOp,
    GraphBreakOp,
    TransferCostOp,
)
from compgen.ir.recipe.ops_provenance import (
    FeedbackOp,
    FromAgentOp,
    LineageOp,
    PromoteOp,
)
from compgen.ir.recipe.ops_scope import (
    AnchorOp,
    BindPayloadOp,
    RecipeGuardOp,
    RecipeRegionOp,
    SegmentOp,
)
from compgen.ir.recipe.ops_verify import (
    RequireCheckFileOp,
    RequireDiffTestOp,
    RequireMemoryBoundOp,
)
from xdsl.context import Context
from xdsl.dialects.builtin import (
    ArrayAttr,
    IntegerAttr,
    IntegerType,
    ModuleOp,
    StringAttr,
    SymbolRefAttr,
)
from xdsl.ir import Block, Region
from xdsl.parser import Parser
from xdsl.printer import Printer


def _i64(val: int) -> IntegerAttr:
    return IntegerAttr(val, IntegerType(64))


def _round_trip(ops: list) -> ModuleOp:
    """Print ops inside a module, parse back, print again, compare."""
    block = Block()
    for op in ops:
        block.add_op(op)
    module = ModuleOp(Region(block))

    buf = io.StringIO()
    Printer(stream=buf).print_op(module)
    text1 = buf.getvalue()

    ctx = Context()
    ctx.register_dialect("recipe", lambda: Recipe)
    from xdsl.dialects import builtin as builtin_dialect

    ctx.register_dialect("builtin", lambda: builtin_dialect.Builtin)
    parsed = Parser(ctx, text1).parse_module()

    buf2 = io.StringIO()
    Printer(stream=buf2).print_op(parsed)
    text2 = buf2.getvalue()

    assert text2 == text1, f"Round-trip mismatch:\n--- first ---\n{text1}\n--- second ---\n{text2}"
    return parsed


# -- Family A: Scope ops round-trip -------------------------------------------


def test_round_trip_recipe_region() -> None:
    op = RecipeRegionOp.build(
        properties={
            "sym_name": StringAttr("matmul0"),
            "payload_region_id": StringAttr("payload_r0"),
        }
    )
    _round_trip([op])


def test_round_trip_segment() -> None:
    op = SegmentOp.build(
        properties={
            "sym_name": StringAttr("seg0"),
            "region_refs": ArrayAttr([SymbolRefAttr("r0"), SymbolRefAttr("r1")]),
        }
    )
    _round_trip([op])


def test_round_trip_anchor() -> None:
    op = AnchorOp.build(
        properties={
            "sym_name": StringAttr("anchor0"),
            "payload_op_name": StringAttr("linalg.matmul"),
        }
    )
    _round_trip([op])


def test_round_trip_bind_payload() -> None:
    op = BindPayloadOp.build(
        properties={
            "region_ref": SymbolRefAttr("r0"),
            "payload_module_id": StringAttr("mod0"),
        }
    )
    _round_trip([op])


def test_round_trip_recipe_guard() -> None:
    op = RecipeGuardOp.build(
        properties={
            "sym_name": StringAttr("guard_fusion"),
            "guard_key": StringAttr("guard.fusion.legality.TRITON_FRIENDLY.abcd1234"),
            "transform_family": StringAttr("fusion"),
            "guard_kind": StringAttr("legality"),
        }
    )
    _round_trip([op])


# -- Family B: Fact ops round-trip --------------------------------------------


def test_round_trip_backend_available() -> None:
    op = BackendAvailableOp.build(
        properties={
            "region_ref": SymbolRefAttr("r0"),
            "backend": StringAttr("triton"),
        }
    )
    _round_trip([op])


def test_round_trip_transfer_cost() -> None:
    cost = CostAttr(500, "measured")
    op = TransferCostOp.build(
        properties={
            "src_region": SymbolRefAttr("r0"),
            "dst_region": SymbolRefAttr("r1"),
            "cost": cost,
        }
    )
    _round_trip([op])


def test_round_trip_graph_break() -> None:
    op = GraphBreakOp.build(
        properties={
            "location": StringAttr("line 42"),
            "reason": StringAttr("data-dependent control flow"),
        }
    )
    _round_trip([op])


# -- Family C: Candidate ops round-trip ---------------------------------------


def test_round_trip_tile() -> None:
    op = TileOp.build(
        properties={
            "region_ref": SymbolRefAttr("seg0"),
            "tile_sizes": ArrayAttr([_i64(64), _i64(32)]),
        }
    )
    _round_trip([op])


def test_round_trip_tile_with_provenance() -> None:
    prov = ProvenanceAttr("agent", 1)
    op = TileOp.build(
        properties={
            "region_ref": SymbolRefAttr("seg0"),
            "tile_sizes": ArrayAttr([_i64(128)]),
            "provenance": prov,
        }
    )
    _round_trip([op])


def test_round_trip_tile_with_symbol_and_guard_refs() -> None:
    op = TileOp.build(
        properties={
            "sym_name": StringAttr("cand_tile_r0"),
            "region_ref": SymbolRefAttr("seg0"),
            "tile_sizes": ArrayAttr([_i64(128)]),
            "guard_refs": ArrayAttr([SymbolRefAttr("guard_fusion")]),
        }
    )
    _round_trip([op])


def test_round_trip_fuse() -> None:
    op = FuseOp.build(
        properties={
            "fuse_regions": ArrayAttr([SymbolRefAttr("r0"), SymbolRefAttr("r1")]),
        }
    )
    _round_trip([op])


def test_round_trip_vectorize() -> None:
    op = VectorizeOp.build(
        properties={
            "region_ref": SymbolRefAttr("r0"),
            "vector_width": _i64(8),
        }
    )
    _round_trip([op])


def test_round_trip_place_on_device() -> None:
    device = DeviceRefAttr(0, "gpu0")
    op = PlaceOnDeviceOp.build(
        properties={
            "region_ref": SymbolRefAttr("r0"),
            "device": device,
        }
    )
    _round_trip([op])


def test_round_trip_blackbox() -> None:
    op = BlackboxOp.build(
        properties={
            "region_ref": SymbolRefAttr("r0"),
            "blackbox_class": StringAttr("opaque"),
        }
    )
    _round_trip([op])


# -- Family D: Choice ops round-trip ------------------------------------------


def test_round_trip_alternatives_empty() -> None:
    alt = AlternativesOp.build(
        properties={"region_ref": SymbolRefAttr("seg0")},
        regions=[Region(Block())],
    )
    _round_trip([alt])


def test_round_trip_rank() -> None:
    op = RankOp.build(
        properties={
            "candidate_ref": SymbolRefAttr("c0"),
            "priority": _i64(1),
        }
    )
    _round_trip([op])


def test_round_trip_search_budget() -> None:
    op = SearchBudgetOp.build(
        properties={
            "max_iterations": _i64(100),
        }
    )
    _round_trip([op])


# -- Family E: Verify ops round-trip ------------------------------------------


def test_round_trip_require_diff_test() -> None:
    op = RequireDiffTestOp.build(
        properties={
            "region_ref": SymbolRefAttr("r0"),
        }
    )
    _round_trip([op])


def test_round_trip_require_memory_bound() -> None:
    op = RequireMemoryBoundOp.build(
        properties={
            "region_ref": SymbolRefAttr("r0"),
            "max_bytes": _i64(1_073_741_824),
        }
    )
    _round_trip([op])


def test_round_trip_require_check_file() -> None:
    op = RequireCheckFileOp.build(
        properties={
            "check_file_path": StringAttr("checks/matmul.check"),
        }
    )
    _round_trip([op])


# -- Family F: Provenance ops round-trip --------------------------------------


def test_round_trip_from_agent() -> None:
    op = FromAgentOp.build(
        properties={
            "agent_id": StringAttr("agent-1"),
            "iteration": _i64(3),
        }
    )
    _round_trip([op])


def test_round_trip_feedback() -> None:
    cost = CostAttr(120, "measured")
    op = FeedbackOp.build(
        properties={
            "candidate_ref": SymbolRefAttr("c0"),
            "outcome": StringAttr("passed"),
            "measured_cost": cost,
        }
    )
    _round_trip([op])


def test_round_trip_promote() -> None:
    op = PromoteOp.build(
        properties={
            "candidate_ref": SymbolRefAttr("c0"),
            "recipe_key": StringAttr("matmul_f32_gpu0"),
            "version": _i64(1),
        }
    )
    _round_trip([op])


def test_round_trip_lineage() -> None:
    op = LineageOp.build(
        properties={
            "candidate_ref": SymbolRefAttr("c2"),
            "parent_refs": ArrayAttr([SymbolRefAttr("c0"), SymbolRefAttr("c1")]),
            "generation": _i64(3),
        }
    )
    _round_trip([op])


# -- Complex multi-family module round-trip ------------------------------------


def test_round_trip_complex_module() -> None:
    """A module with ops from all 6 families survives round-trip."""
    ops = [
        # Family A
        RecipeRegionOp.build(
            properties={
                "sym_name": StringAttr("matmul0"),
                "payload_region_id": StringAttr("payload_r0"),
            }
        ),
        SegmentOp.build(
            properties={
                "sym_name": StringAttr("seg0"),
                "region_refs": ArrayAttr([SymbolRefAttr("matmul0")]),
            }
        ),
        # Family B
        BackendAvailableOp.build(
            properties={
                "region_ref": SymbolRefAttr("matmul0"),
                "backend": StringAttr("triton"),
            }
        ),
        GraphBreakOp.build(
            properties={
                "location": StringAttr("line 10"),
                "reason": StringAttr("unsupported op"),
            }
        ),
        # Family C
        TileOp.build(
            properties={
                "region_ref": SymbolRefAttr("matmul0"),
                "tile_sizes": ArrayAttr([_i64(64), _i64(32)]),
            }
        ),
        # Family D
        SearchBudgetOp.build(
            properties={
                "max_iterations": _i64(200),
            }
        ),
        # Family E
        RequireDiffTestOp.build(
            properties={
                "region_ref": SymbolRefAttr("matmul0"),
            }
        ),
        # Family F
        FromAgentOp.build(
            properties={
                "agent_id": StringAttr("gemini"),
                "iteration": _i64(0),
            }
        ),
    ]
    parsed = _round_trip(ops)
    # Verify the parsed module has the expected number of ops
    block_ops = list(parsed.body.block.ops)
    assert len(block_ops) == 8
