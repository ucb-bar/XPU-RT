"""M-27 tests for the extended :class:`PromoteOp` (Section 19).

Covers the five new optional attrs added to ``recipe.promote`` —
``recipe_signature``, ``applies_when``, ``evidence_summary``,
``fallback_chain``, ``target_class`` — plus the lowering + JSON
round-trip + validate.py integration.
"""

from __future__ import annotations

import io

from xdsl.dialects.builtin import (
    ArrayAttr,
    Block,
    IntegerAttr,
    IntegerType,
    ModuleOp,
    Region,
    StringAttr,
    SymbolRefAttr,
)
from xdsl.printer import Printer

from compgen.ir.recipe.lower import lower_recipe
from compgen.ir.recipe.ops_provenance import PromoteOp
from compgen.ir.recipe.ops_scope import RecipeRegionOp
from compgen.ir.recipe.serialize import recipe_module_to_json
from compgen.ir.recipe.validate import validate_recipe_module


def _i64(val: int) -> IntegerAttr:
    return IntegerAttr(val, IntegerType(64))


def _build_promote_full() -> PromoteOp:
    """A PromoteOp with every M-27 attr populated."""
    return PromoteOp.build(
        properties={
            "candidate_ref": SymbolRefAttr("c0"),
            "recipe_key": StringAttr("matmul_f32_host_v1"),
            "version": _i64(1),
            "recipe_signature": StringAttr("abc123def456"),
            "applies_when": ArrayAttr([
                SymbolRefAttr("fact_tile_divisible"),
                SymbolRefAttr("fact_contiguous_layout"),
            ]),
            "evidence_summary": StringAttr(
                '{"differential":"pass","analytical_cost":"present"}'
            ),
            "fallback_chain": ArrayAttr([
                SymbolRefAttr("c1"),
                SymbolRefAttr("c2"),
            ]),
            "target_class": StringAttr("host_cpu"),
        }
    )


# -- Op construction & introspection -----------------------------------------


def test_promote_with_all_m27_attrs() -> None:
    op = _build_promote_full()
    assert op.recipe_signature.data == "abc123def456"
    assert len(op.applies_when.data) == 2
    assert op.evidence_summary.data.startswith('{"differential"')
    assert len(op.fallback_chain.data) == 2
    assert op.target_class.data == "host_cpu"


def test_promote_legacy_three_field_still_works() -> None:
    """Existing callers without M-27 attrs must keep building cleanly."""
    op = PromoteOp.build(
        properties={
            "candidate_ref": SymbolRefAttr("c0"),
            "recipe_key": StringAttr("legacy"),
            "version": _i64(1),
        }
    )
    assert op.recipe_signature is None
    assert op.applies_when is None
    assert op.evidence_summary is None
    assert op.fallback_chain is None
    assert op.target_class is None


def test_promote_printable_with_m27_attrs() -> None:
    """xDSL printer round-trip must surface every populated attr."""
    op = _build_promote_full()
    buf = io.StringIO()
    Printer(stream=buf).print_op(op)
    text = buf.getvalue()
    assert "recipe.promote" in text
    assert "abc123def456" in text
    assert "host_cpu" in text
    assert "fact_tile_divisible" in text


# -- MLIR text round-trip ----------------------------------------------------


def test_promote_mlir_round_trip() -> None:
    """A PromoteOp with all M-27 attrs survives MLIR text round-trip."""
    from compgen.ir.recipe.serialize import mlir_to_recipe, recipe_to_mlir

    module = ModuleOp(Region(Block([_build_promote_full()])))
    text_a = recipe_to_mlir(module)
    parsed = mlir_to_recipe(text_a)
    text_b = recipe_to_mlir(parsed)
    assert text_a == text_b


# -- JSON projection ---------------------------------------------------------


def test_promote_json_projection_carries_m27_attrs() -> None:
    module = ModuleOp(Region(Block([_build_promote_full()])))
    payload = recipe_module_to_json(module)
    # Sorted-key JSON; just check the populated fields surface.
    assert '"recipe_signature":"abc123def456"' in payload
    assert '"target_class":"host_cpu"' in payload
    assert '"applies_when"' in payload
    assert "fact_tile_divisible" in payload
    assert "fallback_chain" in payload


def test_recipe_module_to_json_is_deterministic() -> None:
    """Two encodings of the same module produce byte-identical JSON."""
    module_a = ModuleOp(Region(Block([_build_promote_full()])))
    module_b = ModuleOp(Region(Block([_build_promote_full()])))
    assert recipe_module_to_json(module_a) == recipe_module_to_json(module_b)


# -- Lowering ----------------------------------------------------------------


def test_lower_promote_emits_promoted_recipe_record() -> None:
    """``recipe.promote`` ops appear in ``LoweringOutput.promoted_recipe_records``."""
    module = ModuleOp(Region(Block([_build_promote_full()])))
    out = lower_recipe(module)

    assert len(out.promoted_recipe_records) == 1
    record = out.promoted_recipe_records[0]
    assert record["candidate_ref"] == "c0"
    assert record["recipe_key"] == "matmul_f32_host_v1"
    assert record["version"] == 1
    assert record["recipe_signature"] == "abc123def456"
    assert record["applies_when"] == [
        "fact_tile_divisible",
        "fact_contiguous_layout",
    ]
    assert record["fallback_chain"] == ["c1", "c2"]
    assert record["target_class"] == "host_cpu"


def test_lower_promote_legacy_op_yields_empty_optional_fields() -> None:
    """Legacy 3-field PromoteOp still lowers — optional fields are empty."""
    legacy = PromoteOp.build(
        properties={
            "candidate_ref": SymbolRefAttr("c0"),
            "recipe_key": StringAttr("legacy"),
            "version": _i64(1),
        }
    )
    module = ModuleOp(Region(Block([legacy])))
    out = lower_recipe(module)

    assert len(out.promoted_recipe_records) == 1
    record = out.promoted_recipe_records[0]
    assert record["recipe_signature"] == ""
    assert record["applies_when"] == []
    assert record["fallback_chain"] == []
    assert record["target_class"] == ""


# -- Validation --------------------------------------------------------------


def test_validate_unknown_target_class_is_rejected() -> None:
    """target_class must be a known string from the allowlist."""
    op = PromoteOp.build(
        properties={
            "candidate_ref": SymbolRefAttr("c0"),
            "recipe_key": StringAttr("k"),
            "version": _i64(1),
            "target_class": StringAttr("definitely_not_a_real_target"),
        }
    )
    module = ModuleOp(Region(Block([op])))
    result = validate_recipe_module(module)
    assert not result.valid
    assert any(
        "target_class" in err.message and err.op_type == "PromoteOp"
        for err in result.errors
    )


def test_validate_known_target_class_accepted() -> None:
    """An allowlisted target_class passes validation."""
    op = PromoteOp.build(
        properties={
            "candidate_ref": SymbolRefAttr("c0"),
            "recipe_key": StringAttr("k"),
            "version": _i64(1),
            "target_class": StringAttr("cuda_sm75"),
        }
    )
    # Need a region symbol so the op-level checks are satisfied.
    region = RecipeRegionOp.build(
        properties={
            "sym_name": StringAttr("c0"),
            "payload_region_id": StringAttr("payload_0"),
        },
    )
    module = ModuleOp(Region(Block([region, op])))
    result = validate_recipe_module(module)
    # No target_class-related errors:
    assert not any(
        "target_class" in err.message for err in result.errors
    )


def test_validate_missing_target_class_accepted() -> None:
    """Legacy PromoteOps without target_class don't trip validation."""
    op = PromoteOp.build(
        properties={
            "candidate_ref": SymbolRefAttr("c0"),
            "recipe_key": StringAttr("k"),
            "version": _i64(1),
        }
    )
    region = RecipeRegionOp.build(
        properties={
            "sym_name": StringAttr("c0"),
            "payload_region_id": StringAttr("payload_0"),
        },
    )
    module = ModuleOp(Region(Block([region, op])))
    result = validate_recipe_module(module)
    assert not any(
        "target_class" in err.message for err in result.errors
    )
