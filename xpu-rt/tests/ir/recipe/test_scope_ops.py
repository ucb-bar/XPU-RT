"""Tests for Recipe IR Family A: Scope/Anchor operations.

Covers RecipeRegionOp, SegmentOp, AnchorOp, BindPayloadOp.
"""

from __future__ import annotations

import io

from compgen.ir.recipe.attrs import EffectClassAttr, ShapeSummaryAttr
from compgen.ir.recipe.ops_scope import (
    AnchorOp,
    BindPayloadOp,
    RecipeGuardOp,
    RecipeRegionOp,
    SegmentOp,
)
from xdsl.dialects.builtin import (
    ArrayAttr,
    IntegerAttr,
    IntegerType,
    StringAttr,
    SymbolRefAttr,
)
from xdsl.printer import Printer


def _print_op(op) -> str:
    """Print a single op to string."""
    buf = io.StringIO()
    Printer(stream=buf).print_op(op)
    return buf.getvalue()


# -- RecipeRegionOp ------------------------------------------------------------


def test_recipe_region_minimal() -> None:
    """Build with only required properties."""
    op = RecipeRegionOp.build(properties={
        "sym_name": StringAttr("matmul0"),
        "payload_region_id": StringAttr("payload_r0"),
    })
    assert op.sym_name.data == "matmul0"
    assert op.payload_region_id.data == "payload_r0"
    assert op.shape_summary is None
    assert op.effect_class is None
    assert op.op_count is None


def test_recipe_region_with_optionals() -> None:
    """Build with all optional properties populated."""
    shape = ShapeSummaryAttr([128, 64], "f32")
    effect = EffectClassAttr("pure")
    op = RecipeRegionOp.build(properties={
        "sym_name": StringAttr("conv1"),
        "payload_region_id": StringAttr("payload_r1"),
        "shape_summary": shape,
        "effect_class": effect,
        "op_count": IntegerAttr(12, IntegerType(64)),
    })
    assert op.shape_summary is not None
    assert op.shape_summary.dtype.data == "f32"
    assert op.effect_class is not None
    assert op.effect_class.kind.data == "pure"
    assert op.op_count.value.data == 12


def test_recipe_region_name() -> None:
    assert RecipeRegionOp.name == "recipe.region"


def test_recipe_region_verify_ok() -> None:
    """Verify succeeds for a well-formed op."""
    op = RecipeRegionOp.build(properties={
        "sym_name": StringAttr("r0"),
        "payload_region_id": StringAttr("p0"),
    })
    op.verify()


def test_recipe_region_printable() -> None:
    op = RecipeRegionOp.build(properties={
        "sym_name": StringAttr("r0"),
        "payload_region_id": StringAttr("p0"),
    })
    text = _print_op(op)
    assert "recipe.region" in text


# -- SegmentOp -----------------------------------------------------------------


def test_segment_op_build() -> None:
    """Build a SegmentOp grouping two regions."""
    op = SegmentOp.build(properties={
        "sym_name": StringAttr("seg0"),
        "region_refs": ArrayAttr([SymbolRefAttr("r0"), SymbolRefAttr("r1")]),
    })
    assert op.sym_name.data == "seg0"
    assert len(op.region_refs.data) == 2


def test_segment_op_name() -> None:
    assert SegmentOp.name == "recipe.segment"


def test_segment_op_verify_ok() -> None:
    op = SegmentOp.build(properties={
        "sym_name": StringAttr("seg0"),
        "region_refs": ArrayAttr([SymbolRefAttr("r0")]),
    })
    op.verify()


# -- AnchorOp ------------------------------------------------------------------


def test_anchor_op_build() -> None:
    op = AnchorOp.build(properties={
        "sym_name": StringAttr("anchor0"),
        "payload_op_name": StringAttr("linalg.matmul"),
    })
    assert op.sym_name.data == "anchor0"
    assert op.payload_op_name.data == "linalg.matmul"


def test_anchor_op_name() -> None:
    assert AnchorOp.name == "recipe.anchor"


def test_anchor_op_verify_ok() -> None:
    op = AnchorOp.build(properties={
        "sym_name": StringAttr("a0"),
        "payload_op_name": StringAttr("arith.addf"),
    })
    op.verify()


# -- RecipeGuardOp ------------------------------------------------------------


def test_recipe_guard_build() -> None:
    op = RecipeGuardOp.build(properties={
        "sym_name": StringAttr("guard_fusion"),
        "guard_key": StringAttr("guard.fusion.legality.TRITON_FRIENDLY.1"),
        "transform_family": StringAttr("fusion"),
        "guard_kind": StringAttr("legality"),
    })
    assert op.sym_name.data == "guard_fusion"
    assert op.transform_family.data == "fusion"


def test_recipe_guard_name() -> None:
    assert RecipeGuardOp.name == "recipe.guard"


# -- BindPayloadOp ------------------------------------------------------------


def test_bind_payload_build() -> None:
    op = BindPayloadOp.build(properties={
        "region_ref": SymbolRefAttr("r0"),
        "payload_module_id": StringAttr("module_main"),
    })
    assert op.payload_module_id.data == "module_main"


def test_bind_payload_name() -> None:
    assert BindPayloadOp.name == "recipe.bind_payload"


def test_bind_payload_printable() -> None:
    op = BindPayloadOp.build(properties={
        "region_ref": SymbolRefAttr("r0"),
        "payload_module_id": StringAttr("mod0"),
    })
    text = _print_op(op)
    assert "recipe.bind_payload" in text
