"""Tests for Recipe IR Family C: Candidate Action operations.

Covers all 12 ops including verify_() custom validators for TileOp,
FuseOp, VectorizeOp, ReassociateOp, RequestTritonKernelOp, PlaceOnDeviceOp.
"""

from __future__ import annotations

import pytest
from compgen.ir.recipe.attrs import DeviceRefAttr, ProvenanceAttr
from compgen.ir.recipe.ops_candidate import (
    BlackboxOp,
    FuseOp,
    InsertCopyBoundaryOp,
    LayoutNormalizeOp,
    LowerToAccelOp,
    MaterializeUkernelOp,
    PlaceOnDeviceOp,
    ReassociateOp,
    RequestTritonKernelOp,
    SegmentBoundaryOp,
    TileOp,
    VectorizeOp,
)
from xdsl.dialects.builtin import (
    ArrayAttr,
    IntegerAttr,
    IntegerType,
    StringAttr,
    SymbolRefAttr,
)
from xdsl.utils.exceptions import VerifyException


def _i64(val: int) -> IntegerAttr:
    return IntegerAttr(val, IntegerType(64))


# -- TileOp -------------------------------------------------------------------


def test_tile_build_minimal() -> None:
    op = TileOp.build(
        properties={
            "region_ref": SymbolRefAttr("seg0"),
            "tile_sizes": ArrayAttr([_i64(64), _i64(32)]),
        }
    )
    assert len(op.tile_sizes.data) == 2
    assert op.sym_name is None
    assert op.interchange is None
    assert op.guard_refs is None
    assert op.provenance is None


def test_tile_build_with_provenance() -> None:
    prov = ProvenanceAttr("agent", 1)
    op = TileOp.build(
        properties={
            "region_ref": SymbolRefAttr("seg0"),
            "tile_sizes": ArrayAttr([_i64(128)]),
            "provenance": prov,
        }
    )
    assert op.provenance.source.data == "agent"


def test_tile_build_with_symbol_and_guard_refs() -> None:
    op = TileOp.build(
        properties={
            "sym_name": StringAttr("cand_tile_r0"),
            "region_ref": SymbolRefAttr("seg0"),
            "tile_sizes": ArrayAttr([_i64(128)]),
            "guard_refs": ArrayAttr([SymbolRefAttr("guard_fusion")]),
        }
    )
    assert op.sym_name.data == "cand_tile_r0"
    assert len(op.guard_refs.data) == 1


def test_tile_verify_positive_sizes() -> None:
    op = TileOp.build(
        properties={
            "region_ref": SymbolRefAttr("seg0"),
            "tile_sizes": ArrayAttr([_i64(64), _i64(32)]),
        }
    )
    op.verify()  # should not raise


def test_tile_verify_zero_size_fails() -> None:
    op = TileOp.build(
        properties={
            "region_ref": SymbolRefAttr("seg0"),
            "tile_sizes": ArrayAttr([_i64(0)]),
        }
    )
    with pytest.raises(VerifyException, match="positive"):
        op.verify()


def test_tile_verify_negative_size_fails() -> None:
    op = TileOp.build(
        properties={
            "region_ref": SymbolRefAttr("seg0"),
            "tile_sizes": ArrayAttr([_i64(-1)]),
        }
    )
    with pytest.raises(VerifyException, match="positive"):
        op.verify()


def test_tile_name() -> None:
    assert TileOp.name == "recipe.tile"


# -- FuseOp -------------------------------------------------------------------


def test_fuse_build() -> None:
    op = FuseOp.build(
        properties={
            "fuse_regions": ArrayAttr([SymbolRefAttr("r0"), SymbolRefAttr("r1")]),
        }
    )
    assert len(op.fuse_regions.data) == 2
    assert op.sym_name is None
    assert op.guard_refs is None


def test_fuse_build_with_fusion_kind() -> None:
    op = FuseOp.build(
        properties={
            "fuse_regions": ArrayAttr([SymbolRefAttr("r0"), SymbolRefAttr("r1")]),
            "fusion_kind": StringAttr("producer_consumer"),
        }
    )
    assert op.fusion_kind.data == "producer_consumer"


def test_fuse_name() -> None:
    assert FuseOp.name == "recipe.fuse"


def test_fuse_verify_single_region_fails() -> None:
    """FuseOp.verify_() rejects fewer than 2 regions."""
    op = FuseOp.build(
        properties={
            "fuse_regions": ArrayAttr([SymbolRefAttr("r0")]),
        }
    )
    with pytest.raises(VerifyException, match="at least 2 regions"):
        op.verify()


# -- VectorizeOp --------------------------------------------------------------


def test_vectorize_build() -> None:
    op = VectorizeOp.build(
        properties={
            "region_ref": SymbolRefAttr("r0"),
            "vector_width": _i64(4),
        }
    )
    assert op.vector_width.value.data == 4


def test_vectorize_verify_ok() -> None:
    op = VectorizeOp.build(
        properties={
            "region_ref": SymbolRefAttr("r0"),
            "vector_width": _i64(8),
        }
    )
    op.verify()


def test_vectorize_verify_zero_fails() -> None:
    op = VectorizeOp.build(
        properties={
            "region_ref": SymbolRefAttr("r0"),
            "vector_width": _i64(0),
        }
    )
    with pytest.raises(VerifyException, match="positive"):
        op.verify()


def test_vectorize_verify_negative_fails() -> None:
    op = VectorizeOp.build(
        properties={
            "region_ref": SymbolRefAttr("r0"),
            "vector_width": _i64(-2),
        }
    )
    with pytest.raises(VerifyException, match="positive"):
        op.verify()


# -- ReassociateOp ------------------------------------------------------------


def test_reassociate_valid_strategies() -> None:
    for strategy in ("left", "right", "balanced"):
        op = ReassociateOp.build(
            properties={
                "region_ref": SymbolRefAttr("r0"),
                "strategy": StringAttr(strategy),
            }
        )
        op.verify()


def test_reassociate_invalid_strategy_fails() -> None:
    op = ReassociateOp.build(
        properties={
            "region_ref": SymbolRefAttr("r0"),
            "strategy": StringAttr("invalid"),
        }
    )
    with pytest.raises(VerifyException, match="Invalid reassociation strategy"):
        op.verify()


# -- LayoutNormalizeOp ---------------------------------------------------------


def test_layout_normalize_build() -> None:
    op = LayoutNormalizeOp.build(
        properties={
            "region_ref": SymbolRefAttr("r0"),
            "target_layout": StringAttr("NCHW"),
        }
    )
    assert op.target_layout.data == "NCHW"


# -- LowerToAccelOp -----------------------------------------------------------


def test_lower_to_accel_minimal() -> None:
    op = LowerToAccelOp.build(
        properties={
            "region_ref": SymbolRefAttr("r0"),
        }
    )
    assert op.accel_cluster is None


def test_lower_to_accel_with_cluster() -> None:
    op = LowerToAccelOp.build(
        properties={
            "region_ref": SymbolRefAttr("r0"),
            "accel_cluster": StringAttr("cluster_0"),
        }
    )
    assert op.accel_cluster.data == "cluster_0"


# -- RequestTritonKernelOp ----------------------------------------------------


def test_request_triton_kernel_build() -> None:
    op = RequestTritonKernelOp.build(
        properties={
            "region_ref": SymbolRefAttr("r0"),
            "search_budget": _i64(100),
        }
    )
    assert op.search_budget.value.data == 100


def test_request_triton_kernel_verify_ok() -> None:
    op = RequestTritonKernelOp.build(
        properties={
            "region_ref": SymbolRefAttr("r0"),
            "search_budget": _i64(50),
        }
    )
    op.verify()


def test_request_triton_kernel_verify_zero_budget_fails() -> None:
    op = RequestTritonKernelOp.build(
        properties={
            "region_ref": SymbolRefAttr("r0"),
            "search_budget": _i64(0),
        }
    )
    with pytest.raises(VerifyException, match="positive"):
        op.verify()


# -- MaterializeUkernelOp -----------------------------------------------------


def test_materialize_ukernel_build() -> None:
    op = MaterializeUkernelOp.build(
        properties={
            "region_ref": SymbolRefAttr("r0"),
            "kernel_name": StringAttr("matmul_f32"),
        }
    )
    assert op.kernel_name.data == "matmul_f32"


# -- PlaceOnDeviceOp ----------------------------------------------------------


def test_place_on_device_build() -> None:
    device = DeviceRefAttr(0, "gpu0")
    op = PlaceOnDeviceOp.build(
        properties={
            "region_ref": SymbolRefAttr("r0"),
            "device": device,
        }
    )
    assert op.device.index.value.data == 0


def test_place_on_device_verify_ok() -> None:
    device = DeviceRefAttr(0, "gpu0")
    op = PlaceOnDeviceOp.build(
        properties={
            "region_ref": SymbolRefAttr("r0"),
            "device": device,
        }
    )
    op.verify()


def test_place_on_device_verify_negative_index_fails() -> None:
    device = DeviceRefAttr(-1, "bad")
    op = PlaceOnDeviceOp.build(
        properties={
            "region_ref": SymbolRefAttr("r0"),
            "device": device,
        }
    )
    with pytest.raises(VerifyException, match="non-negative"):
        op.verify()


# -- InsertCopyBoundaryOp -----------------------------------------------------


def test_insert_copy_boundary_build() -> None:
    op = InsertCopyBoundaryOp.build(
        properties={
            "src_region": SymbolRefAttr("r0"),
            "dst_region": SymbolRefAttr("r1"),
            "tensor_name": StringAttr("weight"),
        }
    )
    assert op.tensor_name.data == "weight"
    assert op.is_async is None


# -- SegmentBoundaryOp --------------------------------------------------------


def test_segment_boundary_build() -> None:
    op = SegmentBoundaryOp.build(
        properties={
            "after_region": SymbolRefAttr("r0"),
        }
    )
    assert op.reason is None


# -- BlackboxOp ----------------------------------------------------------------


def test_blackbox_build() -> None:
    op = BlackboxOp.build(
        properties={
            "region_ref": SymbolRefAttr("r0"),
            "blackbox_class": StringAttr("opaque"),
        }
    )
    assert op.blackbox_class.data == "opaque"


def test_blackbox_name() -> None:
    assert BlackboxOp.name == "recipe.blackbox"
