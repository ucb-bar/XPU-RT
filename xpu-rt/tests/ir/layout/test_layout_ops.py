"""Tests for Layout IR operations."""

from __future__ import annotations

import pytest
from xdsl.dialects.builtin import IntegerAttr, IntegerType, StringAttr, SymbolRefAttr
from xdsl.utils.exceptions import VerifyException

from compgen.ir.layout.attrs import LayoutEncodingAttr, PackSpecAttr
from compgen.ir.layout.ops import PackOp, SetLayoutOp, UnpackOp, UnsetLayoutOp


class TestSetLayoutOp:
    def test_build(self) -> None:
        enc = LayoutEncodingAttr("matmul", 0, "tiled", [128, 64], ["f32"])
        op = SetLayoutOp.build(properties={
            "encoding": enc,
            "source_ref": SymbolRefAttr("test_tensor"),
        })
        assert op.encoding.op_type.data == "matmul"
        assert op.source_ref.root_reference.data == "test_tensor"

    def test_build_with_provenance(self) -> None:
        from compgen.ir.recipe.attrs import ProvenanceAttr
        enc = LayoutEncodingAttr("conv", 1, "blocked", [], ["f16"])
        prov = ProvenanceAttr("agent", 0)
        op = SetLayoutOp.build(properties={
            "encoding": enc,
            "source_ref": SymbolRefAttr("conv_input"),
            "provenance": prov,
        })
        assert op.provenance is not None
        assert op.provenance.source.data == "agent"

    def test_encoding_accessible(self) -> None:
        enc = LayoutEncodingAttr("generic", 0, "rowmajor", [], ["f32"])
        op = SetLayoutOp.build(properties={
            "encoding": enc,
            "source_ref": SymbolRefAttr("generic_ref"),
        })
        assert op.encoding.logical_layout.data == "rowmajor"

    def test_op_name(self) -> None:
        assert SetLayoutOp.name == "layout.set_layout"


class TestUnsetLayoutOp:
    def test_build(self) -> None:
        op = UnsetLayoutOp.build(properties={
            "source_ref": SymbolRefAttr("result_tensor"),
        })
        assert op.source_ref.root_reference.data == "result_tensor"

    def test_build_with_reason(self) -> None:
        op = UnsetLayoutOp.build(properties={
            "source_ref": SymbolRefAttr("boundary"),
            "boundary_reason": StringAttr("custom_call_boundary"),
        })
        assert op.boundary_reason.data == "custom_call_boundary"

    def test_optional_provenance_absent(self) -> None:
        op = UnsetLayoutOp.build(properties={
            "source_ref": SymbolRefAttr("ref"),
        })
        assert op.provenance is None

    def test_op_name(self) -> None:
        assert UnsetLayoutOp.name == "layout.unset_layout"


class TestPackOp:
    def test_build(self) -> None:
        spec = PackSpecAttr([16, 16], [0, 1], "zero")
        op = PackOp.build(properties={
            "source_ref": SymbolRefAttr("weight"),
            "pack_spec": spec,
            "is_prepack": IntegerAttr(1, IntegerType(1)),
        })
        assert op.pack_spec.padding_value.data == "zero"

    def test_verify_prepack_valid_values(self) -> None:
        spec = PackSpecAttr([32, 32], [0, 1], "none")
        # is_prepack=0 (not a prepack)
        op = PackOp.build(properties={
            "source_ref": SymbolRefAttr("activation"),
            "pack_spec": spec,
            "is_prepack": IntegerAttr(0, IntegerType(1)),
        })
        op.verify_()  # Should not raise

    def test_verify_prepack_one(self) -> None:
        spec = PackSpecAttr([32, 32], [0, 1], "none")
        op = PackOp.build(properties={
            "source_ref": SymbolRefAttr("weight"),
            "pack_spec": spec,
            "is_prepack": IntegerAttr(1, IntegerType(1)),
        })
        op.verify_()  # Should not raise

    def test_op_name(self) -> None:
        assert PackOp.name == "layout.pack"


class TestUnpackOp:
    def test_build(self) -> None:
        spec = PackSpecAttr([16, 16], [0, 1], "zero")
        op = UnpackOp.build(properties={
            "source_ref": SymbolRefAttr("packed_weight"),
            "pack_spec": spec,
        })
        assert op.source_ref.root_reference.data == "packed_weight"

    def test_optional_provenance_absent(self) -> None:
        spec = PackSpecAttr([16, 16], [0, 1], "zero")
        op = UnpackOp.build(properties={
            "source_ref": SymbolRefAttr("ref"),
            "pack_spec": spec,
        })
        assert op.provenance is None

    def test_op_name(self) -> None:
        assert UnpackOp.name == "layout.unpack"
