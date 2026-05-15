"""Tests for Layout IR attributes."""

from __future__ import annotations

from compgen.ir.layout.attrs import LayoutEncodingAttr, PackSpecAttr


class TestLayoutEncodingAttr:
    def test_construction_from_primitives(self) -> None:
        enc = LayoutEncodingAttr("matmul", 0, "tiled", [128, 64], ["f32"])
        assert enc.op_type.data == "matmul"
        assert enc.operand_index.value.data == 0
        assert enc.logical_layout.data == "tiled"
        assert len(enc.tile_dims.data) == 2
        assert enc.tile_dims.data[0].value.data == 128
        assert enc.tile_dims.data[1].value.data == 64
        assert len(enc.element_types.data) == 1
        assert enc.element_types.data[0].data == "f32"

    def test_construction_empty_tiles(self) -> None:
        enc = LayoutEncodingAttr("elementwise", 0, "rowmajor", [], ["f32"])
        assert enc.logical_layout.data == "rowmajor"
        assert len(enc.tile_dims.data) == 0

    def test_different_layouts(self) -> None:
        for layout in ("rowmajor", "colmajor", "tiled", "blocked"):
            enc = LayoutEncodingAttr("matmul", 0, layout, [], ["f32"])
            assert enc.logical_layout.data == layout

    def test_multiple_element_types(self) -> None:
        enc = LayoutEncodingAttr("matmul", 0, "tiled", [16, 16], ["f32", "f16"])
        assert len(enc.element_types.data) == 2
        assert enc.element_types.data[0].data == "f32"
        assert enc.element_types.data[1].data == "f16"

    def test_operand_index_preserved(self) -> None:
        for idx in (0, 1, 2):
            enc = LayoutEncodingAttr("matmul", idx, "tiled", [], ["f32"])
            assert enc.operand_index.value.data == idx

    def test_dialect_qualified_name(self) -> None:
        assert LayoutEncodingAttr.name == "layout.encoding"


class TestPackSpecAttr:
    def test_construction_from_primitives(self) -> None:
        spec = PackSpecAttr([16, 16], [0, 1], "zero")
        assert spec.padding_value.data == "zero"
        tiles = [a.value.data for a in spec.inner_tiles.data]
        assert tiles == [16, 16]

    def test_different_tile_sizes(self) -> None:
        spec = PackSpecAttr([128, 64], [1, 0], "none")
        tiles = [a.value.data for a in spec.inner_tiles.data]
        assert tiles == [128, 64]
        perm = [a.value.data for a in spec.outer_perm.data]
        assert perm == [1, 0]

    def test_no_padding(self) -> None:
        spec = PackSpecAttr([32, 32], [0, 1], "none")
        assert spec.padding_value.data == "none"

    def test_identity_permutation(self) -> None:
        spec = PackSpecAttr([64, 64], [0, 1], "zero")
        perm = [a.value.data for a in spec.outer_perm.data]
        assert perm == [0, 1]

    def test_dialect_qualified_name(self) -> None:
        assert PackSpecAttr.name == "layout.pack_spec"
