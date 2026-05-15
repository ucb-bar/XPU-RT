"""Tests for Tile IR custom attributes.

Covers all 3 ParametrizedAttribute types: MemoryClassAttr,
FragmentLayoutAttr, TileShapeAttr. Tests construction from both
Python types and xDSL types, and parameter access.
"""

from __future__ import annotations

import io

from compgen.ir.tile.attrs import FragmentLayoutAttr, MemoryClassAttr, TileShapeAttr
from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, IntegerType, StringAttr
from xdsl.printer import Printer

# -- MemoryClassAttr -----------------------------------------------------------


def test_memory_class_from_str() -> None:
    """Convenience __init__ accepts plain str."""
    attr = MemoryClassAttr("shared")
    assert attr.kind.data == "shared"


def test_memory_class_from_string_attr() -> None:
    """Accepts pre-built StringAttr."""
    attr = MemoryClassAttr(StringAttr("global"))
    assert attr.kind.data == "global"


def test_memory_class_name() -> None:
    """Dialect-qualified name is tile.memory_class."""
    assert MemoryClassAttr.name == "tile.memory_class"


def test_memory_class_all_valid_kinds() -> None:
    """All valid memory kinds can be constructed."""
    for kind in ("global", "shared", "local", "accumulator", "register"):
        attr = MemoryClassAttr(kind)
        assert attr.kind.data == kind


# -- FragmentLayoutAttr --------------------------------------------------------


def test_fragment_layout_from_str() -> None:
    """Convenience __init__ accepts plain str."""
    attr = FragmentLayoutAttr("row_major")
    assert attr.layout.data == "row_major"


def test_fragment_layout_from_string_attr() -> None:
    """Accepts pre-built StringAttr."""
    attr = FragmentLayoutAttr(StringAttr("col_major"))
    assert attr.layout.data == "col_major"


def test_fragment_layout_name() -> None:
    """Dialect-qualified name is tile.fragment_layout."""
    assert FragmentLayoutAttr.name == "tile.fragment_layout"


# -- TileShapeAttr -------------------------------------------------------------


def test_tile_shape_from_list() -> None:
    """Convenience __init__ accepts plain list of ints."""
    attr = TileShapeAttr([16, 16])
    assert len(attr.dims.data) == 2
    assert attr.dims.data[0].value.data == 16
    assert attr.dims.data[1].value.data == 16


def test_tile_shape_from_array_attr() -> None:
    """Accepts pre-built ArrayAttr."""
    dims = ArrayAttr([IntegerAttr(32, IntegerType(64)), IntegerAttr(64, IntegerType(64))])
    attr = TileShapeAttr(dims)
    assert len(attr.dims.data) == 2
    assert attr.dims.data[0].value.data == 32


def test_tile_shape_3d() -> None:
    """Three-dimensional shape (M, N, K)."""
    attr = TileShapeAttr([128, 64, 32])
    assert len(attr.dims.data) == 3
    assert attr.dims.data[2].value.data == 32


def test_tile_shape_name() -> None:
    """Dialect-qualified name is tile.shape."""
    assert TileShapeAttr.name == "tile.shape"


def test_tile_shape_printing() -> None:
    """Attribute can be printed without error."""
    attr = TileShapeAttr([8, 8])
    buf = io.StringIO()
    Printer(stream=buf).print_attribute(attr)
    text = buf.getvalue()
    assert "tile.shape" in text
