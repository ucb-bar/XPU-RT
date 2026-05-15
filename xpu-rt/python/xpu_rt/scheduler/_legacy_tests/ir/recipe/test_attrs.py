"""Tests for Recipe IR custom attributes.

Covers all 5 ParametrizedAttribute types: ShapeSummaryAttr, EffectClassAttr,
CostAttr, ProvenanceAttr, DeviceRefAttr.
"""

from __future__ import annotations

import io

from compgen.ir.recipe.attrs import (
    CostAttr,
    DeviceRefAttr,
    EffectClassAttr,
    ProvenanceAttr,
    ShapeSummaryAttr,
)
from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, IntegerType, StringAttr
from xdsl.printer import Printer

# -- ShapeSummaryAttr ----------------------------------------------------------


def test_shape_summary_from_python_types() -> None:
    """Convenience __init__ accepts plain ints and str."""
    attr = ShapeSummaryAttr([128, 64], "f32")
    assert attr.dtype.data == "f32"
    assert len(attr.dims.data) == 2
    assert attr.dims.data[0].value.data == 128
    assert attr.dims.data[1].value.data == 64


def test_shape_summary_from_xdsl_types() -> None:
    """Accepts pre-built xDSL ArrayAttr and StringAttr."""
    dims = ArrayAttr([IntegerAttr(8, IntegerType(64))])
    dtype = StringAttr("bf16")
    attr = ShapeSummaryAttr(dims, dtype)
    assert attr.dtype.data == "bf16"
    assert len(attr.dims.data) == 1


def test_shape_summary_empty_dims() -> None:
    """Scalar shapes have zero-length dims."""
    attr = ShapeSummaryAttr([], "f64")
    assert len(attr.dims.data) == 0
    assert attr.dtype.data == "f64"


def test_shape_summary_name() -> None:
    """Dialect-qualified name is recipe.shape_summary."""
    assert ShapeSummaryAttr.name == "recipe.shape_summary"


def test_shape_summary_printing() -> None:
    """Attribute can be printed without error."""
    attr = ShapeSummaryAttr([4, 8, 16], "i32")
    buf = io.StringIO()
    Printer(stream=buf).print_attribute(attr)
    text = buf.getvalue()
    assert "recipe.shape_summary" in text


# -- EffectClassAttr -----------------------------------------------------------


def test_effect_class_from_str() -> None:
    """Convenience __init__ accepts plain str."""
    attr = EffectClassAttr("pure")
    assert attr.kind.data == "pure"


def test_effect_class_from_string_attr() -> None:
    """Accepts pre-built StringAttr."""
    attr = EffectClassAttr(StringAttr("readwrite"))
    assert attr.kind.data == "readwrite"


def test_effect_class_name() -> None:
    assert EffectClassAttr.name == "recipe.effect_class"


# -- CostAttr ------------------------------------------------------------------


def test_cost_from_python_types() -> None:
    """Convenience __init__ accepts int and str."""
    attr = CostAttr(100, "measured")
    assert attr.value_us.value.data == 100
    assert attr.confidence.data == "measured"


def test_cost_from_xdsl_types() -> None:
    """Accepts pre-built IntegerAttr and StringAttr."""
    attr = CostAttr(IntegerAttr(50, IntegerType(64)), StringAttr("estimated"))
    assert attr.value_us.value.data == 50
    assert attr.confidence.data == "estimated"


def test_cost_name() -> None:
    assert CostAttr.name == "recipe.cost"


def test_cost_printing() -> None:
    attr = CostAttr(42, "unknown")
    buf = io.StringIO()
    Printer(stream=buf).print_attribute(attr)
    text = buf.getvalue()
    assert "recipe.cost" in text


# -- ProvenanceAttr ------------------------------------------------------------


def test_provenance_from_python_types() -> None:
    attr = ProvenanceAttr("agent", 5)
    assert attr.source.data == "agent"
    assert attr.iteration.value.data == 5


def test_provenance_from_xdsl_types() -> None:
    attr = ProvenanceAttr(StringAttr("eqsat"), IntegerAttr(3, IntegerType(64)))
    assert attr.source.data == "eqsat"
    assert attr.iteration.value.data == 3


def test_provenance_name() -> None:
    assert ProvenanceAttr.name == "recipe.provenance"


# -- DeviceRefAttr -------------------------------------------------------------


def test_device_ref_from_python_types() -> None:
    attr = DeviceRefAttr(0, "gpu0")
    assert attr.index.value.data == 0
    assert attr.device_name.data == "gpu0"


def test_device_ref_from_xdsl_types() -> None:
    attr = DeviceRefAttr(IntegerAttr(1, IntegerType(64)), StringAttr("tpu0"))
    assert attr.index.value.data == 1
    assert attr.device_name.data == "tpu0"


def test_device_ref_name() -> None:
    assert DeviceRefAttr.name == "recipe.device_ref"


def test_device_ref_printing() -> None:
    attr = DeviceRefAttr(2, "npu0")
    buf = io.StringIO()
    Printer(stream=buf).print_attribute(attr)
    text = buf.getvalue()
    assert "recipe.device_ref" in text
