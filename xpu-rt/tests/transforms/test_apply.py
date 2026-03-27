"""Tests for transforms/apply.py -- transform application."""

from __future__ import annotations

import pytest
from compgen.transforms.apply import TransformDiagnostic, TransformedIR


def test_transform_diagnostic_construction() -> None:
    d = TransformDiagnostic(
        transform_name="tile",
        level="warning",
        message="tile size not a power of 2",
    )
    assert d.transform_name == "tile"
    assert d.level == "warning"
    assert d.message == "tile size not a power of 2"
    assert d.op_name == ""


def test_transformed_ir_defaults() -> None:
    t = TransformedIR(module=None)
    assert t.module is None
    assert t.scripts_applied == []
    assert t.diagnostics == []


@pytest.mark.skip(reason="scaffold only -- implementation pending")
def test_transform_applicator_apply() -> None:
    """TransformApplicator.apply should return a TransformedIR."""


@pytest.mark.skip(reason="scaffold only -- implementation pending")
def test_apply_transforms_convenience() -> None:
    """apply_transforms should work with default settings."""
