"""Tests for accelerator dialect registration."""

from __future__ import annotations

import pytest
from compgen.ir.accel.dialect import AccelDialect


def test_accel_dialect_defaults() -> None:
    dialect = AccelDialect()
    assert dialect.name == "compgen.accel"
    assert dialect.vendor == ""


def test_accel_dialect_custom_vendor() -> None:
    dialect = AccelDialect(vendor="nki")
    assert dialect.vendor == "nki"
    assert dialect.name == "compgen.accel"


@pytest.mark.skip(reason="scaffold only -- implementation pending")
def test_accel_dialect_register() -> None:
    """AccelDialect.register should register the dialect with xDSL."""


@pytest.mark.skip(reason="scaffold only -- implementation pending")
def test_accel_dialect_ops_registered() -> None:
    """After register(), the dialect's ops should be available in the context."""
