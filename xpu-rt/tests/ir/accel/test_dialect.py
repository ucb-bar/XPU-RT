"""Tests for accelerator dialect registration."""

from __future__ import annotations

from xdsl.ir import Dialect

from compgen.ir.accel.dialect import AccelDialect, AccelDialectConfig
from compgen.ir.accel.ops import ACCEL_IR_OPS


def test_accel_dialect_is_xdsl_dialect() -> None:
    """AccelDialect should be an xDSL Dialect instance."""
    assert isinstance(AccelDialect, Dialect)


def test_accel_dialect_name() -> None:
    """AccelDialect should have the correct name."""
    assert AccelDialect.name == "compgen.accel"


def test_accel_dialect_config_defaults() -> None:
    """AccelDialectConfig (legacy) should preserve default values."""
    config = AccelDialectConfig()
    assert config.name == "compgen.accel"
    assert config.vendor == ""


def test_accel_dialect_config_custom_vendor() -> None:
    """AccelDialectConfig (legacy) should accept a vendor."""
    config = AccelDialectConfig(vendor="nki")
    assert config.vendor == "nki"
    assert config.name == "compgen.accel"


def test_accel_dialect_config_register_returns_dialect() -> None:
    """AccelDialectConfig.register() should return the xDSL Dialect object."""
    config = AccelDialectConfig()
    dialect = config.register()
    assert dialect is AccelDialect
    assert isinstance(dialect, Dialect)


def test_accel_dialect_ops_registered() -> None:
    """The dialect should contain all ACCEL_IR_OPS."""
    for op_cls in ACCEL_IR_OPS:
        assert op_cls.name.startswith("compgen.accel.")


def test_accel_dialect_ops_count() -> None:
    """The dialect should have exactly 6 ops."""
    assert len(ACCEL_IR_OPS) == 6
