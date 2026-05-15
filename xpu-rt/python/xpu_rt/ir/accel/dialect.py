"""Accelerator dialect registration.

Registers the ``xpu_rt.accel`` dialect with xDSL. The dialect provides
ops for custom accelerator primitives.

Two exports:
    - ``AccelDialect`` -- xDSL ``Dialect`` object for registration.
    - ``AccelDialectConfig`` -- legacy dataclass kept for backward compat.
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.ir import Dialect

from xpu_rt.ir.accel.ops import ACCEL_IR_OPS

AccelDialect = Dialect("xpu_rt.accel", ACCEL_IR_OPS, [])
"""The Accel IR dialect -- register with ``ctx.register_dialect("xpu_rt.accel", lambda: AccelDialect)``."""


@dataclass
class AccelDialectConfig:
    """Legacy accelerator dialect configuration.

    Kept for backward compatibility. Prefer using :data:`AccelDialect` directly.

    Attributes:
        name: Dialect name (default: "xpu_rt.accel").
        vendor: Optional vendor prefix for vendor-specific extensions.
    """

    name: str = "xpu_rt.accel"
    vendor: str = ""

    def register(self) -> Dialect:
        """Return the xDSL Dialect object for ``xpu_rt.accel``.

        Returns:
            The :data:`AccelDialect` xDSL ``Dialect`` instance.
        """
        return AccelDialect


__all__ = ["AccelDialect", "AccelDialectConfig"]
