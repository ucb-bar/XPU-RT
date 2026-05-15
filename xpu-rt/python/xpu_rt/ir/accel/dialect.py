"""Accelerator dialect registration.

Registers the ``compgen.accel`` dialect with xDSL. The dialect provides
ops for custom accelerator primitives.

Two exports:
    - ``AccelDialect`` -- xDSL ``Dialect`` object for registration.
    - ``AccelDialectConfig`` -- legacy dataclass kept for backward compat.
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.ir import Dialect

from compgen.ir.accel.ops import ACCEL_IR_OPS

AccelDialect = Dialect("compgen.accel", ACCEL_IR_OPS, [])
"""The Accel IR dialect -- register with ``ctx.register_dialect("compgen.accel", lambda: AccelDialect)``."""


@dataclass
class AccelDialectConfig:
    """Legacy accelerator dialect configuration.

    Kept for backward compatibility. Prefer using :data:`AccelDialect` directly.

    Attributes:
        name: Dialect name (default: "compgen.accel").
        vendor: Optional vendor prefix for vendor-specific extensions.
    """

    name: str = "compgen.accel"
    vendor: str = ""

    def register(self) -> Dialect:
        """Return the xDSL Dialect object for ``compgen.accel``.

        Returns:
            The :data:`AccelDialect` xDSL ``Dialect`` instance.
        """
        return AccelDialect


__all__ = ["AccelDialect", "AccelDialectConfig"]
