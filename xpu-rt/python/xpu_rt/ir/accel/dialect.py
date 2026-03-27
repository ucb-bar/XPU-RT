"""Accelerator dialect registration.

Registers the ``compgen.accel`` dialect with xDSL. The dialect provides
ops for custom accelerator primitives.

Invariants:
    - The dialect is registered lazily (only when needed).
    - All ops have explicit memory/effect semantics.
    - The dialect is extensible per-vendor.

TODO: Implement xDSL Dialect subclass for compgen.accel.
TODO: Register with xDSL's dialect registry.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AccelDialect:
    """Custom accelerator dialect registration.

    Attributes:
        name: Dialect name (default: "compgen.accel").
        vendor: Optional vendor prefix for vendor-specific extensions.

    TODO: Implement as xDSL Dialect subclass.
    TODO: Register ops from ops.py.
    """

    name: str = "compgen.accel"
    vendor: str = ""

    def register(self) -> None:
        """Register this dialect with the xDSL context.

        TODO: Create xDSL Dialect, register all ops.
        """
        raise NotImplementedError("AccelDialect.register is not yet implemented")


__all__ = ["AccelDialect"]
