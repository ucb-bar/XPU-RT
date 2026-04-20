"""LayoutResolver protocol and default implementation.

Provides the ``LayoutResolver`` protocol for target-specific layout
specialization. Each target (or extension pack) provides a resolver
that converts generic layout encodings into target-specific pack
specifications.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from compgen.ir.layout.attrs import PackSpecAttr


@runtime_checkable
class LayoutResolver(Protocol):
    """Protocol for target-specific layout specialization.

    Each target or extension pack provides a resolver that knows how
    to map generic layout encodings (e.g., ``tiled_128x64``) to
    concrete ``PackSpecAttr`` values for that hardware.
    """

    def specialize(
        self,
        encoding_str: str,
        target_caps: Any,
    ) -> PackSpecAttr | None:
        """Convert a generic encoding to a target-specific pack spec.

        Args:
            encoding_str: The generic layout encoding string.
            target_caps: Target CapabilitySpec for hardware-aware decisions.

        Returns:
            A PackSpecAttr if specialization applies, None otherwise.
        """
        ...

    def materialize(self, specialized: PackSpecAttr) -> dict[str, Any]:
        """Return metadata for materializing a specialized pack.

        Args:
            specialized: The target-specific pack specification.

        Returns:
            Dict with materialization metadata (inner_tiles, perm, etc.).
        """
        ...


@dataclass(frozen=True)
class DefaultLayoutResolver:
    """Default resolver that keeps generic encodings unchanged.

    Returns None for specialization (no target-specific packing)
    and plain metadata for materialization.
    """

    def specialize(
        self,
        encoding_str: str,
        target_caps: Any,
    ) -> PackSpecAttr | None:
        """No specialization — return None."""
        return None

    def materialize(self, specialized: PackSpecAttr) -> dict[str, Any]:
        """Return basic metadata from the pack spec."""
        return {
            "inner_tiles": [a.value.data if hasattr(a, "value") else 0 for a in specialized.inner_tiles.data],
            "outer_perm": [a.value.data if hasattr(a, "value") else 0 for a in specialized.outer_perm.data],
            "padding": specialized.padding_value.data,
        }


__all__ = ["DefaultLayoutResolver", "LayoutResolver"]
