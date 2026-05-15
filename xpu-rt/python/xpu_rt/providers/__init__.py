"""Unified provider substrate.

Generalizes the kernel-only provider layer at
:mod:`compgen.kernels.provider` into a card-driven extension surface
covering kernel providers, MLIR-dialect lowerings, and pass tools.

See ``docs/architecture/EXTENSION_PROVIDER_ARCHITECTURE.md`` for the
hard contract every provider, dialect lowering, and pass tool must obey.
"""

from __future__ import annotations

from compgen.providers.provider_types import (
    BLOCKED_REASONS,
    INTEGRATION_LEVELS,
    PAPER_CLAIMABLE_LEVELS,
    PROBE_STATUSES,
    BlockedReason,
    IntegrationLevel,
    ProviderCard,
    ProviderProbeResult,
    ProviderProbeStatus,
)

__all__ = [
    "BLOCKED_REASONS",
    "BlockedReason",
    "INTEGRATION_LEVELS",
    "IntegrationLevel",
    "PAPER_CLAIMABLE_LEVELS",
    "PROBE_STATUSES",
    "ProviderCard",
    "ProviderProbeResult",
    "ProviderProbeStatus",
]
