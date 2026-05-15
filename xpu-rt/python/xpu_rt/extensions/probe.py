"""spec'd path for extension probing.

Re-exports :mod:`compgen.providers.provider_probe` so user spec
imports of ``compgen.extensions.probe`` resolve.
"""

from __future__ import annotations

from compgen.providers.provider_probe import (
    PROBE_SCHEMA_VERSION,
    probe_dialect_provider,
    probe_provider,
)

__all__ = [
    "PROBE_SCHEMA_VERSION",
    "probe_provider",
    "probe_dialect_provider",
]
