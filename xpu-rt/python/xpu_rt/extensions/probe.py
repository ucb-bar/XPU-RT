"""spec'd path for extension probing.

Re-exports :mod:`xpu_rt.providers.provider_probe` so user spec
imports of ``xpu_rt.extensions.probe`` resolve.
"""

from __future__ import annotations

from xpu_rt.providers.provider_probe import (
    PROBE_SCHEMA_VERSION,
    probe_dialect_provider,
    probe_provider,
)

__all__ = [
    "PROBE_SCHEMA_VERSION",
    "probe_provider",
    "probe_dialect_provider",
]
