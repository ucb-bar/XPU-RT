"""spec'd path for extension probe reports.

Re-exports :mod:`compgen.providers.provider_reports` so user spec
imports of ``compgen.extensions.reports`` resolve.
"""

from __future__ import annotations

from compgen.providers.provider_reports import (
    PROBE_REPORT_SCHEMA_VERSION,
    write_probe_reports,
)

__all__ = [
    "PROBE_REPORT_SCHEMA_VERSION",
    "write_probe_reports",
]
