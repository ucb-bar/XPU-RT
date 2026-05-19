"""spec'd path for extension probe reports.

Re-exports :mod:`xpu_rt.providers.provider_reports` so user spec
imports of ``xpu_rt.extensions.reports`` resolve.
"""

from __future__ import annotations

from xpu_rt.providers.provider_reports import (
    PROBE_REPORT_SCHEMA_VERSION,
    write_probe_reports,
)

__all__ = [
    "PROBE_REPORT_SCHEMA_VERSION",
    "write_probe_reports",
]
