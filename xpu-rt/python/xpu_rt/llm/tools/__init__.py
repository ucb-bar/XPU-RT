"""LLM tool implementations — wrappers around existing compgen analyzers
and ported IREE/XLA passes.

Two categories:
- ``observability`` + ``verification``: read-only and gate-shaped tools.
- Ported passes: full-surface Phase 2/3 transformations from
  :mod:`compgen.ir.payload.passes` (auto-registers on import).

Importing this package triggers registration of every tool into the
global registry. Real ports have ``stub=False``; scaffolded stubs have
``stub=True``. See ``user_perspective/reports/stage_b_third_wave_status.md``
for status.
"""

from __future__ import annotations

# Trigger registration of Phase 2/3 ported passes. The module's import
# side-effect populates the registry with 14 passes (3 real + 11 stubs
# as of the third wave).
import compgen.ir.payload.passes as _ported_passes  # noqa: F401
from compgen.llm.tools import megakernel, observability, verification

__all__ = ["megakernel", "observability", "verification"]
