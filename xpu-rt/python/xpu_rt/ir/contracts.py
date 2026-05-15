"""DEPRECATED: Import from compgen.ir.payload.contracts instead.

This shim will be removed at the end of Phase 1.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "compgen.ir.contracts is deprecated. Use compgen.ir.payload.contracts instead.",
    DeprecationWarning,
    stacklevel=2,
)

from compgen.ir.payload.contracts import *  # noqa: F401, F403, E402
