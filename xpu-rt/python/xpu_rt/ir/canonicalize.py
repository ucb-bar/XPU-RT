"""DEPRECATED: Import from compgen.ir.payload.canonicalize instead.

This shim will be removed at the end of Phase 1.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "compgen.ir.canonicalize is deprecated. Use compgen.ir.payload.canonicalize instead.",
    DeprecationWarning,
    stacklevel=2,
)

from compgen.ir.payload.canonicalize import *  # noqa: F401, F403, E402
