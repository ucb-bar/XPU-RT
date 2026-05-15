"""DEPRECATED: Import from compgen.ir.payload.import_fx instead.

This shim will be removed at the end of Phase 1.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "compgen.ir.import_fx is deprecated. Use compgen.ir.payload.import_fx instead.",
    DeprecationWarning,
    stacklevel=2,
)

from compgen.ir.payload.import_fx import *  # noqa: F401, F403, E402
