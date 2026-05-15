"""DEPRECATED: Import from xpu_rt.ir.payload.import_fx instead.

This shim will be removed at the end of Phase 1.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "xpu_rt.ir.import_fx is deprecated. Use xpu_rt.ir.payload.import_fx instead.",
    DeprecationWarning,
    stacklevel=2,
)

from xpu_rt.ir.payload.import_fx import *  # noqa: F401, F403, E402
