"""Backwards-compat shim — content moved to
:mod:`xpu_rt.spike_harness.templates.saturn`.

Prefer importing from the new location for new code.
"""

from __future__ import annotations

from xpu_rt.spike_harness.templates.saturn import (
    render_driver_c as _new_render_driver_c,
    render_init_c,
    stage_contract_dir,
)


def render_driver_c(M: int, K: int, N: int) -> str:
    """Compatibility wrapper — original signature was positional."""
    return _new_render_driver_c(M=M, K=K, N=N)


__all__ = ["render_driver_c", "render_init_c", "stage_contract_dir"]
