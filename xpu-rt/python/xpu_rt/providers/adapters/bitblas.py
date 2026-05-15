"""BitBLAS adapter shell."""

from __future__ import annotations

from xpu_rt.providers.adapters.blocked_shell import BlockedShellAdapter


class BitBlasProvider(BlockedShellAdapter):
    provider_id = "bitblas"
