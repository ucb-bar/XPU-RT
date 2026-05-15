"""TileLang adapter shell."""

from __future__ import annotations

from xpu_rt.providers.adapters.blocked_shell import BlockedShellAdapter


class TileLangProvider(BlockedShellAdapter):
    provider_id = "tilelang"
