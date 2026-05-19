"""Mirage / MPK adapter shell."""

from __future__ import annotations

from xpu_rt.providers.adapters.blocked_shell import BlockedShellAdapter


class MirageProvider(BlockedShellAdapter):
    provider_id = "mirage"
