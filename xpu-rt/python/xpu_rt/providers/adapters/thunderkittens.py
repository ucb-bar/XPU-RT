"""ThunderKittens adapter shell."""

from __future__ import annotations

from compgen.providers.adapters.blocked_shell import BlockedShellAdapter


class ThunderKittensProvider(BlockedShellAdapter):
    provider_id = "thunderkittens"
