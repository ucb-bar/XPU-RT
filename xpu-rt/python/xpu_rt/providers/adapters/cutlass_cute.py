"""CUTLASS / CuTe adapter shell."""

from __future__ import annotations

from compgen.providers.adapters.blocked_shell import BlockedShellAdapter


class CutlassCuteProvider(BlockedShellAdapter):
    provider_id = "cutlass_cute"
