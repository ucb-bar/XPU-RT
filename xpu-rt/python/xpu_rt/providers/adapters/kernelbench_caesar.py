"""KernelBench + Caesar adapter shell."""

from __future__ import annotations

from compgen.providers.adapters.blocked_shell import BlockedShellAdapter


class KernelBenchCaesarProvider(BlockedShellAdapter):
    provider_id = "kernelbench_caesar"
