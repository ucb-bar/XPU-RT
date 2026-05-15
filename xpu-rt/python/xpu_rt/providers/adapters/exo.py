"""Exo dialect shell.

The kernel-provider for Exo lives at
``compgen.kernels.providers.exo_riscv_opu:ExoRiscvOpuProvider`` and
goes through the legacy shim. The dialect-provider shell here
satisfies the dialect-card entrypoint.
"""

from __future__ import annotations

from compgen.providers.adapters.blocked_shell import BlockedShellAdapter


class ExoDialectProvider(BlockedShellAdapter):
    provider_id = "exo"
