"""CUDA Tile IR adapter shell.

Both the kernel-provider shell (``CudaTileIRProvider``) and the
dialect-provider shell (``CudaTileDialectProvider``) live here; the
cards reference them by name.
"""

from __future__ import annotations

from compgen.providers.adapters.blocked_shell import BlockedShellAdapter


class CudaTileIRProvider(BlockedShellAdapter):
    provider_id = "cuda_tile_ir"


class CudaTileDialectProvider(BlockedShellAdapter):
    """Dialect-provider shell. Shares the kernel-provider card so the
    blocked-reason ladder is identical."""

    provider_id = "cuda_tile_ir"
