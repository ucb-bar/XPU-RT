"""Hexagon-MLIR adapter shell.

Hardware-gated — uses remote scaffold to ship kernels to a
Qualcomm Hexagon dev board via
``configs/remote_targets/hexagon_dev_1.yaml``.
"""

from __future__ import annotations

from compgen.providers.adapters.blocked_shell import BlockedShellAdapter
from compgen.providers.adapters.remote_shell import RemoteShellAdapter


class HexagonMLIRProvider(RemoteShellAdapter):
    provider_id = "hexagon_mlir"
    remote_config_filename = "hexagon_dev_1.yaml"


class HexagonDialectProvider(BlockedShellAdapter):
    provider_id = "hexagon_mlir"
