"""Gemmini C / Chipyard adapter shell (M-91b).

Hardware-gated — uses remote scaffold to ship RISC-V Gemmini
kernels to a Chipyard / Spike / FireSim simulator via
``configs/remote_targets/firesim_gemmini.yaml``.
"""

from __future__ import annotations

from compgen.providers.adapters.blocked_shell import BlockedShellAdapter
from compgen.providers.adapters.remote_shell import RemoteShellAdapter


class GemminiCProvider(RemoteShellAdapter):
    provider_id = "gemmini_c"
    remote_config_filename = "firesim_gemmini.yaml"


class GemminiDialectProvider(BlockedShellAdapter):
    provider_id = "gemmini_c"
