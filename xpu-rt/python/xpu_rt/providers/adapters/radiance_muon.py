"""Chipyard Radiance / Muon adapter shell (M-91b).

Hardware-gated — uses remote scaffold to ship RISC-V Muon
SIMT kernels to a Chipyard FireSim simulator via
``configs/remote_targets/firesim_radiance.yaml``.
"""

from __future__ import annotations

from compgen.providers.adapters.blocked_shell import BlockedShellAdapter
from compgen.providers.adapters.remote_shell import RemoteShellAdapter


class RadianceMuonProvider(RemoteShellAdapter):
    provider_id = "radiance_muon"
    remote_config_filename = "firesim_radiance.yaml"


class RadianceDialectProvider(BlockedShellAdapter):
    provider_id = "radiance_muon"
