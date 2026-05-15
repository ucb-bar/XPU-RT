"""Google TPU / Pallas adapter shell.

Hardware-gated provider — relies on the remote-target
scaffold to ship kernels to a remote TPU pod. Until the user
populates ``configs/remote_targets/tpu_v5e_pod_1.yaml`` with a
real SSH host, this provider probes ``blocked`` with reason
``hardware_unavailable``.
"""

from __future__ import annotations

from compgen.providers.adapters.blocked_shell import BlockedShellAdapter
from compgen.providers.adapters.remote_shell import RemoteShellAdapter


class PallasProvider(RemoteShellAdapter):
    provider_id = "pallas"
    remote_config_filename = "tpu_v5e_pod_1.yaml"


class PallasDialectProvider(BlockedShellAdapter):
    provider_id = "pallas"
