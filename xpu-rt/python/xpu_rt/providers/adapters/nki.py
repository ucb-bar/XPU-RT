"""AWS NKI (Neuron) adapter shell (M-91b).

Hardware-gated — uses remote scaffold to ship kernels to a
remote Trainium / Inferentia instance via
``configs/remote_targets/aws_trn1_inst.yaml``.
"""

from __future__ import annotations

from compgen.providers.adapters.blocked_shell import BlockedShellAdapter
from compgen.providers.adapters.remote_shell import RemoteShellAdapter


class NkiProvider(RemoteShellAdapter):
    provider_id = "nki"
    remote_config_filename = "aws_trn1_inst.yaml"


class NkiDialectProvider(BlockedShellAdapter):
    provider_id = "nki"
