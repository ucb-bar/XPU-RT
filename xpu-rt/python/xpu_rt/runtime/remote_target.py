"""Remote-target execution scaffold.

Provides a typed :class:`RemoteTargetRunner` interface so
hardware-gated providers (TPU/Pallas, AWS Neuron/NKI, Hexagon-MLIR,
Gemmini-FireSim, Radiance-FireSim) can ship a kernel to the user's
remote hardware, run it, and return real execution evidence.

This module is **transport-agnostic**. Concrete backends live in
:mod:`compgen.runtime.remote_runners.*`:

* ``ssh_runner.py``   — generic SSH-over-public-key transport.
* (future) ``modal_runner.py`` — Modal Labs remote execution.
* (future) ``k8s_runner.py``   — Kubernetes job dispatch.

The contract every backend obeys:

1. ``probe()`` checks the remote endpoint is reachable AND the
   declared toolchain is installed there. Returns a typed
   :class:`RemoteProbeResult` — never raises.
2. ``ship_and_run(payload)`` ships the kernel source + build
   artifacts + contract inputs, runs the remote command, captures
   stdout/stderr/runtime stats. Returns a typed
   :class:`RemoteRunResult` — never raises.
3. Failures (SSH dead, remote OOM, build failure, runtime crash,
   timeout) become typed :class:`RemoteRunResult` records with the
   specific ``failure_mode``.

The audit (``check_execution_evidence`` + the upcoming
``remote_execution_evidence`` check) verifies the remote receipt
is real before flipping a HW-gated provider's certificate to
``passed``.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

REMOTE_SCHEMA_VERSION: Final[str] = "remote_execution_v1"

REMOTE_PROBE_STATUSES: Final[tuple[str, ...]] = (
    "available",
    "unreachable",
    "toolchain_missing",
    "auth_failed",
    "blocked",
    "probe_error",
)

REMOTE_RUN_STATUSES: Final[tuple[str, ...]] = (
    "succeeded",
    "build_failed",
    "runtime_failed",
    "timed_out",
    "transport_error",
    "remote_oom",
    "remote_killed",
    "skipped",
)


class RemoteExecutionError(RuntimeError):
    """Typed parent of remote-execution errors (used only for
    programming-bug cases; expected failures land in the typed
    result records)."""


@dataclass(frozen=True)
class RemoteTargetConfig:
    """Loaded from ``configs/remote_targets/*.yaml``."""

    target_id: str
    transport: str  # "ssh" | "modal" | "k8s" | ...
    host: str = ""
    user: str = ""
    workdir: str = "/tmp/compgen_remote"
    toolchain_probe_cmd: str = ""
    build_cmd_template: str = ""
    run_cmd_template: str = ""
    timeout_s: int = 1800
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RemoteProbeResult:
    """Typed remote-probe outcome — mirrors typed status."""

    schema_version: str
    target_id: str
    status: str  # one of REMOTE_PROBE_STATUSES
    detail: str = ""
    toolchain_version: str = ""
    probed_utc: str = ""

    def __post_init__(self) -> None:
        if self.status not in REMOTE_PROBE_STATUSES:
            raise ValueError(
                f"remote probe status={self.status!r} must be in {REMOTE_PROBE_STATUSES}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "target_id": self.target_id,
            "status": self.status,
            "detail": self.detail,
            "toolchain_version": self.toolchain_version,
            "probed_utc": self.probed_utc,
        }


@dataclass(frozen=True)
class RemoteRunPayload:
    """What the runner ships to the remote.

    The runner reads ``kernel_source_path`` from disk, ships its
    contents (and any companion files) to the remote workdir, runs
    the build + run commands, and captures the receipt.
    """

    task_id: str
    provider_id: str
    contract_hash: str
    kernel_source_path: Path
    extra_files: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RemoteRunResult:
    """Typed remote-run record. Lands as ``remote_receipt.json``
    in the per-provider evidence directory."""

    schema_version: str
    target_id: str
    task_id: str
    provider_id: str
    status: str  # one of REMOTE_RUN_STATUSES
    started_utc: str
    finished_utc: str
    elapsed_s: float
    transport: str
    host: str
    stdout_tail: str = ""
    stderr_tail: str = ""
    runtime_stats: dict[str, Any] = field(default_factory=dict)
    failure_mode: str = ""
    detail: str = ""

    def __post_init__(self) -> None:
        if self.status not in REMOTE_RUN_STATUSES:
            raise ValueError(
                f"remote run status={self.status!r} must be in {REMOTE_RUN_STATUSES}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "target_id": self.target_id,
            "task_id": self.task_id,
            "provider_id": self.provider_id,
            "status": self.status,
            "started_utc": self.started_utc,
            "finished_utc": self.finished_utc,
            "elapsed_s": self.elapsed_s,
            "transport": self.transport,
            "host": self.host,
            "stdout_tail": self.stdout_tail[:2048],
            "stderr_tail": self.stderr_tail[:2048],
            "runtime_stats": dict(self.runtime_stats),
            "failure_mode": self.failure_mode,
            "detail": self.detail,
        }


class RemoteTargetRunner(ABC):
    """Backend ABC. Concrete runners live in
    :mod:`compgen.runtime.remote_runners.*`."""

    transport: str = ""

    def __init__(self, config: RemoteTargetConfig) -> None:
        self.config = config

    @abstractmethod
    def probe(self) -> RemoteProbeResult:
        """Check reachability + toolchain presence. Never raises."""

    @abstractmethod
    def ship_and_run(self, payload: RemoteRunPayload) -> RemoteRunResult:
        """Ship payload, run, capture, return typed result. Never raises."""


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_remote_target_config(path: str | Path) -> RemoteTargetConfig:
    """Load a remote-target descriptor from YAML."""

    import yaml

    body = yaml.safe_load(Path(path).read_text())
    if not isinstance(body, dict):
        raise ValueError(
            f"remote target config at {path} must be a YAML mapping"
        )
    return RemoteTargetConfig(
        target_id=str(body["target_id"]),
        transport=str(body["transport"]),
        host=str(body.get("host", "")),
        user=str(body.get("user", "")),
        workdir=str(body.get("workdir", "/tmp/compgen_remote")),
        toolchain_probe_cmd=str(body.get("toolchain_probe_cmd", "")),
        build_cmd_template=str(body.get("build_cmd_template", "")),
        run_cmd_template=str(body.get("run_cmd_template", "")),
        timeout_s=int(body.get("timeout_s", 1800)),
        extras=dict(body.get("extras", {}) or {}),
    )


def build_runner(config: RemoteTargetConfig) -> RemoteTargetRunner:
    """Factory: returns the right backend for ``config.transport``."""

    if config.transport == "ssh":
        from compgen.runtime.remote_runners.ssh_runner import SshRunner
        return SshRunner(config)
    raise RemoteExecutionError(
        f"unsupported remote transport: {config.transport!r}"
    )


__all__ = [
    "REMOTE_PROBE_STATUSES",
    "REMOTE_RUN_STATUSES",
    "REMOTE_SCHEMA_VERSION",
    "RemoteExecutionError",
    "RemoteProbeResult",
    "RemoteRunPayload",
    "RemoteRunResult",
    "RemoteTargetConfig",
    "RemoteTargetRunner",
    "build_runner",
    "load_remote_target_config",
]
