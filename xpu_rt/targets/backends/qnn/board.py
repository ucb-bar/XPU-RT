"""Board contract for the QRB5165 / Q_DEV_BOAR target.

Owns the dataclass that everything in the QNN flow (heterogeneous_loop,
run_on_board_flow, the MCP tools, the CLI startup interview) uses to
describe *how to reach the board*. Sources values, in priority order:

1. Explicit CLI / constructor overrides.
2. ``XPURT_*`` environment variables.
3. A dotenv file (default: ``/scratch2/agustin/XPU-RT/.env``) — the
   pre-existing source of truth in this workspace, which spells the
   variable ``Q_DEV_BOAR`` (no trailing D). ``Q_DEV_BOARD`` is also
   accepted as a forward-compatible alias.
4. Legacy fall-back: the historically-hardcoded ``DIMA_SLICE`` private
   key two levels above ``scripts/``.

The dotenv parser here is deliberately tiny so the rest of XPU-RT
doesn't gain a python-dotenv dependency for one file with three lines.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path

DEFAULT_ENV_FILE = Path("/scratch2/agustin/XPU-RT/.env")
DEFAULT_BOARD_BIN = "/root/merlin-dispatch-flow-runner"
DEFAULT_VMFB_DIR = "/root/dispatch_flow"
DEFAULT_QNN_LIB_DIR = "/root/qairt/lib/target"
LEGACY_SSH_KEY = Path("/scratch2/agustin/DIMA_SLICE")


@dataclass(frozen=True)
class BoardConfig:
    """How to reach the QRB5165 board.

    ``ssh_host`` is in standard SSH "user@host" form. ``ssh_identity``
    is the private key path (``None`` lets ssh-agent / system defaults
    decide). The other fields point at on-board paths that the merlin
    runtime expects to find.
    """

    ssh_host: str
    ssh_identity: Path | None
    board_bin: str = DEFAULT_BOARD_BIN
    vmfb_dir: str = DEFAULT_VMFB_DIR
    qnn_lib_dir: str = DEFAULT_QNN_LIB_DIR
    source: str = "default"  # for logging / dashboard

    def with_overrides(
        self,
        *,
        ssh_host: str | None = None,
        ssh_identity: Path | None = None,
    ) -> BoardConfig:
        """Return a copy with the supplied non-``None`` fields applied."""
        return replace(
            self,
            ssh_host=ssh_host or self.ssh_host,
            ssh_identity=(
                ssh_identity if ssh_identity is not None else self.ssh_identity
            ),
            source=self.source + ("+override" if (ssh_host or ssh_identity) else ""),
        )

    def to_dict(self) -> dict:
        return {
            "ssh_host": self.ssh_host,
            "ssh_identity": str(self.ssh_identity) if self.ssh_identity else None,
            "board_bin": self.board_bin,
            "vmfb_dir": self.vmfb_dir,
            "qnn_lib_dir": self.qnn_lib_dir,
            "source": self.source,
        }


def _parse_dotenv(path: Path) -> dict[str, str]:
    """Read a ``KEY=value`` dotenv file. Ignores blank lines and #-comments."""
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key:
            out[key] = val
    return out


def load_board_config(
    *,
    env_file: Path | None = DEFAULT_ENV_FILE,
    override_ip: str | None = None,
    override_key: Path | None = None,
) -> BoardConfig:
    """Resolve a ``BoardConfig`` from CLI overrides / env / dotenv / legacy.

    ``override_ip`` may be in either ``user@host`` or bare-IP form. When
    bare, ``root@`` is prepended (matching the original Q_DEV_BOAR
    spelling that uses the ``root`` user).
    """
    env = dict(os.environ)
    dotenv = _parse_dotenv(env_file) if env_file else {}

    host_candidates: list[tuple[str, str]] = []
    if override_ip:
        host_candidates.append(("override", override_ip))
    for key in ("XPURT_BOARD_HOST", "XPURT_SSH_HOST"):
        if env.get(key):
            host_candidates.append((f"env:{key}", env[key]))
    for key in ("Q_DEV_BOAR", "Q_DEV_BOARD"):
        if dotenv.get(key):
            host_candidates.append((f"dotenv:{key}", dotenv[key]))
        if env.get(key):
            host_candidates.append((f"env:{key}", env[key]))

    if not host_candidates:
        source = "fallback"
        ssh_host = "root@10.44.120.201"
    else:
        source, ssh_host = host_candidates[0]
    if "@" not in ssh_host:
        ssh_host = f"root@{ssh_host}"

    key_candidates: list[tuple[str, Path]] = []
    if override_key:
        key_candidates.append(("override", Path(override_key)))
    for ek in ("XPURT_SSH_KEY", "Q_DEV_BOAR_KEY", "Q_DEV_BOARD_KEY"):
        if env.get(ek):
            key_candidates.append((f"env:{ek}", Path(env[ek])))
    if dotenv.get("XPURT_SSH_KEY"):
        key_candidates.append(("dotenv:XPURT_SSH_KEY", Path(dotenv["XPURT_SSH_KEY"])))
    if LEGACY_SSH_KEY.is_file():
        key_candidates.append(("legacy:DIMA_SLICE", LEGACY_SSH_KEY))

    ssh_identity: Path | None = None
    for src, cand in key_candidates:
        if cand.is_file():
            ssh_identity = cand
            source = f"{source}+key:{src}"
            break

    return BoardConfig(
        ssh_host=ssh_host,
        ssh_identity=ssh_identity,
        source=source,
    )


@dataclass(frozen=True)
class BoardProbeResult:
    reachable: bool
    detail: str
    uname: str | None = None
    runner_present: bool = False


def probe_board(cfg: BoardConfig, *, timeout: float = 5.0) -> BoardProbeResult:
    """SSH-poke the board and report whether it's reachable + has the runner.

    Never raises; turns subprocess errors into a structured ``detail``
    string so the CLI dashboard can render the failure mode without a
    traceback.
    """
    if shutil.which("ssh") is None:
        return BoardProbeResult(reachable=False, detail="ssh not on PATH")

    cmd = ["ssh", "-o", f"ConnectTimeout={int(timeout)}",
           "-o", "BatchMode=yes",
           "-o", "StrictHostKeyChecking=accept-new"]
    if cfg.ssh_identity is not None:
        cmd.extend(["-i", str(cfg.ssh_identity)])
    cmd.extend([cfg.ssh_host,
                f"uname -a; test -x {shlex.quote(cfg.board_bin)} && "
                f"echo RUNNER_PRESENT || echo RUNNER_MISSING"])
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout + 2.0,
        )
    except subprocess.TimeoutExpired:
        return BoardProbeResult(reachable=False, detail="ssh timeout")
    except OSError as exc:
        return BoardProbeResult(reachable=False, detail=f"ssh error: {exc}")
    if proc.returncode != 0:
        return BoardProbeResult(
            reachable=False,
            detail=f"ssh rc={proc.returncode}: {proc.stderr.strip()[:200]}",
        )
    out = proc.stdout.strip()
    runner = "RUNNER_PRESENT" in out
    uname = out.splitlines()[0] if out else None
    return BoardProbeResult(
        reachable=True,
        detail="ok",
        uname=uname,
        runner_present=runner,
    )
