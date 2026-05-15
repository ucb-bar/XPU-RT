"""KernelBlaster GPU-sidecar manager (task #146).

KernelBlaster's RL search spawns binaries onto a separate "GPU server"
process (`src.kernelblaster.servers.gpu`, a FastAPI/uvicorn app
bound to localhost:2002 by default). The search loop POSTs each
candidate binary to ``/gpu/binary`` and waits for the run result.

Before this module, CompGen's KB integration relied on
``scripts/run_single_kernelblaster.sh`` to start the sidecar inline —
which fails if KB's deps aren't installed in the active venv, and is
opaque to the CompGen audit gate. This module makes the sidecar a
first-class CompGen artifact: typed errors when prerequisites are
missing, a real ``/health`` probe before declaring success, and a
``sidecar_health.json`` receipt that the evidence pack can verify.

Usage::

    from compgen.kernels.kernelblaster_sidecar import (
        KernelBlasterSidecar,
        SidecarUnavailable,
    )

    try:
        with KernelBlasterSidecar.start() as sidecar:
            # KB now reachable at sidecar.url
            ...
    except SidecarUnavailable as e:
        # typed reason — `missing_dep:fastapi`, `repo_not_found`,
        # `health_timeout`, `bind_failed:<port>`, etc.
        ...

The manager never raises bare ``Exception`` — every failure mode is a
``SidecarUnavailable`` with a typed ``reason`` enum + free-form
``detail``. The audit gate uses ``reason`` to decide whether the block
is honest (missing toolchain) or a regression (sidecar crashed after
``/health`` succeeded).
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import structlog

log = structlog.get_logger()


_REQUIRED_IMPORTS: tuple[str, ...] = (
    "fastapi",
    "uvicorn",
    "pydantic",
    "loguru",
    "requests",
    "dotenv",  # python-dotenv installs as `dotenv`
    "psutil",
)
"""Modules KB's GPU sidecar imports at startup. Missing any of these
makes the subprocess crash with returncode != 0, which we'd rather
catch as a typed ``missing_dep:<name>`` than a generic SIGCHLD."""

DEFAULT_PORT = 2002
DEFAULT_HEALTH_TIMEOUT_S = 20.0


class SidecarUnavailable(RuntimeError):
    """Typed failure when the GPU sidecar cannot be brought up.

    ``reason`` is one of:

    - ``repo_not_found`` — no KernelBlaster checkout on disk.
    - ``missing_dep:<name>`` — a required Python import failed.
    - ``port_in_use:<port>`` — the requested port is already bound by
      another process (and ``allow_reuse`` is False).
    - ``health_timeout`` — subprocess started but ``/health`` never
      responded inside the deadline.
    - ``process_died:<rc>`` — subprocess exited before ``/health``
      responded.
    - ``bind_failed:<port>`` — subprocess could not bind to the port.
    - ``unknown`` — anything else; ``detail`` carries the diagnostic.
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class SidecarHealthReceipt:
    """Snapshot the audit gate can verify.

    Written next to the per-provider evidence (see
    :meth:`KernelBlasterSidecar.write_receipt`).
    """

    schema_version: str
    url: str
    port: int
    pid: int
    health_probe_ms: float
    health_response: dict[str, str]
    repo_root: str
    started_utc: str
    nvidia_smi_seen: bool

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "url": self.url,
            "port": self.port,
            "pid": self.pid,
            "health_probe_ms": round(self.health_probe_ms, 3),
            "health_response": dict(self.health_response),
            "repo_root": self.repo_root,
            "started_utc": self.started_utc,
            "nvidia_smi_seen": self.nvidia_smi_seen,
        }


def _check_imports(names: Iterable[str]) -> tuple[bool, str]:
    import importlib

    for name in names:
        try:
            importlib.import_module(name)
        except Exception:  # noqa: BLE001
            return False, name
    return True, ""


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.25)
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return True
    return False


def _resolve_repo_root(explicit: Path | None) -> Path | None:
    """Find the KB checkout: explicit arg > env var > conventional path."""
    if explicit is not None:
        return explicit if explicit.exists() else None
    env_root = os.environ.get("COMPGEN_KERNELBLASTER_ROOT", "").strip()
    if env_root:
        cand = Path(env_root).expanduser()
        return cand if cand.exists() else None
    conventional = Path.cwd() / "third_party" / "kernelblaster"
    return conventional if conventional.exists() else None


@dataclass
class KernelBlasterSidecar:
    """Manage the KB GPU sidecar lifecycle.

    Construct via :meth:`start`. Instances are context managers; exit
    sends SIGTERM and waits up to ``shutdown_timeout_s`` for a clean
    exit before SIGKILL.
    """

    repo_root: Path
    port: int
    process: subprocess.Popen[bytes]
    url: str
    health_response: dict[str, str]
    health_probe_ms: float
    shutdown_timeout_s: float = 5.0
    _started_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    _terminated: bool = False

    # ---- Lifecycle --------------------------------------------------

    @classmethod
    def start(
        cls,
        *,
        repo_root: Path | None = None,
        port: int = DEFAULT_PORT,
        health_timeout_s: float = DEFAULT_HEALTH_TIMEOUT_S,
        allow_reuse: bool = False,
        env_overlay: dict[str, str] | None = None,
        log_path: Path | None = None,
    ) -> "KernelBlasterSidecar":
        """Start the sidecar and block until ``/health`` returns 200.

        Raises :class:`SidecarUnavailable` with a typed ``reason`` on
        any failure mode. The caller is responsible for context-
        managing the returned instance (or calling :meth:`terminate`).
        """
        # 0. Repo
        root = _resolve_repo_root(repo_root)
        if root is None:
            raise SidecarUnavailable(
                "repo_not_found",
                f"no KernelBlaster checkout (set COMPGEN_KERNELBLASTER_ROOT={repo_root!s}/...)",
            )

        # 1. Python deps
        ok, missing = _check_imports(_REQUIRED_IMPORTS)
        if not ok:
            raise SidecarUnavailable(
                f"missing_dep:{missing}",
                f"install with `uv sync --extra kernelblaster-sidecar` (missing: {missing})",
            )

        # 2. Port
        if _port_in_use(port) and not allow_reuse:
            raise SidecarUnavailable(
                f"port_in_use:{port}",
                "another process is bound (pass allow_reuse=True if reusing intentionally)",
            )

        # 3. Subprocess
        env = os.environ.copy()
        env.setdefault("KERNELBLASTER_GPU_SERVER_SKIP_PROCESS_CHECK", "1")
        if env_overlay:
            env.update(env_overlay)

        # Make KB's `src.kernelblaster.servers.gpu` resolvable.
        kb_pythonpath = str(root)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{kb_pythonpath}{os.pathsep}{existing}" if existing else kb_pythonpath
        )

        argv = [
            sys.executable,
            "-m",
            "src.kernelblaster.servers.gpu",
            "--port",
            str(port),
        ]
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            argv += ["--log_path", str(log_path)]

        try:
            proc = subprocess.Popen(  # noqa: S603
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=str(root),
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            raise SidecarUnavailable("unknown", f"cannot spawn KB sidecar: {exc}") from exc

        url = f"http://127.0.0.1:{port}"

        # 4. Health probe
        t0 = time.monotonic()
        response: dict[str, str] = {}
        deadline = t0 + health_timeout_s
        last_err: str = ""
        try:
            import requests as _req
        except ImportError as exc:  # pragma: no cover — checked above
            proc.terminate()
            raise SidecarUnavailable("missing_dep:requests", str(exc)) from exc

        while time.monotonic() < deadline:
            rc = proc.poll()
            if rc is not None:
                # Process exited; capture tail for diagnostic.
                try:
                    out = (proc.stdout.read() if proc.stdout else b"") or b""
                    tail = out[-2048:].decode("utf-8", errors="replace")
                except Exception:  # noqa: BLE001
                    tail = ""
                raise SidecarUnavailable(
                    f"process_died:{rc}", tail.strip()[-512:] or last_err
                )
            try:
                r = _req.get(f"{url}/health", timeout=1.0)
                if r.status_code == 200:
                    response = dict(r.json() or {})
                    break
                last_err = f"status={r.status_code}"
            except _req.exceptions.ConnectionError:
                pass
            except Exception as exc:  # noqa: BLE001
                last_err = str(exc)
            time.sleep(0.25)
        else:
            proc.terminate()
            raise SidecarUnavailable("health_timeout", last_err or f"no /health on {url}")

        elapsed_ms = (time.monotonic() - t0) * 1000.0

        log.info(
            "kernelblaster.sidecar.up",
            url=url,
            pid=proc.pid,
            health_probe_ms=round(elapsed_ms, 2),
        )

        return cls(
            repo_root=root,
            port=port,
            process=proc,
            url=url,
            health_response=response,
            health_probe_ms=elapsed_ms,
        )

    def terminate(self) -> None:
        """Best-effort shutdown. Idempotent."""
        if self._terminated:
            return
        self._terminated = True
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=self.shutdown_timeout_s)
            except subprocess.TimeoutExpired:
                log.warning("kernelblaster.sidecar.kill", pid=self.process.pid)
                self.process.kill()
                try:
                    self.process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:  # pragma: no cover
                    pass

    def __enter__(self) -> "KernelBlasterSidecar":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.terminate()

    # ---- Evidence ---------------------------------------------------

    def receipt(self) -> SidecarHealthReceipt:
        return SidecarHealthReceipt(
            schema_version="kernelblaster_sidecar_v1",
            url=self.url,
            port=self.port,
            pid=self.process.pid,
            health_probe_ms=self.health_probe_ms,
            health_response=dict(self.health_response),
            repo_root=str(self.repo_root),
            started_utc=self._started_utc,
            nvidia_smi_seen=self._detect_nvidia_smi(),
        )

    def write_receipt(self, out_dir: Path) -> Path:
        """Write ``sidecar_health.json`` for the audit gate."""
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "sidecar_health.json"
        path.write_text(json.dumps(self.receipt().to_json(), indent=2) + "\n")
        return path

    @staticmethod
    def _detect_nvidia_smi() -> bool:
        import shutil

        return shutil.which("nvidia-smi") is not None


__all__ = [
    "DEFAULT_HEALTH_TIMEOUT_S",
    "DEFAULT_PORT",
    "KernelBlasterSidecar",
    "SidecarHealthReceipt",
    "SidecarUnavailable",
]
