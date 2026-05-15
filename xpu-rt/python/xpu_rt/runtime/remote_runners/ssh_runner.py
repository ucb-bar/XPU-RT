"""SSH remote-target runner.

Generic ``ssh user@host`` + ``scp`` based transport. Suitable for:

* user-provisioned TPU pods accessed via SSH-gateway,
* AWS Trainium/Inferentia instances,
* Chipyard FireSim runs on a remote dev box,
* Hexagon dev boards exposed over SSH,
* anything else that speaks OpenSSH.

The runner does **not** keep persistent connections; each
``probe()`` / ``ship_and_run()`` opens a fresh ssh process. That
keeps the transport stateless and easy to test in isolation —
sophisticated transports (Modal, k8s) can subclass and add
session reuse if needed.
"""

from __future__ import annotations

import shlex
import subprocess
import time
from pathlib import Path

from compgen.runtime.remote_target import (
    REMOTE_SCHEMA_VERSION,
    RemoteProbeResult,
    RemoteRunPayload,
    RemoteRunResult,
    RemoteTargetConfig,
    RemoteTargetRunner,
    _now_iso,
)


class SshRunner(RemoteTargetRunner):
    transport = "ssh"

    def __init__(self, config: RemoteTargetConfig) -> None:
        super().__init__(config)
        if not config.host:
            raise ValueError(
                f"ssh transport requires config.host on target_id={config.target_id!r}"
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ssh_target(self) -> str:
        return f"{self.config.user}@{self.config.host}" if self.config.user else self.config.host

    def _run_ssh(
        self,
        cmd: str,
        *,
        timeout: int | None = None,
    ) -> tuple[int, str, str]:
        full = ["ssh", "-o", "BatchMode=yes", self._ssh_target(), cmd]
        try:
            r = subprocess.run(
                full,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout or self.config.timeout_s,
            )
            return r.returncode, r.stdout, r.stderr
        except subprocess.TimeoutExpired as exc:
            return (
                124,
                exc.stdout.decode("utf-8", "replace") if exc.stdout else "",
                f"ssh command timed out after {exc.timeout}s",
            )
        except FileNotFoundError:
            return 127, "", "ssh: command not found"

    def _run_scp(
        self,
        src: Path,
        dst_remote: str,
        *,
        timeout: int | None = None,
    ) -> tuple[int, str, str]:
        full = [
            "scp",
            "-B",  # batch mode — fail if password prompt
            str(src),
            f"{self._ssh_target()}:{dst_remote}",
        ]
        try:
            r = subprocess.run(
                full,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout or self.config.timeout_s,
            )
            return r.returncode, r.stdout, r.stderr
        except subprocess.TimeoutExpired:
            return 124, "", "scp timed out"
        except FileNotFoundError:
            return 127, "", "scp: command not found"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def probe(self) -> RemoteProbeResult:
        # 1. reach the host with a trivial command
        rc, stdout, stderr = self._run_ssh("echo compgen_remote_probe", timeout=15)
        if rc == 127:
            return RemoteProbeResult(
                schema_version=REMOTE_SCHEMA_VERSION,
                target_id=self.config.target_id,
                status="probe_error",
                detail="ssh binary missing on local host",
                probed_utc=_now_iso(),
            )
        if rc == 124:
            return RemoteProbeResult(
                schema_version=REMOTE_SCHEMA_VERSION,
                target_id=self.config.target_id,
                status="unreachable",
                detail="ssh reachability check timed out",
                probed_utc=_now_iso(),
            )
        if rc != 0:
            if "Permission denied" in stderr or "publickey" in stderr.lower():
                return RemoteProbeResult(
                    schema_version=REMOTE_SCHEMA_VERSION,
                    target_id=self.config.target_id,
                    status="auth_failed",
                    detail=stderr.strip()[:512],
                    probed_utc=_now_iso(),
                )
            return RemoteProbeResult(
                schema_version=REMOTE_SCHEMA_VERSION,
                target_id=self.config.target_id,
                status="unreachable",
                detail=stderr.strip()[:512] or f"ssh rc={rc}",
                probed_utc=_now_iso(),
            )

        # 2. toolchain probe (optional)
        if self.config.toolchain_probe_cmd:
            rc2, stdout2, stderr2 = self._run_ssh(
                self.config.toolchain_probe_cmd, timeout=30
            )
            if rc2 != 0:
                return RemoteProbeResult(
                    schema_version=REMOTE_SCHEMA_VERSION,
                    target_id=self.config.target_id,
                    status="toolchain_missing",
                    detail=(stderr2.strip() or f"toolchain probe rc={rc2}")[:512],
                    probed_utc=_now_iso(),
                )
            tool_version = stdout2.strip().splitlines()[0] if stdout2 else ""
        else:
            tool_version = ""

        return RemoteProbeResult(
            schema_version=REMOTE_SCHEMA_VERSION,
            target_id=self.config.target_id,
            status="available",
            detail="",
            toolchain_version=tool_version,
            probed_utc=_now_iso(),
        )

    def ship_and_run(self, payload: RemoteRunPayload) -> RemoteRunResult:
        started_utc = _now_iso()
        started = time.perf_counter()

        # 1. prep remote workdir
        workdir = f"{self.config.workdir.rstrip('/')}/{payload.task_id}"
        rc, _, stderr = self._run_ssh(f"mkdir -p {shlex.quote(workdir)}", timeout=30)
        if rc != 0:
            return self._fail(
                payload, started_utc, started,
                status="transport_error", failure_mode="workdir_setup",
                stderr=stderr,
            )

        # 2. scp the kernel source
        if not Path(payload.kernel_source_path).is_file():
            return self._fail(
                payload, started_utc, started,
                status="transport_error", failure_mode="missing_local_source",
                stderr=f"source path {payload.kernel_source_path} not a file",
            )
        rc, _, stderr = self._run_scp(
            payload.kernel_source_path,
            f"{workdir}/{Path(payload.kernel_source_path).name}",
        )
        if rc != 0:
            return self._fail(
                payload, started_utc, started,
                status="transport_error", failure_mode="scp_kernel_source",
                stderr=stderr,
            )

        # 2b. scp any companion files
        for rel, content in payload.extra_files.items():
            local = Path(payload.kernel_source_path).parent / rel
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_text(content)
            rc, _, stderr = self._run_scp(local, f"{workdir}/{rel}")
            if rc != 0:
                return self._fail(
                    payload, started_utc, started,
                    status="transport_error", failure_mode=f"scp_extra:{rel}",
                    stderr=stderr,
                )

        # 3. build (optional)
        env_prefix = " ".join(
            f"{k}={shlex.quote(v)}" for k, v in payload.env.items()
        )
        if self.config.build_cmd_template:
            build_cmd = self.config.build_cmd_template.format(
                workdir=workdir,
                source=Path(payload.kernel_source_path).name,
            )
            full_build = f"{env_prefix} bash -lc {shlex.quote(f'cd {workdir} && {build_cmd}')}"
            rc, stdout, stderr = self._run_ssh(full_build)
            if rc != 0:
                return self._fail(
                    payload, started_utc, started,
                    status="build_failed", failure_mode="remote_build",
                    stdout=stdout, stderr=stderr,
                )

        # 4. run
        if not self.config.run_cmd_template:
            return self._fail(
                payload, started_utc, started,
                status="skipped", failure_mode="no_run_cmd_template",
                stderr="config.run_cmd_template is empty",
            )
        run_cmd = self.config.run_cmd_template.format(
            workdir=workdir,
            source=Path(payload.kernel_source_path).name,
        )
        full_run = f"{env_prefix} bash -lc {shlex.quote(f'cd {workdir} && {run_cmd}')}"
        rc, stdout, stderr = self._run_ssh(full_run)
        if rc == 124:
            return self._fail(
                payload, started_utc, started,
                status="timed_out", failure_mode="remote_timeout",
                stdout=stdout, stderr=stderr,
            )
        if rc == 137 or "out of memory" in stderr.lower():
            return self._fail(
                payload, started_utc, started,
                status="remote_oom", failure_mode="oom_signal",
                stdout=stdout, stderr=stderr,
            )
        if rc != 0:
            return self._fail(
                payload, started_utc, started,
                status="runtime_failed", failure_mode="non_zero_exit",
                stdout=stdout, stderr=stderr,
            )

        # 5. parse runtime stats if a tail line is JSON
        runtime_stats: dict = {}
        for line in reversed(stdout.splitlines()):
            stripped = line.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                try:
                    import json
                    runtime_stats = json.loads(stripped)
                    break
                except json.JSONDecodeError:
                    continue

        elapsed = time.perf_counter() - started
        return RemoteRunResult(
            schema_version=REMOTE_SCHEMA_VERSION,
            target_id=self.config.target_id,
            task_id=payload.task_id,
            provider_id=payload.provider_id,
            status="succeeded",
            started_utc=started_utc,
            finished_utc=_now_iso(),
            elapsed_s=elapsed,
            transport=self.transport,
            host=self.config.host,
            stdout_tail=stdout[-2048:],
            stderr_tail=stderr[-2048:],
            runtime_stats=runtime_stats,
        )

    def _fail(
        self,
        payload: RemoteRunPayload,
        started_utc: str,
        started: float,
        *,
        status: str,
        failure_mode: str,
        stdout: str = "",
        stderr: str = "",
    ) -> RemoteRunResult:
        return RemoteRunResult(
            schema_version=REMOTE_SCHEMA_VERSION,
            target_id=self.config.target_id,
            task_id=payload.task_id,
            provider_id=payload.provider_id,
            status=status,
            started_utc=started_utc,
            finished_utc=_now_iso(),
            elapsed_s=time.perf_counter() - started,
            transport=self.transport,
            host=self.config.host,
            stdout_tail=stdout[-2048:],
            stderr_tail=stderr[-2048:],
            failure_mode=failure_mode,
            detail=f"{failure_mode}: {stderr.strip()[:512]}",
        )


__all__ = ["SshRunner"]
