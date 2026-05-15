"""RemoteTargetRunner ABC + SSH backend tests.

Uses subprocess monkey-patching to avoid actually shelling out to
``ssh`` / ``scp`` — the SSH backend's failure paths (rc=124 timeout,
rc=127 missing binary, rc=non-zero remote failures) are exercised
by injecting fake completed-processes.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from compgen.runtime.remote_target import (
    REMOTE_PROBE_STATUSES,
    REMOTE_RUN_STATUSES,
    REMOTE_SCHEMA_VERSION,
    RemoteProbeResult,
    RemoteRunPayload,
    RemoteRunResult,
    RemoteTargetConfig,
    build_runner,
    load_remote_target_config,
)
from compgen.runtime.remote_runners.ssh_runner import SshRunner


def _config(**overrides) -> RemoteTargetConfig:
    body = {
        "target_id": "test_target",
        "transport": "ssh",
        "host": "test.host.example",
        "user": "compgen",
        "workdir": "/tmp/compgen_remote",
        "toolchain_probe_cmd": "echo TOOLCHAIN_1.0",
        "build_cmd_template": "",
        "run_cmd_template": "python {source}",
        "timeout_s": 30,
    }
    body.update(overrides)
    return RemoteTargetConfig(**body)


class _FakeCompleted:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


# ---------------------------------------------------------------------------
# Enum + schema discipline
# ---------------------------------------------------------------------------


def test_probe_statuses_enum_complete():
    assert "available" in REMOTE_PROBE_STATUSES
    assert "unreachable" in REMOTE_PROBE_STATUSES
    assert "toolchain_missing" in REMOTE_PROBE_STATUSES
    assert "auth_failed" in REMOTE_PROBE_STATUSES


def test_run_statuses_enum_complete():
    assert "succeeded" in REMOTE_RUN_STATUSES
    assert "build_failed" in REMOTE_RUN_STATUSES
    assert "timed_out" in REMOTE_RUN_STATUSES
    assert "remote_oom" in REMOTE_RUN_STATUSES


def test_probe_result_unknown_status_rejected():
    with pytest.raises(ValueError, match="probe status"):
        RemoteProbeResult(
            schema_version=REMOTE_SCHEMA_VERSION,
            target_id="x",
            status="wave_hands",
        )


def test_run_result_unknown_status_rejected():
    with pytest.raises(ValueError, match="run status"):
        RemoteRunResult(
            schema_version=REMOTE_SCHEMA_VERSION,
            target_id="x",
            task_id="t",
            provider_id="p",
            status="wave_hands",
            started_utc="x",
            finished_utc="y",
            elapsed_s=1.0,
            transport="ssh",
            host="x",
        )


# ---------------------------------------------------------------------------
# SSH probe paths
# ---------------------------------------------------------------------------


def _patch_subprocess(monkeypatch, *, responses: list[_FakeCompleted]):
    """Each subprocess.run call consumes one queued response."""

    calls = []

    def fake_run(args, **kwargs):
        calls.append((tuple(args), kwargs))
        if not responses:
            raise AssertionError(
                f"unexpected extra subprocess.run call: {args}"
            )
        return responses.pop(0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def test_probe_available_when_remote_responds(monkeypatch):
    _patch_subprocess(
        monkeypatch,
        responses=[
            _FakeCompleted(0, "compgen_remote_probe\n"),  # reachability
            _FakeCompleted(0, "TOOLCHAIN_1.0\n"),         # toolchain probe
        ],
    )
    runner = SshRunner(_config())
    r = runner.probe()
    assert r.status == "available"
    assert r.toolchain_version.startswith("TOOLCHAIN_1.0")


def test_probe_unreachable_on_timeout(monkeypatch):
    def fake_run(args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args, timeout=5)

    monkeypatch.setattr(subprocess, "run", fake_run)
    runner = SshRunner(_config())
    r = runner.probe()
    assert r.status == "unreachable"


def test_probe_auth_failed_on_publickey_reject(monkeypatch):
    _patch_subprocess(
        monkeypatch,
        responses=[
            _FakeCompleted(255, "", "Permission denied (publickey).\n"),
        ],
    )
    runner = SshRunner(_config())
    r = runner.probe()
    assert r.status == "auth_failed"


def test_probe_toolchain_missing(monkeypatch):
    _patch_subprocess(
        monkeypatch,
        responses=[
            _FakeCompleted(0, "compgen_remote_probe\n"),
            _FakeCompleted(127, "", "neuronx-cc: command not found\n"),
        ],
    )
    runner = SshRunner(_config(toolchain_probe_cmd="neuronx-cc --version"))
    r = runner.probe()
    assert r.status == "toolchain_missing"


def test_probe_ssh_binary_missing(monkeypatch):
    def fake_run(args, **kwargs):
        raise FileNotFoundError("ssh: command not found")

    monkeypatch.setattr(subprocess, "run", fake_run)
    runner = SshRunner(_config())
    r = runner.probe()
    assert r.status == "probe_error"


# ---------------------------------------------------------------------------
# ship_and_run failure paths
# ---------------------------------------------------------------------------


def _payload(tmp_path: Path) -> RemoteRunPayload:
    src = tmp_path / "kernel.py"
    src.write_text("print('hello from kernel')\n")
    return RemoteRunPayload(
        task_id="kcodegen_test",
        provider_id="pallas",
        contract_hash="abc",
        kernel_source_path=src,
    )


def test_ship_and_run_succeeded(monkeypatch, tmp_path: Path):
    _patch_subprocess(
        monkeypatch,
        responses=[
            _FakeCompleted(0, "", ""),  # mkdir
            _FakeCompleted(0, "", ""),  # scp kernel
            _FakeCompleted(0,
                           '{"correct": true, "latency_ms": 1.5}\n',
                           ""),  # run
        ],
    )
    runner = SshRunner(_config(build_cmd_template=""))
    result = runner.ship_and_run(_payload(tmp_path))
    assert result.status == "succeeded"
    assert result.runtime_stats == {"correct": True, "latency_ms": 1.5}


def test_ship_and_run_missing_local_source(monkeypatch, tmp_path: Path):
    _patch_subprocess(
        monkeypatch,
        responses=[
            _FakeCompleted(0, "", ""),  # mkdir
        ],
    )
    payload = RemoteRunPayload(
        task_id="t",
        provider_id="pallas",
        contract_hash="abc",
        kernel_source_path=tmp_path / "does_not_exist.py",
    )
    runner = SshRunner(_config())
    result = runner.ship_and_run(payload)
    assert result.status == "transport_error"
    assert result.failure_mode == "missing_local_source"


def test_ship_and_run_scp_failure(monkeypatch, tmp_path: Path):
    _patch_subprocess(
        monkeypatch,
        responses=[
            _FakeCompleted(0, "", ""),                    # mkdir
            _FakeCompleted(1, "", "scp: write failed\n"), # scp
        ],
    )
    runner = SshRunner(_config())
    result = runner.ship_and_run(_payload(tmp_path))
    assert result.status == "transport_error"
    assert result.failure_mode == "scp_kernel_source"


def test_ship_and_run_build_failure(monkeypatch, tmp_path: Path):
    _patch_subprocess(
        monkeypatch,
        responses=[
            _FakeCompleted(0, "", ""),                    # mkdir
            _FakeCompleted(0, "", ""),                    # scp
            _FakeCompleted(1, "", "cc1: error: ...\n"),   # build
        ],
    )
    runner = SshRunner(_config(build_cmd_template="make build"))
    result = runner.ship_and_run(_payload(tmp_path))
    assert result.status == "build_failed"


def test_ship_and_run_remote_oom(monkeypatch, tmp_path: Path):
    _patch_subprocess(
        monkeypatch,
        responses=[
            _FakeCompleted(0, "", ""),                       # mkdir
            _FakeCompleted(0, "", ""),                       # scp
            _FakeCompleted(137, "", "out of memory: kernel killed\n"),
        ],
    )
    runner = SshRunner(_config(build_cmd_template=""))
    result = runner.ship_and_run(_payload(tmp_path))
    assert result.status == "remote_oom"


def test_ship_and_run_timeout(monkeypatch, tmp_path: Path):
    _patch_subprocess(
        monkeypatch,
        responses=[
            _FakeCompleted(0, "", ""),
            _FakeCompleted(0, "", ""),
            _FakeCompleted(124, "", "ssh command timed out after 30s"),
        ],
    )
    runner = SshRunner(_config(build_cmd_template=""))
    result = runner.ship_and_run(_payload(tmp_path))
    assert result.status == "timed_out"


def test_ship_and_run_runtime_failed(monkeypatch, tmp_path: Path):
    _patch_subprocess(
        monkeypatch,
        responses=[
            _FakeCompleted(0, "", ""),
            _FakeCompleted(0, "", ""),
            _FakeCompleted(1, "stdout content", "Traceback ..."),
        ],
    )
    runner = SshRunner(_config(build_cmd_template=""))
    result = runner.ship_and_run(_payload(tmp_path))
    assert result.status == "runtime_failed"
    assert result.failure_mode == "non_zero_exit"


# ---------------------------------------------------------------------------
# Config loader + factory
# ---------------------------------------------------------------------------


def test_load_remote_target_config(tmp_path: Path):
    p = tmp_path / "ssh.yaml"
    p.write_text(
        """\
target_id: my_target
transport: ssh
host: x.example
user: y
workdir: /tmp/x
toolchain_probe_cmd: "echo ok"
build_cmd_template: ""
run_cmd_template: "python {source}"
timeout_s: 60
"""
    )
    cfg = load_remote_target_config(p)
    assert cfg.target_id == "my_target"
    assert cfg.transport == "ssh"
    assert cfg.host == "x.example"


def test_load_shipped_remote_target_configs():
    """All 5 shipped remote-target descriptors load cleanly."""

    for name in (
        "tpu_v5e_pod_1",
        "aws_trn1_inst",
        "hexagon_dev_1",
        "firesim_gemmini",
        "firesim_radiance",
    ):
        cfg = load_remote_target_config(
            Path("configs/remote_targets") / f"{name}.yaml"
        )
        assert cfg.transport == "ssh"
        assert cfg.target_id


def test_build_runner_ssh(tmp_path: Path):
    cfg = _config()
    runner = build_runner(cfg)
    assert isinstance(runner, SshRunner)
    assert runner.transport == "ssh"


def test_build_runner_unsupported_transport():
    from compgen.runtime.remote_target import RemoteExecutionError

    cfg = _config(transport="modal")
    with pytest.raises(RemoteExecutionError, match="modal"):
        build_runner(cfg)


def test_ssh_runner_requires_host():
    cfg = _config(host="")
    with pytest.raises(ValueError, match="host"):
        SshRunner(cfg)


# ---------------------------------------------------------------------------
# Shipped configs probe as unreachable (no real hostname set)
# ---------------------------------------------------------------------------


def test_shipped_remote_configs_have_empty_host_initially():
    """The shipped descriptors carry empty host strings — the user
    is expected to fill them in. The audit must surface
    this as a typed blocked_proof until they're populated."""

    for name in (
        "tpu_v5e_pod_1",
        "aws_trn1_inst",
        "hexagon_dev_1",
        "firesim_gemmini",
        "firesim_radiance",
    ):
        cfg = load_remote_target_config(
            Path("configs/remote_targets") / f"{name}.yaml"
        )
        assert cfg.host == "", (
            f"{name} has a non-empty host {cfg.host!r} — set up done?"
        )
