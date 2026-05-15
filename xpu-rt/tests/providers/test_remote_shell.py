"""M-91b — remote-aware adapter shell tests."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from compgen.providers.adapters.blocked_shell import BlockedShellAdapter
from compgen.providers.adapters.gemmini_c import GemminiCProvider
from compgen.providers.adapters.hexagon_mlir import HexagonMLIRProvider
from compgen.providers.adapters.nki import NkiProvider
from compgen.providers.adapters.pallas import PallasProvider
from compgen.providers.adapters.radiance_muon import RadianceMuonProvider
from compgen.providers.adapters.remote_shell import (
    DEFAULT_REMOTE_CONFIG_ROOT,
    RemoteShellAdapter,
)
from compgen.providers.kernel_provider import (
    KernelCodegenRequest,
    KernelProvider,
)
from compgen.providers.provider_types import PROBE_STATUSES
from compgen.providers.result_v1 import ProviderResultV1


HW_GATED = {
    "pallas": PallasProvider,
    "nki": NkiProvider,
    "hexagon_mlir": HexagonMLIRProvider,
    "gemmini_c": GemminiCProvider,
    "radiance_muon": RadianceMuonProvider,
}


# ---------------------------------------------------------------------------
# Each HW-gated provider is a real KernelProvider
# ---------------------------------------------------------------------------


def test_every_hw_gated_provider_is_a_kernel_provider():
    for pid, cls in HW_GATED.items():
        inst = cls()
        assert isinstance(inst, KernelProvider), f"{pid} not a KernelProvider"
        assert isinstance(inst, RemoteShellAdapter), f"{pid} not a RemoteShellAdapter"


def test_every_hw_gated_provider_has_a_remote_config_filename():
    for pid, cls in HW_GATED.items():
        assert cls.remote_config_filename, f"{pid} has no remote_config_filename"


def test_every_hw_gated_remote_config_exists():
    for pid, cls in HW_GATED.items():
        cfg_path = DEFAULT_REMOTE_CONFIG_ROOT / cls.remote_config_filename
        assert cfg_path.is_file(), f"{pid} missing remote config {cfg_path}"


# ---------------------------------------------------------------------------
# probe() returns typed status — local fail OR remote unreachable
# ---------------------------------------------------------------------------


def test_every_hw_gated_probe_returns_typed_status(monkeypatch):
    # Set ALL local env vars so the local probe passes for each
    # provider, exposing the remote-probe path.
    monkeypatch.setenv("NEURON_HOME", "/tmp/fake")
    monkeypatch.setenv("HEXAGON_MLIR_ROOT", "/tmp/fake")
    monkeypatch.setenv("CHIPYARD_ROOT", "/tmp/fake")
    monkeypatch.setenv("GEMMINI_ROOT", "/tmp/fake")
    monkeypatch.setenv("RADIANCE_KERNELS_ROOT", "/tmp/fake")
    monkeypatch.setenv("RISCV_TOOLCHAIN_PATH", "/tmp/fake")

    for pid, cls in HW_GATED.items():
        inst = cls()
        probe = inst.probe()
        assert probe.status in PROBE_STATUSES, (
            f"{pid}: status {probe.status!r} not typed"
        )
        if probe.status != "available":
            assert probe.blocked_reason
            assert probe.detail


def test_pallas_remote_unreachable_when_host_empty():
    """The shipped remote config has empty host → blocked +
    hardware_unavailable."""

    inst = PallasProvider()
    # Pallas has python_package_missing locally (no jax). Patch the
    # local probe to "available" so we test the REMOTE path.
    from compgen.providers.provider_types import ProviderProbeResult

    with patch(
        "compgen.providers.adapters.remote_shell.probe_provider"
    ) as mock_local:
        mock_local.return_value = ProviderProbeResult(
            schema_version="provider_status_v1",
            provider_id="pallas",
            status="available",
        )
        probe = inst.probe()
    assert probe.status == "blocked"
    assert probe.blocked_reason == "hardware_unavailable"
    assert "host field is empty" in probe.detail


# ---------------------------------------------------------------------------
# propose() returns a typed v1 result
# ---------------------------------------------------------------------------


class _Target:
    name = "x"


def test_propose_returns_blocked_v1_when_remote_down():
    inst = PallasProvider()
    req = KernelCodegenRequest(
        task_id="t",
        contract=None,
        target=_Target(),
        artifact_dir="/tmp",
    )
    result = inst.propose(req)
    assert isinstance(result, ProviderResultV1)
    assert result.status == "blocked"
    assert result.detail
    assert result.claims.get("adapter_kind") == "remote_shell"


def test_propose_returns_blocked_when_remote_available_but_no_backend():
    """When the remote IS available but the provider-specific
    codegen hasn't been written, the shell honestly says so."""

    from compgen.providers.adapters import remote_shell
    from compgen.providers.provider_types import ProviderProbeResult

    inst = PallasProvider()
    with patch.object(
        inst,
        "probe",
        return_value=ProviderProbeResult(
            schema_version="provider_status_v1",
            provider_id="pallas",
            status="available",
        ),
    ):
        req = KernelCodegenRequest(
            task_id="t",
            contract=None,
            target=_Target(),
            artifact_dir="/tmp",
        )
        result = inst.propose(req)
    assert isinstance(result, ProviderResultV1)
    assert result.status == "blocked"
    assert "no provider-specific codegen backend" in result.detail


# ---------------------------------------------------------------------------
# execute_on_remote_and_record helper records the quartet when
# the remote run succeeds.
# ---------------------------------------------------------------------------


def test_execute_on_remote_records_quartet_on_success(tmp_path: Path):
    """When the remote runner returns status=succeeded, the helper
    records the full available-quartet ."""

    # Populate a remote config with a non-empty host so the helper
    # gets past the empty-host guard.
    config_root = tmp_path / "configs"
    config_root.mkdir()
    cfg_path = config_root / "fake.yaml"
    cfg_path.write_text(
        """\
target_id: fake_target
transport: ssh
host: fake.host
user: fake_user
workdir: /tmp/fake
toolchain_probe_cmd: echo ok
build_cmd_template: ""
run_cmd_template: "python {source}"
timeout_s: 30
"""
    )

    class _FakeAdapter:
        provider_id = "fake_remote"
        remote_config_filename = "fake.yaml"
        remote_config_path = cfg_path

    from compgen.providers.adapters.remote_shell import (
        execute_on_remote_and_record,
    )
    from compgen.runtime.remote_target import (
        REMOTE_SCHEMA_VERSION,
        RemoteRunResult,
    )

    fake_run_result = RemoteRunResult(
        schema_version=REMOTE_SCHEMA_VERSION,
        target_id="fake_target",
        task_id="t",
        provider_id="fake_remote",
        status="succeeded",
        started_utc="2026-05-12T00:00:00Z",
        finished_utc="2026-05-12T00:00:01Z",
        elapsed_s=1.0,
        transport="ssh",
        host="fake.host",
        runtime_stats={"correct": True, "latency_ms": 0.5, "samples": 10},
    )

    pack = tmp_path / "evidence_pack"
    with patch(
        "compgen.runtime.remote_target.build_runner"
    ) as mock_build:
        fake_runner = type(
            "FakeRunner",
            (),
            {
                "ship_and_run": lambda self, payload: fake_run_result,
                "probe": lambda self: None,
            },
        )()
        mock_build.return_value = fake_runner
        outcome = execute_on_remote_and_record(
            _FakeAdapter(),
            kernel_source="import jax\n",
            language="python",
            contract_hash="abc",
            target_id="fake_target",
            evidence_pack=pack,
        )
    assert outcome["status"] == "succeeded"
    pp = pack / "per_provider" / "fake_remote"
    assert (pp / "kernel_source.py").is_file()
    assert (pp / "run_report.json").is_file()
    assert (pp / "certificate.json").is_file()
    assert (pp / "remote_receipt.json").is_file()
    receipt = json.loads((pp / "remote_receipt.json").read_text())
    assert receipt["status"] == "succeeded"
    assert receipt["host"] == "fake.host"


def test_execute_on_remote_records_block_on_empty_host(tmp_path: Path):
    """When the remote config has an empty host, the helper writes a
    typed blocked_proof — no fake kernel artifacts."""

    config_root = tmp_path / "configs"
    config_root.mkdir()
    cfg_path = config_root / "empty.yaml"
    cfg_path.write_text(
        """\
target_id: empty_target
transport: ssh
host: ""
user: ""
workdir: /tmp/x
"""
    )

    class _FakeAdapter:
        provider_id = "fake_remote_empty"
        remote_config_filename = "empty.yaml"
        remote_config_path = cfg_path

    from compgen.providers.adapters.remote_shell import (
        execute_on_remote_and_record,
    )

    pack = tmp_path / "evidence_pack"
    outcome = execute_on_remote_and_record(
        _FakeAdapter(),
        kernel_source="x",
        language="python",
        contract_hash="abc",
        target_id="empty_target",
        evidence_pack=pack,
    )
    assert outcome["status"] == "blocked"
    pp = pack / "per_provider" / "fake_remote_empty"
    assert (pp / "blocked_proof.json").is_file()
    # No kernel source written.
    assert not list(pp.glob("kernel_source.*"))


# ---------------------------------------------------------------------------
# All 5 HW-gated provider cards have recorded blocked_proof.json on
# this machine (shipped via record_hw_gated_blocked_proofs.py).
# ---------------------------------------------------------------------------


def test_shipped_evidence_pack_has_blocked_proof_for_each_hw_gated_provider():
    pack = Path("results/extension_provider_evidence_pack")
    if not pack.is_dir():
        pytest.skip("no canonical evidence pack on disk")
    for pid in HW_GATED:
        proof_path = pack / "per_provider" / pid / "blocked_proof.json"
        assert proof_path.is_file(), (
            f"{pid}: no blocked_proof.json at {proof_path}"
        )
        body = json.loads(proof_path.read_text())
        assert body["provider_id"] == pid
        assert body["status"] in (
            "blocked",
            "unsupported",
            "probe_error",
            "not_installed",
        )
        assert body["blocked_reason"]
        assert body["detail"]
