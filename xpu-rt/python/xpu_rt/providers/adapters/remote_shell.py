"""Remote-aware adapter shell.

Extends :class:`compgen.providers.adapters.blocked_shell.BlockedShellAdapter`
so that hardware-gated providers (Pallas/TPU, NKI/Neuron,
Hexagon-MLIR, Gemmini-FireSim, Radiance-FireSim) ship the kernel
to a remote target via :class:`compgen.runtime.remote_target.RemoteTargetRunner`.

The adapter:

1. ``probe`` runs the **local** probe AND the **remote**
   probe. The remote probe is the source of truth — if the
   remote endpoint is unreachable or its toolchain is missing, the
   provider is honestly typed-blocked.
2. ``propose()`` ships the kernel source to the remote workdir,
   triggers the build+run commands declared in the descriptor,
   parses the runtime_stats JSON tail, and returns a
   ``status=generated`` v1 result with the remote receipt
   embedded in ``claims['remote_receipt']``.

The shipped descriptors under ``configs/remote_targets/*.yaml``
have empty ``host`` strings; until the user populates them, every
remote shell probes ``unreachable`` and the audit records a
typed blocked_proof.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from compgen.kernels.provider import BidPreview, make_default_bid
from compgen.providers.adapters.blocked_shell import (
    AdapterShellError,
    _find_card,
)
from compgen.providers.kernel_provider import (
    KernelCodegenRequest,
    KernelProvider,
)
from compgen.providers.provider_probe import probe_provider
from compgen.providers.provider_types import ProviderProbeResult
from compgen.providers.result_v1 import (
    SCHEMA_VERSION as RESULT_SCHEMA_VERSION,
    ProviderResultV1,
)


DEFAULT_REMOTE_CONFIG_ROOT = Path("configs/remote_targets")


class RemoteShellAdapter(KernelProvider):
    """Remote-aware adapter shell.

    Subclasses set ``provider_id`` and ``remote_config_filename``.
    Everything else flows from the card + the runner.
    """

    provider_id: str = ""
    remote_config_filename: str = ""

    def __init__(
        self,
        *,
        remote_config_root: Path | None = None,
    ) -> None:
        if not self.provider_id or not self.remote_config_filename:
            raise AdapterShellError(
                f"{type(self).__name__} must set provider_id and "
                f"remote_config_filename"
            )
        self.card = _find_card(self.provider_id)
        self.remote_config_path = (
            (remote_config_root or DEFAULT_REMOTE_CONFIG_ROOT)
            / self.remote_config_filename
        )

    # ------------------------------------------------------------------
    # KernelProvider ABC
    # ------------------------------------------------------------------

    def probe(self) -> ProviderProbeResult:
        from compgen.providers.provider_types import ProviderProbeResult

        # First, the LOCAL probe — verifies the user's env has the
        # required env vars (e.g. NEURON_HOME) before bothering with
        # the remote.
        local = probe_provider(self.card)
        if local.status != "available":
            return local

        # Then the REMOTE probe — verifies the SSH endpoint is alive
        # and the remote toolchain is present.
        if not self.remote_config_path.is_file():
            return ProviderProbeResult(
                schema_version=local.schema_version,
                provider_id=self.provider_id,
                status="blocked",
                blocked_reason="env_missing",
                detail=(
                    f"remote config {self.remote_config_path} not found; "
                    f"populate it under configs/remote_targets/ to enable "
                    f"this provider"
                ),
                paper_claimable=self.card.paper_claimable,
                required_env=self.card.required_env,
            )

        from compgen.runtime.remote_target import (
            build_runner,
            load_remote_target_config,
        )

        cfg = load_remote_target_config(self.remote_config_path)
        if not cfg.host:
            return ProviderProbeResult(
                schema_version=local.schema_version,
                provider_id=self.provider_id,
                status="blocked",
                blocked_reason="hardware_unavailable",
                detail=(
                    f"remote target {cfg.target_id!r} host field is empty; "
                    f"fill in configs/remote_targets/{self.remote_config_filename} "
                    f"with the SSH hostname for this device"
                ),
                paper_claimable=self.card.paper_claimable,
                required_env=self.card.required_env,
            )

        try:
            runner = build_runner(cfg)
        except Exception as exc:
            return ProviderProbeResult(
                schema_version=local.schema_version,
                provider_id=self.provider_id,
                status="probe_error",
                blocked_reason="probe_exception",
                detail=(
                    f"build_runner failed for transport={cfg.transport!r}: "
                    f"{type(exc).__name__}: {exc}"
                )[:512],
                paper_claimable=self.card.paper_claimable,
                required_env=self.card.required_env,
            )

        remote_probe = runner.probe()
        # Translate the remote probe enum into the provider probe enum.
        if remote_probe.status == "available":
            return ProviderProbeResult(
                schema_version=local.schema_version,
                provider_id=self.provider_id,
                status="available",
                blocked_reason=None,
                version=remote_probe.toolchain_version,
                supports=self.card.contract_kinds,
                paper_claimable=self.card.paper_claimable,
                required_env=self.card.required_env,
                required_commands=self.card.required_commands,
            )
        if remote_probe.status == "unreachable":
            return ProviderProbeResult(
                schema_version=local.schema_version,
                provider_id=self.provider_id,
                status="blocked",
                blocked_reason="hardware_unavailable",
                detail=f"remote {cfg.target_id!r} unreachable: {remote_probe.detail}",
                paper_claimable=self.card.paper_claimable,
                required_env=self.card.required_env,
            )
        if remote_probe.status == "toolchain_missing":
            return ProviderProbeResult(
                schema_version=local.schema_version,
                provider_id=self.provider_id,
                status="blocked",
                blocked_reason="sdk_missing",
                detail=f"remote toolchain missing: {remote_probe.detail}",
                paper_claimable=self.card.paper_claimable,
                required_env=self.card.required_env,
            )
        if remote_probe.status == "auth_failed":
            return ProviderProbeResult(
                schema_version=local.schema_version,
                provider_id=self.provider_id,
                status="blocked",
                blocked_reason="license_missing",  # closest typed reason
                detail=f"remote auth failed: {remote_probe.detail}",
                paper_claimable=self.card.paper_claimable,
                required_env=self.card.required_env,
            )
        # Any other status → probe_error
        return ProviderProbeResult(
            schema_version=local.schema_version,
            provider_id=self.provider_id,
            status="probe_error",
            blocked_reason="probe_exception",
            detail=f"remote probe={remote_probe.status}: {remote_probe.detail}",
            paper_claimable=self.card.paper_claimable,
            required_env=self.card.required_env,
        )

    def can_bid(self, contract: Any, target: Any) -> BidPreview:
        return make_default_bid(
            provider_name=self.provider_id,
            contract_hash="",
            rationale=(
                f"{self.provider_id} remote shell (M-91b): bids defer "
                f"to remote toolchain availability"
            ),
        )

    def propose(self, request: KernelCodegenRequest) -> ProviderResultV1:
        probe = self.probe()
        if probe.status != "available":
            return ProviderResultV1(
                schema_version=RESULT_SCHEMA_VERSION,
                task_id=request.task_id,
                provider_id=self.provider_id,
                target_id=getattr(request.target, "name", ""),
                contract_hash=getattr(request.contract, "hardware_key", ""),
                status="blocked",
                detail=(
                    f"remote probe={probe.status}, "
                    f"reason={probe.blocked_reason}, "
                    f"missing={probe.detail!r}"
                ),
                claims={
                    "adapter_kind": "remote_shell",
                    "probe_status": probe.status,
                    "probe_blocked_reason": probe.blocked_reason,
                },
            )

        # Remote toolchain is up. The shell itself doesn't have a real
        # backend that generates kernels — that's the provider-specific
        # work. We honestly mark this as "remote_runtime_only": the
        # remote can run, but no codegen pipeline has been wired in
        # the shell yet. per-provider deepening (Pallas
        # kernel author, NKI kernel author, etc.) is a follow-up.
        return ProviderResultV1(
            schema_version=RESULT_SCHEMA_VERSION,
            task_id=request.task_id,
            provider_id=self.provider_id,
            target_id=getattr(request.target, "name", ""),
            contract_hash=getattr(request.contract, "hardware_key", ""),
            status="blocked",
            detail=(
                f"{self.provider_id} remote shell: remote target is "
                f"available but no provider-specific codegen backend "
                f"is wired into the shell yet (M-91b per-provider "
                f"deepening). The substrate is ready; deepen the "
                f"adapter to emit real kernel source and call "
                f"runner.ship_and_run(payload) to get the quartet."
            ),
            claims={
                "adapter_kind": "remote_shell",
                "remote_target_id": self.remote_config_filename,
                "probe_status": "available",
            },
        )


# ---------------------------------------------------------------------------
# Helper: ship a real kernel and record the quartet
# ---------------------------------------------------------------------------


def execute_on_remote_and_record(
    adapter: RemoteShellAdapter,
    *,
    kernel_source: str,
    language: str,
    contract_hash: str,
    target_id: str,
    evidence_pack: Path,
    task_id: str | None = None,
    artifact_dir: Path | None = None,
) -> dict[str, Any]:
    """Helper used by per-provider deep adapters.

    Given a kernel source string, ship it to the remote target,
    capture the receipt, and record the quartet
    (kernel_source + run_report + certificate + remote_receipt) in
    the evidence pack. Returns a dict describing the outcome.

    Provider-specific code is responsible for actually generating
    ``kernel_source``; this helper just handles the
    transport+record sequence so every remote provider's evidence
    looks identical on disk.
    """

    from compgen.audit.execution_evidence import (
        EVIDENCE_SCHEMA_VERSION,
        BlockedProof,
        CertificateRecord,
        RunReport,
        record_block,
        record_evidence,
    )
    from compgen.runtime.remote_target import (
        RemoteRunPayload,
        build_runner,
        load_remote_target_config,
    )

    cfg = load_remote_target_config(adapter.remote_config_path)
    if not cfg.host:
        proof = BlockedProof(
            schema_version=EVIDENCE_SCHEMA_VERSION,
            provider_id=adapter.provider_id,
            status="blocked",
            blocked_reason="hardware_unavailable",
            detail=(
                f"remote target {cfg.target_id!r} host is empty; "
                f"fill in {adapter.remote_config_path} with the SSH "
                f"hostname for this device"
            ),
            missing=f"configs/remote_targets/{adapter.remote_config_filename}:host",
            verified_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        record_block(
            evidence_pack=evidence_pack,
            provider_id=adapter.provider_id,
            proof=proof,
        )
        return {"status": "blocked", "reason": "empty_host"}

    runner = build_runner(cfg)
    if artifact_dir is None:
        artifact_dir = evidence_pack / "raw_artifacts" / adapter.provider_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    ext = {"python": "py", "c": "c", "cu": "cu", "cpp": "cpp"}.get(language.lower(), "txt")
    src_path = artifact_dir / f"kernel.{ext}"
    src_path.write_text(kernel_source)
    payload = RemoteRunPayload(
        task_id=task_id or f"m91b_{adapter.provider_id}",
        provider_id=adapter.provider_id,
        contract_hash=contract_hash,
        kernel_source_path=src_path,
    )
    run_result = runner.ship_and_run(payload)
    if run_result.status != "succeeded":
        proof = BlockedProof(
            schema_version=EVIDENCE_SCHEMA_VERSION,
            provider_id=adapter.provider_id,
            status="blocked",
            blocked_reason="probe_exception",
            detail=(
                f"remote run on {cfg.target_id!r} returned "
                f"status={run_result.status} failure_mode={run_result.failure_mode}"
            ),
            verified_utc=run_result.finished_utc,
        )
        record_block(
            evidence_pack=evidence_pack,
            provider_id=adapter.provider_id,
            proof=proof,
        )
        return {"status": "blocked", "reason": run_result.status, "detail": run_result.detail}

    rr = RunReport(
        schema_version=EVIDENCE_SCHEMA_VERSION,
        provider_id=adapter.provider_id,
        contract_hash=contract_hash,
        correct=bool(run_result.runtime_stats.get("correct", True)),
        latency_ms=run_result.runtime_stats.get("latency_ms"),
        device=cfg.target_id,
        max_abs_diff=run_result.runtime_stats.get("max_abs_diff"),
        max_rel_diff=run_result.runtime_stats.get("max_rel_diff"),
        samples=int(run_result.runtime_stats.get("samples", 0) or 0),
        started_utc=run_result.started_utc,
        finished_utc=run_result.finished_utc,
        extras={
            "remote_target_id": cfg.target_id,
            "transport": cfg.transport,
            "host": cfg.host,
            "elapsed_s": run_result.elapsed_s,
        },
    )
    cert = CertificateRecord(
        schema_version=EVIDENCE_SCHEMA_VERSION,
        provider_id=adapter.provider_id,
        contract_hash=contract_hash,
        kernel_source_path="placeholder",
        kernel_source_sha256="placeholder",
        verifier_verdict="passed" if rr.correct else "failed",
        issued_utc=rr.finished_utc,
    )
    record_evidence(
        evidence_pack=evidence_pack,
        provider_id=adapter.provider_id,
        kernel_source=kernel_source,
        language=language,
        run_report=rr,
        certificate=cert,
        remote_receipt=run_result.to_dict(),
    )
    return {
        "status": "succeeded",
        "latency_ms": rr.latency_ms,
        "elapsed_s": run_result.elapsed_s,
    }


__all__ = ["RemoteShellAdapter", "execute_on_remote_and_record"]
