"""Execution-plan emission with M-46 region_kernel_bindings.

Phase C M-46: emit ``05_execution_plan/execution_plan.yaml`` plus a
sidecar ``05_execution_plan/region_kernel_bindings.json`` describing
the (region_id, contract_hash, certificate_path, kernel_artifact,
dispatch_model) tuples for every region whose M-45 certificate exists.

Hard rule: a region is bound only if its certificate file exists AND
the certificate's ``contract_hash`` matches what M-42's request +
M-44's verification + M-45's emit produced. Otherwise the region is
recorded as ``unbound`` in the sidecar with a typed reason; M-47's
plan executor will refuse to call those regions.

This stage runs after ``--stop-after kernel-codegen-request`` (where
M-40-M-42 emit the contract + request) and after the M-43/M-44/M-45
chain has had a chance to write certificates. In the operator-driven
flow, the operator has already submitted a provider response and the
certificate is on disk before this stage runs.

When no certificates exist (e.g. fresh greedy run with no provider
response yet), the plan is emitted with an empty
``region_kernel_bindings`` list and a sidecar that names the
unbound regions with ``no_certificate`` as the reason. This makes
M-47's downstream check honest — a plan that bound zero kernels
fails the runtime-bound-kernel check rather than silently running.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from compgen.runtime.execution_plan import (
    ExecutionPlan,
    RegionKernelBinding,
    RegionPlacement,
    Resource,
)


@dataclass(frozen=True)
class BindingRow:
    region_id: str
    status: str  # "bound" | "unbound"
    contract_hash: str = ""
    certificate_path: str = ""
    kernel_artifact: str = ""
    dispatch_model: str = "sync"
    unbound_reason: str = ""
    canonical_contract_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "status": self.status,
            "contract_hash": self.contract_hash,
            "canonical_contract_hash": self.canonical_contract_hash,
            "certificate_path": self.certificate_path,
            "kernel_artifact": self.kernel_artifact,
            "dispatch_model": self.dispatch_model,
            "unbound_reason": self.unbound_reason,
        }


@dataclass(frozen=True)
class ExecutionPlanEmitResult:
    out_dir: Path
    plan_path: Path
    bindings_path: Path
    overall: str  # "pass" | "skipped" | "fail"
    bound_count: int
    unbound_count: int
    bindings: tuple[BindingRow, ...] = ()


def _utcnow() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_or_none(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return _read_json(path)
    except json.JSONDecodeError:
        return None


def _resolve_target_id(run_dir: Path) -> str:
    """Pull target_id from the recipe summary (most authoritative on
    disk) with a defensive fallback to the run manifest."""
    rs = _read_json_or_none(run_dir / "03_recipe_planning" / "recipe_summary.json")
    if rs and rs.get("target_id"):
        return str(rs["target_id"])
    rm = _read_json_or_none(run_dir / "run_manifest.json")
    if rm:
        return str((rm.get("target") or {}).get("target_id", "") or "")
    return ""


def _resolve_workload_id(run_dir: Path) -> str:
    rm = _read_json_or_none(run_dir / "run_manifest.json")
    if rm:
        return str((rm.get("model") or {}).get("model_id", "") or "")
    return ""


def _bindings_for_run(run_dir: Path) -> list[BindingRow]:
    """Walk every kernel-codegen request and produce a binding row.

    The request is the source of truth for which regions need a
    binding. The certificate (M-45) is the source of truth for whether
    that region is bindable.
    """
    rows: list[BindingRow] = []
    requests_dir = run_dir / "04_kernel_codegen" / "requests"
    if not requests_dir.is_dir():
        return rows
    for req_path in sorted(requests_dir.glob("*.request.json")):
        request = _read_json_or_none(req_path)
        if request is None:
            continue
        region_id = request.get("region_id", "") or ""
        contract_hash = request.get("contract_hash", "") or ""
        request_kind = request.get("request_kind", "") or ""

        if request_kind != "kernel_codegen":
            rows.append(BindingRow(
                region_id=region_id, status="unbound",
                unbound_reason=(
                    f"request_kind={request_kind!r} (no kernel codegen "
                    "expected for this candidate)"
                ),
            ))
            continue

        if not contract_hash:
            rows.append(BindingRow(
                region_id=region_id, status="unbound",
                unbound_reason="contract_hash missing on request",
            ))
            continue

        cert_rel = (
            f"04_kernel_codegen/certificates/{contract_hash}.json"
        )
        cert_path = run_dir / cert_rel
        if not cert_path.exists():
            rows.append(BindingRow(
                region_id=region_id, status="unbound",
                contract_hash=contract_hash,
                certificate_path=cert_rel,
                unbound_reason=(
                    "no_certificate; provider response not yet committed "
                    "(or M-44 verification rejected)"
                ),
            ))
            continue

        cert = _read_json(cert_path)
        # Sanity-check the certificate's contract_hash matches.
        if cert.get("contract_hash") != contract_hash:
            rows.append(BindingRow(
                region_id=region_id, status="unbound",
                contract_hash=contract_hash,
                certificate_path=cert_rel,
                unbound_reason=(
                    f"certificate contract_hash="
                    f"{cert.get('contract_hash')!r} mismatched binding "
                    f"contract_hash={contract_hash!r}"
                ),
            ))
            continue

        # Pick the kernel_source artifact to record; M-47 imports from
        # this path.
        kernel_artifact = ""
        for name, p in (cert.get("artifact_paths") or {}).items():
            if name == "kernel_source":
                kernel_artifact = p
                break

        # Read dispatch_model from the materialised contract.
        contract_path = (
            request.get("contract_paths", {}).get("full") or ""
        )
        dispatch_model = "sync"
        if contract_path:
            full = run_dir / contract_path
            full_body = _read_json_or_none(full)
            if full_body:
                dispatch_model = (
                    (full_body.get("orchestration") or {})
                    .get("dispatch", {}).get("model", "sync")
                )

        canonical_hash = str(cert.get("canonical_contract_hash", "") or "")

        rows.append(BindingRow(
            region_id=region_id,
            status="bound",
            contract_hash=contract_hash,
            certificate_path=cert_rel,
            kernel_artifact=kernel_artifact,
            dispatch_model=dispatch_model,
            canonical_contract_hash=canonical_hash,
        ))
    return rows


def emit_execution_plan(run_dir: Path) -> ExecutionPlanEmitResult:
    """Build a minimal ExecutionPlan with M-46 region_kernel_bindings,
    serialise it to ``05_execution_plan/execution_plan.yaml``, and
    write the per-region binding sidecar.

    The plan is intentionally minimal at M-46 — placement, dependency,
    sync, and copy edges land in M-47/M-51/M-52 when the executor is
    actually emitted. M-46's only goal is to bind certified kernels
    to regions and validate that the bindings agree with on-disk
    certificates.
    """
    run_dir = Path(run_dir).resolve()
    out_dir = run_dir / "05_execution_plan"
    out_dir.mkdir(parents=True, exist_ok=True)

    binding_rows = _bindings_for_run(run_dir)
    bound = [r for r in binding_rows if r.status == "bound"]
    unbound = [r for r in binding_rows if r.status == "unbound"]

    bindings = [
        RegionKernelBinding(
            region_id=r.region_id,
            contract_hash=r.contract_hash,
            certificate_path=r.certificate_path,
            kernel_artifact=r.kernel_artifact,
            dispatch_model=r.dispatch_model,
            canonical_contract_hash=r.canonical_contract_hash,
        )
        for r in bound
    ]

    workload = _resolve_workload_id(run_dir) or "unknown"
    target = _resolve_target_id(run_dir) or "host_cpu"

    placements = [
        RegionPlacement(
            region_id=r.region_id,
            device=target,
            queue=f"queue_{target}",
        )
        for r in binding_rows
    ]
    resources = [
        Resource(
            id=f"queue_{target}", kind="compute",
            device=target, capacity=1.0,
        ),
    ]
    plan = ExecutionPlan(
        workload=workload,
        target=target,
        target_resource_model_hash="",
        resources=resources,
        region_placement=placements,
        region_kernel_bindings=bindings,
        summary={
            "schema_origin": "m46_execution_plan_emit",
            "generated_at_utc": _utcnow(),
            "bound_regions": [r.region_id for r in bound],
            "unbound_regions": [r.region_id for r in unbound],
        },
    )
    # Structural validation (run-dir-agnostic).
    plan.validate()
    # Strict validation against on-disk certificates — M-46's load-bearing
    # check. If a cert references a missing/mismatched contract_hash we
    # surface it as an unbound row (fall back gracefully); validate_with_run_dir
    # only fires on bindings that truly have a cert path.
    plan.validate_with_run_dir(run_dir)

    # Persist plan as YAML when PyYAML is available; otherwise JSON.
    plan_dict = plan.to_dict()
    plan_path = out_dir / "execution_plan.yaml"
    try:
        import yaml  # type: ignore[import-untyped]
        plan_path.write_text(
            yaml.safe_dump(plan_dict, sort_keys=True, default_flow_style=False),
            encoding="utf-8",
        )
    except ImportError:
        plan_path = out_dir / "execution_plan.json"
        plan_path.write_text(
            json.dumps(plan_dict, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    bindings_path = out_dir / "region_kernel_bindings.json"
    bindings_body = {
        "schema_version": "region_kernel_bindings_v1",
        "generated_at_utc": _utcnow(),
        "workload": workload,
        "target": target,
        "bound_count": len(bound),
        "unbound_count": len(unbound),
        "bindings": [r.to_dict() for r in binding_rows],
    }
    bindings_path.write_text(
        json.dumps(bindings_body, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    overall = "pass" if bound else "skipped"
    return ExecutionPlanEmitResult(
        out_dir=out_dir,
        plan_path=plan_path,
        bindings_path=bindings_path,
        overall=overall,
        bound_count=len(bound),
        unbound_count=len(unbound),
        bindings=tuple(binding_rows),
    )
