"""Kernel-codegen task emission (M-42, supersedes M-39's request emit).

Phase C M-42: replaces M-39's ``kernel_specialization_request_v1`` with
``kernel_codegen_request_v1`` and migrates the directory layout from
``04_kernel_specialization/`` to ``04_kernel_codegen/``. The new
schema is leaner: the request POINTS to the materialised contract
files (M-40 wrote them) instead of embedding shape/tile/layout/dtype
inline. The contract files are the source of truth; the request
bounds **which** of those files the kernel-codegen provider may read,
and **where** it may write its artifacts.

Directory layout under ``04_kernel_codegen/``:

::

    04_kernel_codegen/
      requests/<task_id>.request.json     (this module emits)
      contracts/<region>.<hash>.json      (M-40 emits)
      views/<region>.kernel_facing.json   (M-40 emits)
      artifacts/<task_id>/                (sandbox; provider writes here at M-43)
      kernel_codegen_summary.json

The sandbox dir is created empty so M-43's commit tool can enforce
"every response artifact path lives under ``artifact_dir``" without
having to recreate the directory. Forbidden-path-write is a fatal
protocol violation in M-43.

Critical invariant: the subagent (M-43+) only sees
``contract_paths.kernel_facing`` — never ``contract_paths.full``. The
``compiler_only()`` projection (wait_on, blocking, lifetimes,
fusion, observability, max_concurrent_invocations, providers,
metadata) lives only in the full contract; the request is wired so a
correct provider implementation never has cause to read the full
contract.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


_SCHEMA_VERSION = "kernel_codegen_request_v1"
_NOT_APPLICABLE_REASONS = {
    "fuse_producer_consumer": (
        "M-42 supports only candidate_kind='set_tile_params'; widens "
        "to fusion when the contract registry grows past COMPUTE_TILED."
    ),
    "create_kernel_contract": (
        "create_kernel_contract is a planning op, not a kernel-codegen "
        "trigger; M-42 emits not_applicable."
    ),
}


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _ContractPaths:
    full: str               # 04_kernel_codegen/contracts/<region>.<hash>.json
    kernel_facing: str      # 04_kernel_codegen/views/<region>.kernel_facing.json

    def to_dict(self) -> dict[str, Any]:
        return {"full": self.full, "kernel_facing": self.kernel_facing}


@dataclass(frozen=True)
class KernelCodegenRequest:
    """Bounded specification of the kernel-codegen task a provider
    (cffi-C, Triton template, Claude Code subagent) is expected to
    fulfil. The request POINTS to the materialised contract files;
    the provider reads ``contract_paths.kernel_facing`` for what it
    needs to know.
    """

    schema_version: str
    task_id: str
    generated_at_utc: str
    request_kind: str  # "kernel_codegen" | "not_applicable"

    region_id: str
    candidate_id: str
    recipe_op_id: str
    contract_hash: str

    contract_paths: _ContractPaths

    allowed_backends: tuple[str, ...]
    required_outputs: tuple[str, ...]
    forbidden: tuple[str, ...]
    artifact_dir: str  # relative to run_dir

    not_applicable_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "generated_at_utc": self.generated_at_utc,
            "request_kind": self.request_kind,
            "region_id": self.region_id,
            "candidate_id": self.candidate_id,
            "recipe_op_id": self.recipe_op_id,
            "contract_hash": self.contract_hash,
            "contract_paths": self.contract_paths.to_dict(),
            "allowed_backends": list(self.allowed_backends),
            "required_outputs": list(self.required_outputs),
            "forbidden": list(self.forbidden),
            "artifact_dir": self.artifact_dir,
            "not_applicable_reason": self.not_applicable_reason,
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _utcnow() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_or_none(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _read_json(path)


def _task_id(*, candidate_id: str, region_id: str) -> str:
    """Deterministic task id derived from (region_id, candidate_id).
    Stable across reruns of the same selection."""
    raw = f"{region_id}|{candidate_id}".encode("utf-8")
    h = hashlib.sha256(raw).hexdigest()[:8]
    return f"kcodegen_{h}"


# Forbidden-mutations the request lists explicitly. Names match the
# M-43 commit-tool failure-class taxonomy (so `forbidden` strings on
# the request can be cross-referenced with the rejection reasons the
# commit tool emits).
_FORBIDDEN_MUTATIONS: tuple[str, ...] = (
    "modify_contract",
    "modify_payload_ir",
    "change_tolerance",
    "invent_shape",
    "emit_unverified_success",
    "widen_tolerance_beyond_higham_bound",
    "ignore_layout",
)

# Required output filenames the provider MUST produce inside
# artifact_dir/. M-43's commit tool checks each one exists before
# routing to the M-44 verifier.
_REQUIRED_OUTPUTS: tuple[str, ...] = (
    "kernel_source",       # kernel.py | kernel.c
    "kernel_metadata",     # kernel_metadata.json
    "launch_config",       # launch_config.json
    "provider_claims",     # provider_claims.json
)


def _allowed_backends_for(target_class: str) -> tuple[str, ...]:
    """Pick the allowed-backend list per target class.

    M-42 keeps this small. M-50+ widens (e.g., adds 'persistent' for
    MEGA dispatch). The list is the bounded surface the provider may
    pick from; choosing anything outside it is rejected by M-43 as
    ``unsupported_backend``.
    """
    tc = target_class.lower()
    if "cuda" in tc or "gpu" in tc:
        # Triton template is the proven GPU path (Phase B M-19/M-20).
        return ("triton",)
    # Default — host_cpu and other CPU-class targets.
    return ("c_reference",)


# --------------------------------------------------------------------------- #
# Public emitter
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class KernelCodegenResult:
    out_dir: Path
    request_path: Path
    artifact_dir: Path
    request_id: str
    request_kind: str
    overall: str  # "pass" | "skipped"


def build_kernel_codegen_request(run_dir: Path) -> KernelCodegenRequest:
    """Build the M-42 request from on-disk recipe-planning + M-40
    materialised-contract artifacts.

    Pure function — reads only on-disk state. The wrapper
    ``run_kernel_codegen_request`` is the pipeline-stage entry point
    that calls this and writes the artifact + summary.
    """
    run_dir = Path(run_dir).resolve()
    rp = run_dir / "03_recipe_planning"

    sel = _read_json(rp / "candidate_selection.json")
    summary = _read_json_or_none(rp / "recipe_summary.json") or {}

    candidate_kind = sel.get("candidate_kind", "")
    candidate_id = sel.get("selected_candidate_id", "") or ""
    region_id = sel.get("region_id", "") or ""
    target_id = (
        summary.get("target_id") or sel.get("target_id") or "host_cpu"
    )
    task_id = _task_id(candidate_id=candidate_id, region_id=region_id)

    # Empty contract-paths struct — populated below for set_tile_params.
    contract_paths = _ContractPaths(full="", kernel_facing="")

    # M-40 contract files for the same selected candidate. Read the
    # M-40 summary to find the contract_hash + paths, NOT the global
    # filesystem (avoids races on rerun + scope creep).
    mat_summary = _read_json_or_none(
        run_dir / "04_kernel_codegen" / "contract_materialization_summary.json"
    )
    materialized_row = None
    if mat_summary:
        for row in mat_summary.get("rows", []) or []:
            if row.get("region_id") == region_id and row.get("candidate_id") == candidate_id:
                materialized_row = row
                break

    # Non-set_tile_params: emit a typed not_applicable request. M-40
    # would also have emitted not_applicable; we cross-reference.
    if candidate_kind != "set_tile_params":
        reason = _NOT_APPLICABLE_REASONS.get(
            candidate_kind,
            f"M-42 supports only set_tile_params; got candidate_kind="
            f"{candidate_kind!r}",
        )
        return KernelCodegenRequest(
            schema_version=_SCHEMA_VERSION,
            task_id=task_id,
            generated_at_utc=_utcnow(),
            request_kind="not_applicable",
            region_id=region_id,
            candidate_id=candidate_id,
            recipe_op_id="recipe_0000",
            contract_hash="",
            contract_paths=contract_paths,
            allowed_backends=(),
            required_outputs=(),
            forbidden=(),
            artifact_dir=str(
                Path("04_kernel_codegen") / "artifacts" / task_id
            ),
            not_applicable_reason=reason,
        )

    # set_tile_params with materialized contract.
    if materialized_row is None or materialized_row.get("status") != "materialized":
        # M-40 should have materialised this; if it didn't, surface
        # the gap as an error-typed request (this is a real bug, not
        # a not_applicable case).
        return KernelCodegenRequest(
            schema_version=_SCHEMA_VERSION,
            task_id=task_id,
            generated_at_utc=_utcnow(),
            request_kind="not_applicable",
            region_id=region_id,
            candidate_id=candidate_id,
            recipe_op_id="recipe_0000",
            contract_hash="",
            contract_paths=contract_paths,
            allowed_backends=(),
            required_outputs=(),
            forbidden=(),
            artifact_dir=str(
                Path("04_kernel_codegen") / "artifacts" / task_id
            ),
            not_applicable_reason=(
                "M-40 contract materialization did not produce a "
                f"contract for region={region_id!r} "
                f"candidate={candidate_id!r}; M-42 cannot emit a "
                "kernel_codegen request without a materialised contract."
            ),
        )

    contract_hash = str(materialized_row["contract_hash"])
    contract_paths = _ContractPaths(
        full=str(materialized_row["contract_path"]),
        kernel_facing=str(materialized_row["kernel_facing_path"]),
    )

    # Resolve target_class from the materialised contract (more
    # authoritative than target_id alone — for example, a synthetic
    # cuda_sm75 target_class on a host_cpu run).
    target_class = target_id
    full_contract_path = run_dir / contract_paths.full
    if full_contract_path.exists():
        body = _read_json(full_contract_path)
        target_class = (
            (body.get("orchestration") or {})
            .get("execution", {})
            .get("hardware", {})
            .get("target_name") or target_id
        )

    return KernelCodegenRequest(
        schema_version=_SCHEMA_VERSION,
        task_id=task_id,
        generated_at_utc=_utcnow(),
        request_kind="kernel_codegen",
        region_id=region_id,
        candidate_id=candidate_id,
        recipe_op_id="recipe_0000",
        contract_hash=contract_hash,
        contract_paths=contract_paths,
        allowed_backends=_allowed_backends_for(str(target_class)),
        required_outputs=_REQUIRED_OUTPUTS,
        forbidden=_FORBIDDEN_MUTATIONS,
        artifact_dir=str(
            Path("04_kernel_codegen") / "artifacts" / task_id
        ),
    )


# --------------------------------------------------------------------------- #
# M-55 — registry resolution
# --------------------------------------------------------------------------- #


_REGISTRY_RESOLUTION_SCHEMA = "registry_resolution_v1"


def _emit_registry_resolution(
    *,
    run_dir: Path,
    request: KernelCodegenRequest,
) -> Path:
    """Write ``04_kernel_codegen/registry_resolution.json`` listing the
    static-metadata applicability of every registered kernel provider.

    M-55 — purely informational. The file records, byte-deterministically,
    which entry-point providers *could* bid on the materialized contract
    for this task. M-57's auction will consume this list; until then,
    today's Claude-Code subagent path runs unchanged via M-43.

    The file is also written for ``not_applicable`` requests with an
    empty applicable list — readers can rely on the file existing
    whenever the M-42 stage runs.
    """
    out_dir = run_dir / "04_kernel_codegen"
    out_dir.mkdir(parents=True, exist_ok=True)
    resolution_path = out_dir / "registry_resolution.json"

    applicable_rows: list[dict[str, Any]] = []
    fallback_used = True
    error: str | None = None

    if request.request_kind == "kernel_codegen" and request.contract_hash:
        try:
            from compgen.kernels.contract_v3 import KernelContractV3
            from compgen.kernels.registry import default_registry

            # Reconstruct the V3 contract from the materialized JSON the
            # request points at — same projection M-43/M-44 use.
            full_path = run_dir / request.contract_paths.full
            if full_path.exists():
                from compgen.graph_compilation.kernel_codegen_response import (
                    _reconstruct_contract_from_dict,
                )

                body = _read_json(full_path)
                contract_v3 = _reconstruct_contract_from_dict(body)
                if isinstance(contract_v3, KernelContractV3):
                    reg = default_registry()
                    applicability = reg.applicable(contract_v3)
                    applicable_rows = [r.to_dict() for r in applicability]
                    if any(r["applicable"] for r in applicable_rows):
                        fallback_used = False
        except Exception as exc:  # noqa: BLE001
            # Fall through cleanly; the registry log is informational.
            error = f"{type(exc).__name__}: {exc}"

    payload: dict[str, Any] = {
        "schema_version": _REGISTRY_RESOLUTION_SCHEMA,
        "generated_at_utc": _utcnow(),
        "task_id": request.task_id,
        "region_id": request.region_id,
        "candidate_id": request.candidate_id,
        "contract_hash": request.contract_hash,
        "request_kind": request.request_kind,
        "providers_considered": applicable_rows,
        "applicable_provider_names": [
            r["provider_name"] for r in applicable_rows if r["applicable"]
        ],
        "fallback_used": fallback_used,
        "fallback_path": "claude_code_subagent_via_m43" if fallback_used else "",
    }
    if error is not None:
        payload["resolution_error"] = error

    resolution_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return resolution_path


def run_kernel_codegen_request(run_dir: Path) -> KernelCodegenResult:
    """Pipeline-stage entry point. Build the request, write it to
    ``04_kernel_codegen/requests/<task_id>.request.json``, create the
    sandboxed ``artifact_dir``, and emit the summary."""
    run_dir = Path(run_dir).resolve()
    request = build_kernel_codegen_request(run_dir)

    out_dir = run_dir / "04_kernel_codegen"
    requests_dir = out_dir / "requests"
    requests_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = run_dir / request.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)

    request_path = requests_dir / f"{request.task_id}.request.json"
    request_path.write_text(
        json.dumps(request.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # M-55 — emit registry resolution alongside the request.
    _emit_registry_resolution(run_dir=run_dir, request=request)

    summary_path = out_dir / "kernel_codegen_summary.json"
    summary = {
        "schema_version": "kernel_codegen_summary_v1",
        "generated_at_utc": _utcnow(),
        "requests": [
            {
                "task_id": request.task_id,
                "request_kind": request.request_kind,
                "region_id": request.region_id,
                "candidate_id": request.candidate_id,
                "contract_hash": request.contract_hash,
                "contract_paths": request.contract_paths.to_dict(),
                "allowed_backends": list(request.allowed_backends),
                "artifact_dir": request.artifact_dir,
                "not_applicable_reason": request.not_applicable_reason,
            }
        ],
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return KernelCodegenResult(
        out_dir=out_dir,
        request_path=request_path,
        artifact_dir=artifact_dir,
        request_id=request.task_id,
        request_kind=request.request_kind,
        overall="pass" if request.request_kind == "kernel_codegen" else "skipped",
    )
