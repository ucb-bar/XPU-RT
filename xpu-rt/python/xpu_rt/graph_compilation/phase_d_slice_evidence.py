"""M-65 — Phase D vertical-slice evidence emitter.

The plan calls for two cross-cutting stress slices:

* **Slice 2** — ``proxy_vla`` on ``host_cpu`` (fusion path).
* **Slice 3** — ``merlin_mlp_wide`` on ``cuda_sm75`` via Triton.

Both slices exercise the M-55..M-64 substrate end-to-end. M-65 is
documentation + evidence: it walks an existing ``run_dir``, summarises
what each Phase D milestone produced, and emits a slice-specific
evidence JSON. The honest output stays honest — when a slice can't
run (e.g. proxy_vla's selected candidate is a fusion, which M-42
deliberately routes to ``not_applicable``), the evidence file
records the gap rather than papering over it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


_SLICE_SCHEMA = "phase_d_slice_evidence_v1"


def _utcnow() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json_or_none(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


@dataclass(frozen=True)
class SliceEvidence:
    schema_version: str
    slice_id: str
    slice_name: str
    model: str
    target: str
    overall: str  # "green" | "honest_gap" | "deferred"
    overall_reason: str
    auction_summary: dict[str, Any]
    coverage_summary: dict[str, Any]
    specialization_summary: dict[str, Any]
    bindings_summary: dict[str, Any]
    contract_versioning_summary: dict[str, Any]
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "slice_id": self.slice_id,
            "slice_name": self.slice_name,
            "model": self.model,
            "target": self.target,
            "overall": self.overall,
            "overall_reason": self.overall_reason,
            "auction_summary": dict(self.auction_summary),
            "coverage_summary": dict(self.coverage_summary),
            "specialization_summary": dict(self.specialization_summary),
            "bindings_summary": dict(self.bindings_summary),
            "contract_versioning_summary": dict(self.contract_versioning_summary),
            "notes": self.notes,
            "generated_at_utc": _utcnow(),
        }


def _summarise_auction(run_dir: Path) -> dict[str, Any]:
    auction_root = run_dir / "04_kernel_codegen" / "auction"
    if not auction_root.exists():
        return {
            "ran": False,
            "reason": (
                "auction stage did not run; the M-42 request is "
                "request_kind='not_applicable' or the pipeline stopped "
                "before --stop-after kernel-auction"
            ),
        }
    reports = sorted(auction_root.glob("*/auction_report.json"))
    if not reports:
        return {"ran": False, "reason": "auction directory empty"}
    body = _read_json_or_none(reports[0]) or {}
    return {
        "ran": True,
        "overall": body.get("overall"),
        "mode": body.get("mode"),
        "winner_provider": body.get("winner_provider", ""),
        "n_bids": len(body.get("bids", []) or []),
        "n_fulfilled": len(body.get("fulfilled", []) or []),
        "n_verified": len(body.get("verified", []) or []),
        "contract_hash": body.get("contract_hash", ""),
    }


def _summarise_coverage(run_dir: Path) -> dict[str, Any]:
    body = _read_json_or_none(
        run_dir / "04_kernel_codegen" / "coverage_report.json"
    )
    if body is None:
        return {"ran": False, "reason": "coverage_report.json missing"}
    return {
        "ran": True,
        "n_groups": body.get("summary", {}).get("n_groups", 0),
        "n_groups_with_cert": body.get("summary", {}).get("n_groups_with_cert", 0),
        "max_group_size": body.get("summary", {}).get("max_group_size", 0),
        "coverage_inflation_total": body.get("coverage_inflation_total", 0),
    }


def _summarise_specialization(run_dir: Path) -> dict[str, Any]:
    body = _read_json_or_none(
        run_dir / "04_kernel_codegen" / "specialization_report.json"
    )
    if body is None:
        return {"ran": False, "reason": "specialization_report.json missing"}
    summary = body.get("summary") or {}
    return {
        "ran": True,
        "n_regions_total": summary.get("n_regions_total", 0),
        "n_covered": summary.get("n_covered", 0),
        "n_uncovered": summary.get("n_uncovered", 0),
        "n_specialization_targets": len(
            body.get("recommended_specialization_targets") or []
        ),
    }


def _summarise_bindings(run_dir: Path) -> dict[str, Any]:
    body = _read_json_or_none(
        run_dir / "05_execution_plan" / "region_kernel_bindings.json"
    )
    if body is None:
        return {"ran": False, "reason": "region_kernel_bindings.json missing"}
    return {
        "ran": True,
        "bound_count": int(body.get("bound_count", 0) or 0),
        "unbound_count": int(body.get("unbound_count", 0) or 0),
        "coverage_inflated_count": int(body.get("coverage_inflated_count") or 0),
        "n_bindings": len(body.get("bindings") or []),
    }


def _summarise_contract_versioning(run_dir: Path) -> dict[str, Any]:
    """Re-derive every cert's canonical hash post-migration; fail-fast
    summary so the slice evidence records whether M-64's invariant
    holds for the slice-specific certs."""
    cert_dir = run_dir / "04_kernel_codegen" / "certificates"
    if not cert_dir.exists():
        return {"ran": False, "reason": "no certificates directory"}
    cert_paths = sorted(cert_dir.glob("*.json"))
    if not cert_paths:
        return {"ran": False, "reason": "no certificates"}

    try:
        from compgen.graph_compilation.kernel_codegen_response import (
            _reconstruct_contract_from_dict,
        )
        from compgen.kernels.contract_migration import (
            migrate_contract_body_v3_to_v3_1,
        )
        from compgen.promotion.contract_hash import canonical_contract_hash
    except Exception as exc:  # noqa: BLE001
        return {"ran": False, "reason": f"import error: {exc}"}

    drifted = 0
    checked = 0
    for cp in cert_paths:
        body = _read_json_or_none(cp)
        if body is None:
            continue
        recorded = str(body.get("canonical_contract_hash") or "")
        contract_rel = body.get("contract_path") or ""
        if not recorded or not contract_rel:
            continue
        contract_full = run_dir / contract_rel
        contract_body = _read_json_or_none(contract_full)
        if contract_body is None:
            continue
        try:
            migrated = migrate_contract_body_v3_to_v3_1(contract_body)
            contract = _reconstruct_contract_from_dict(migrated)
            re_derived = canonical_contract_hash(contract)
        except Exception:  # noqa: BLE001
            continue
        checked += 1
        if recorded != re_derived:
            drifted += 1
    return {
        "ran": True,
        "n_certs": len(cert_paths),
        "n_checked": checked,
        "n_drifted": drifted,
    }


def emit_slice_evidence(
    *,
    run_dir: Path,
    slice_id: str,
    slice_name: str,
    model: str,
    target: str,
    overall: str = "green",
    overall_reason: str = "",
    notes: str = "",
) -> Path:
    """Write the slice-evidence JSON under the run directory.

    Path: ``<run_dir>/phase_d_slice_<slice_id>_evidence.json``.

    The four-summary block (auction / coverage / specialization /
    bindings) is computed by walking the existing on-disk artifacts;
    no pipeline stage is re-invoked.
    """
    run_dir = Path(run_dir).resolve()
    evidence = SliceEvidence(
        schema_version=_SLICE_SCHEMA,
        slice_id=slice_id,
        slice_name=slice_name,
        model=model,
        target=target,
        overall=overall,
        overall_reason=overall_reason,
        auction_summary=_summarise_auction(run_dir),
        coverage_summary=_summarise_coverage(run_dir),
        specialization_summary=_summarise_specialization(run_dir),
        bindings_summary=_summarise_bindings(run_dir),
        contract_versioning_summary=_summarise_contract_versioning(run_dir),
        notes=notes,
    )
    out_path = run_dir / f"phase_d_slice_{slice_id}_evidence.json"
    out_path.write_text(
        json.dumps(evidence.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out_path


def emit_deferred_slice_evidence(
    *,
    out_dir: Path,
    slice_id: str,
    slice_name: str,
    model: str,
    target: str,
    deferred_reason: str,
) -> Path:
    """Write a slice-evidence JSON for a slice that couldn't run.

    Used by Slice 3 (cuda_sm75) on hosts without a CUDA target YAML
    or a CUDA-capable runtime. The evidence honestly records the
    deferral rather than producing a misleading green report.
    """
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    body = SliceEvidence(
        schema_version=_SLICE_SCHEMA,
        slice_id=slice_id,
        slice_name=slice_name,
        model=model,
        target=target,
        overall="deferred",
        overall_reason=deferred_reason,
        auction_summary={"ran": False, "reason": "slice deferred"},
        coverage_summary={"ran": False, "reason": "slice deferred"},
        specialization_summary={"ran": False, "reason": "slice deferred"},
        bindings_summary={"ran": False, "reason": "slice deferred"},
        contract_versioning_summary={"ran": False, "reason": "slice deferred"},
        notes=f"Slice deferred — re-run on a host with the prerequisites: {deferred_reason}",
    ).to_dict()
    path = out_dir / f"phase_d_slice_{slice_id}_evidence.json"
    path.write_text(
        json.dumps(body, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


__all__ = [
    "SliceEvidence",
    "emit_deferred_slice_evidence",
    "emit_slice_evidence",
]
