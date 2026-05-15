"""Materialize KernelContractV3 from selected Recipe IR decisions.

Section 21 /. Pipeline-stage wrapper around
:meth:`compgen.kernels.contract_v3.KernelContractV3.from_recipe`. Reads
on-disk artifacts (``candidate_selection.json``, the region dossier,
the target YAML, the recipe-gate verdict) and emits two artifacts per
selected kernel-bearing region:

- ``04_kernel_codegen/contracts/<region_id>.<contract_hash>.json`` —
  the full materialized contract.
- ``04_kernel_codegen/views/<region_id>.kernel_facing.json`` — the
  ``kernel_facing`` projection ONLY. hand this to the
  Claude Code subagent as the bounded surface.

Plus a per-run summary at
``04_kernel_codegen/contract_materialization_summary.json``.

Non-``set_tile_params`` candidates (today: fusion, kernel-contract
creation) emit a typed ``not_applicable`` row in the summary instead
of a contract file. widens the supported kinds.

This module is intentionally thin: it's a reader + an adapter + a
serialiser. The semantic work (mapping fields) lives in
``KernelContractV3.from_recipe``; the canonical hash lives in
``compgen.promotion.contract_hash.hash_contract``. Phase C will
unify all callers on the latter.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from compgen.kernels.contract_v3 import (
    KernelContractV3,
    KernelFacingView,
    LayoutKind,
    MemoryTier,
)
from compgen.promotion.contract_hash import hash_contract


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ContractMaterializationRow:
    region_id: str
    candidate_id: str
    candidate_kind: str
    status: str  # "materialized" | "not_applicable" | "error"
    contract_hash: str = ""
    contract_path: str = ""  # relative to run_dir
    kernel_facing_path: str = ""
    not_applicable_reason: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "candidate_id": self.candidate_id,
            "candidate_kind": self.candidate_kind,
            "status": self.status,
            "contract_hash": self.contract_hash,
            "contract_path": self.contract_path,
            "kernel_facing_path": self.kernel_facing_path,
            "not_applicable_reason": self.not_applicable_reason,
            "error": self.error,
        }


@dataclass(frozen=True)
class ContractMaterializationResult:
    out_dir: Path
    summary_path: Path
    rows: tuple[ContractMaterializationRow, ...] = ()
    overall: str = "pass"  # "pass" if at least one materialized; else "skipped"


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


def _read_yaml_or_none(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        import yaml  # type: ignore[import-untyped]
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 — degrade with empty profile
        return {}


def _resolve_region_dossier(run_dir: Path, region_id: str) -> Path | None:
    """Locate the region's dossier JSON. Mirrors the lookup logic in
    's emitter so the same precedence applies everywhere."""
    rd_dir = run_dir / "02_graph_analysis" / "region_dossiers"
    if rd_dir.is_dir():
        exact = rd_dir / f"{region_id}.json"
        if exact.exists():
            return exact
        prefix_matches = sorted(rd_dir.glob(f"{region_id}__*.json"))
        if prefix_matches:
            return prefix_matches[0]
    legacy = run_dir / "02_graph_analysis" / f"region_dossier__{region_id}.json"
    if legacy.exists():
        return legacy
    return None


def _resolve_target_profile(run_dir: Path, target_id: str) -> Path | None:
    """Locate the target YAML. Tries:
    1. ``configs/targets/<target_id>.yaml`` under repo root.
    2. ``run_manifest.json::target.config_path`` if present.
    """
    repo_root = Path(__file__).resolve().parents[3]
    cfg = repo_root / "configs" / "targets" / f"{target_id}.yaml"
    if cfg.exists():
        return cfg
    manifest = run_dir / "run_manifest.json"
    if manifest.exists():
        body = _read_json(manifest)
        cfg_path = (body.get("target") or {}).get("config_path")
        if cfg_path:
            cfg_p = Path(cfg_path)
            if cfg_p.exists():
                return cfg_p
    return None


def _declared_refinement_for(
    run_dir: Path, candidate_id: str, fallback_kind: str | None = None,
) -> str:
    """Read the recipe-gate verdict for the declared refinement.

    's single_k_iter rule writes ``declared_refinement`` per
    checked recipe op into ``recipe_gate_verdict.json``. We look up
    by source_candidate.
    """
    verdict = _read_json_or_none(
        run_dir / "03_recipe_planning" / "recipe_gate_verdict.json"
    )
    if not verdict:
        return "unknown"
    for op in verdict.get("checked_recipe_ops") or []:
        if op.get("source_candidate") == candidate_id:
            return str(op.get("declared_refinement") or "unknown")
    return "unknown"


# --------------------------------------------------------------------------- #
# Serialization (KernelContractV3 → JSON-friendly dict)
# --------------------------------------------------------------------------- #


def contract_to_dict(c: KernelContractV3) -> dict[str, Any]:
    """Lossy-but-stable JSON serialization of a contract for on-disk
    storage. Round-trippable enough for inspection + audit.

    Note: this is NOT the canonical hash projection. The hash uses
    :func:`hash_contract` which canonicalises the ``kernel_facing()``
    view internally. This serialization is for human-readable + audit
    artifacts.
    """
    orch = c.orchestration
    exe = orch.execution
    return {
        "schema_version": "kernel_contract_v3_serialized_v1",
        "op_name": c.op_name,
        "archetype": c.archetype.value,
        "granularity": c.granularity.value,
        "contract_version": c.contract_version,
        "io": _io_to_dict(c.io),
        "orchestration": {
            "execution": {
                "hardware": {
                    "target_name": exe.hardware.target_name,
                    "vector_lanes": exe.hardware.vector_lanes,
                    "scratchpad_bytes": exe.hardware.scratchpad_bytes,
                    "register_bytes": exe.hardware.register_bytes,
                    "native_dtypes": list(exe.hardware.native_dtypes),
                    "peak_bandwidth_gbps": exe.hardware.peak_bandwidth_gbps,
                    # extended hardware envelope.
                    "codegen_hints": list(exe.hardware.codegen_hints),
                    "mma_shapes": {
                        k: list(v) for k, v in exe.hardware.mma_shapes.items()
                    },
                    "peak_compute_per_dtype": dict(exe.hardware.peak_compute_per_dtype),
                    "register_quota_per_thread": exe.hardware.register_quota_per_thread,
                    "max_concurrent_blocks": exe.hardware.max_concurrent_blocks,
                },
                "memory_budget_bytes": exe.memory_budget_bytes,
                "concurrency_unit": exe.concurrency_unit.value,
                "padding": exe.padding.value,
                "priority": exe.priority.value,
            } if exe is not None else None,
            "sync": {
                "event_decls": [
                    {"name": e.name, "scope": e.scope, "wait_count": e.wait_count}
                    for e in orch.sync.event_decls
                ],
                "wait_on": list(orch.sync.wait_on),
                "aliasing": [
                    {"input_idx": a.input_idx, "output_idx": a.output_idx}
                    for a in orch.sync.aliasing
                ],
                "blocking": orch.sync.blocking,
            },
            "memory": {
                "input_tiers": [t.value for t in orch.memory.input_tiers],
                "output_tiers": [t.value for t in orch.memory.output_tiers],
                "in_place_safe": orch.memory.in_place_safe,
                "lifetimes": [
                    {"output_idx": l.output_idx, "live_after": l.live_after}
                    for l in orch.memory.lifetimes
                ],
            },
            "fusion": {
                "is_boundary": orch.fusion.is_boundary,
                "fusable_with": list(orch.fusion.fusable_with),
                "prefer_inline_into": orch.fusion.prefer_inline_into,
            },
            "dispatch": {
                "model": orch.dispatch.model.value,
                "max_concurrent_invocations": orch.dispatch.max_concurrent_invocations,
                "retry_on_recoverable_error": orch.dispatch.retry_on_recoverable_error,
            },
            "observability": {
                "emit_dispatch_event": orch.observability.emit_dispatch_event,
                "emit_completion_event": orch.observability.emit_completion_event,
                "cost_emit_period": orch.observability.cost_emit_period,
            },
        },
        "selection": {
            "providers": [
                {"name": p.name, "weight": p.weight, "rationale": p.rationale}
                for p in c.selection.providers
            ],
        },
        # typed pre/post-condition predicates.
        "preconditions": [
            p.to_dict() for p in (c.preconditions or ())
        ],
        "postconditions": [
            p.to_dict() for p in (c.postconditions or ())
        ],
        # forward-compatible refinement slot.
        "optional_v3_1_fields": dict(c.optional_v3_1_fields or {}),
        "metadata": dict(c.metadata),
    }


def _io_to_dict(io: Any) -> dict[str, Any]:
    def _tensor_io(t: Any) -> dict[str, Any]:
        return {
            "name": t.name,
            "shape": {
                "dims": list(t.shape.dims),
                "max_dims": list(t.shape.max_dims) if t.shape.max_dims else None,
                "divisibility": list(t.shape.divisibility) if t.shape.divisibility else None,
            },
            "dtype_class": list(t.dtype_class),
            "layout": t.layout.value,
            "alignment_bytes": t.alignment_bytes,
            "broadcast_pattern": t.broadcast_pattern,
        }

    return {
        "inputs": [_tensor_io(t) for t in io.inputs],
        "outputs": [_tensor_io(t) for t in io.outputs],
        "attributes": [
            {"name": a.name, "value": a.value} for a in io.attributes
        ],
        "numerics": {
            "accumulator_dtype": io.numerics.accumulator_dtype,
            "fast_math": io.numerics.fast_math,
            "max_relative_error": io.numerics.max_relative_error,
            "deterministic": io.numerics.deterministic,
        },
    }


def kernel_facing_to_dict(view: KernelFacingView) -> dict[str, Any]:
    """JSON-friendly projection of the kernel-facing view ONLY.

    This is the BOUNDED surface a kernel-codegen provider may read
    (+ hands this file to the spawned Claude Code agent). It MUST
    exclude every field present only in the compiler-only view: no
    ``wait_on``, no ``blocking``, no ``lifetimes``, no ``fusion``, no
    ``observability``, no ``dispatch.max_concurrent_invocations``, no
    ``selection`` providers (those are scheduler hints), no
    ``metadata`` (compiler bookkeeping).

    A negative control in the tests asserts none of these
    field names appear in the serialized output.
    """
    exe = view.execution
    return {
        "schema_version": "kernel_facing_view_v1",
        "op_name": view.op_name,
        "archetype": view.archetype.value,
        "granularity": view.granularity.value,
        "io": _io_to_dict(view.io),
        "execution": {
            "hardware": {
                "target_name": exe.hardware.target_name,
                "vector_lanes": exe.hardware.vector_lanes,
                "scratchpad_bytes": exe.hardware.scratchpad_bytes,
                "register_bytes": exe.hardware.register_bytes,
                "native_dtypes": list(exe.hardware.native_dtypes),
                "peak_bandwidth_gbps": exe.hardware.peak_bandwidth_gbps,
            },
            "memory_budget_bytes": exe.memory_budget_bytes,
            "concurrency_unit": exe.concurrency_unit.value,
            "padding": exe.padding.value,
            "priority": exe.priority.value,
        } if exe is not None else None,
        "memory_residency": {
            "input_tiers": [t.value for t in view.memory_residency.input_tiers],
            "output_tiers": [t.value for t in view.memory_residency.output_tiers],
            "in_place_safe": view.memory_residency.in_place_safe,
            "aliasing": [
                {"input_idx": a.input_idx, "output_idx": a.output_idx}
                for a in view.memory_residency.aliasing
            ],
        },
        "event_decls": [
            {"name": e.name, "scope": e.scope, "wait_count": e.wait_count}
            for e in view.event_decls
        ],
        "dispatch_model": view.dispatch_model.value,
    }


# --------------------------------------------------------------------------- #
# In-memory materialization (used by every contract_hash caller)
# --------------------------------------------------------------------------- #


def _dispatch_mode_override_for(
    *, run_dir: Path, region_id: str,
) -> str | None:
    """scan candidate_selection for a sibling SetDispatchMode op
    that sets the dispatch mode for this region. Returns the mode
    string ("sync"|"async"|"persistent"|"inline") or None when no
    override was selected.

    The agent records SetDispatchMode picks under the same
    ``recipe_delta`` field that records SetTileParams. We walk the
    delta looking for ``op == "SetDispatchMode"`` entries scoped to
    ``region_id``.
    """
    sel_path = run_dir / "03_recipe_planning" / "candidate_selection.json"
    body = _read_json_or_none(sel_path)
    if not body:
        return None
    for op in body.get("recipe_delta") or []:
        if (
            op.get("op") == "SetDispatchMode"
            and op.get("region") == region_id
        ):
            return str(op.get("mode") or "")
    # Also look at sibling selections — the agent may have committed
    # multiple decisions across runs. , we only honour the
    # in-current-selection delta.
    return None


def materialize_contract_from_run_dir(
    *,
    run_dir: Path,
    candidate_selection: dict[str, Any],
    region_id: str,
    target_id: str,
) -> KernelContractV3 | None:
    """Materialize a ``KernelContractV3`` in memory from on-disk
    artifacts, without writing the contract back to disk.

    This is the single helper every ``hash_contract`` caller routes
    through, so promotion-write, promotion-read, 's request emit,
    and any future hashing site all derive from byte-identical
    contract content.

    Returns ``None`` for unsupported candidate kinds (today: anything
    other than ``set_tile_params``). Callers fall back to an empty
    string for the contract_hash in that case — region-signature-only
    retrieval still works.

    when a sibling SetDispatchMode op is recorded for the same
    region, the contract's dispatch.model reflects the agent's
    chosen mode rather than the SYNC default.
    """
    candidate_kind = candidate_selection.get("candidate_kind", "")
    if candidate_kind != "set_tile_params":
        return None
    dossier_path = _resolve_region_dossier(run_dir, region_id)
    if dossier_path is None:
        return None
    region_dossier = _read_json(dossier_path)
    target_yaml_path = _resolve_target_profile(run_dir, target_id)
    target_profile = (
        _read_yaml_or_none(target_yaml_path) if target_yaml_path else {}
    ) or {}
    declared_refinement = _declared_refinement_for(
        run_dir, candidate_selection.get("selected_candidate_id", "") or "",
    )
    dispatch_mode_override = _dispatch_mode_override_for(
        run_dir=run_dir, region_id=region_id,
    )
    try:
        return KernelContractV3.from_recipe(
            candidate_selection=candidate_selection,
            region_dossier=region_dossier,
            target_profile=target_profile,
            declared_refinement=declared_refinement,
            dispatch_mode_override=dispatch_mode_override,
        )
    except ValueError:
        return None


def hash_contract_from_run_dir(
    *,
    run_dir: Path,
    candidate_selection: dict[str, Any],
    region_id: str,
    target_id: str,
) -> str:
    """Convenience: materialize then hash. Returns empty string when
    the contract cannot be materialized (unsupported kind, missing
    dossier, etc.)."""
    contract = materialize_contract_from_run_dir(
        run_dir=run_dir,
        candidate_selection=candidate_selection,
        region_id=region_id,
        target_id=target_id,
    )
    if contract is None:
        return ""
    return hash_contract(contract)


# --------------------------------------------------------------------------- #
# Public emitter (writes to disk)
# --------------------------------------------------------------------------- #


def materialize_contract_for_run(
    run_dir: Path,
) -> ContractMaterializationResult:
    """Read recipe-planning artifacts; materialize the kernel contract
    for the selected candidate; write the contract + kernel-facing
    projection + summary to ``04_kernel_codegen/``.

    Idempotent: re-running on the same run dir produces byte-identical
    output (modulo ``generated_at_utc`` in the summary)."""
    run_dir = Path(run_dir).resolve()
    rp = run_dir / "03_recipe_planning"
    out_dir = run_dir / "04_kernel_codegen"
    contracts_dir = out_dir / "contracts"
    views_dir = out_dir / "views"
    contracts_dir.mkdir(parents=True, exist_ok=True)
    views_dir.mkdir(parents=True, exist_ok=True)

    sel = _read_json(rp / "candidate_selection.json")
    summary = _read_json_or_none(rp / "recipe_summary.json") or {}
    candidate_kind = sel.get("candidate_kind", "")
    candidate_id = sel.get("selected_candidate_id", "") or ""
    region_id = sel.get("region_id", "") or ""
    target_id = (
        summary.get("target_id")
        or sel.get("target_id")
        or "host_cpu"
    )

    rows: list[ContractMaterializationRow] = []

    if candidate_kind == "fuse_producer_consumer":
        #  closure (Phase D Batch B): materialise a fusion
        # contract for pointwise→pointwise fusions. The producer +
        # consumer dossier paths come in via candidate_selection.evidence;
        # we synthesize a POINTWISE contract whose IO is the producer's
        # inputs + the consumer's output.
        try:
            dossier_path = _resolve_region_dossier(run_dir, region_id)
            region_dossier = _read_json(dossier_path) if dossier_path else {}
            target_yaml_path = _resolve_target_profile(run_dir, target_id)
            target_profile = (
                _read_yaml_or_none(target_yaml_path) if target_yaml_path else {}
            ) or {}
            evidence = sel.get("evidence") or {}
            producer_path = evidence.get("producer_dossier") or ""
            consumer_path = evidence.get("consumer_dossier") or ""
            producer_dossier: dict[str, Any] = {}
            consumer_dossier: dict[str, Any] = {}
            if producer_path:
                pd = run_dir / producer_path
                if pd.exists():
                    producer_dossier = _read_json(pd)
            if consumer_path:
                cd = run_dir / consumer_path
                if cd.exists():
                    consumer_dossier = _read_json(cd)

            contract = KernelContractV3.from_recipe_fusion(
                candidate_selection=sel,
                producer_dossier=producer_dossier,
                consumer_dossier=consumer_dossier,
                region_dossier=region_dossier,
                target_profile=target_profile,
            )
            ch = hash_contract(contract)
            fusion_region_id = sel.get("label") or region_id
            contract_path = contracts_dir / f"{fusion_region_id}.{ch}.json"
            view_path = views_dir / f"{fusion_region_id}.kernel_facing.json"
            contract_path.write_text(
                json.dumps(contract_to_dict(contract), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            view_path.write_text(
                json.dumps(kernel_facing_to_dict(contract.kernel_facing()),
                           indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            rows.append(ContractMaterializationRow(
                region_id=region_id,
                candidate_id=candidate_id,
                candidate_kind=candidate_kind,
                status="materialized",
                contract_hash=ch,
                contract_path=str(contract_path.relative_to(run_dir)),
                kernel_facing_path=str(view_path.relative_to(run_dir)),
            ))
        except Exception as exc:  # noqa: BLE001
            rows.append(ContractMaterializationRow(
                region_id=region_id,
                candidate_id=candidate_id,
                candidate_kind=candidate_kind,
                status="error",
                not_applicable_reason=(
                    f"M-40 fusion materialization failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            ))
    elif candidate_kind != "set_tile_params":
        rows.append(ContractMaterializationRow(
            region_id=region_id,
            candidate_id=candidate_id,
            candidate_kind=candidate_kind,
            status="not_applicable",
            not_applicable_reason=(
                f"M-40 supports candidate_kind in "
                f"['set_tile_params', 'fuse_producer_consumer']; "
                f"got {candidate_kind!r}."
            ),
        ))
    else:
        try:
            dossier_path = _resolve_region_dossier(run_dir, region_id)
            region_dossier = _read_json(dossier_path) if dossier_path else {}
            target_yaml_path = _resolve_target_profile(run_dir, target_id)
            target_profile = (
                _read_yaml_or_none(target_yaml_path) if target_yaml_path else {}
            ) or {}
            declared_refinement = _declared_refinement_for(run_dir, candidate_id)

            # : COMPGEN_SHAPE_POLICY=class makes from_recipe
            # substitute concrete dims with None so the canonical
            # hash falls all the way to dynamic.
            import os as _os

            shape_policy = _os.environ.get("COMPGEN_SHAPE_POLICY", "concrete")
            if shape_policy not in ("concrete", "class"):
                shape_policy = "concrete"
            contract = KernelContractV3.from_recipe(
                candidate_selection=sel,
                region_dossier=region_dossier,
                target_profile=target_profile,
                declared_refinement=declared_refinement,
                shape_policy=shape_policy,
            )
            ch = hash_contract(contract)
            contract_path = contracts_dir / f"{region_id}.{ch}.json"
            view_path = views_dir / f"{region_id}.kernel_facing.json"

            contract_path.write_text(
                json.dumps(contract_to_dict(contract), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            view_path.write_text(
                json.dumps(kernel_facing_to_dict(contract.kernel_facing()),
                           indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            rows.append(ContractMaterializationRow(
                region_id=region_id,
                candidate_id=candidate_id,
                candidate_kind=candidate_kind,
                status="materialized",
                contract_hash=ch,
                contract_path=str(contract_path.relative_to(run_dir)),
                kernel_facing_path=str(view_path.relative_to(run_dir)),
            ))
        except Exception as exc:  # noqa: BLE001
            rows.append(ContractMaterializationRow(
                region_id=region_id,
                candidate_id=candidate_id,
                candidate_kind=candidate_kind,
                status="error",
                error=f"{type(exc).__name__}: {exc}",
            ))

    summary_path = out_dir / "contract_materialization_summary.json"
    summary_obj = {
        "schema_version": "contract_materialization_summary_v1",
        "generated_at_utc": _utcnow(),
        "rows": [r.to_dict() for r in rows],
    }
    summary_path.write_text(
        json.dumps(summary_obj, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    overall = "pass" if any(r.status == "materialized" for r in rows) else "skipped"
    return ContractMaterializationResult(
        out_dir=out_dir,
        summary_path=summary_path,
        rows=tuple(rows),
        overall=overall,
    )
