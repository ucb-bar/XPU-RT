"""Phase B → promotion bridge (M-26, write side).

Reads a completed graph_compilation run directory and writes a promoted
recipe to ``.compgen_cache/recipes/`` via the existing
:class:`compgen.promotion.promote.RecipePromoter` plus the
``memory.promotions`` SQLite index.

This module is the *seam* described in Section 19: before M-26, every
Phase B run dies in ``results/graph_compilation/<run>/`` and a second
run on the same model (let alone a *different* model with the same
region pattern) re-emits every candidate. After M-26, the run dir's
:file:`03_recipe_planning/candidate_selection.json`,
:file:`recipe.mlir`, and any present differential reports are folded
into a :class:`compgen.runtime.bundle.Bundle`, the bundle's
``verification_report.json`` is synthesised from Phase B evidence, and
a two-tier-keyed promoted recipe lands in the deterministic recipe
library.

Best-effort by contract — never raises. The caller (typically
``run.py`` after :func:`run_kernel_section_readiness`) wraps the call
in a ledger event and continues regardless of outcome. Returned typed
status:

- ``ok`` — recipe was promoted; ``recipe_path`` and ``key`` populated.
- ``not_eligible`` — no candidate selected, or required artifacts
  missing; ``reason`` describes which.
- ``error`` — unexpected exception; ``reason`` carries the message.

The bridge keeps the existing ``RecipeKey`` directory naming
(``target_hash_model_hash_objective_hash_vN``) — adding two more
underscore-separated fields would break the parser at
``cache.py:99-123``. The two M-26 dimensions
(``contract_hash`` and ``region_signature``) ride along inside the
recipe directory's ``promoted_recipe.json`` sidecar plus the
``memory.promotions`` SQLite index, which is what M-28 retrieval
queries against.
"""

from __future__ import annotations

import hashlib
import json
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from compgen.promotion.gates import GateEvaluation, evaluate_gate
from compgen.promotion.promote import (
    PromotedRecipe,
    PromotionResult,
    RecipeKey,
    RecipePromoter,
    write_promoted_recipe_sidecar,
)
from compgen.promotion.region_signature import (
    hash_region_signature,
    make_region_signature,
)
from compgen.runtime.bundle import Bundle

log = structlog.get_logger(__name__)


# Default library path mirrors ``RecipePromoter`` usage elsewhere
# (``.compgen_cache/recipes/`` is the gitignored deterministic
# library). Resolved against the repo root rather than cwd so the
# bridge writes to the same library regardless of where the pipeline
# was invoked from. Caught during M-30 real-workload validation:
# ``Path(".compgen_cache")`` was cwd-relative, which silently put
# different runs into different libraries when cwd drifted.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_LIBRARY_PATH = _REPO_ROOT / ".compgen_cache" / "recipes"


@dataclass(frozen=True)
class PromotionEmissionResult:
    """Typed outcome of a bridge invocation.

    Attributes:
        status: ``"ok"``, ``"not_eligible"``, or ``"error"``.
        reason: Human-readable explanation (always populated).
        recipe_path: On-disk recipe directory when ``status == "ok"``.
        key: :class:`RecipeKey` of the promoted recipe (with two-tier
            fields filled in) when ``status == "ok"``.
    """

    status: str
    reason: str = ""
    recipe_path: Path | None = None
    key: RecipeKey | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "recipe_path": str(self.recipe_path) if self.recipe_path else None,
            "key": self.key.key if self.key else None,
            "contract_hash": self.key.contract_hash if self.key else "",
            "region_signature": self.key.region_signature if self.key else "",
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _short_sha(value: str, *, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _resolve_region_dossier(run_dir: Path, region_id: str) -> Path | None:
    """Find ``02_graph_analysis/region_dossiers/<region_id>__<hash>.json``.

    Phase B writes region dossiers as ``<region_id>__<8-char-hash>.json``.
    The bridge's first cut assumed the hash-less form and silently
    fell back to ``unknown`` op_family when the file was absent —
    smoke-test on merlin_mlp_wide caught it.
    """
    if not region_id:
        return None
    base = run_dir / "02_graph_analysis" / "region_dossiers"
    if not base.is_dir():
        return None
    # Exact match first (legacy fixtures), then prefix match.
    exact = base / f"{region_id}.json"
    if exact.is_file():
        return exact
    matches = sorted(base.glob(f"{region_id}__*.json"))
    return matches[0] if matches else None


def _resolve_payload_path(run_dir: Path) -> Path | None:
    """Return the canonical payload.mlir path for this run.

    Phase B writes payload.mlir at one of two locations depending on
    capture mode: ``01_payload_lowering/export_program/payload.mlir``
    for clean torch.export captures, or
    ``01_payload_lowering/dynamo_partitions/partition_000/payload.mlir``
    for graph-broken models. Earlier runs wrote it to
    ``01_payload_lowering/payload.mlir``; we still check that location
    last for forward compatibility with older fixtures.
    """
    pl_dir = run_dir / "01_payload_lowering"
    candidates = [
        pl_dir / "export_program" / "payload.mlir",
        pl_dir / "payload.mlir",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    # Fall back to the first dynamo partition that has a payload.
    partitions_dir = pl_dir / "dynamo_partitions"
    if partitions_dir.is_dir():
        for partition in sorted(partitions_dir.iterdir()):
            cand = partition / "payload.mlir"
            if cand.is_file():
                return cand
    return None


def _read_payload_text(run_dir: Path) -> str | None:
    """Read the canonical Payload IR text for hashing into ``model_hash``."""
    payload_path = _resolve_payload_path(run_dir)
    if payload_path is None:
        return None
    return payload_path.read_text(encoding="utf-8")


def _gather_differential_outcomes(rp: Path) -> dict[str, Any]:
    """Read whichever Phase B differential reports exist.

    Returns a dict with optional sub-blocks (``post_lowering``,
    ``differential``, ``real_transform``, ``real_fusion``). Missing
    reports are simply absent from the dict. Path layout was
    corrected after the first real-workload smoke test — Phase B
    nests these under per-stage subdirs, not at the recipe-planning
    root, so each candidate lists multiple known-good locations.
    """
    outcomes: dict[str, Any] = {}
    candidates: dict[str, tuple[Path, ...]] = {
        "post_lowering": (
            rp / "post_lowering" / "post_lowering_verification_report.json",
            rp / "post_lowering_verification_report.json",  # legacy
        ),
        "differential": (
            rp / "differential_verification" / "differential_verification_report.json",
            rp / "differential_verification_report.json",  # legacy
        ),
        "real_transform": (
            rp / "real_verification" / "real_differential_report.json",
            rp / "real_transform_differential_report.json",  # legacy
        ),
        "real_fusion": (
            rp / "real_fusion_verification" / "real_fusion_differential_report.json",
            rp / "real_fusion_differential_report.json",  # legacy
        ),
    }
    for label, paths in candidates.items():
        for p in paths:
            report = _read_json(p)
            if report is not None:
                outcomes[label] = report
                break
    return outcomes


def _synthesize_verification_report(
    bundle_root: Path, outcomes: dict[str, Any]
) -> tuple[Path | None, str]:
    """Write a ``verification_report.json`` in the format the gate expects.

    The compgen-promotion gate at
    :func:`compgen.promotion.promote._inspect_verification` requires a
    JSON object with ``passed``, ``levels_run``, ``levels_passed``,
    ``details``. Phase B writes individual differential reports in
    different schemas; we fold them into the canonical shape.

    The synthesised report lands under ``04_promotion/`` rather than at
    ``run_dir`` root: that subdir is not covered by any earlier stage's
    ``output_hash``, so writing into it after the recipe stage record
    has been snapshotted does not break the R009 hash-chain (same
    pattern as M-10B / M-13 / M-22.1).

    Returns a (path, reason) pair. ``path`` is ``None`` when no Phase B
    evidence is present — in that case the caller should bail out
    rather than promote a bogus bundle.
    """
    if not outcomes:
        return None, "no Phase B differential evidence present"

    levels_run: list[str] = []
    levels_passed: list[str] = []
    details: dict[str, str] = {}

    # Structural — sourced from M-08 post-lowering verification.
    pl = outcomes.get("post_lowering")
    if pl is not None:
        levels_run.append("structural")
        status = pl.get("status", "fail")
        details["structural"] = f"post_lowering_verification_report.status={status!r}"
        if status == "pass":
            levels_passed.append("structural")

    # Differential — prefer M-09; fall back to M-12 then M-16.2.
    diff = outcomes.get("differential") or outcomes.get("real_transform") or outcomes.get("real_fusion")
    if diff is not None:
        levels_run.append("differential")
        # Different reports use different field names; check both.
        status = diff.get("status") or diff.get("overall") or "fail"
        details["differential"] = f"diff_status={status!r}"
        if status in {"pass", "tolerance_eps", "bit_equality"}:
            levels_passed.append("differential")

    passed = (
        len(levels_passed) > 0
        and set(levels_passed) >= {"structural", "differential"}
    )

    body = {
        "schema_version": "verification_report_v1",
        "synthesized_by": "graph_compilation.promotion_bridge",
        "passed": passed,
        "max_abs_error": None,
        "levels_run": sorted(set(levels_run)),
        "levels_passed": sorted(set(levels_passed)),
        "details": details,
    }
    promotion_dir = bundle_root / "04_promotion"
    promotion_dir.mkdir(parents=True, exist_ok=True)
    out_path = promotion_dir / "verification_report.json"
    out_path.write_text(
        json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not passed:
        missing = {"structural", "differential"} - set(levels_passed)
        return out_path, f"required levels did not pass: {sorted(missing)}"
    return out_path, "passed"


def derive_region_signature(
    *, run_dir: Path, region_id: str, target_id: str, kind: str
) -> tuple[str, dict[str, str]]:
    """Public M-28 entry point — see :func:`_derive_region_signature`."""
    return _derive_region_signature(
        run_dir=run_dir, region_id=region_id, target_id=target_id, kind=kind,
    )


def derive_contract_hash(
    *,
    candidate_selection: dict[str, Any],
    region_signature_fields: dict[str, str],
) -> str:
    """Synthesize the M-26 exact-kernel ``contract_hash``.

    Phase B does not currently persist :class:`KernelContractV3`
    objects to disk for every region (the lowering manifest reports
    ``kernel_contracts: 0`` for ``SetTileParams`` recipes), so the
    bridge constructs the kernel-identity hash directly from the
    candidate's recipe_delta plus the region's dtype/layout/shape
    facts and target_class. Two regions whose recipe_delta + facts
    canonicalise identically produce the same hash; the M-28
    retrieval can then surface a previously-compiled kernel as an
    exact-contract match without re-codegenning.

    The hash is stable across runs (same model + tile spec + target
    on the same machine produces the same key).
    """
    payload: dict[str, Any] = {
        "candidate_kind": candidate_selection.get("candidate_kind", ""),
        "recipe_delta": list(candidate_selection.get("recipe_delta") or []),
        "region": {
            "op_family": region_signature_fields.get("op_family", ""),
            "dtype": region_signature_fields.get("dtype", ""),
            "layout": region_signature_fields.get("layout", ""),
            "shape_class": region_signature_fields.get("shape_class", ""),
            "target_class": region_signature_fields.get("target_class", ""),
        },
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _derive_region_signature(
    *, run_dir: Path, region_id: str, target_id: str, kind: str
) -> tuple[str, dict[str, str]]:
    """Construct a region pattern signature from the M-10 dossier.

    Fields not derivable from the dossier degrade to ``"unknown"`` —
    M-26 ships with this degradation explicit. Future milestones
    (M-27 ops_provenance, M-28 retrieval) will tighten the signature
    once Recipe IR carries the full pattern.
    """
    dossier_path = _resolve_region_dossier(run_dir, region_id)
    dossier = _read_json(dossier_path) if dossier_path else {}
    dossier = dossier or {}

    # Prefer the dossier's ``kind`` over whatever the caller passed —
    # the write side passes the post-override op_family ("matmul"),
    # but the read side gets the *site* kind from llm_action_space
    # ("tiling", "fusion", ...). Without the override, write and
    # read hash to different signatures and retrieval misses.
    if isinstance(dossier.get("kind"), str) and dossier["kind"]:
        kind = dossier["kind"]

    # Dtype: pick the first dtype tag whose numerical_sensitivity entry
    # reports status=safe (or fall back to fp32).
    dtype = "unknown"
    ns = dossier.get("numerical_sensitivity", {})
    if isinstance(ns, dict):
        for cand in ("fp32", "fp16_accum", "fp8_e4m3"):
            entry = ns.get(cand)
            if isinstance(entry, dict) and entry.get("status") in {"safe", "risky"}:
                dtype = cand
                break

    # Dims: best-effort from working_set_curve's first entry.
    dims: list[Any] = []
    ws = dossier.get("working_set_curve", [])
    if isinstance(ws, list) and ws:
        first = ws[0]
        if isinstance(first, dict):
            for key in ("input_dims", "shape", "dims"):
                val = first.get(key)
                if isinstance(val, list):
                    dims = list(val)
                    break

    sig = make_region_signature(
        op_family=kind or "unknown",
        dtype=dtype,
        layout="row_major",  # Phase B does not surface layout yet.
        dims=dims,
        target_class=target_id or "unknown",
    )
    return hash_region_signature(sig), sig.to_dict()


def _derive_applies_when(dossier: dict[str, Any] | None) -> tuple[str, ...]:
    """Project the region dossier into a tuple of fact predicates.

    Phase B's dossier already encodes the facts a future run needs to
    decide whether a promoted recipe still applies — they're spread
    across ``legality_constraints``, ``numerical_sensitivity``, and
    ``placement_envelope``. We project each into a stable string
    predicate so the M-26 sidecar can carry them and M-28 retrieval
    can filter or rank by them.

    Predicate string forms (informal but stable):

    - ``can_tile`` / ``can_fuse_with_single_consumer`` /
      ``can_quantize_fp8`` — from ``legality_constraints[].constraint``
      with ``ok=True``.
    - ``numerics_safe_fp32`` / ``numerics_safe_fast_math`` /
      ``numerics_risky_fp16_accum`` — derived from
      ``numerical_sensitivity[*].status``.
    - ``memory_fit_<device>`` — from
      ``placement_envelope.devices[].memory_fit``.

    Returns an ordered tuple (sorted for byte-stable output).
    """
    if not dossier:
        return ()
    predicates: set[str] = set()

    for c in dossier.get("legality_constraints", []) or []:
        if not isinstance(c, dict):
            continue
        name = str(c.get("constraint") or "")
        if not name:
            continue
        if c.get("ok"):
            predicates.add(name)

    ns = dossier.get("numerical_sensitivity", {})
    if isinstance(ns, dict):
        for dtype, entry in ns.items():
            if not isinstance(entry, dict):
                continue
            status = str(entry.get("status") or "")
            if status == "safe":
                predicates.add(f"numerics_safe_{dtype}")
            elif status == "risky":
                predicates.add(f"numerics_risky_{dtype}")
            # exceeds_budget / requires_reference are not assertions
            # the recipe relies on; skip.

    pe = dossier.get("placement_envelope", {})
    if isinstance(pe, dict):
        for dev in pe.get("devices", []) or []:
            if not isinstance(dev, dict):
                continue
            if dev.get("memory_fit") and dev.get("device"):
                predicates.add(f"memory_fit_{dev['device']}")

    return tuple(sorted(predicates))


def _build_promoted_recipe(
    *,
    candidate_selection: dict[str, Any],
    region_signature_hash: str,
    region_signature_fields: dict[str, str],
    target_id: str,
    differential_outcomes: dict[str, Any],
    gate_evaluation: GateEvaluation | None = None,
    applies_when: tuple[str, ...] = (),
) -> PromotedRecipe:
    """Construct a :class:`PromotedRecipe` from Phase B evidence."""
    candidate_id = candidate_selection.get("selected_candidate_id") or "unknown"
    candidate_kind = candidate_selection.get("candidate_kind") or "unknown"
    region_id = candidate_selection.get("region_id") or "unknown"

    # Compact recipe id — readable and stable.
    recipe_id = (
        f"recipe_{candidate_kind}_{region_id}_{target_id}_{region_signature_hash[:8]}"
    )

    evidence_summary: dict[str, Any] = {
        "selected_candidate_id": candidate_id,
        "candidate_kind": candidate_kind,
        "region_id": region_id,
        "cost_preview": candidate_selection.get("cost_preview", {}),
        "rationale": candidate_selection.get("rationale", {}),
        "differential_outcomes": {
            label: report.get("status") or report.get("overall") or "unknown"
            for label, report in differential_outcomes.items()
        },
    }

    certificates: dict[str, str] = {}
    for label, report in differential_outcomes.items():
        body = json.dumps(report, sort_keys=True, separators=(",", ":"))
        certificates[f"{label}_sha256"] = _short_sha(body, length=64)

    validity = {
        "target_class": region_signature_fields.get("target_class", ""),
        "op_family": region_signature_fields.get("op_family", ""),
        "dtype": region_signature_fields.get("dtype", ""),
        "layout": region_signature_fields.get("layout", ""),
    }

    # M-29: fold gate-evaluation evidence + level into the recipe.
    if gate_evaluation is not None:
        evidence_summary["gate_level"] = str(gate_evaluation.level)
        evidence_summary["gate_reasons"] = dict(gate_evaluation.reasons)
        # Layer gate evidence on top of differential outcomes — the
        # gate evaluator already projects them, so this is additive.
        for k, v in gate_evaluation.evidence_summary.items():
            evidence_summary.setdefault(k, v)
        gate_level_str = str(gate_evaluation.level)
    else:
        gate_level_str = ""

    return PromotedRecipe(
        recipe_id=recipe_id,
        recipe_signature=region_signature_hash,
        recipe_ir_path="recipe.mlir",
        evidence_summary=evidence_summary,
        applies_when=applies_when,
        fallback_chain=(),
        certificates=certificates,
        validity=validity,
        gate_level=gate_level_str,
    )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def emit(
    run_dir: Path,
    *,
    library_path: Path | None = None,
    memory: Any = None,
) -> PromotionEmissionResult:
    """Read ``run_dir`` and promote its selected candidate to the library.

    Args:
        run_dir: A completed graph_compilation run directory containing
            at minimum ``run_manifest.json``,
            ``01_payload_lowering/payload.mlir``,
            ``03_recipe_planning/recipe.mlir``,
            ``03_recipe_planning/candidate_selection.json``, and any
            differential reports the gate consults.
        library_path: Recipe library root. Defaults to
            ``.compgen_cache/recipes/`` rooted at CWD.
        memory: Optional :class:`compgen.memory.store.CompilerMemory`
            instance — when provided, the promotion is also indexed in
            the SQLite ``promotions`` table with the two-tier dimensions.

    Returns:
        A :class:`PromotionEmissionResult` with typed status. Never
        raises — caller decides how to surface the outcome.
    """
    try:
        return _emit_impl(run_dir, library_path=library_path, memory=memory)
    except Exception as exc:  # noqa: BLE001 - bridge is best-effort
        log.warning(
            "promotion_bridge_unhandled_error",
            run_dir=str(run_dir),
            error=type(exc).__name__,
            message=str(exc),
        )
        return PromotionEmissionResult(
            status="error",
            reason=f"unhandled {type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )


def _emit_impl(
    run_dir: Path,
    *,
    library_path: Path | None,
    memory: Any,
) -> PromotionEmissionResult:
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        return PromotionEmissionResult(
            status="not_eligible", reason=f"run_dir does not exist: {run_dir}"
        )

    manifest = _read_json(run_dir / "run_manifest.json")
    if manifest is None:
        return PromotionEmissionResult(
            status="not_eligible",
            reason="run_manifest.json missing or unreadable",
        )

    model_id = manifest.get("model", {}).get("model_id") or ""
    target_id = manifest.get("target", {}).get("target_id") or ""
    if not model_id or not target_id:
        return PromotionEmissionResult(
            status="not_eligible",
            reason=f"manifest missing model_id={model_id!r} or target_id={target_id!r}",
        )

    rp_dir = run_dir / "03_recipe_planning"
    selection = _read_json(rp_dir / "candidate_selection.json")
    if selection is None:
        return PromotionEmissionResult(
            status="not_eligible",
            reason="03_recipe_planning/candidate_selection.json missing",
        )

    selected_id = selection.get("selected_candidate_id")
    if not selected_id:
        return PromotionEmissionResult(
            status="not_eligible",
            reason="no candidate selected in this run",
        )

    payload_text = _read_payload_text(run_dir)
    if payload_text is None:
        return PromotionEmissionResult(
            status="not_eligible",
            reason="01_payload_lowering/payload.mlir missing",
        )

    differential_outcomes = _gather_differential_outcomes(rp_dir)

    # Synthesize the verification_report.json that the promotion gate
    # reads. We write it under the run dir (which the bundle metadata
    # points at as bundle_root) so the existing copy-into-library logic
    # in RecipePromoter.promote() picks it up alongside the other
    # artifacts.
    verify_path, verify_reason = _synthesize_verification_report(
        run_dir, differential_outcomes
    )
    if verify_path is None or verify_reason != "passed":
        return PromotionEmissionResult(
            status="not_eligible",
            reason=f"verification gate would fail: {verify_reason}",
        )

    # Compute the region signature.
    region_id = selection.get("region_id") or ""
    region_kind = (selection.get("candidate_kind") or "").split("_")[0] or "unknown"
    # Prefer dossier ``kind`` when the dossier is found (real layout
    # uses ``<region_id>__<hash>.json``; the legacy hash-less form
    # is also accepted).
    dossier_path_for_kind = _resolve_region_dossier(run_dir, region_id)
    dossier_for_kind = _read_json(dossier_path_for_kind) if dossier_path_for_kind else None
    if dossier_for_kind and isinstance(dossier_for_kind.get("kind"), str):
        region_kind = dossier_for_kind["kind"]

    region_sig_hash, region_sig_fields = _derive_region_signature(
        run_dir=run_dir,
        region_id=region_id,
        target_id=target_id,
        kind=region_kind,
    )

    # M-26 contract_hash — exact-kernel reuse tier. Synthesized from
    # candidate_selection + region facts because Phase B doesn't yet
    # persist full KernelContractV3 objects to disk for every region.
    contract_hash_str = derive_contract_hash(
        candidate_selection=selection,
        region_signature_fields=region_sig_fields,
    )

    # M-29: evaluate the promotion-gate ladder before constructing
    # the bundle so the gate level rides along in the sidecar +
    # memory index.
    gate_eval: GateEvaluation | None = None
    try:
        library_for_gate = (
            Path(library_path) if library_path else _DEFAULT_LIBRARY_PATH
        )
        gate_eval = evaluate_gate(
            run_dir,
            region_signature=region_sig_hash,
            target_class=target_id,
            library_path=library_for_gate,
        )
    except Exception as exc:  # noqa: BLE001 - gate eval is best-effort
        log.warning(
            "promotion_bridge_gate_eval_failed",
            run_dir=str(run_dir),
            error=type(exc).__name__,
            message=str(exc),
        )

    # Construct the bundle. Artifacts are referenced relative to
    # bundle_root (the run dir) so RecipePromoter copies them into the
    # library as part of promotion.
    payload_path_resolved = _resolve_payload_path(run_dir)
    artifacts: dict[str, str] = {
        "verification_report": "04_promotion/verification_report.json",
    }
    if payload_path_resolved is not None:
        artifacts["payload"] = str(
            payload_path_resolved.relative_to(run_dir).as_posix()
        )
    if (rp_dir / "recipe.mlir").is_file():
        artifacts["recipe_mlir"] = "03_recipe_planning/recipe.mlir"
    if (rp_dir / "candidate_selection.json").is_file():
        artifacts["candidate_selection"] = "03_recipe_planning/candidate_selection.json"
    optional_reports: tuple[tuple[str, Path], ...] = (
        ("post_lowering_report",
         rp_dir / "post_lowering" / "post_lowering_verification_report.json"),
        ("differential_report",
         rp_dir / "differential_verification" / "differential_verification_report.json"),
        ("real_transform_diff_report",
         rp_dir / "real_verification" / "real_differential_report.json"),
        ("real_fusion_diff_report",
         rp_dir / "real_fusion_verification" / "real_fusion_differential_report.json"),
    )
    for label, abs_path in optional_reports:
        if abs_path.is_file():
            artifacts[label] = abs_path.relative_to(run_dir).as_posix()

    model_hash = _short_sha(payload_text, length=16)
    bundle_metadata: dict[str, Any] = {
        "bundle_root": str(run_dir),
        "model_id": model_id,
        "run_id": manifest.get("run_id", ""),
    }
    if gate_eval is not None:
        # M-29: surface the gate level into RecipePromoter so the
        # audit log records it alongside the basic promotion event.
        bundle_metadata["gate_level"] = str(gate_eval.level)
    bundle = Bundle(
        version="1.0",
        target_profile=target_id,
        model_hash=model_hash,
        objective="latency",
        artifacts=artifacts,
        creation_timestamp=manifest.get("created_at_utc", ""),
        metadata=bundle_metadata,
    )

    # Promote — the gate reads the synthesized verification_report.
    library = Path(library_path) if library_path else _DEFAULT_LIBRARY_PATH
    promoter = RecipePromoter(library_path=library)
    try:
        result: PromotionResult = promoter.promote(bundle)
    except Exception as exc:  # noqa: BLE001
        return PromotionEmissionResult(
            status="error",
            reason=f"RecipePromoter.promote raised {type(exc).__name__}: {exc}",
        )

    if not result.promoted or result.key is None or result.recipe_path is None:
        return PromotionEmissionResult(
            status="not_eligible",
            reason=result.reason or "RecipePromoter returned not promoted",
        )

    # Re-key with the two-tier dimensions (the original key from the
    # promoter has only the model-tier fields populated). The directory
    # name on disk is unchanged — the new fields ride in the sidecar +
    # SQLite index.
    new_key = RecipeKey(
        target_hash=result.key.target_hash,
        model_hash=result.key.model_hash,
        objective_hash=result.key.objective_hash,
        version=result.key.version,
        contract_hash=contract_hash_str,
        region_signature=region_sig_hash,
    )

    # M-30 gap #3: derive applies_when from the region dossier.
    dossier_path_for_facts = _resolve_region_dossier(run_dir, region_id)
    dossier_for_facts = (
        _read_json(dossier_path_for_facts) if dossier_path_for_facts else None
    )
    applies_when_tuple = _derive_applies_when(dossier_for_facts)

    promoted = _build_promoted_recipe(
        candidate_selection=selection,
        region_signature_hash=region_sig_hash,
        region_signature_fields=region_sig_fields,
        target_id=target_id,
        differential_outcomes=differential_outcomes,
        gate_evaluation=gate_eval,
        applies_when=applies_when_tuple,
    )
    write_promoted_recipe_sidecar(result.recipe_path, new_key, promoted)

    # Index in memory.promotions when a CompilerMemory is supplied.
    if memory is not None:
        try:
            from compgen.memory.schema import GeneratorKind, ObjectKind

            task = memory.create_task(
                kind=ObjectKind.BACKEND_PLAN,
                workload_key=model_hash,
                target_key=target_id,
                objective="latency",
            )
            cand = memory.record_candidate(
                task_id=task.task_id,
                artifact=json.dumps(bundle.to_dict(), separators=(",", ":")),
                generator_kind=GeneratorKind.TEMPLATE,
            )
            memory.promote_candidate(
                candidate_id=cand.candidate_id,
                promotion_key=new_key.key,
                reason="graph_compilation.promotion_bridge",
                region_signature=region_sig_hash,
                contract_hash=contract_hash_str,
                gate_level=str(gate_eval.level) if gate_eval else "",
            )
        except Exception as exc:  # noqa: BLE001 - memory bridge is best-effort
            log.warning(
                "promotion_bridge_memory_index_failed",
                run_dir=str(run_dir),
                error=type(exc).__name__,
                message=str(exc),
            )

    return PromotionEmissionResult(
        status="ok",
        reason=f"promoted as {new_key.key}",
        recipe_path=result.recipe_path,
        key=new_key,
    )


__all__ = ["PromotionEmissionResult", "derive_region_signature", "emit"]
