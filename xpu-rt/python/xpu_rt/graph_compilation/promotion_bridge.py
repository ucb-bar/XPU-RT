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
# (``.compgen_cache/recipes/`` is the gitignored deterministic library).
_DEFAULT_LIBRARY_PATH = Path(".compgen_cache") / "recipes"


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


def _read_payload_text(run_dir: Path) -> str | None:
    """Read the canonical Payload IR text for hashing into ``model_hash``."""
    payload_path = run_dir / "01_payload_lowering" / "payload.mlir"
    if not payload_path.is_file():
        return None
    return payload_path.read_text(encoding="utf-8")


def _gather_differential_outcomes(rp: Path) -> dict[str, Any]:
    """Read whichever Phase B differential reports exist.

    Returns a dict with three optional sub-blocks:

    - ``post_lowering`` from M-08
    - ``differential`` from M-09
    - ``real_transform`` from M-12
    - ``real_fusion`` from M-16.2

    Missing reports are simply absent from the dict.
    """
    outcomes: dict[str, Any] = {}
    candidates: dict[str, str] = {
        "post_lowering": "post_lowering_verification_report.json",
        "differential": "differential_verification_report.json",
        "real_transform": "real_transform_differential_report.json",
        "real_fusion": "real_fusion_differential_report.json",
    }
    for label, fname in candidates.items():
        report = _read_json(rp / fname)
        if report is not None:
            outcomes[label] = report
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


def _derive_region_signature(
    *, run_dir: Path, region_id: str, target_id: str, kind: str
) -> tuple[str, dict[str, str]]:
    """Construct a region pattern signature from the M-10 dossier.

    Fields not derivable from the dossier degrade to ``"unknown"`` —
    M-26 ships with this degradation explicit. Future milestones
    (M-27 ops_provenance, M-28 retrieval) will tighten the signature
    once Recipe IR carries the full pattern.
    """
    dossier_path = run_dir / "02_graph_analysis" / "region_dossiers" / f"{region_id}.json"
    dossier = _read_json(dossier_path) or {}

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


def _build_promoted_recipe(
    *,
    candidate_selection: dict[str, Any],
    region_signature_hash: str,
    region_signature_fields: dict[str, str],
    target_id: str,
    differential_outcomes: dict[str, Any],
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

    return PromotedRecipe(
        recipe_id=recipe_id,
        recipe_signature=region_signature_hash,
        recipe_ir_path="recipe.mlir",
        evidence_summary=evidence_summary,
        applies_when=(),  # M-27 will populate from PromoteOp.applies_when.
        fallback_chain=(),
        certificates=certificates,
        validity=validity,
        gate_level="",  # M-29 will populate.
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
    # Better op_family from the dossier when available.
    dossier = _read_json(
        run_dir / "02_graph_analysis" / "region_dossiers" / f"{region_id}.json"
    )
    if dossier and isinstance(dossier.get("kind"), str):
        region_kind = dossier["kind"]

    region_sig_hash, region_sig_fields = _derive_region_signature(
        run_dir=run_dir,
        region_id=region_id,
        target_id=target_id,
        kind=region_kind,
    )

    # Construct the bundle. Artifacts are referenced relative to
    # bundle_root (the run dir) so RecipePromoter copies them into the
    # library as part of promotion.
    artifacts: dict[str, str] = {
        "verification_report": "04_promotion/verification_report.json",
        "payload": "01_payload_lowering/payload.mlir",
    }
    if (rp_dir / "recipe.mlir").is_file():
        artifacts["recipe_mlir"] = "03_recipe_planning/recipe.mlir"
    if (rp_dir / "candidate_selection.json").is_file():
        artifacts["candidate_selection"] = "03_recipe_planning/candidate_selection.json"
    for label, fname in (
        ("post_lowering_report", "post_lowering_verification_report.json"),
        ("differential_report", "differential_verification_report.json"),
        ("real_transform_diff_report", "real_transform_differential_report.json"),
        ("real_fusion_diff_report", "real_fusion_differential_report.json"),
    ):
        if (rp_dir / fname).is_file():
            artifacts[label] = f"03_recipe_planning/{fname}"

    model_hash = _short_sha(payload_text, length=16)
    bundle = Bundle(
        version="1.0",
        target_profile=target_id,
        model_hash=model_hash,
        objective="latency",
        artifacts=artifacts,
        creation_timestamp=manifest.get("created_at_utc", ""),
        metadata={
            "bundle_root": str(run_dir),
            "model_id": model_id,
            "run_id": manifest.get("run_id", ""),
        },
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
        contract_hash="",  # M-26 ships without kernel_contracts plumbing.
        region_signature=region_sig_hash,
    )

    promoted = _build_promoted_recipe(
        candidate_selection=selection,
        region_signature_hash=region_sig_hash,
        region_signature_fields=region_sig_fields,
        target_id=target_id,
        differential_outcomes=differential_outcomes,
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
                contract_hash="",
                gate_level="",
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
