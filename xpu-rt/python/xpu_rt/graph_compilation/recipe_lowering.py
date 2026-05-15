"""Recipe Lowering to Lowering Artifacts (Milestone 07).

Lowers every verified recipe op into a family-specific *lowering
artifact* without applying it. Tiling and fusion recipes (and the
numerics/placement families) emit transform scripts; extension-closure
recipes emit kernel contract drafts.

This stage **does not**:

- mutate ``payload.mlir`` or any other Payload IR file,
- generate ``transformed_payload.mlir``,
- apply or simulate transforms,
- run differential / structural verification,
- call kernel codegen, benchmarks, or profilers,
- modify compiler core.

Outputs (under ``03_recipe_planning/``):

- ``lowering_artifacts/transforms/<recipe_op_id>.mlir``
- ``lowering_artifacts/contracts/<recipe_op_id>.kernel_contract.mlir``
- ``lowering_artifacts/README.md``
- ``lowering_artifact_manifest.json``         — inventory + sha256 per artifact
- ``transform_lowering_report.json``          — per-op lowering trace
- ``transform_validation.json``               — hard gate (no_payload_mutation included)
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from compgen.graph_compilation.hashing import sha256_file, sha256_tree
from compgen.graph_compilation.recipe_gate import (  # parser
    _parse_recipe_mlir,
    _ParsedRecipeOp,
)

# --------------------------------------------------------------------------- #
# Result + entry point
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RecipeLoweringResult:
    overall: str  # "pass" | "fail"
    manifest_path: Path
    report_path: Path
    validation_path: Path
    artifact_paths: tuple[Path, ...]


def _utcnow() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> dict[str, Any]:
    obj: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return obj


def _strip_symbol(value: Any) -> str:
    """Extract a symbol name from an attribute value that may be either
    ``@name`` (MLIR symbol form) or ``"@name"`` (legacy quoted form).
    """
    if not isinstance(value, str):
        return ""
    return value.lstrip("@")


# --------------------------------------------------------------------------- #
# Family-specific lowering writers
# --------------------------------------------------------------------------- #


_TRANSFORM_HEADER = (
    "// schema_version: compgen_transform_script_v1\n"
)
_CONTRACT_HEADER = (
    "// schema_version: compgen_kernel_contract_draft_v1\n"
)


def _safe_id(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", s).strip("_") or "x"


def _payload_ref_for_region(region_id: str, region_map: dict[str, Any]) -> str:
    for r in region_map.get("regions", []):
        if r["region_id"] == region_id:
            for po in r.get("payload_ops", []):
                if po.get("payload_ref"):
                    ref: str = po["payload_ref"]
                    return ref
    return ""


def _lower_set_tile_params(
    op: _ParsedRecipeOp,
    *,
    region_map: dict[str, Any],
    transforms_dir: Path,
) -> tuple[str, str, Path, list[str]]:
    """Emit a transform-intent script for a SetTileParams op."""
    region = str(op.attrs.get("region", ""))
    tile = op.attrs.get("tile") or {}
    M = int(tile.get("M", 0))
    N = int(tile.get("N", 0))
    K = int(tile.get("K", 0))
    cand = str(op.attrs.get("source_candidate", ""))
    obl = _strip_symbol(op.attrs.get("semantic_obligation"))
    refinement = str(op.attrs.get("declared_refinement", "bit_equality"))
    payload_ref = _payload_ref_for_region(region, region_map)

    lines: list[str] = [_TRANSFORM_HEADER.rstrip("\n")]
    lines.append(f"// recipe_op_id: {op.recipe_op_id}")
    lines.append("// recipe_kind: SetTileParams")
    lines.append(f"// source_candidate: {cand}")
    lines.append(f"// region: {region}")
    if payload_ref:
        lines.append(f"// payload_ref: {payload_ref}")
    if obl:
        lines.append(f"// semantic_obligation: {obl}")
    lines.append(f"// declared_refinement: {refinement}")
    lines.append("")
    lines.append(
        f"transform.compgen.sequence @{op.recipe_op_id} attributes {{"
    )
    lines.append(f'  recipe_op = "{op.recipe_op_id}",')
    lines.append(f'  region = "{region}",')
    lines.append('  target_op = "linalg.matmul",')
    lines.append('  action = "set_tile_params",')
    lines.append(f"  M = {M} : i64,")
    lines.append(f"  N = {N} : i64,")
    lines.append(f"  K = {K} : i64")
    lines.append("}")
    lines.append("")

    path = transforms_dir / f"{op.recipe_op_id}.mlir"
    path.write_text("\n".join(lines), encoding="utf-8")
    payload_refs = [payload_ref] if payload_ref else []
    return "transform_script", str(path.name), path, payload_refs


def _lower_fuse_producer_consumer(
    op: _ParsedRecipeOp,
    *,
    region_map: dict[str, Any],
    transforms_dir: Path,
) -> tuple[str, str, Path, list[str]]:
    producer = str(op.attrs.get("producer", ""))
    consumer = str(op.attrs.get("consumer", ""))
    via_tensor = str(op.attrs.get("via_tensor", ""))
    cand = str(op.attrs.get("source_candidate", ""))
    obl = _strip_symbol(op.attrs.get("semantic_obligation"))
    refinement = str(op.attrs.get("declared_refinement", "bit_equality"))
    p_ref = _payload_ref_for_region(producer, region_map)
    c_ref = _payload_ref_for_region(consumer, region_map)
    payload_refs = sorted({r for r in (p_ref, c_ref) if r})

    lines: list[str] = [_TRANSFORM_HEADER.rstrip("\n")]
    lines.append(f"// recipe_op_id: {op.recipe_op_id}")
    lines.append("// recipe_kind: FuseProducerConsumer")
    lines.append(f"// source_candidate: {cand}")
    lines.append(f"// producer: {producer}")
    lines.append(f"// consumer: {consumer}")
    lines.append(f"// via_tensor: {via_tensor}")
    if p_ref:
        lines.append(f"// producer_payload_ref: {p_ref}")
    if c_ref:
        lines.append(f"// consumer_payload_ref: {c_ref}")
    if obl:
        lines.append(f"// semantic_obligation: {obl}")
    lines.append(f"// declared_refinement: {refinement}")
    lines.append("")
    lines.append(f"transform.compgen.sequence @{op.recipe_op_id} attributes {{")
    lines.append(f'  recipe_op = "{op.recipe_op_id}",')
    lines.append('  action = "fuse_producer_consumer",')
    lines.append(f'  producer = "{producer}",')
    lines.append(f'  consumer = "{consumer}",')
    lines.append(f'  via_tensor = "{via_tensor}"')
    lines.append("}")
    lines.append("")

    path = transforms_dir / f"{op.recipe_op_id}.mlir"
    path.write_text("\n".join(lines), encoding="utf-8")
    return "transform_script", str(path.name), path, payload_refs


def _lower_kernel_contract(
    op: _ParsedRecipeOp,
    *,
    region_map: dict[str, Any],
    target_id: str,
    contracts_dir: Path,
) -> tuple[str, str, Path, list[str]]:
    region = str(op.attrs.get("region", ""))
    cand = str(op.attrs.get("source_candidate", ""))
    obl = _strip_symbol(op.attrs.get("semantic_obligation"))
    refinement = str(op.attrs.get("declared_refinement", "contract_obligation"))
    region_record = next(
        (r for r in region_map.get("regions", []) if r["region_id"] == region),
        None,
    )
    src_class = (
        region_record.get("source_classification", "unknown")
        if region_record is not None
        else "unknown"
    )
    payload_ref = _payload_ref_for_region(region, region_map)
    payload_refs = [payload_ref] if payload_ref else []

    lines: list[str] = [_CONTRACT_HEADER.rstrip("\n")]
    lines.append(f"// recipe_op_id: {op.recipe_op_id}")
    lines.append(f"// recipe_kind: {op.op_camel}")
    lines.append(f"// source_candidate: {cand}")
    if obl:
        lines.append(f"// semantic_obligation: {obl}")
    lines.append(f"// declared_refinement: {refinement}")
    lines.append("")
    lines.append(
        f"contract.kernel @{op.recipe_op_id}_contract attributes {{"
    )
    lines.append(f'  recipe_op = "{op.recipe_op_id}",')
    lines.append(f'  source_candidate = "{cand}",')
    lines.append(f'  target_id = "{target_id}",')
    lines.append('  proof_stage = "kernel_contract_generation"')
    lines.append("} {")
    lines.append("  contract.region attributes {")
    lines.append(f'    region_id = "{region}",')
    lines.append(f'    source_classification = "{src_class}"')
    if payload_ref:
        lines.append(f'    , payload_ref = "{payload_ref}"')
    lines.append("  }")
    if obl:
        lines.append(f"  sem.obligation_ref @{obl}")
    lines.append("}")
    lines.append("")

    path = contracts_dir / f"{op.recipe_op_id}.kernel_contract.mlir"
    path.write_text("\n".join(lines), encoding="utf-8")
    return "kernel_contract_draft", str(path.name), path, payload_refs


def _lower_keep_as_fallback(
    op: _ParsedRecipeOp,
    *,
    region_map: dict[str, Any],
    target_id: str,
    contracts_dir: Path,
) -> tuple[str, str, Path, list[str]]:
    region = str(op.attrs.get("region", ""))
    cand = str(op.attrs.get("source_candidate", ""))
    obl = _strip_symbol(op.attrs.get("semantic_obligation"))
    payload_ref = _payload_ref_for_region(region, region_map)
    payload_refs = [payload_ref] if payload_ref else []

    lines = [_CONTRACT_HEADER.rstrip("\n")]
    lines.append(f"// recipe_op_id: {op.recipe_op_id}")
    lines.append("// recipe_kind: KeepAsFallback")
    lines.append(f"// source_candidate: {cand}")
    lines.append("// declared_refinement: fallback_obligation")
    lines.append("")
    lines.append(
        f"contract.fallback @{op.recipe_op_id}_fallback attributes {{"
    )
    lines.append(f'  recipe_op = "{op.recipe_op_id}",')
    lines.append(f'  region = "{region}",')
    lines.append(f'  target_id = "{target_id}",')
    lines.append('  proof_stage = "always_pass"')
    lines.append("}")
    if obl:
        lines.append(f"// semantic_obligation: {obl}")
    lines.append("")

    path = contracts_dir / f"{op.recipe_op_id}.kernel_contract.mlir"
    path.write_text("\n".join(lines), encoding="utf-8")
    return "kernel_contract_draft", str(path.name), path, payload_refs


def _lower_numerics(
    op: _ParsedRecipeOp,
    *,
    region_map: dict[str, Any],
    transforms_dir: Path,
) -> tuple[str, str, Path, list[str]]:
    region = str(op.attrs.get("region", ""))
    cand = str(op.attrs.get("source_candidate", ""))
    obl = _strip_symbol(op.attrs.get("semantic_obligation"))
    refinement = str(op.attrs.get("declared_refinement", "tolerance_eps"))
    payload_ref = _payload_ref_for_region(region, region_map)
    payload_refs = [payload_ref] if payload_ref else []

    action = {
        "QuantizeFP8": "quantize_fp8",
        "SetAccumulator": "set_accumulator_fp16",
        "EnableFastMath": "enable_fast_math",
    }.get(op.op_camel, op.op_snake)

    lines = [_TRANSFORM_HEADER.rstrip("\n")]
    lines.append(f"// recipe_op_id: {op.recipe_op_id}")
    lines.append(f"// recipe_kind: {op.op_camel}")
    lines.append(f"// source_candidate: {cand}")
    lines.append(f"// region: {region}")
    if payload_ref:
        lines.append(f"// payload_ref: {payload_ref}")
    if obl:
        lines.append(f"// semantic_obligation: {obl}")
    lines.append(f"// declared_refinement: {refinement}")
    lines.append("")
    lines.append(f"transform.compgen.sequence @{op.recipe_op_id} attributes {{")
    lines.append(f'  recipe_op = "{op.recipe_op_id}",')
    lines.append(f'  region = "{region}",')
    lines.append(f'  action = "{action}"')
    lines.append("}")
    lines.append("")

    path = transforms_dir / f"{op.recipe_op_id}.mlir"
    path.write_text("\n".join(lines), encoding="utf-8")
    return "transform_script", str(path.name), path, payload_refs


def _lower_assign_device(
    op: _ParsedRecipeOp,
    *,
    region_map: dict[str, Any],
    transforms_dir: Path,
) -> tuple[str, str, Path, list[str]]:
    region = str(op.attrs.get("region", ""))
    device = str(op.attrs.get("device", ""))
    cand = str(op.attrs.get("source_candidate", ""))
    obl = _strip_symbol(op.attrs.get("semantic_obligation"))
    payload_ref = _payload_ref_for_region(region, region_map)
    payload_refs = [payload_ref] if payload_ref else []

    lines = [_TRANSFORM_HEADER.rstrip("\n")]
    lines.append(f"// recipe_op_id: {op.recipe_op_id}")
    lines.append("// recipe_kind: AssignDevice")
    lines.append(f"// source_candidate: {cand}")
    lines.append(f"// region: {region}")
    lines.append(f"// device: {device}")
    if obl:
        lines.append(f"// semantic_obligation: {obl}")
    lines.append("// declared_refinement: placement_obligation")
    lines.append("")
    lines.append(f"transform.compgen.sequence @{op.recipe_op_id} attributes {{")
    lines.append(f'  recipe_op = "{op.recipe_op_id}",')
    lines.append(f'  region = "{region}",')
    lines.append('  action = "assign_device",')
    lines.append(f'  device = "{device}"')
    lines.append("}")
    lines.append("")

    path = transforms_dir / f"{op.recipe_op_id}.mlir"
    path.write_text("\n".join(lines), encoding="utf-8")
    return "transform_script", str(path.name), path, payload_refs


# Op camel name → (lowerer fn, expected artifact_kind)
_LOWERING_DISPATCH: dict[str, tuple[Any, str]] = {
    "SetTileParams":                       (_lower_set_tile_params,           "transform_script"),
    "FuseProducerConsumer":                (_lower_fuse_producer_consumer,    "transform_script"),
    "QuantizeFP8":                         (_lower_numerics,                  "transform_script"),
    "SetAccumulator":                      (_lower_numerics,                  "transform_script"),
    "EnableFastMath":                      (_lower_numerics,                  "transform_script"),
    "AssignDevice":                        (_lower_assign_device,             "transform_script"),
    "CreatePayloadLoweringExtension":      (_lower_kernel_contract,           "kernel_contract_draft"),
    "CreateKernelContract":                (_lower_kernel_contract,           "kernel_contract_draft"),
    "KeepAsFallback":                      (_lower_keep_as_fallback,          "kernel_contract_draft"),
}


# --------------------------------------------------------------------------- #
# Top-level entry point
# --------------------------------------------------------------------------- #


def run_recipe_lowering(run_dir: Path) -> RecipeLoweringResult:
    """Run the lowering against an existing recipe-verification output.

    Reads (read-only):

    - ``03_recipe_planning/verified_recipe.mlir``
    - ``03_recipe_planning/semantic_obligations.{mlir,json}``
    - ``03_recipe_planning/recipe_gate_verdict.json``
    - ``02_graph_analysis/region_map.json``

    Writes:

    - ``03_recipe_planning/lowering_artifacts/transforms/<id>.mlir``
    - ``03_recipe_planning/lowering_artifacts/contracts/<id>.kernel_contract.mlir``
    - ``03_recipe_planning/lowering_artifacts/README.md``
    - ``03_recipe_planning/lowering_artifact_manifest.json``
    - ``03_recipe_planning/transform_lowering_report.json``
    - ``03_recipe_planning/transform_validation.json``

    Crucially: ``01_payload_lowering/`` must be byte-identical before and
    after this call. The validator records both sha256s and fails if they
    differ.
    """
    run_dir = Path(run_dir).resolve()
    rp = run_dir / "03_recipe_planning"
    if not rp.is_dir():
        raise FileNotFoundError(f"03_recipe_planning/ missing under {run_dir}")
    verified_path = rp / "verified_recipe.mlir"
    sem_mlir_path = rp / "semantic_obligations.mlir"
    sem_json_path = rp / "semantic_obligations.json"
    if not verified_path.exists():
        raise FileNotFoundError(f"verified_recipe.mlir missing: {verified_path}")
    if not sem_mlir_path.exists():
        raise FileNotFoundError(f"semantic_obligations.mlir missing: {sem_mlir_path}")
    if not sem_json_path.exists():
        raise FileNotFoundError(f"semantic_obligations.json missing: {sem_json_path}")

    pl_dir = run_dir / "01_payload_lowering"
    payload_pre_sha = sha256_tree(pl_dir)

    verified_text = verified_path.read_text(encoding="utf-8")
    sem_mlir_text = sem_mlir_path.read_text(encoding="utf-8")
    sem_json = _read_json(sem_json_path)
    module_attrs, ops = _parse_recipe_mlir(verified_text)
    verified_sha = "sha256:" + hashlib.sha256(verified_text.encode("utf-8")).hexdigest()
    sem_sha = "sha256:" + hashlib.sha256(sem_mlir_text.encode("utf-8")).hexdigest()

    # Read region_map for payload_ref resolution
    region_map = _read_json(run_dir / "02_graph_analysis" / "region_map.json")
    target_id = str(module_attrs.get("target_id", "host_cpu"))
    model_id = str(module_attrs.get("model_id", "model"))

    # Build the obligation set so we can verify each lowered op references one.
    declared_obligations = {
        ob["id"] for ob in sem_json.get("obligations", [])
    }

    out_root = rp / "lowering_artifacts"
    transforms_dir = out_root / "transforms"
    contracts_dir = out_root / "contracts"
    transforms_dir.mkdir(parents=True, exist_ok=True)
    contracts_dir.mkdir(parents=True, exist_ok=True)

    # Wipe stale artifacts so a stale-fragment-from-prior-run can't masquerade.
    for stale_dir in (transforms_dir, contracts_dir):
        for p in stale_dir.glob("*.mlir"):
            p.unlink()

    artifacts: list[dict[str, Any]] = []
    lowered_ops: list[dict[str, Any]] = []
    artifact_paths: list[Path] = []
    fail_reasons: list[str] = []

    for op in ops:
        if op.attrs.get("gate_status") != "pass":
            fail_reasons.append(
                f"recipe op {op.recipe_op_id!r} has gate_status="
                f"{op.attrs.get('gate_status')!r} — not eligible for lowering"
            )
            continue
        dispatch = _LOWERING_DISPATCH.get(op.op_camel)
        if dispatch is None:
            fail_reasons.append(
                f"recipe op {op.recipe_op_id!r} kind={op.op_camel!r} has no lowering family"
            )
            continue
        lowerer, expected_kind = dispatch

        # Family-aware kwargs
        if expected_kind == "transform_script":
            artifact_kind, fname, path, payload_refs = lowerer(
                op, region_map=region_map, transforms_dir=transforms_dir,
            )
        else:
            artifact_kind, fname, path, payload_refs = lowerer(
                op,
                region_map=region_map,
                target_id=target_id,
                contracts_dir=contracts_dir,
            )

        # Sanity: did the lowerer produce the expected family?
        if artifact_kind != expected_kind:
            fail_reasons.append(
                f"recipe op {op.recipe_op_id!r}: expected {expected_kind!r} "
                f"but got {artifact_kind!r}"
            )

        sha = sha256_file(path)
        rel = path.relative_to(run_dir).as_posix()
        artifacts.append(
            {
                "recipe_op_id": op.recipe_op_id,
                "recipe_kind": op.op_camel,
                "artifact_kind": artifact_kind,
                "path": rel,
                "sha256": sha,
                "size_bytes": path.stat().st_size,
                "status": "emitted",
            }
        )
        sem_obligation = _strip_symbol(op.attrs.get("semantic_obligation"))
        cand = str(op.attrs.get("source_candidate", ""))
        refinement = str(op.attrs.get("declared_refinement", ""))
        proof_stage = next(
            (
                ob["proof_stage"]
                for ob in sem_json.get("obligations", [])
                if ob["id"] == sem_obligation
            ),
            "",
        )

        # Per-op lowering checks.
        op_checks: list[str] = ["verified_recipe_gate_status_pass"]
        if sem_obligation in declared_obligations:
            op_checks.append("semantic_obligation_exists")
        else:
            fail_reasons.append(
                f"recipe op {op.recipe_op_id!r}: semantic obligation "
                f"{sem_obligation!r} not in semantic_obligations.json"
            )
        if op.attrs.get("region") and any(
            r["region_id"] == op.attrs["region"] for r in region_map.get("regions", [])
        ):
            op_checks.append("region_exists")
        elif op.op_camel == "FuseProducerConsumer":
            # Fusion uses producer/consumer not region.
            if (
                any(r["region_id"] == op.attrs.get("producer") for r in region_map.get("regions", []))
                and any(r["region_id"] == op.attrs.get("consumer") for r in region_map.get("regions", []))
            ):
                op_checks.append("region_exists")
            else:
                fail_reasons.append(
                    f"recipe op {op.recipe_op_id!r}: producer/consumer regions missing"
                )
        else:
            fail_reasons.append(
                f"recipe op {op.recipe_op_id!r}: region "
                f"{op.attrs.get('region')!r} missing in region_map"
            )
        # Verify each declared payload_ref exists.
        bad_refs: list[str] = []
        for pr in payload_refs:
            if not (run_dir / pr).exists():
                bad_refs.append(pr)
        if not bad_refs:
            op_checks.append("payload_ref_exists")
        else:
            fail_reasons.append(
                f"recipe op {op.recipe_op_id!r}: payload_refs missing on disk: {bad_refs}"
            )
        op_checks.append("artifact_emitted")

        lowered_ops.append(
            {
                "recipe_op_id": op.recipe_op_id,
                "recipe_kind": op.op_camel,
                "source_candidate": cand,
                "region": op.attrs.get("region"),
                "artifact_kind": artifact_kind,
                "artifact_path": rel,
                "payload_refs": payload_refs,
                "semantic_obligation": sem_obligation,
                "declared_refinement": refinement,
                "proof_stage": proof_stage,
                "lowering_checks": op_checks,
            }
        )
        artifact_paths.append(path)

    # README.md so reviewers landing in lowering_artifacts/ can orient quickly.
    (out_root / "README.md").write_text(
        (
            "# Lowering artifacts (M-07)\n\n"
            "Each verified recipe op produced exactly one lowering artifact.\n\n"
            "- `transforms/<recipe_op_id>.mlir` — transform-intent script for\n"
            "  tiling / fusion / numerics / placement recipes.\n"
            "- `contracts/<recipe_op_id>.kernel_contract.mlir` — kernel contract\n"
            "  draft for extension-closure / fallback recipes.\n\n"
            "M-07 does **not** apply these scripts. Application + structural\n"
            "verification land in M-08 (which will write\n"
            "`transformed_payload.mlir` to a copy of `01_payload_lowering/`).\n"
        ),
        encoding="utf-8",
    )

    payload_post_sha = sha256_tree(pl_dir)

    # ---------------- manifest -------------------------------------- #
    manifest = {
        "schema_version": "lowering_artifact_manifest_v1",
        "model_id": model_id,
        "target_id": target_id,
        "generated_at_utc": _utcnow(),
        "source": {
            "verified_recipe": "03_recipe_planning/verified_recipe.mlir",
            "verified_recipe_sha256": verified_sha,
            "semantic_obligations": "03_recipe_planning/semantic_obligations.mlir",
            "semantic_obligations_sha256": sem_sha,
        },
        "artifacts": artifacts,
        "summary": {
            "recipe_ops_total": len(ops),
            "artifacts_emitted": len(artifacts),
            "transform_scripts": sum(
                1 for a in artifacts if a["artifact_kind"] == "transform_script"
            ),
            "kernel_contracts": sum(
                1 for a in artifacts if a["artifact_kind"] == "kernel_contract_draft"
            ),
        },
    }
    manifest_path = rp / "lowering_artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    # ---------------- report ---------------------------------------- #
    report = {
        "schema_version": "transform_lowering_report_v1",
        "status": "pass" if not fail_reasons else "fail",
        "model_id": model_id,
        "target_id": target_id,
        "generated_at_utc": _utcnow(),
        "source": {
            "verified_recipe_sha256": verified_sha,
            "semantic_obligations_sha256": sem_sha,
        },
        "lowered_ops": lowered_ops,
        "failure_reasons": fail_reasons,
    }
    report_path = rp / "transform_lowering_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )

    # ---------------- validation ------------------------------------- #
    checks: list[dict[str, Any]] = []

    def _add(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "status": "pass" if ok else "fail", "detail": detail})

    _add(
        "all_verified_recipe_ops_lowered",
        len(artifacts) == len(ops),
        f"ops={len(ops)}, artifacts={len(artifacts)}",
    )
    _add(
        "semantic_obligations_resolved",
        all(
            lo["semantic_obligation"] and lo["semantic_obligation"] in declared_obligations
            for lo in lowered_ops
        ),
        f"obligations={sorted(declared_obligations)}",
    )
    _add(
        "payload_refs_exist",
        all(
            (run_dir / pr).exists()
            for lo in lowered_ops for pr in lo["payload_refs"]
        ),
        "",
    )
    _add(
        "no_payload_mutation",
        payload_pre_sha == payload_post_sha,
        f"pre={payload_pre_sha[:16]}... post={payload_post_sha[:16]}...",
    )
    # Recompute hashes from disk and compare to manifest.
    _add(
        "artifact_hashes_match_manifest",
        all(
            sha256_file(run_dir / a["path"]) == a["sha256"] for a in artifacts
        ),
        "",
    )
    # Family-specific routing: every artifact landed in the right subdir.
    routing_ok = True
    for a in artifacts:
        if a["artifact_kind"] == "transform_script" and "/transforms/" not in a["path"]:
            routing_ok = False
        elif a["artifact_kind"] == "kernel_contract_draft" and "/contracts/" not in a["path"]:
            routing_ok = False
    _add("family_specific_artifact_kind", routing_ok, "")

    overall = "pass" if all(c["status"] == "pass" for c in checks) else "fail"
    if fail_reasons and overall == "pass":
        overall = "fail"

    validation = {
        "schema_version": "transform_validation_v1",
        "overall": overall,
        "model_id": model_id,
        "target_id": target_id,
        "checks": checks,
        "failure_reasons": fail_reasons,
        "source": {
            "verified_recipe_sha256": verified_sha,
            "payload_pre_sha256": payload_pre_sha,
            "payload_post_sha256": payload_post_sha,
        },
    }
    validation_path = rp / "transform_validation.json"
    validation_path.write_text(
        json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8"
    )

    return RecipeLoweringResult(
        overall=overall,
        manifest_path=manifest_path,
        report_path=report_path,
        validation_path=validation_path,
        artifact_paths=tuple(artifact_paths),
    )
