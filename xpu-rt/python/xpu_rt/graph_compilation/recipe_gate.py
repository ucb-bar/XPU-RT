"""Recipe Verification Gate (Milestone 06).

Verifies every Recipe op committed by M-05 against the canonical action
space + region graph + dossier facts. This is a **pre-lowering gate**:
it proves source consistency and family-specific preconditions, and
declares the **semantic / refinement obligation** that a later
lowering / verification stage (M-07 / M-08) must discharge.

Specifically, M-06:

1. Replays every ``source_candidate`` against
   ``02_graph_analysis/action_space.mlir`` via :mod:`action_space_resolver`
   (hash chain + recipe_delta cross-check).
2. Re-checks the candidate's legality at the time of verification (not
   at the time of selection — defends against later corruption).
3. Runs family-specific preconditions for each recipe family:
   ``SetTileParams``, ``FuseProducerConsumer``, ``CreateKernelContract``
   / ``CreatePayloadLoweringExtension`` / ``KeepAsFallback``,
   ``QuantizeFP8`` / ``SetAccumulator`` / ``EnableFastMath``,
   ``AssignDevice``.
4. Declares per-op semantic obligations with refinement type +
   proof_stage + verifier_chain.
5. Emits:

   - ``03_recipe_planning/recipe_gate_verdict.json``     (top-level result)
   - ``03_recipe_planning/recipe_gate_trace.jsonl``      (per-check audit)
   - ``03_recipe_planning/semantic_obligations.mlir``    (canonical IR)
   - ``03_recipe_planning/semantic_obligations.json``    (projection)
   - ``03_recipe_planning/verified_recipe.mlir``         (recipe + gate annotations)

It also amends the existing ``recipe_validation.json`` /
``recipe_summary.json`` with the gate's overall verdict.

This stage does **not**:

- mutate ``payload.mlir`` or any Payload IR file,
- generate transform scripts,
- call kernel codegen,
- benchmark or profile,
- modify compiler core.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from compgen.graph_compilation.action_space_resolver import (
    HashMismatchError,
    IllegalCandidateError,
    RecipeDeltaMismatchError,
    ResolvedCandidate,
    ResolverError,
    _parse_attrs_body,  # reuse the tiny MLIR-attr parser
    resolve_candidate,
)
from compgen.graph_compilation.region_dossier import (
    TargetProfile,
    load_target_profile,
)


class RecipeGateError(RuntimeError):
    """Raised by :func:`run_recipe_gate` when verification cannot proceed."""


# --------------------------------------------------------------------------- #
# Recipe IR parser
# --------------------------------------------------------------------------- #

_RECIPE_OP_LINE_RE = re.compile(
    r"^\s*recipe\.(?P<op>[A-Za-z_][A-Za-z0-9_]*)\s+@(?P<id>[A-Za-z_][A-Za-z0-9_]*)\s+"
    r"attributes\s*\{\s*(?P<body>.*?)\s*\}\s*$"
)
_RECIPE_MODULE_HEADER_RE = re.compile(
    r"^\s*recipe\.module\s+@(?P<id>[A-Za-z_][A-Za-z0-9_]*)\s+attributes\s*\{(?P<attrs>[^}]*)\}\s*\{\s*$"
)


@dataclass(frozen=True)
class _ParsedRecipeOp:
    recipe_op_id: str
    op_snake: str  # e.g. set_tile_params
    op_camel: str  # e.g. SetTileParams (reverse of snake)
    attrs: dict[str, Any]


def _snake_to_camel(s: str) -> str:
    return "".join(p.title() if p else "" for p in s.split("_"))


def _parse_recipe_mlir(text: str) -> tuple[dict[str, Any], list[_ParsedRecipeOp]]:
    """Parse ``recipe.mlir`` into its module attributes plus a list of
    individual ``recipe.<op> @<id>`` lines.

    Raises :class:`RecipeGateError` when the file is malformed.
    """
    module_attrs: dict[str, Any] = {}
    ops: list[_ParsedRecipeOp] = []
    in_module = False
    for line in text.splitlines():
        if not in_module:
            m = _RECIPE_MODULE_HEADER_RE.match(line)
            if m:
                module_attrs = _parse_attrs_body(m.group("attrs"))
                in_module = True
            continue
        if line.strip() == "}":
            in_module = False
            continue
        m = _RECIPE_OP_LINE_RE.match(line)
        if not m:
            continue
        attrs = _parse_attrs_body(m.group("body"))
        op_snake = m.group("op")
        ops.append(
            _ParsedRecipeOp(
                recipe_op_id=m.group("id"),
                op_snake=op_snake,
                op_camel=_snake_to_camel(op_snake),
                attrs=attrs,
            )
        )
    return module_attrs, ops


# --------------------------------------------------------------------------- #
# Family-specific gate functions
# --------------------------------------------------------------------------- #


@dataclass
class GateOpVerdict:
    recipe_op_id: str
    source_candidate: str
    op: str  # CamelCase name as recorded in the recipe_delta
    region: str | None
    gate_status: str  # "pass" | "fail"
    declared_refinement: str
    proof_stage: str
    verifier_chain: list[str]
    semantic_obligation: str  # symbol id, e.g. obl_recipe_0000
    discharged_now: list[str] = field(default_factory=list)
    deferred_until_lowering: list[str] = field(default_factory=list)
    failure_reasons: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class _GateContext:
    run_dir: Path
    region_map: dict[str, Any]
    use_def: dict[str, Any]
    region_graph: dict[str, Any]
    region_dossier_paths: dict[str, str]  # region_id -> dossier ref
    profile: TargetProfile
    gap_lookup: dict[str, dict[str, Any]]  # region_id -> gap record
    allow_risky_numerics: bool


def _region_kind(ctx: _GateContext, region_id: str) -> str:
    for r in ctx.region_map.get("regions", []):
        if r["region_id"] == region_id:
            return str(r.get("kind", ""))
    return ""


def _is_opaque(kind: str) -> bool:
    return kind.startswith("opaque_")


def _gate_set_tile_params(
    ctx: _GateContext,
    op: _ParsedRecipeOp,
    resolved: ResolvedCandidate,
) -> GateOpVerdict:
    region_id = op.attrs.get("region", "") or resolved.region_id
    discharged: list[str] = []
    failures: list[str] = []
    extra: dict[str, Any] = {}

    # Region exists and is structured.
    kind = _region_kind(ctx, region_id)
    if not kind:
        failures.append(f"region {region_id!r} not in region_map")
    else:
        discharged.append("region_exists")
        if _is_opaque(kind):
            failures.append(f"region {region_id!r} is opaque ({kind})")
        elif kind not in {"matmul", "conv"}:
            failures.append(
                f"region {region_id!r} kind={kind!r} is not a matmul/conv tile target"
            )
        else:
            discharged.append("region_is_matmul_or_conv")

    # Tile must come from the dossier's working_set_curve.
    dossier_ref = ctx.region_dossier_paths.get(region_id)
    tile_match: dict[str, Any] | None = None
    if dossier_ref is None:
        failures.append(f"region dossier missing for {region_id!r}")
    else:
        dossier_path = ctx.run_dir / dossier_ref
        if not dossier_path.exists():
            failures.append(f"region dossier file missing: {dossier_ref}")
        else:
            d = json.loads(dossier_path.read_text(encoding="utf-8"))
            discharged.append("region_dossier_exists")
            # The recipe-op's tile is nested: ``tile = { K = 16 : i64, M = ... }``.
            # Fall back to flat M/N/K attrs for forward-compat with simpler emitters.
            tile_block = op.attrs.get("tile")
            if isinstance(tile_block, dict):
                our_tile = {
                    k: int(v) for k, v in tile_block.items()
                    if isinstance(v, (int, float))
                }
            else:
                our_tile = {
                    k: int(v) for k, v in op.attrs.items()
                    if k in {"M", "N", "K"} and isinstance(v, (int, float))
                }
            for entry in d.get("working_set_curve", []):
                if entry["tile"] == our_tile:
                    tile_match = entry
                    break
            if tile_match is None:
                failures.append(
                    f"tile {our_tile!r} not in working_set_curve for region {region_id!r}"
                )
            else:
                discharged.append("tile_exists_in_working_set_curve")
                extra["live_bytes"] = tile_match["live_bytes"]
                extra["fits_scratchpad"] = tile_match["fits_scratchpad"]
                extra["fits_l2"] = tile_match["fits_l2"]
                # Tile legality matches fits_l2 policy (M-04 invariant).
                if tile_match["fits_l2"]:
                    discharged.append("working_set_fits_required_memory_tier")
                else:
                    failures.append("tile working set does not fit L2")

            # M-37.9 Fix 3a: derive declared_refinement from tile-vs-shape
            # divisibility. SetTileParams preserves bit_equality only when
            # every region dimension is divisible by its tile size — that
            # keeps accumulation order stable. When boundary handling is
            # required (any non-clean divide), we downgrade to
            # tolerance_eps so the recipe-level claim matches what the
            # M-15B differential check measures.
            region_shape = d.get("region_shape") or {}
            extra["region_shape_summary"] = region_shape.get("summary", "")
            extra["clean_divide"] = None
            extra["boundary_required"] = None
            if (
                region_shape
                and region_shape.get("kind") == "matmul"
                and our_tile
            ):
                inp = region_shape.get("input_shapes") or []
                if (
                    len(inp) >= 2
                    and len(inp[0]) == 2
                    and len(inp[1]) == 2
                    and inp[0][1] == inp[1][0]
                ):
                    M_dim, K_dim = inp[0]
                    _, N_dim = inp[1]
                    tM = our_tile.get("M", 0) or 0
                    tN = our_tile.get("N", 0) or 0
                    tK = our_tile.get("K", 0) or 0
                    clean = (
                        tM > 0 and M_dim % tM == 0
                        and tN > 0 and N_dim % tN == 0
                        and tK > 0 and K_dim % tK == 0
                    )
                    # M-37.12 Fix: bit_equality requires NOT just
                    # clean-divide but also a single K iteration. With
                    # K_iters > 1 the partial sums accumulate in a
                    # different order than eager (eager: one running
                    # sum across K; tiled: sum-of-tile-sums). The
                    # values agree mathematically but float rounding
                    # differs at ~1e-6. So bit_equality only holds
                    # when tK >= K_dim (whole K in one tile).
                    single_k_iter = tK > 0 and tK >= K_dim
                    extra["clean_divide"] = clean
                    extra["boundary_required"] = not clean
                    extra["single_k_iter"] = single_k_iter
                    extra["region_dims"] = {
                        "M": M_dim, "N": N_dim, "K": K_dim,
                    }

    # source payload_ref must exist
    for po in resolved.recipe_delta:
        # SetTileParams recipe_delta entries don't have payload_ref directly;
        # we rely on the resolved candidate's evidence.
        pass
    payload_ref = resolved.evidence.get("payload_ref", "")
    if payload_ref:
        if (ctx.run_dir / payload_ref).exists():
            discharged.append("source_payload_ref_exists")
        else:
            failures.append(f"payload_ref does not exist: {payload_ref}")

    # M-37.12 Fix: claimable refinement matches the differential reality.
    # bit_equality requires BOTH clean_divide AND single K iteration —
    # multiple K iters reorder accumulation even with clean divides
    # (verified empirically on tiny_mlp's tile_M4_N16_K16, K_iters=4 →
    # max_abs ~5.7e-6). When boundary handling is needed (any non-clean
    # divide) OR when K_iters > 1, downgrade to tolerance_eps.
    # ``unknown`` is the conservative fallback when shape is missing.
    if extra.get("clean_divide") is True and extra.get("single_k_iter") is True:
        derived_refinement = "bit_equality"
    elif extra.get("clean_divide") is True:
        derived_refinement = "tolerance_eps"  # clean but K_iters > 1
    elif extra.get("clean_divide") is False:
        derived_refinement = "tolerance_eps"
    else:
        derived_refinement = "unknown"

    return GateOpVerdict(
        recipe_op_id=op.recipe_op_id,
        source_candidate=op.attrs.get("source_candidate", resolved.candidate_id),
        op=op.op_camel,
        region=region_id,
        gate_status="pass" if not failures else "fail",
        declared_refinement=derived_refinement,
        proof_stage="post_lowering",
        verifier_chain=["structural", "differential"],
        semantic_obligation=f"obl_{op.recipe_op_id}",
        discharged_now=discharged,
        deferred_until_lowering=[
            "payload_transform_structural_check",
            "post_lowering_differential_check",
        ],
        failure_reasons=failures,
        extra=extra,
    )


_REDUCTION_KINDS = {"matmul", "conv", "softmax", "layer_norm", "batch_norm"}


def _gate_fuse_producer_consumer(
    ctx: _GateContext,
    op: _ParsedRecipeOp,
    resolved: ResolvedCandidate,
) -> GateOpVerdict:
    producer = op.attrs.get("producer", "")
    consumer = op.attrs.get("consumer", "")
    via_tensor = op.attrs.get("via_tensor", "")
    discharged: list[str] = []
    failures: list[str] = []

    # Producer / consumer regions exist
    pkind = _region_kind(ctx, producer)
    ckind = _region_kind(ctx, consumer)
    if not pkind:
        failures.append(f"producer region {producer!r} not in region_map")
    else:
        discharged.append("producer_region_exists")
    if not ckind:
        failures.append(f"consumer region {consumer!r} not in region_map")
    else:
        discharged.append("consumer_region_exists")
    if pkind and ckind:
        if _is_opaque(pkind) or _is_opaque(ckind):
            failures.append(
                f"opaque endpoint(s): producer={pkind!r}, consumer={ckind!r}"
            )
        else:
            discharged.append("endpoints_are_structured")

    # via_tensor exists with single consumer + transient lifetime + reuse_horizon ≤ 1
    tensor_record: dict[str, Any] | None = None
    for t in ctx.use_def.get("tensors", []):
        if t["tensor_id"] == via_tensor:
            tensor_record = t
            break
    if tensor_record is None:
        failures.append(f"via_tensor {via_tensor!r} not in tensor_use_def_graph")
    else:
        discharged.append("via_tensor_exists")
        if tensor_record.get("consumer_count") != 1:
            failures.append(
                f"via_tensor {via_tensor!r} consumer_count={tensor_record.get('consumer_count')}, expected 1"
            )
        else:
            discharged.append("via_tensor_consumer_count_is_one")
        horizon = tensor_record.get("reuse_horizon", -1)
        if horizon < 0 or horizon > 1:
            failures.append(
                f"via_tensor {via_tensor!r} reuse_horizon={horizon}, expected ≤ 1"
            )
        else:
            discharged.append("via_tensor_reuse_horizon_is_immediate")
        if tensor_record.get("producer_lifetime_class") != "transient":
            failures.append(
                f"via_tensor {via_tensor!r} producer_lifetime_class="
                f"{tensor_record.get('producer_lifetime_class')!r}, expected transient"
            )
        else:
            discharged.append("via_tensor_is_transient")

    # Region-graph edge from producer → consumer carrying via_tensor
    edge_found = False
    for e in ctx.region_graph.get("edges", []):
        if e["src"] == producer and e["dst"] == consumer and e["tensor_id"] == via_tensor:
            edge_found = True
            break
    if edge_found:
        discharged.append("region_graph_edge_exists")
    else:
        failures.append(
            f"no region_graph edge {producer!r}→{consumer!r} via {via_tensor!r}"
        )

    # Refinement: bit_equality unless either endpoint is reduction-class
    refinement = "bit_equality"
    if pkind in _REDUCTION_KINDS or ckind in _REDUCTION_KINDS:
        refinement = "tolerance_eps"

    return GateOpVerdict(
        recipe_op_id=op.recipe_op_id,
        source_candidate=op.attrs.get("source_candidate", resolved.candidate_id),
        op=op.op_camel,
        region=producer,  # convention: fusion is anchored at the producer
        gate_status="pass" if not failures else "fail",
        declared_refinement=refinement,
        proof_stage="post_lowering",
        verifier_chain=["structural", "differential"],
        semantic_obligation=f"obl_{op.recipe_op_id}",
        discharged_now=discharged,
        deferred_until_lowering=[
            "fused_payload_transform_structural_check",
            "post_lowering_differential_check",
        ],
        failure_reasons=failures,
        extra={"consumer": consumer, "via_tensor": via_tensor},
    )


def _gate_extension_closure(
    ctx: _GateContext,
    op: _ParsedRecipeOp,
    resolved: ResolvedCandidate,
) -> GateOpVerdict:
    region_id = op.attrs.get("region", "") or resolved.region_id
    discharged: list[str] = []
    failures: list[str] = []

    # Look up the source_classification (the canonical "this is an opaque
    # fallback" signal) directly from region_map. The ``kind`` field is a
    # heuristic name (``elementwise_gelu`` for crgtoy.affine_gelu post-lowering),
    # while ``source_classification`` reliably says ``opaque_fallback``.
    region_record = next(
        (r for r in ctx.region_map.get("regions", []) if r["region_id"] == region_id),
        None,
    )
    if region_record is None:
        failures.append(f"region {region_id!r} not in region_map")
    else:
        discharged.append("region_exists")
        src_cls = region_record.get("source_classification", "")
        if op.op_camel == "KeepAsFallback":
            discharged.append("keep_as_fallback_always_legal")
        elif src_cls == "opaque_fallback":
            discharged.append("region_is_opaque_fallback")
        else:
            failures.append(
                f"extension_closure on non-opaque region {region_id!r} "
                f"(source_classification={src_cls!r})"
            )

    # Shape/dtype evidence: the candidate's recipe_delta or the resolved
    # candidate's evidence must have payload_ops.
    has_signatures = bool(resolved.cost_preview) or bool(resolved.evidence)
    if has_signatures:
        discharged.append("shape_dtype_evidence_present")
    else:
        failures.append("no shape/dtype evidence in resolved candidate")

    # Gap-discovery linkage (when gap_action_queue is available)
    gap = ctx.gap_lookup.get(region_id)
    if gap is not None:
        discharged.append("gap_record_present")
    else:
        # Not required, but recorded.
        pass

    # source payload_ref exists (when present)
    payload_ref = resolved.evidence.get("payload_ref", "")
    if payload_ref:
        if (ctx.run_dir / payload_ref).exists():
            discharged.append("source_payload_ref_exists")
        else:
            failures.append(f"payload_ref does not exist: {payload_ref}")

    if op.op_camel == "CreatePayloadLoweringExtension":
        refinement = "extension_obligation"
        proof_stage = "extension_verify"
        chain = ["contract_schema", "differential", "locked_reference"]
    elif op.op_camel == "CreateKernelContract":
        refinement = "contract_obligation"
        proof_stage = "kernel_contract_generation"
        chain = ["contract_schema", "differential", "locked_reference"]
    else:  # KeepAsFallback
        refinement = "fallback_obligation"
        proof_stage = "always_pass"
        chain = ["differential"]

    return GateOpVerdict(
        recipe_op_id=op.recipe_op_id,
        source_candidate=op.attrs.get("source_candidate", resolved.candidate_id),
        op=op.op_camel,
        region=region_id,
        gate_status="pass" if not failures else "fail",
        declared_refinement=refinement,
        proof_stage=proof_stage,
        verifier_chain=chain,
        semantic_obligation=f"obl_{op.recipe_op_id}",
        discharged_now=discharged,
        deferred_until_lowering=[
            "extension_locked_files_audit",
            "extension_differential_test",
        ] if op.op_camel != "KeepAsFallback" else [],
        failure_reasons=failures,
        extra={"gap_id": (gap or {}).get("gap_id", "")},
    )


def _gate_numerics(
    ctx: _GateContext,
    op: _ParsedRecipeOp,
    resolved: ResolvedCandidate,
) -> GateOpVerdict:
    region_id = op.attrs.get("region", "") or resolved.region_id
    discharged: list[str] = []
    failures: list[str] = []

    dossier_ref = ctx.region_dossier_paths.get(region_id)
    sens_key, dtype_required = _numerics_kind_to_sens_key(op.op_camel)
    if dossier_ref is None:
        failures.append(f"region dossier missing for {region_id!r}")
    else:
        d = json.loads((ctx.run_dir / dossier_ref).read_text(encoding="utf-8"))
        sens = d["numerical_sensitivity"].get(sens_key)
        if sens is None:
            failures.append(f"numerical_sensitivity[{sens_key!r}] missing")
        else:
            discharged.append("numerical_sensitivity_entry_present")
            if sens["status"] == "safe":
                discharged.append("numerical_sensitivity_safe")
            elif sens["status"] == "risky" and ctx.allow_risky_numerics:
                discharged.append("numerical_sensitivity_risky_explicitly_allowed")
            else:
                failures.append(
                    f"numerical_sensitivity[{sens_key!r}].status={sens['status']!r} "
                    f"(allow_risky_numerics={ctx.allow_risky_numerics})"
                )

    if dtype_required and dtype_required not in ctx.profile.supported_dtypes:
        failures.append(
            f"target {ctx.profile.target_id!r} does not support dtype {dtype_required!r}"
        )
    elif dtype_required:
        discharged.append("target_supports_required_dtype")

    return GateOpVerdict(
        recipe_op_id=op.recipe_op_id,
        source_candidate=op.attrs.get("source_candidate", resolved.candidate_id),
        op=op.op_camel,
        region=region_id,
        gate_status="pass" if not failures else "fail",
        declared_refinement="tolerance_eps",
        proof_stage="post_lowering",
        verifier_chain=["differential"],
        semantic_obligation=f"obl_{op.recipe_op_id}",
        discharged_now=discharged,
        deferred_until_lowering=[
            "post_lowering_differential_check",
            "tolerance_window_validation",
        ],
        failure_reasons=failures,
        extra={"sens_key": sens_key, "dtype_required": dtype_required or ""},
    )


def _numerics_kind_to_sens_key(op_camel: str) -> tuple[str, str | None]:
    if op_camel == "QuantizeFP8":
        return "fp8_e4m3", None  # FP8 not enumerated in supported_dtypes by default
    if op_camel == "SetAccumulator":
        return "fp16_accum", "fp16"
    if op_camel == "EnableFastMath":
        return "fast_math", None
    return "fp32", None


def _gate_assign_device(
    ctx: _GateContext,
    op: _ParsedRecipeOp,
    resolved: ResolvedCandidate,
) -> GateOpVerdict:
    region_id = op.attrs.get("region", "") or resolved.region_id
    device = op.attrs.get("device", "")
    discharged: list[str] = []
    failures: list[str] = []

    dossier_ref = ctx.region_dossier_paths.get(region_id)
    if dossier_ref is None:
        failures.append(f"region dossier missing for {region_id!r}")
    else:
        d = json.loads((ctx.run_dir / dossier_ref).read_text(encoding="utf-8"))
        envelope = d["placement_envelope"]["devices"]
        env = next((e for e in envelope if e["device"] == device), None)
        if env is None:
            failures.append(f"device {device!r} not in placement envelope")
        else:
            discharged.append("placement_envelope_includes_device")
            if env["memory_fit"]:
                discharged.append("placement_memory_fit")
            else:
                failures.append(f"device {device!r} memory_fit=False")

    return GateOpVerdict(
        recipe_op_id=op.recipe_op_id,
        source_candidate=op.attrs.get("source_candidate", resolved.candidate_id),
        op=op.op_camel,
        region=region_id,
        gate_status="pass" if not failures else "fail",
        declared_refinement="placement_obligation",
        proof_stage="runtime_emission",
        verifier_chain=["placement_check"],
        semantic_obligation=f"obl_{op.recipe_op_id}",
        discharged_now=discharged,
        deferred_until_lowering=[
            "runtime_dispatch_dispatchability",
        ],
        failure_reasons=failures,
        extra={"device": device},
    )


_GATE_DISPATCH = {
    "SetTileParams": _gate_set_tile_params,
    "FuseProducerConsumer": _gate_fuse_producer_consumer,
    "CreatePayloadLoweringExtension": _gate_extension_closure,
    "CreateKernelContract": _gate_extension_closure,
    "KeepAsFallback": _gate_extension_closure,
    "QuantizeFP8": _gate_numerics,
    "SetAccumulator": _gate_numerics,
    "EnableFastMath": _gate_numerics,
    "AssignDevice": _gate_assign_device,
}


# --------------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------------- #


def _utcnow() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _mlir_attr(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return f"{v} : i64"
    if isinstance(v, float):
        return f"{v} : f64"
    if v is None:
        return '"null"'
    if isinstance(v, list):
        return "[" + ", ".join(_mlir_attr(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{k} = {_mlir_attr(val)}" for k, val in sorted(v.items())) + " }"
    s = str(v).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{s}"'


def _emit_attrs(d: dict[str, Any]) -> str:
    return ", ".join(f"{k} = {_mlir_attr(d[k])}" for k in sorted(d))


def _emit_semantic_obligations_mlir(
    *, model_id: str, target_id: str, source_recipe: str,
    verdicts: list[GateOpVerdict],
) -> str:
    lines = []
    head = {
        "model_id": model_id,
        "target_id": target_id,
        "source_recipe": source_recipe,
        "obligation_count": len(verdicts),
    }
    lines.append(f"sem.module @{model_id}_{target_id} attributes {{ {_emit_attrs(head)} }} {{")
    for v in verdicts:
        attrs = {
            "recipe_op": f"@{v.recipe_op_id}",
            "source_candidate": v.source_candidate,
            "recipe_kind": v.op,
            "region": v.region or "",
            "refinement": v.declared_refinement,
            "proof_stage": v.proof_stage,
            "verifier_chain": list(v.verifier_chain),
            "status": "declared" if v.gate_status == "pass" else "declared_failed",
            "gate_status": v.gate_status,
        }
        for k, val in v.extra.items():
            if k not in attrs:
                attrs[k] = val
        lines.append(f"  sem.obligation @{v.semantic_obligation} attributes {{ {_emit_attrs(attrs)} }}")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _emit_verified_recipe_mlir(
    *,
    module_attrs: dict[str, Any],
    ops: list[_ParsedRecipeOp],
    verdicts_by_id: dict[str, GateOpVerdict],
    overall: str,
) -> str:
    head = dict(module_attrs)
    head["recipe_gate_status"] = overall
    lines = [f"recipe.module @verified_recipe attributes {{ {_emit_attrs(head)} }} {{"]
    for op in ops:
        v = verdicts_by_id[op.recipe_op_id]
        body = dict(op.attrs)
        body["gate_status"] = v.gate_status
        body["declared_refinement"] = v.declared_refinement
        # ``semantic_obligation`` is an MLIR symbol reference (``@<name>``)
        # not a string — emit it unquoted so it round-trips like other
        # symbol attrs in the canonical IR.
        attrs_text = _emit_attrs(body)
        attrs_text += (
            f', semantic_obligation = @{v.semantic_obligation}'
        )
        lines.append(
            f"  recipe.{op.op_snake} @{op.recipe_op_id} attributes "
            f"{{ {attrs_text} }}"
        )
    lines.append("}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RecipeGateResult:
    overall: str  # "pass" | "fail"
    verdict_path: Path
    trace_path: Path
    semantic_obligations_mlir_path: Path
    semantic_obligations_json_path: Path
    verified_recipe_path: Path
    verdicts: tuple[GateOpVerdict, ...]


def _read_json(path: Path) -> dict[str, Any]:
    obj: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return obj


def run_recipe_gate(
    run_dir: Path,
    *,
    target_yaml_path: Path | None = None,
    allow_risky_numerics: bool = False,
) -> RecipeGateResult:
    """Run the M-06 verification gate against an existing recipe_planning output.

    Inputs (read-only):

    - ``03_recipe_planning/recipe.mlir``
    - ``03_recipe_planning/candidate_selection.json``
    - ``02_graph_analysis/`` (region_map, region_dossiers, tensor_use_def_graph,
      region_graph, action_space.mlir, candidate_actions.json, ...)
    - optionally ``04_gap_discovery/gap_action_queue.json``

    Outputs (written):

    - ``03_recipe_planning/recipe_gate_verdict.json``
    - ``03_recipe_planning/recipe_gate_trace.jsonl``
    - ``03_recipe_planning/semantic_obligations.mlir``
    - ``03_recipe_planning/semantic_obligations.json``
    - ``03_recipe_planning/verified_recipe.mlir``

    Also amends ``03_recipe_planning/recipe_validation.json`` and
    ``03_recipe_planning/recipe_summary.json`` with the gate verdict.
    """
    run_dir = Path(run_dir).resolve()
    rp_dir = run_dir / "03_recipe_planning"
    ga_dir = run_dir / "02_graph_analysis"
    if not rp_dir.is_dir():
        raise FileNotFoundError(f"03_recipe_planning/ missing under {run_dir}")
    if not ga_dir.is_dir():
        raise FileNotFoundError(f"02_graph_analysis/ missing under {run_dir}")

    if target_yaml_path is None:
        repo_root = Path(__file__).resolve().parents[3]
        target_yaml_path = repo_root / "configs" / "targets" / "host_cpu.yaml"
    profile = load_target_profile(Path(target_yaml_path))

    recipe_mlir_path = rp_dir / "recipe.mlir"
    selection_path = rp_dir / "candidate_selection.json"
    if not recipe_mlir_path.exists():
        raise FileNotFoundError(f"recipe.mlir missing: {recipe_mlir_path}")
    if not selection_path.exists():
        raise FileNotFoundError(f"candidate_selection.json missing: {selection_path}")

    recipe_text = recipe_mlir_path.read_text(encoding="utf-8")
    module_attrs, ops = _parse_recipe_mlir(recipe_text)
    selection = _read_json(selection_path)

    region_map = _read_json(ga_dir / "region_map.json")
    use_def = _read_json(ga_dir / "tensor_use_def_graph.json")
    region_graph = _read_json(ga_dir / "region_graph.json")
    graph_dossier = _read_json(ga_dir / "graph_dossier_v2.json")
    region_dossier_paths = dict(graph_dossier["region_dossiers"])

    # Optional gap_action_queue.json — index by region_id.
    from compgen.graph_compilation.artifacts import stage_dir

    gap_lookup: dict[str, dict[str, Any]] = {}
    gd_dir = stage_dir(run_dir, "gap_discovery")
    queue_path = Path(gd_dir) / "gap_action_queue.json"
    if queue_path.exists():
        try:
            queue = _read_json(queue_path)
            for g in queue.get("gaps", []):
                rid = g.get("region_id", "")
                if rid and rid not in gap_lookup:
                    gap_lookup[rid] = g
        except (json.JSONDecodeError, OSError):
            pass

    ctx = _GateContext(
        run_dir=run_dir,
        region_map=region_map,
        use_def=use_def,
        region_graph=region_graph,
        region_dossier_paths=region_dossier_paths,
        profile=profile,
        gap_lookup=gap_lookup,
        allow_risky_numerics=allow_risky_numerics,
    )

    # Trace events: every check we make (resolver pass + per-op gate dispatch).
    trace: list[dict[str, Any]] = []

    def _trace(event: str, op_id: str, status: str, detail: str = "") -> None:
        trace.append(
            {
                "schema_version": "recipe_gate_trace_event_v1",
                "timestamp_utc": _utcnow(),
                "event": event,
                "recipe_op_id": op_id,
                "status": status,
                "detail": detail,
            }
        )

    verdicts: list[GateOpVerdict] = []
    overall_pass = True

    for op in ops:
        cand_id = op.attrs.get("source_candidate")
        if not isinstance(cand_id, str) or not cand_id:
            verdicts.append(
                GateOpVerdict(
                    recipe_op_id=op.recipe_op_id,
                    source_candidate="",
                    op=op.op_camel,
                    region=op.attrs.get("region"),
                    gate_status="fail",
                    declared_refinement="undefined",
                    proof_stage="undefined",
                    verifier_chain=[],
                    semantic_obligation=f"obl_{op.recipe_op_id}",
                    failure_reasons=[
                        "recipe op missing source_candidate attribute"
                    ],
                )
            )
            overall_pass = False
            _trace("source_candidate_missing", op.recipe_op_id, "fail", "")
            continue

        # 1. Resolve via M-04.5 (hash-chain + recipe_delta cross-check).
        try:
            resolved, _ = resolve_candidate(run_dir, cand_id)
            _trace("resolver_pass", op.recipe_op_id, "pass", cand_id)
        except (
            HashMismatchError,
            IllegalCandidateError,
            RecipeDeltaMismatchError,
            ResolverError,
        ) as exc:
            verdicts.append(
                GateOpVerdict(
                    recipe_op_id=op.recipe_op_id,
                    source_candidate=cand_id,
                    op=op.op_camel,
                    region=op.attrs.get("region"),
                    gate_status="fail",
                    declared_refinement="undefined",
                    proof_stage="undefined",
                    verifier_chain=[],
                    semantic_obligation=f"obl_{op.recipe_op_id}",
                    failure_reasons=[f"{type(exc).__name__}: {exc}"],
                )
            )
            overall_pass = False
            _trace("resolver_fail", op.recipe_op_id, "fail", f"{type(exc).__name__}: {exc}")
            continue

        # 2. Family dispatch.
        gate_fn = _GATE_DISPATCH.get(op.op_camel)
        if gate_fn is None:
            verdicts.append(
                GateOpVerdict(
                    recipe_op_id=op.recipe_op_id,
                    source_candidate=cand_id,
                    op=op.op_camel,
                    region=op.attrs.get("region"),
                    gate_status="fail",
                    declared_refinement="undefined",
                    proof_stage="undefined",
                    verifier_chain=[],
                    semantic_obligation=f"obl_{op.recipe_op_id}",
                    failure_reasons=[f"no gate dispatch for op kind {op.op_camel!r}"],
                )
            )
            overall_pass = False
            _trace("no_gate_dispatch", op.recipe_op_id, "fail", op.op_camel)
            continue

        verdict = gate_fn(ctx, op, resolved)
        for d in verdict.discharged_now:
            _trace(f"discharged:{d}", op.recipe_op_id, "pass", "")
        for fail_reason in verdict.failure_reasons:
            _trace("gate_failure", op.recipe_op_id, "fail", fail_reason)
        verdicts.append(verdict)
        if verdict.gate_status != "pass":
            overall_pass = False

    overall = "pass" if overall_pass else "fail"

    # ------------------------------------------------------------------ #
    # Emit artifacts.
    # ------------------------------------------------------------------ #
    model_id = selection.get("model_id", "model")
    target_id = selection.get("target_id", profile.target_id)
    action_space_ir_sha = selection["source"].get("action_space_ir_sha256", "")
    import hashlib

    recipe_sha = "sha256:" + hashlib.sha256(recipe_text.encode("utf-8")).hexdigest()

    # 1. semantic_obligations.mlir
    sem_mlir_path = rp_dir / "semantic_obligations.mlir"
    sem_mlir_path.write_text(
        _emit_semantic_obligations_mlir(
            model_id=_safe_id(model_id),
            target_id=_safe_id(target_id),
            source_recipe="03_recipe_planning/recipe.mlir",
            verdicts=verdicts,
        ),
        encoding="utf-8",
    )

    # 2. semantic_obligations.json (projection)
    sem_json_path = rp_dir / "semantic_obligations.json"
    sem_json_path.write_text(
        json.dumps(
            {
                "schema_version": "semantic_obligations_v1",
                "model_id": model_id,
                "target_id": target_id,
                "source": {
                    "recipe": "03_recipe_planning/recipe.mlir",
                    "recipe_sha256": recipe_sha,
                    "action_space_ir_sha256": action_space_ir_sha,
                },
                "obligations": [
                    {
                        "id": v.semantic_obligation,
                        "recipe_op_id": v.recipe_op_id,
                        "source_candidate": v.source_candidate,
                        "recipe_kind": v.op,
                        "region": v.region,
                        "refinement": v.declared_refinement,
                        "proof_stage": v.proof_stage,
                        "verifier_chain": list(v.verifier_chain),
                        "status": "declared" if v.gate_status == "pass" else "declared_failed",
                        "gate_status": v.gate_status,
                    }
                    for v in verdicts
                ],
            },
            indent=2, sort_keys=True,
        ),
        encoding="utf-8",
    )

    # 3. verified_recipe.mlir
    verified_path = rp_dir / "verified_recipe.mlir"
    verified_path.write_text(
        _emit_verified_recipe_mlir(
            module_attrs=module_attrs,
            ops=ops,
            verdicts_by_id={v.recipe_op_id: v for v in verdicts},
            overall=overall,
        ),
        encoding="utf-8",
    )

    # 4. recipe_gate_verdict.json
    verdict_path = rp_dir / "recipe_gate_verdict.json"
    verdict_path.write_text(
        json.dumps(
            {
                "schema_version": "recipe_gate_verdict_v1",
                "status": overall,
                "model_id": model_id,
                "target_id": target_id,
                "recipe": "03_recipe_planning/recipe.mlir",
                "verified_recipe": "03_recipe_planning/verified_recipe.mlir",
                "source": {
                    "action_space_ir": "02_graph_analysis/action_space.mlir",
                    "action_space_ir_sha256": action_space_ir_sha,
                    "recipe_sha256": recipe_sha,
                },
                "checked_recipe_ops": [
                    {
                        "recipe_op_id": v.recipe_op_id,
                        "source_candidate": v.source_candidate,
                        "op": v.op,
                        "region": v.region,
                        "gate_status": v.gate_status,
                        "declared_refinement": v.declared_refinement,
                        "proof_stage": v.proof_stage,
                        "verifier_chain": list(v.verifier_chain),
                        "semantic_obligation": v.semantic_obligation,
                        "discharged_now": list(v.discharged_now),
                        "deferred_until_lowering": list(v.deferred_until_lowering),
                        "failure_reasons": list(v.failure_reasons),
                        "extra": dict(v.extra),
                    }
                    for v in verdicts
                ],
                "summary": {
                    "recipe_ops_total": len(verdicts),
                    "passed": sum(1 for v in verdicts if v.gate_status == "pass"),
                    "failed": sum(1 for v in verdicts if v.gate_status != "pass"),
                    "deferred_semantic_obligations": sum(
                        1 for v in verdicts if v.deferred_until_lowering
                    ),
                },
            },
            indent=2, sort_keys=True,
        ),
        encoding="utf-8",
    )

    # 5. recipe_gate_trace.jsonl
    trace_path = rp_dir / "recipe_gate_trace.jsonl"
    with trace_path.open("w", encoding="utf-8") as f:
        for ev in trace:
            f.write(json.dumps(ev) + "\n")

    # 6. Amend recipe_validation.json + recipe_summary.json
    _amend_recipe_validation(rp_dir, overall, verdicts)
    _amend_recipe_summary(rp_dir, overall, verdicts)

    return RecipeGateResult(
        overall=overall,
        verdict_path=verdict_path,
        trace_path=trace_path,
        semantic_obligations_mlir_path=sem_mlir_path,
        semantic_obligations_json_path=sem_json_path,
        verified_recipe_path=verified_path,
        verdicts=tuple(verdicts),
    )


def _safe_id(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", s).strip("_") or "x"


def _amend_recipe_validation(
    rp_dir: Path, gate_overall: str, verdicts: list[GateOpVerdict]
) -> None:
    path = rp_dir / "recipe_validation.json"
    if not path.exists():
        return
    obj = json.loads(path.read_text(encoding="utf-8"))
    checks = obj.setdefault("checks", [])
    checks.append(
        {
            "name": "recipe_gate_overall",
            "status": gate_overall,
            "detail": f"verdicts={len(verdicts)}; failures="
            f"{sum(1 for v in verdicts if v.gate_status != 'pass')}",
        }
    )
    if gate_overall != "pass":
        obj["overall"] = "fail"
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _amend_recipe_summary(
    rp_dir: Path, gate_overall: str, verdicts: list[GateOpVerdict]
) -> None:
    path = rp_dir / "recipe_summary.json"
    if not path.exists():
        return
    obj = json.loads(path.read_text(encoding="utf-8"))
    obj["recipe_gate_status"] = gate_overall
    obj["recipe_gate_obligation_count"] = len(verdicts)
    obj["recipe_gate_passed_op_count"] = sum(
        1 for v in verdicts if v.gate_status == "pass"
    )
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
