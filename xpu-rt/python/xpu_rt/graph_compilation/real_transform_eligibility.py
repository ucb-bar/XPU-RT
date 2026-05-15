"""Real Transform Eligibility Audit (Milestone 11A).

Read-only audit that classifies each model's selected recipe with respect
to a narrow real-matmul-tiling MVP. The MVP target is intentionally
restrictive:

- selected recipe kind == ``SetTileParams``
- target op == ``linalg.matmul``
- tensor shapes are static, rank-2, ``f32``
- target op appears exactly once for the selected ``compgen.region_id``
- tile (M, N, K) parsed from ``verified_recipe.mlir`` matches the tile
  parsed from ``applied_transform_manifest.json``
- the tile came from the legal candidate menu (i.e. the action-space
  working-set curve)
- source ``payload.mlir`` is byte-identical pre/post audit

This stage is purely an audit. It does not apply transforms, it does not
mutate Payload IR, and it does not emit ``transformed_payload.real.mlir``.
will use the resulting eligibility report to drive a real
SetTileParams transform for the eligible cases.

Hard non-goals:

No real loop tiling.
No differential verification.
- No codegen, benchmarks, profiler feedback.
- No compiler-core changes.

When the selected recipe is not a SetTileParams (e.g. fusion or contract
draft), the audit cleanly marks ``eligible=false`` with a precise
rejection reason. Such cases are NOT pipeline failures — the audit's own
``status`` stays ``pass`` because the audit itself ran cleanly.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# Result dataclass + entry point
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RealTransformEligibilityResult:
    overall: str  # "pass" | "fail" — audit-side status, NOT eligibility
    eligible: bool
    out_dir: Path
    json_path: Path
    md_path: Path
    rejection_reasons: tuple[str, ...]


def _utcnow() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> dict[str, Any]:
    obj: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return obj


def _read_json_or_none(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _read_json(path)


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# Parsers
# --------------------------------------------------------------------------- #


_RECIPE_TILE_RE = re.compile(
    r"recipe\.set_tile_params\s+@(?P<rop>[A-Za-z_][A-Za-z0-9_]*)\b"
    r"(?P<body>.*?)"
    r"tile\s*=\s*\{\s*"
    r"K\s*=\s*(?P<K>\d+)\s*:\s*i\d+\s*,\s*"
    r"M\s*=\s*(?P<M>\d+)\s*:\s*i\d+\s*,\s*"
    r"N\s*=\s*(?P<N>\d+)\s*:\s*i\d+\s*\}",
    re.DOTALL,
)


def _parse_tile_from_verified_recipe(
    text: str, recipe_op_id: str
) -> tuple[int, int, int] | None:
    for m in _RECIPE_TILE_RE.finditer(text):
        if m.group("rop") == recipe_op_id:
            return int(m.group("M")), int(m.group("N")), int(m.group("K"))
    return None


# Match a single `linalg.matmul` op line, capturing its inline attribute
# block + LHS/RHS input tensor types and result tensor type. Restricted
# to single-line ops (the importer always emits matmuls on one line).
_MATMUL_LINE_RE = re.compile(
    r"^\s*%\w+\s*=\s*linalg\.matmul\s*"
    r"(?:\{(?P<attrs>[^}]*)\}\s*)?"
    r"ins\(\s*(?P<ins>[^)]*?)\s*:\s*(?P<in_types>[^)]*)\)\s*"
    r"outs\(\s*[^)]*?\s*:\s*(?P<out_type>[^)]*)\)\s*"
    r"->\s*(?P<ret_type>tensor<[^>]+>).*$",
    re.MULTILINE,
)

_TENSOR_TYPE_RE = re.compile(r"tensor<\s*([^>]+?)\s*>")
_TENSOR_DIMS_DTYPE_RE = re.compile(r"^([0-9x?]*)x([a-zA-Z][a-zA-Z0-9_]*)$")


def _parse_tensor_type(tt: str) -> dict[str, Any] | None:
    """Parse a ``tensor<DxDxDx...DxELEMENT>`` text into shape/dtype."""
    m = _TENSOR_TYPE_RE.match(tt.strip())
    if not m:
        return None
    body = m.group(1)
    dm = _TENSOR_DIMS_DTYPE_RE.match(body)
    if dm is None:
        # Could be 0-d (e.g. tensor<f32>) — treat as rank-0.
        return {"rank": 0, "dims": [], "dtype": body, "dynamic": False}
    dims_text = dm.group(1)
    dtype = dm.group(2)
    if not dims_text:
        return {"rank": 0, "dims": [], "dtype": dtype, "dynamic": False}
    parts = dims_text.split("x")
    dims: list[int | None] = []
    dynamic = False
    for p in parts:
        if p == "?":
            dims.append(None)
            dynamic = True
        else:
            dims.append(int(p))
    return {"rank": len(dims), "dims": dims, "dtype": dtype, "dynamic": dynamic}


def _split_tensor_list(tt_list: str) -> list[str]:
    """Split a ``ins(...)`` operand-types list like
    ``tensor<4x64xf32>, tensor<64x128xf32>`` into individual tensor types,
    respecting the matching ``<...>`` brackets."""
    out: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in tt_list:
        if ch == "<":
            depth += 1
            cur.append(ch)
        elif ch == ">":
            depth -= 1
            cur.append(ch)
        elif ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        out.append(tail)
    return out


def _attr_value(attrs_text: str, key: str) -> str | None:
    """Extract a string attribute value (``key = "value"``) from the
    inline attribute block of an op line."""
    pat = re.compile(rf'\b{re.escape(key)}\s*=\s*"([^"]*)"')
    m = pat.search(attrs_text)
    return m.group(1) if m else None


# --------------------------------------------------------------------------- #
# Auditor
# --------------------------------------------------------------------------- #


def _resolve_payload_ref_for_region(
    region_map: dict[str, Any], region_id: str
) -> str | None:
    """Find the payload.mlir path that contains the given region (taking
    the first ``payload_ops[].payload_ref`` for the region)."""
    for region in region_map.get("regions", []):
        if region.get("region_id") != region_id:
            continue
        for op in region.get("payload_ops", []):
            ref = op.get("payload_ref")
            if ref:
                return ref
    return None


def _find_matmul_for_region(
    payload_text: str, region_id: str
) -> list[dict[str, Any]]:
    """Return all ``linalg.matmul`` op-line records whose attribute block
    contains ``compgen.region_id = "<region_id>"``. Preserves order."""
    results: list[dict[str, Any]] = []
    for m in _MATMUL_LINE_RE.finditer(payload_text):
        attrs = m.group("attrs") or ""
        if _attr_value(attrs, "compgen.region_id") != region_id:
            continue
        in_types = _split_tensor_list(m.group("in_types"))
        out_type = m.group("ret_type")
        lhs = _parse_tensor_type(in_types[0]) if in_types else None
        rhs = _parse_tensor_type(in_types[1]) if len(in_types) > 1 else None
        out = _parse_tensor_type(out_type)
        results.append(
            {
                "line_offset": m.start(),
                "attrs": attrs,
                "lhs": lhs,
                "rhs": rhs,
                "out": out,
                "transposed_b": _attr_value(attrs, "compgen.transposed_b") == "true",
            }
        )
    return results


def _matmul_signature(
    matmul: dict[str, Any]
) -> dict[str, Any] | None:
    """Derive M/N/K + dtype + rank/dynamic from a parsed ``linalg.matmul``.

    The ``linalg.matmul`` op semantics are fixed regardless of any upstream
    ``compgen.transposed_b`` marker (which only records that the importer
    absorbed an ``aten.permute`` to feed this matmul):

    - LHS tensor is ``MxK``
    - RHS tensor is ``KxN``
    - OUT tensor is ``MxN``
    """
    lhs = matmul.get("lhs")
    rhs = matmul.get("rhs")
    out = matmul.get("out")
    if not (lhs and rhs and out):
        return None
    if lhs.get("rank") != 2 or rhs.get("rank") != 2 or out.get("rank") != 2:
        return None
    if lhs.get("dynamic") or rhs.get("dynamic") or out.get("dynamic"):
        return None
    M_lhs, K_lhs = lhs["dims"]
    K_rhs, N_rhs = rhs["dims"]
    M_out, N_out = out["dims"]
    if M_lhs != M_out or N_rhs != N_out or K_lhs != K_rhs:
        return None
    return {
        "M": int(M_lhs),
        "N": int(N_rhs),
        "K": int(K_lhs),
        "lhs_dtype": lhs["dtype"],
        "rhs_dtype": rhs["dtype"],
        "out_dtype": out["dtype"],
        "rank": 2,
        "dynamic_dims": False,
    }


def _selected_candidate_for_region(
    candidate_actions: dict[str, Any], candidate_id: str
) -> dict[str, Any] | None:
    for c in candidate_actions.get("candidates", []):
        if c.get("candidate_id") == candidate_id:
            return c
    return None


def _md_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    lines = [
        "| model | eligible | recipe_kind | reason |",
        "|---|---|---|---|",
    ]
    for r in rows:
        reason = ", ".join(r["rejection_reasons"]) or "—"
        lines.append(
            f"| {r['model_id']} | {r['eligible']} | "
            f"{r['recipe_kind'] or '—'} | {reason} |"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Top-level entry point
# --------------------------------------------------------------------------- #


def run_real_transform_eligibility(
    run_dir: Path,
) -> RealTransformEligibilityResult:
    run_dir = Path(run_dir).resolve()
    rp = run_dir / "03_recipe_planning"
    ga = run_dir / "02_graph_analysis"
    pl = run_dir / "01_payload_lowering"
    if not rp.is_dir():
        raise FileNotFoundError(f"03_recipe_planning/ missing under {run_dir}")
    if not ga.is_dir():
        raise FileNotFoundError(f"02_graph_analysis/ missing under {run_dir}")

    region_map_path = ga / "region_map.json"
    candidate_actions_path = ga / "candidate_actions.json"
    verified_recipe_path = rp / "verified_recipe.mlir"
    candidate_selection_path = rp / "candidate_selection.json"
    semantic_obligations_path = rp / "semantic_obligations.json"
    applied_manifest_path = rp / "post_lowering" / "applied_transform_manifest.json"

    if not region_map_path.exists():
        raise FileNotFoundError(f"region_map.json missing: {region_map_path}")
    if not candidate_actions_path.exists():
        raise FileNotFoundError(
            f"candidate_actions.json missing: {candidate_actions_path}"
        )

    region_map = _read_json(region_map_path)
    candidate_actions = _read_json(candidate_actions_path)
    candidate_selection = _read_json_or_none(candidate_selection_path)
    semantic_obligations = _read_json_or_none(semantic_obligations_path)
    applied_manifest = _read_json_or_none(applied_manifest_path)
    verified_recipe_text = (
        verified_recipe_path.read_text(encoding="utf-8")
        if verified_recipe_path.exists() else ""
    )

    # Pre-snapshot SHA of every payload.mlir under 01_payload_lowering/.
    pre_payload_shas: dict[str, str] = {
        str(p.relative_to(run_dir)): _sha256_file(p)
        for p in sorted(pl.rglob("payload.mlir"))
    }

    out_dir = rp
    out_dir.mkdir(parents=True, exist_ok=True)

    model_id = (
        (candidate_selection or {}).get("model_id", "")
        or (applied_manifest or {}).get("model_id", "")
        or (semantic_obligations or {}).get("model_id", "")
        or ""
    )
    target_id = (
        (candidate_selection or {}).get("target_id", "")
        or (semantic_obligations or {}).get("target_id", "")
        or ""
    )

    checks: list[dict[str, Any]] = []
    rejections: list[str] = []

    def _add(name: str, ok: bool, detail: str = "") -> None:
        checks.append(
            {
                "name": name,
                "status": "pass" if ok else "fail",
                "detail": detail,
            }
        )

    # ------------------------------------------------------------------ #
    # 1. selected recipe exists and is SetTileParams
    # ------------------------------------------------------------------ #
    selected_kind = (candidate_selection or {}).get("candidate_kind", "")
    selected_recipe_kind = ""  # capitalized form (SetTileParams etc.)
    selected_block: dict[str, Any] = {}
    if candidate_selection is None:
        _add("recipe_kind_set_tile_params", False, "no candidate_selection.json")
        rejections.append("no selected recipe (candidate_selection.json missing)")
    else:
        # candidate_selection.candidate_kind is snake_case; map to recipe op.
        kind_map = {
            "set_tile_params": "SetTileParams",
            "fuse_producer_consumer": "FuseProducerConsumer",
            "create_kernel_contract": "CreateKernelContract",
            "create_payload_lowering_extension":
                "CreatePayloadLoweringExtension",
            "keep_as_fallback": "KeepAsFallback",
            "quantize_fp8": "QuantizeFP8",
            "set_accumulator": "SetAccumulator",
            "enable_fast_math": "EnableFastMath",
            "assign_device": "AssignDevice",
        }
        selected_recipe_kind = kind_map.get(selected_kind, selected_kind)
        ok = selected_recipe_kind == "SetTileParams"
        _add(
            "recipe_kind_set_tile_params", ok,
            "" if ok else f"selected recipe kind is {selected_recipe_kind!r}"
        )
        if not ok:
            rejections.append(
                f"selected recipe is {selected_recipe_kind or 'unknown'}, "
                f"not SetTileParams"
            )
        selected_block = {
            "recipe_op_id": "",  # filled below from semantic_obligations
            "recipe_kind": selected_recipe_kind,
            "region": candidate_selection.get("region_id", ""),
            "tile": None,
            "semantic_obligation": "",
            "selected_candidate_id": candidate_selection.get(
                "selected_candidate_id", ""
            ),
        }
        # Resolve recipe_op_id + obligation via semantic_obligations.
        if semantic_obligations is not None:
            for ob in semantic_obligations.get("obligations", []):
                if ob.get("source_candidate") == selected_block[
                    "selected_candidate_id"
                ]:
                    selected_block["recipe_op_id"] = ob.get("recipe_op_id", "")
                    selected_block["semantic_obligation"] = ob.get("id", "")
                    break

    # If the recipe isn't SetTileParams, short-circuit the rest of the
    # checklist with eligible=false. Audit-side status stays "pass".
    if selected_recipe_kind != "SetTileParams" or not rejections == []:
        # (Continue building artifact even when ineligible — the rest of
        # the checks just won't be exercised.)
        pass

    # ------------------------------------------------------------------ #
    # 2. region exists in region_map AND is not opaque_fallback
    # ------------------------------------------------------------------ #
    region_id = selected_block.get("region", "")
    region_map_record: dict[str, Any] | None = None
    if region_id:
        for r in region_map.get("regions", []):
            if r.get("region_id") == region_id:
                region_map_record = r
                break
    region_in_map = region_map_record is not None
    region_classification = (
        region_map_record.get("source_classification", "")
        if region_map_record else ""
    )
    region_kind = (
        region_map_record.get("kind", "") if region_map_record else ""
    )
    if selected_recipe_kind == "SetTileParams":
        _add(
            "region_in_region_map", region_in_map,
            "" if region_in_map
            else f"region {region_id!r} not in region_map"
        )
        if not region_in_map:
            rejections.append(f"region {region_id!r} not in region_map")
        is_opaque = region_classification == "opaque_fallback" or region_kind.startswith(
            "opaque_"
        )
        _add(
            "region_not_opaque", region_in_map and not is_opaque,
            "" if not is_opaque
            else f"region kind={region_kind!r} classification={region_classification!r}"
        )
        if region_in_map and is_opaque:
            rejections.append(
                f"region {region_id!r} is opaque (kind={region_kind!r})"
            )

    # ------------------------------------------------------------------ #
    # 3. target op == linalg.matmul, found exactly once for this region
    # ------------------------------------------------------------------ #
    payload_ref: str | None = None
    matmul_signature: dict[str, Any] | None = None
    occurrences = 0
    if selected_recipe_kind == "SetTileParams" and region_in_map:
        payload_ref = _resolve_payload_ref_for_region(region_map, region_id)
        _add(
            "payload_ref_resolves",
            payload_ref is not None and (run_dir / payload_ref).exists(),
            "" if payload_ref else "no payload_ref for region",
        )
        if not payload_ref:
            rejections.append(f"no payload_ref recorded for region {region_id!r}")
        elif not (run_dir / payload_ref).exists():
            rejections.append(f"payload_ref does not exist on disk: {payload_ref}")
        else:
            payload_text = (run_dir / payload_ref).read_text(encoding="utf-8")
            matmuls = _find_matmul_for_region(payload_text, region_id)
            occurrences = len(matmuls)
            _add(
                "target_op_linalg_matmul",
                occurrences >= 1,
                "" if occurrences >= 1
                else f"no linalg.matmul with compgen.region_id={region_id!r}"
            )
            if occurrences == 0:
                rejections.append(
                    f"selected SetTileParams targets region {region_id!r} "
                    f"but no linalg.matmul carries that compgen.region_id "
                    f"in {payload_ref}"
                )
            _add(
                "payload_region_found_once",
                occurrences == 1,
                "" if occurrences == 1
                else f"found {occurrences} linalg.matmul ops for region {region_id!r}"
            )
            if occurrences > 1:
                rejections.append(
                    f"region {region_id!r} ambiguous: "
                    f"{occurrences} linalg.matmul ops match"
                )
            if occurrences == 1:
                matmul_signature = _matmul_signature(matmuls[0])
                # Static rank-2 + f32 dtype check.
                ok_static = matmul_signature is not None
                _add(
                    "static_rank2_shapes", ok_static,
                    "" if ok_static
                    else "matmul shapes are not static rank-2 (saw "
                         f"lhs={matmuls[0].get('lhs')}, rhs={matmuls[0].get('rhs')}, "
                         f"out={matmuls[0].get('out')})"
                )
                if not ok_static:
                    rejections.append(
                        "matmul shapes are not static rank-2"
                    )
                if matmul_signature is not None:
                    f32_ok = (
                        matmul_signature["lhs_dtype"] == "f32"
                        and matmul_signature["rhs_dtype"] == "f32"
                        and matmul_signature["out_dtype"] == "f32"
                    )
                    _add(
                        "dtype_f32", f32_ok,
                        "" if f32_ok
                        else f"dtypes lhs={matmul_signature['lhs_dtype']!r} "
                             f"rhs={matmul_signature['rhs_dtype']!r} "
                             f"out={matmul_signature['out_dtype']!r}"
                    )
                    if not f32_ok:
                        rejections.append(
                            "matmul dtype is not f32 across LHS/RHS/OUT"
                        )

    # ------------------------------------------------------------------ #
    # 4. tile from verified_recipe.mlir matches applied tile
    # ------------------------------------------------------------------ #
    recipe_tile: tuple[int, int, int] | None = None
    applied_tile: tuple[int, int, int] | None = None
    if (
        selected_recipe_kind == "SetTileParams"
        and selected_block.get("recipe_op_id")
        and verified_recipe_text
    ):
        recipe_tile = _parse_tile_from_verified_recipe(
            verified_recipe_text, selected_block["recipe_op_id"]
        )
    if applied_manifest is not None:
        for a in applied_manifest.get("applied", []):
            if (
                a.get("recipe_op_id") == selected_block.get("recipe_op_id")
                and a.get("recipe_kind") == "SetTileParams"
            ):
                t = a.get("tile") or {}
                if all(k in t for k in ("M", "N", "K")):
                    applied_tile = (int(t["M"]), int(t["N"]), int(t["K"]))
                break
    if selected_recipe_kind == "SetTileParams":
        if recipe_tile is not None:
            selected_block["tile"] = {
                "M": recipe_tile[0], "N": recipe_tile[1], "K": recipe_tile[2],
            }
        ok_match = (
            recipe_tile is not None
            and applied_tile is not None
            and recipe_tile == applied_tile
        )
        _add(
            "tile_matches_verified_recipe",
            ok_match,
            "" if ok_match
            else f"verified_recipe tile={recipe_tile} applied tile={applied_tile}"
        )
        if not ok_match:
            rejections.append(
                f"tile mismatch: verified_recipe={recipe_tile}, "
                f"applied_transform_manifest={applied_tile}"
            )

    # ------------------------------------------------------------------ #
    # 5. tile is one of the legal candidate tiles for the region
    #    (i.e. came from the working-set curve)
    # ------------------------------------------------------------------ #
    if selected_recipe_kind == "SetTileParams":
        cand_id = selected_block.get("selected_candidate_id", "")
        cand = _selected_candidate_for_region(candidate_actions, cand_id)
        cand_legal = bool(cand and (cand.get("legality") or {}).get("ok"))
        cand_cost = (cand or {}).get("cost_preview") or {}
        fits_scratchpad = bool(cand_cost.get("fits_scratchpad"))
        fits_l2 = bool(cand_cost.get("fits_l2"))
        ok_curve = cand_legal and (fits_scratchpad or fits_l2)
        _add(
            "tile_exists_in_working_set_curve",
            ok_curve,
            "" if ok_curve
            else f"candidate_id={cand_id!r} legal={cand_legal} "
                 f"fits_scratchpad={fits_scratchpad} fits_l2={fits_l2}"
        )
        if not ok_curve:
            rejections.append(
                "selected tile is not in the legal working-set curve "
                f"(candidate_id={cand_id!r})"
            )

    # ------------------------------------------------------------------ #
    # 6. divides-or-boundary observation (informational; never rejects).
    # Per the spec the tile may either divide the matmul dimensions
    # cleanly OR be handled via explicit boundary tiles — including the
    # degenerate "tile >= dim" case which lowers to a single-iteration
    # loop. Passing `tile_exists_in_working_set_curve` is the legality
    # gate; this check is only a recorded hint for the transformer
    # so it knows whether to emit a clean tiling or boundary-aware code.
    # ------------------------------------------------------------------ #
    tile_geometry: dict[str, Any] | None = None
    if (
        selected_recipe_kind == "SetTileParams"
        and recipe_tile is not None
        and matmul_signature is not None
    ):
        M_op, N_op, K_op = (
            matmul_signature["M"], matmul_signature["N"], matmul_signature["K"]
        )
        tM, tN, tK = recipe_tile
        divides_M = M_op % tM == 0 if tM <= M_op else (tM == M_op)
        divides_N = N_op % tN == 0 if tN <= N_op else (tN == N_op)
        divides_K = K_op % tK == 0 if tK <= K_op else (tK == K_op)
        divides_cleanly = divides_M and divides_N and divides_K
        boundary_required = not divides_cleanly
        degenerate_single_iter = tM >= M_op or tN >= N_op or tK >= K_op
        tile_geometry = {
            "divides_cleanly": divides_cleanly,
            "boundary_required": boundary_required,
            "degenerate_single_iter": degenerate_single_iter,
            "iters_M": max(1, (M_op + tM - 1) // tM),
            "iters_N": max(1, (N_op + tN - 1) // tN),
            "iters_K": max(1, (K_op + tK - 1) // tK),
        }
        _add(
            "tile_geometry_recorded", True,
            f"divides_cleanly={divides_cleanly} "
            f"boundary_required={boundary_required} "
            f"degenerate_single_iter={degenerate_single_iter}"
        )

    # ------------------------------------------------------------------ #
    # 7. source payload tree byte-identical pre/post audit
    # ------------------------------------------------------------------ #
    post_payload_shas = {
        str(p.relative_to(run_dir)): _sha256_file(p)
        for p in sorted(pl.rglob("payload.mlir"))
    }
    payloads_unchanged = pre_payload_shas == post_payload_shas
    _add(
        "source_payload_unchanged", payloads_unchanged,
        "" if payloads_unchanged
        else f"pre={list(pre_payload_shas.values())[:1]} "
             f"post={list(post_payload_shas.values())[:1]}"
    )
    if not payloads_unchanged:
        rejections.append("source payload mutated during audit")

    # ------------------------------------------------------------------ #
    # Eligibility verdict + audit-side status
    # ------------------------------------------------------------------ #
    eligible = (
        selected_recipe_kind == "SetTileParams"
        and not rejections
        and all(c["status"] == "pass" for c in checks)
    )
    # Audit-side status is "pass" unless the audit itself failed (e.g.
    # source payload mutation). Ineligibility from a non-SetTileParams
    # recipe is NOT an audit failure.
    audit_failed = not payloads_unchanged
    audit_status = "fail" if audit_failed else "pass"

    payload_ref_field = payload_ref or ""
    payload_sha_before = (
        pre_payload_shas.get(payload_ref_field, "") if payload_ref_field else ""
    )
    payload_sha_after = (
        post_payload_shas.get(payload_ref_field, "") if payload_ref_field else ""
    )

    artifact = {
        "schema_version": "real_transform_eligibility_v1",
        "status": audit_status,
        "model_id": model_id,
        "target_id": target_id,
        "eligible": eligible,
        "selected_recipe": {
            "recipe_op_id": selected_block.get("recipe_op_id", ""),
            "recipe_kind": selected_recipe_kind,
            "region": selected_block.get("region", ""),
            "tile": selected_block.get("tile"),
            "semantic_obligation": selected_block.get(
                "semantic_obligation", ""
            ),
            "selected_candidate_id": selected_block.get(
                "selected_candidate_id", ""
            ),
        },
        "payload": {
            "payload_ref": payload_ref_field,
            "payload_sha256_before": payload_sha_before,
            "payload_sha256_after": payload_sha_after,
        },
        "matmul_signature": matmul_signature,
        "tile_geometry": tile_geometry,
        "checks": checks,
        "rejection_reasons": rejections,
    }
    json_path = out_dir / "real_transform_eligibility.json"
    json_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    # Markdown summary.
    md_lines: list[str] = []
    md_lines.append(f"# Real Transform Eligibility — {model_id or '(unknown)'}\n")
    md_lines.append(f"_Generated_: {_utcnow()}\n")
    md_lines.append(
        f"- **eligible**: `{eligible}`  "
        f"\n- **audit status**: `{audit_status}`  "
        f"\n- **selected recipe**: "
        f"`{selected_recipe_kind or '—'}` on region "
        f"`{selected_block.get('region', '—')}`  "
        f"\n- **tile**: `{selected_block.get('tile')}`\n"
    )
    if matmul_signature is not None:
        md_lines.append(
            "## Matmul signature\n"
            f"- M = {matmul_signature['M']}, "
            f"N = {matmul_signature['N']}, "
            f"K = {matmul_signature['K']}\n"
            f"- dtype: lhs={matmul_signature['lhs_dtype']}, "
            f"rhs={matmul_signature['rhs_dtype']}, "
            f"out={matmul_signature['out_dtype']}\n"
            f"- rank: {matmul_signature['rank']}; "
            f"dynamic_dims: {matmul_signature['dynamic_dims']}\n"
        )
    md_lines.append("## Checks\n")
    md_lines.append("| name | status | detail |")
    md_lines.append("|---|---|---|")
    for c in checks:
        md_lines.append(f"| {c['name']} | {c['status']} | {c['detail']} |")
    if rejections:
        md_lines.append("\n## Rejection reasons\n")
        for r in rejections:
            md_lines.append(f"- {r}")
    md_path = out_dir / "real_transform_eligibility.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return RealTransformEligibilityResult(
        overall=audit_status,
        eligible=eligible,
        out_dir=out_dir,
        json_path=json_path,
        md_path=md_path,
        rejection_reasons=tuple(rejections),
    )
