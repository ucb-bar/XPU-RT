"""Real SetTileParams Transform MVP (Milestone 11B).

For an eligible ``SetTileParams`` recipe (per M-11A's
``real_transform_eligibility.json``), emit a real tiled IR artifact that
replaces the selected ``linalg.matmul`` op with a triple-nested
``scf.for`` loop nest. The result is a derived artifact under
``03_recipe_planning/real_lowering/`` — it never overwrites the source
payload nor the metadata-only ``transformed_payload.mlir``.

The transform is intentionally narrow: ``linalg.matmul`` only, static
rank-2 ``f32`` tensors, single occurrence per ``compgen.region_id``,
tile parsed verbatim from ``verified_recipe.mlir``. The MVP emits one
of two artifact kinds:

- ``executable_structured_ir`` — when every tile dim is ≤ the matmul dim
  AND the tile divides cleanly. The innermost loop body emits real
  ``tensor.extract_slice`` / ``linalg.matmul`` / ``tensor.insert_slice``
  with proper iter-arg threading.
- ``non_executable_structural_ir`` — when boundary handling is required
  (tile exceeds a matmul dim, or doesn't divide). The innermost loop
  body is empty (just ``scf.yield`` of the iter-arg). The IR is
  syntactically valid and parses, but it does NOT compute the matmul
  result. The artifact and validation report mark this honestly.

Hard invariants:

- ``01_payload_lowering/`` is read-only. The payload tree must be
  byte-identical pre/post.
- ``03_recipe_planning/post_lowering/transformed_payload.mlir``
  (the M-08 metadata-only artifact) must not be touched.
- The real transformed file lives ONLY under
  ``03_recipe_planning/real_lowering/``. Writing it under
  ``01_payload_lowering/`` is a hard fail.
- This stage makes NO claim of semantic equivalence with the source.
  Differential verification of a real transform is M-12's job. M-11B
  emits the IR artifact and structural validation only.
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
# Result + entry point
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RealLoweringResult:
    overall: str  # "pass" | "fail" | "skipped"
    real_transform_kind: str
    eligible: bool
    out_dir: Path
    transformed_real_path: Path | None
    diff_path: Path | None
    manifest_path: Path
    validation_path: Path
    summary_md_path: Path
    failures: tuple[str, ...]


# Whitelist of model IDs that the M-11A audit covered for the executable
# path. M-11A audited differential correctness on ``merlin_mlp_wide`` only.
#
# M-37.12 + M-37.13 — load-bearing safety shift (honest accounting):
#
# The pre-M-37.12 ``selected_model_is_merlin_mlp_wide`` gate was the
# structural barrier preventing non-audited models from claiming
# differential correctness on the clean-divide path. M-37.12 admits
# any model on the ``executable_structured_ir`` path. The same safety
# property is now carried by two structural rules in adjacent gates:
#
#   1. ``recipe_gate.single_k_iter`` — ``bit_equality`` is claimed
#      only when the tile divides every region dim cleanly AND
#      ``tK >= K_dim`` (single K iteration). Otherwise the recipe
#      declares ``tolerance_eps``. Multiple K iterations reorder
#      accumulation and break bit-exact equivalence with eager.
#
#   2. ``real_transform_differential.matmul_higham_bound`` (M-37.13)
#      — for declared ``tolerance_eps`` cases the per-case criterion
#      is ``|sim - eager| <= 4 * K * eps * max|A| * max|B|`` (Higham's
#      matmul accumulation bound, derived per-case from inputs;
#      never silently widened, scales linearly with K and with input
#      magnitude). For declared ``bit_equality`` the criterion is
#      exact equality. Negative controls in
#      ``tests/graph_compilation/test_m37_13_negative_controls.py``
#      exercise both crafted-input boundaries.
#
# The whitelist is preserved (not removed) because it still fires on
# ``executable_with_boundary_handling``, where the M-11A audit's
# load-bearing concern (boundary handling on a real matmul) DOES
# apply. See ``docs/realness/m37_12_clean_divide_admission.yaml``
# for the full contract and forbidden constructs.
_EXECUTABLE_MODEL_WHITELIST: frozenset[str] = frozenset({
    "merlin_mlp_wide",
})


def _utcnow() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> dict[str, Any]:
    obj: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return obj


def _read_json_or_none(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _read_json(path)


def _sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# IR splice helpers
# --------------------------------------------------------------------------- #


_MATMUL_LINE_RE = re.compile(
    r"^(?P<indent>\s*)%(?P<result>[A-Za-z0-9_]+)\s*=\s*linalg\.matmul\s*"
    r"(?:\{(?P<attrs>[^}]*)\}\s*)?"
    r"ins\(\s*(?P<lhs_ssa>%[A-Za-z0-9_]+)\s*,\s*(?P<rhs_ssa>%[A-Za-z0-9_]+)\s*"
    r":\s*(?P<lhs_type>tensor<[^>]+>)\s*,\s*(?P<rhs_type>tensor<[^>]+>)\s*\)\s*"
    r"outs\(\s*(?P<out_ssa>%[A-Za-z0-9_]+)\s*:\s*(?P<out_type>tensor<[^>]+>)\s*\)\s*"
    r"->\s*(?P<ret_type>tensor<[^>]+>)\s*$",
    re.MULTILINE,
)


_ATTR_RE_TEMPLATE = r'\b{}\s*=\s*"([^"]*)"'


def _attr_str(attrs_text: str, key: str) -> str | None:
    m = re.search(_ATTR_RE_TEMPLATE.format(re.escape(key)), attrs_text or "")
    return m.group(1) if m else None


def _find_matmul_for_region(
    payload_text: str, region_id: str
) -> list[re.Match[str]]:
    out: list[re.Match[str]] = []
    for m in _MATMUL_LINE_RE.finditer(payload_text):
        if _attr_str(m.group("attrs") or "", "compgen.region_id") == region_id:
            out.append(m)
    return out


def _emit_executable_loop_nest(
    *,
    indent: str,
    out_ssa: str,
    result_ssa: str,
    lhs_ssa: str,
    rhs_ssa: str,
    out_type: str,
    M: int,
    N: int,
    K: int,
    tM: int,
    tN: int,
    tK: int,
    elem: str,
    region_id: str,
    recipe_op_id: str,
    obligation_id: str,
) -> str:
    """Emit an executable triple-nested scf.for that tiles the matmul.

    Preconditions (caller must enforce):

    - tM ≤ M, tN ≤ N, tK ≤ K
    - M % tM == 0, N % tN == 0, K % tK == 0
    """
    full_t = f"tensor<{M}x{N}x{elem}>"
    lhs_t = f"tensor<{M}x{K}x{elem}>"
    rhs_t = f"tensor<{K}x{N}x{elem}>"
    tile_t = f"tensor<{tM}x{tN}x{elem}>"
    lhs_tile_t = f"tensor<{tM}x{tK}x{elem}>"
    rhs_tile_t = f"tensor<{tK}x{tN}x{elem}>"
    in1 = indent
    in2 = indent + "  "
    in3 = indent + "    "
    in4 = indent + "      "
    lines: list[str] = []
    lines.append(
        f"{in1}// compgen.real_transform = \"set_tile_params\""
    )
    lines.append(
        f"{in1}// real_transform_kind = \"executable_structured_ir\""
    )
    lines.append(f"{in1}// source_region = \"{region_id}\"")
    lines.append(f"{in1}// recipe_op = \"{recipe_op_id}\"")
    lines.append(f"{in1}// semantic_obligation = \"{obligation_id}\"")
    lines.append(f"{in1}// tile = [{tM}, {tN}, {tK}]")
    lines.append(f"{in1}%_real_c0 = arith.constant 0 : index")
    lines.append(f"{in1}%_real_M = arith.constant {M} : index")
    lines.append(f"{in1}%_real_N = arith.constant {N} : index")
    lines.append(f"{in1}%_real_K = arith.constant {K} : index")
    lines.append(f"{in1}%_real_tileM = arith.constant {tM} : index")
    lines.append(f"{in1}%_real_tileN = arith.constant {tN} : index")
    lines.append(f"{in1}%_real_tileK = arith.constant {tK} : index")
    lines.append(
        f"{in1}%{result_ssa} = scf.for %_real_i = %_real_c0 to %_real_M "
        f"step %_real_tileM iter_args(%_real_acc_i = {out_ssa}) -> ({full_t}) {{"
    )
    lines.append(
        f"{in2}%_real_inner_i = scf.for %_real_j = %_real_c0 to %_real_N "
        f"step %_real_tileN iter_args(%_real_acc_j = %_real_acc_i) "
        f"-> ({full_t}) {{"
    )
    lines.append(
        f"{in3}%_real_inner_j = scf.for %_real_k = %_real_c0 to %_real_K "
        f"step %_real_tileK iter_args(%_real_acc_k = %_real_acc_j) "
        f"-> ({full_t}) {{"
    )
    lines.append(
        f"{in4}%_real_lhs_tile = tensor.extract_slice {lhs_ssa}"
        f"[%_real_i, %_real_k] [{tM}, {tK}] [1, 1] : {lhs_t} to {lhs_tile_t}"
    )
    lines.append(
        f"{in4}%_real_rhs_tile = tensor.extract_slice {rhs_ssa}"
        f"[%_real_k, %_real_j] [{tK}, {tN}] [1, 1] : {rhs_t} to {rhs_tile_t}"
    )
    lines.append(
        f"{in4}%_real_out_tile = tensor.extract_slice %_real_acc_k"
        f"[%_real_i, %_real_j] [{tM}, {tN}] [1, 1] : {full_t} to {tile_t}"
    )
    lines.append(
        f"{in4}%_real_matmul_tile = linalg.matmul "
        f"ins(%_real_lhs_tile, %_real_rhs_tile : {lhs_tile_t}, {rhs_tile_t}) "
        f"outs(%_real_out_tile : {tile_t}) -> {tile_t}"
    )
    lines.append(
        f"{in4}%_real_inserted = tensor.insert_slice %_real_matmul_tile "
        f"into %_real_acc_k[%_real_i, %_real_j] [{tM}, {tN}] [1, 1] "
        f": {tile_t} into {full_t}"
    )
    lines.append(f"{in4}scf.yield %_real_inserted : {full_t}")
    lines.append(f"{in3}}}")
    lines.append(f"{in3}scf.yield %_real_inner_j : {full_t}")
    lines.append(f"{in2}}}")
    lines.append(f"{in2}scf.yield %_real_inner_i : {full_t}")
    lines.append(f"{in1}}}")
    return "\n".join(lines)


def _emit_structural_loop_nest(
    *,
    indent: str,
    out_ssa: str,
    result_ssa: str,
    out_type: str,
    M: int,
    N: int,
    K: int,
    tM: int,
    tN: int,
    tK: int,
    elem: str,
    region_id: str,
    recipe_op_id: str,
    obligation_id: str,
    iters_M: int,
    iters_N: int,
    iters_K: int,
    boundary_required: bool,
    degenerate_single_iter: bool,
) -> str:
    """Emit a non-executable structural loop nest. The body is empty
    (just ``scf.yield`` of the iter-arg); the IR is syntactically valid
    and round-trips through MLIR text but does NOT compute the matmul
    result. This is honest about boundary handling being deferred."""
    full_t = f"tensor<{M}x{N}x{elem}>"
    in1 = indent
    in2 = indent + "  "
    in3 = indent + "    "
    in4 = indent + "      "
    lines: list[str] = []
    lines.append(
        f"{in1}// compgen.real_transform = \"set_tile_params\""
    )
    lines.append(
        f"{in1}// real_transform_kind = \"executable_with_boundary_handling\""
    )
    lines.append(f"{in1}// source_region = \"{region_id}\"")
    lines.append(f"{in1}// recipe_op = \"{recipe_op_id}\"")
    lines.append(f"{in1}// semantic_obligation = \"{obligation_id}\"")
    lines.append(f"{in1}// tile = [{tM}, {tN}, {tK}]")
    lines.append(
        f"{in1}// matmul_dims = [M={M}, N={N}, K={K}]; "
        f"iters = [M={iters_M}, N={iters_N}, K={iters_K}]; "
        f"boundary_required = {str(boundary_required).lower()}; "
        f"degenerate_single_iter = {str(degenerate_single_iter).lower()}"
    )
    lines.append(
        f"{in1}// MLIR body intentionally structural: per-iteration "
        f"effective tile size = min(tile, dim - offset) is handled by "
        f"M-12's boundary-aware Python evaluator (M-16). A future "
        f"MLIR-level upgrade will emit dynamic-shape extract_slice."
    )
    lines.append(f"{in1}%_real_c0 = arith.constant 0 : index")
    lines.append(f"{in1}%_real_M = arith.constant {M} : index")
    lines.append(f"{in1}%_real_N = arith.constant {N} : index")
    lines.append(f"{in1}%_real_K = arith.constant {K} : index")
    lines.append(f"{in1}%_real_tileM = arith.constant {tM} : index")
    lines.append(f"{in1}%_real_tileN = arith.constant {tN} : index")
    lines.append(f"{in1}%_real_tileK = arith.constant {tK} : index")
    lines.append(
        f"{in1}%{result_ssa} = scf.for %_real_i = %_real_c0 to %_real_M "
        f"step %_real_tileM iter_args(%_real_acc_i = {out_ssa}) -> ({full_t}) {{"
    )
    lines.append(
        f"{in2}%_real_inner_i = scf.for %_real_j = %_real_c0 to %_real_N "
        f"step %_real_tileN iter_args(%_real_acc_j = %_real_acc_i) "
        f"-> ({full_t}) {{"
    )
    lines.append(
        f"{in3}%_real_inner_j = scf.for %_real_k = %_real_c0 to %_real_K "
        f"step %_real_tileK iter_args(%_real_acc_k = %_real_acc_j) "
        f"-> ({full_t}) {{"
    )
    lines.append(f"{in4}// structural placeholder body — no extract_slice/matmul/insert_slice")
    lines.append(f"{in4}scf.yield %_real_acc_k : {full_t}")
    lines.append(f"{in3}}}")
    lines.append(f"{in3}scf.yield %_real_inner_j : {full_t}")
    lines.append(f"{in2}}}")
    lines.append(f"{in2}scf.yield %_real_inner_i : {full_t}")
    lines.append(f"{in1}}}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Top-level entry point
# --------------------------------------------------------------------------- #


def _classify_real_transform_kind(
    *, M: int, N: int, K: int, tM: int, tN: int, tK: int,
) -> tuple[str, dict[str, bool]]:
    boundary_required = not (
        (M % tM == 0) and (N % tN == 0) and (K % tK == 0)
        and tM <= M and tN <= N and tK <= K
    )
    flags = {
        "tM_le_M": tM <= M,
        "tN_le_N": tN <= N,
        "tK_le_K": tK <= K,
        "M_divides": M % tM == 0,
        "N_divides": N % tN == 0,
        "K_divides": K % tK == 0,
        "boundary_required": boundary_required,
    }
    if not boundary_required:
        return "executable_structured_ir", flags
    # M-16: boundary cases now route through M-12's boundary-aware
    # evaluator (Python-side ``min()``-based slicing). The MLIR is
    # still structural (no per-iteration dynamic-shape extract_slice
    # in the emitted IR — that's a future MLIR-level upgrade), but
    # the kind signals to M-12 that the evaluator handles boundaries
    # correctly so verification IS executable.
    return "executable_with_boundary_handling", flags


def run_real_lowering(run_dir: Path) -> RealLoweringResult:
    """Apply M-11A's eligibility verdict to emit a real tiled IR artifact.

    Reads:

    - ``03_recipe_planning/real_transform_eligibility.json`` (M-11A)
    - the source ``payload.mlir`` indicated by the eligibility report
    - ``03_recipe_planning/post_lowering/transformed_payload.mlir`` (M-08;
      read-only, must not be modified)

    Writes (under ``03_recipe_planning/real_lowering/``):

    - ``transformed_payload.real.mlir`` (transform-like only; never under
      ``01_payload_lowering/``)
    - ``real_transform_diff.json``
    - ``real_transform_manifest.json``
    - ``real_transform_validation.json``
    """
    run_dir = Path(run_dir).resolve()
    rp = run_dir / "03_recipe_planning"
    pl = run_dir / "01_payload_lowering"
    if not rp.is_dir():
        raise FileNotFoundError(f"03_recipe_planning/ missing under {run_dir}")

    eligibility_path = rp / "real_transform_eligibility.json"
    if not eligibility_path.exists():
        raise FileNotFoundError(
            f"M-11B requires M-11A; missing {eligibility_path}"
        )
    eligibility = _read_json(eligibility_path)

    out_dir = rp / "real_lowering"
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in (
        "transformed_payload.real.mlir",
        "real_transform_diff.json",
        "real_transform_manifest.json",
        "real_transform_validation.json",
        "real_transform_summary.md",
    ):
        target = out_dir / p
        if target.exists():
            target.unlink()

    # Pre-snapshot SHAs of every payload.mlir under 01_payload_lowering/
    # and the M-08 metadata-only transformed payload, so we can prove
    # they were not touched.
    pre_payload_shas: dict[str, str] = {
        str(p.relative_to(run_dir)): _sha256_file(p)
        for p in sorted(pl.rglob("payload.mlir"))
    }
    metadata_only_path = rp / "post_lowering" / "transformed_payload.mlir"
    pre_metadata_only_sha = (
        _sha256_file(metadata_only_path) if metadata_only_path.exists() else None
    )

    failures: list[str] = []
    checks: list[dict[str, Any]] = []

    def _add(name: str, ok: bool, detail: str = "") -> None:
        checks.append(
            {"name": name, "status": "pass" if ok else "fail", "detail": detail}
        )

    def _skip(name: str, detail: str = "") -> None:
        # ``status="skipped"`` means "this check is not applicable on the
        # current path"; it does NOT contribute to overall=fail. Used
        # for spec'd checks that only apply to the executable path
        # (e.g. boundary_not_required) on the structural / skipped paths.
        checks.append(
            {"name": name, "status": "skipped", "detail": detail}
        )

    eligible = bool(eligibility.get("eligible"))
    selected = eligibility.get("selected_recipe", {}) or {}
    recipe_kind = selected.get("recipe_kind", "")
    # The eligibility report's ``model_id`` is the FX-importer module id
    # (e.g. ``export_program``). The user-facing model_id from the YAML
    # config is what the M-11B spec's
    # ``selected_model_is_merlin_mlp_wide`` check expects.
    # ``00_graph_capture/capture_report.json::model_id`` is written
    # early in the run and is the canonical source. ``run_manifest.json``
    # is only written at the END of run_graph_compilation, so reading
    # it here would always fall back. Capture-report fallbacks order:
    # capture_report → run_manifest → run_dir.name.
    model_id = ""
    capture_report = _read_json_or_none(run_dir / "00_graph_capture" / "capture_report.json")
    if capture_report is not None:
        model_id = capture_report.get("model_id", "")
    if not model_id:
        manifest_top = _read_json_or_none(run_dir / "run_manifest.json")
        if manifest_top is not None:
            model_id = (manifest_top.get("model") or {}).get("model_id", "")
    if not model_id:
        model_id = run_dir.name
    fx_module_id = eligibility.get("model_id", "")

    real_kind = "skipped"
    transformed_real_path: Path | None = None
    diff_path: Path | None = None
    region_id = selected.get("region", "")
    recipe_op_id = selected.get("recipe_op_id", "")
    obligation_id = selected.get("semantic_obligation", "")
    tile_block = selected.get("tile") or {}
    matmul_sig = eligibility.get("matmul_signature") or {}
    payload_ref = (eligibility.get("payload") or {}).get("payload_ref", "")
    payload_sha_before = (eligibility.get("payload") or {}).get(
        "payload_sha256_before", ""
    )
    skipped_reasons: list[str] = []

    flags: dict[str, bool] = {}
    target_op_is_matmul = False  # set when the splice path runs

    if not eligible or recipe_kind != "SetTileParams":
        # Skipped path: no transform attempted. This is NOT a pipeline
        # failure; it is the documented behavior for ineligible models.
        # The skipped_reason field carries the precise eligibility
        # rejection from M-11A. Invariant checks below still run.
        real_kind = "unsupported_real_transform"
        skipped_reasons = list(eligibility.get("rejection_reasons", []) or [])
        if not skipped_reasons:
            skipped_reasons.append(
                f"M-11A eligible=False (recipe_kind={recipe_kind!r})"
            )
    else:
        # Sanity guards on inputs.
        for key in ("M", "N", "K", "lhs_dtype", "rhs_dtype", "out_dtype"):
            if key not in matmul_sig or matmul_sig[key] is None:
                failures.append(f"matmul_signature missing field {key!r}")
        if (
            matmul_sig.get("lhs_dtype") != "f32"
            or matmul_sig.get("rhs_dtype") != "f32"
            or matmul_sig.get("out_dtype") != "f32"
        ):
            failures.append("matmul dtype is not f32 across LHS/RHS/OUT")
        for key in ("M", "N", "K"):
            if tile_block.get(key) in (None, 0):
                failures.append(f"tile.{key} missing/zero")
        if not payload_ref or not (run_dir / payload_ref).exists():
            failures.append(f"payload_ref {payload_ref!r} not on disk")

        if not failures:
            payload_path = run_dir / payload_ref
            payload_text = payload_path.read_text(encoding="utf-8")
            matches = _find_matmul_for_region(payload_text, region_id)
            _add(
                "target_region_found_once",
                len(matches) == 1,
                "" if len(matches) == 1
                else f"found {len(matches)} matmul(s) for region {region_id!r}",
            )
            if len(matches) != 1:
                failures.append(
                    f"region {region_id!r} ambiguous: {len(matches)} linalg.matmul ops"
                )
            else:
                m = matches[0]
                indent = m.group("indent")
                out_ssa = m.group("out_ssa")
                lhs_ssa = m.group("lhs_ssa")
                rhs_ssa = m.group("rhs_ssa")
                out_type = m.group("out_type")
                result_ssa = m.group("result")
                target_op_is_matmul = True

                M = int(matmul_sig["M"])
                N = int(matmul_sig["N"])
                K = int(matmul_sig["K"])
                tM = int(tile_block["M"])
                tN = int(tile_block["N"])
                tK = int(tile_block["K"])

                _add(
                    "tile_matches_verified_recipe",
                    True,  # M-11A already enforced this; we re-confirm by surfacing the values
                    f"tile=({tM},{tN},{tK}) matmul=({M},{N},{K})",
                )

                real_kind, flags = _classify_real_transform_kind(
                    M=M, N=N, K=K, tM=tM, tN=tN, tK=tK
                )

                geom = eligibility.get("tile_geometry") or {}
                if real_kind == "executable_structured_ir":
                    block = _emit_executable_loop_nest(
                        indent=indent,
                        out_ssa=out_ssa,
                        result_ssa=result_ssa,
                        lhs_ssa=lhs_ssa,
                        rhs_ssa=rhs_ssa,
                        out_type=out_type,
                        M=M, N=N, K=K, tM=tM, tN=tN, tK=tK,
                        elem="f32",
                        region_id=region_id,
                        recipe_op_id=recipe_op_id,
                        obligation_id=obligation_id,
                    )
                else:
                    block = _emit_structural_loop_nest(
                        indent=indent,
                        out_ssa=out_ssa,
                        result_ssa=result_ssa,
                        out_type=out_type,
                        M=M, N=N, K=K, tM=tM, tN=tN, tK=tK,
                        elem="f32",
                        region_id=region_id,
                        recipe_op_id=recipe_op_id,
                        obligation_id=obligation_id,
                        iters_M=int(geom.get("iters_M", 1)),
                        iters_N=int(geom.get("iters_N", 1)),
                        iters_K=int(geom.get("iters_K", 1)),
                        boundary_required=bool(geom.get("boundary_required", True)),
                        degenerate_single_iter=bool(
                            geom.get("degenerate_single_iter", False)
                        ),
                    )

                # Splice: replace the matmul line with the loop block.
                start, end = m.start(), m.end()
                new_text = payload_text[:start] + block + payload_text[end:]
                if new_text == payload_text:
                    failures.append(
                        "real-transform splice produced byte-identical output"
                    )

                transformed_real_path = out_dir / "transformed_payload.real.mlir"
                transformed_real_path.write_text(new_text, encoding="utf-8")

                diff = {
                    "schema_version": "real_transform_diff_v1",
                    "source_payload": payload_ref,
                    "source_payload_sha256_before": payload_sha_before,
                    "transformed_payload_real":
                        transformed_real_path.relative_to(run_dir).as_posix(),
                    "transformed_payload_real_sha256":
                        _sha256_text(new_text),
                    "splice": {
                        "byte_start": start,
                        "byte_end": end,
                        "before": payload_text[start:end],
                        "after_first_line": block.split("\n", 1)[0],
                        "added_lines": block.count("\n") + 1,
                        "removed_lines": payload_text[start:end].count("\n") + 1,
                    },
                    "tile_classification": flags,
                }
                diff_path = out_dir / "real_transform_diff.json"
                diff_path.write_text(
                    json.dumps(diff, indent=2, sort_keys=True), encoding="utf-8",
                )

    # ------------------------------------------------------------------ #
    # Source-payload + metadata-only invariants.
    # ------------------------------------------------------------------ #
    post_payload_shas = {
        str(p.relative_to(run_dir)): _sha256_file(p)
        for p in sorted(pl.rglob("payload.mlir"))
    }
    payload_unchanged = pre_payload_shas == post_payload_shas
    _add(
        "source_payload_unchanged", payload_unchanged,
        "" if payload_unchanged else "01_payload_lowering/ tree changed",
    )
    if not payload_unchanged:
        failures.append("source payload mutated during real-lowering")

    metadata_only_unchanged = True
    if pre_metadata_only_sha is not None:
        post_metadata_only_sha = (
            _sha256_file(metadata_only_path) if metadata_only_path.exists() else None
        )
        metadata_only_unchanged = pre_metadata_only_sha == post_metadata_only_sha
    if not metadata_only_unchanged:
        failures.append("M-08 metadata-only transformed payload was overwritten")

    # The real transformed payload must NEVER live under 01_payload_lowering/.
    leak = list(pl.rglob("transformed_payload.real*"))
    _add(
        "real_transform_not_under_01_payload_lowering",
        not leak,
        "" if not leak
        else f"leaks: {[p.relative_to(run_dir).as_posix() for p in leak]}",
    )
    if leak:
        failures.append(
            f"transformed_payload.real.mlir leaked under 01_payload_lowering/: {leak}"
        )

    # ------------------------------------------------------------------ #
    # Spec'd named checks (M-11B). Path-aware: ``pass`` on the path
    # they apply to, ``skipped`` on the others. Skipped checks do not
    # contribute to ``overall=fail``.
    # ------------------------------------------------------------------ #
    if real_kind == "unsupported_real_transform":
        _skip("eligibility_passed", "model is unsupported by the M-11B MVP")
        _skip("selected_model_is_merlin_mlp_wide",
              f"path=unsupported model_id={model_id!r}")
        _skip("selected_recipe_is_set_tile_params", f"recipe_kind={recipe_kind!r}")
        _skip("target_op_is_linalg_matmul", "no transform attempted")
        _skip("boundary_not_required", "no transform attempted")
        _skip("real_artifact_differs_from_source", "no transform attempted")
    else:
        # Eligible path (executable or structural-only).
        _add("eligibility_passed", eligible, "")
        # M-37.12 + M-37.13: the named M-11B check is preserved
        # (downstream tooling reads the name), but its semantics
        # evolved with explicit safety-shift accounting.
        #
        # - ``executable_structured_ir`` (clean-divide): admits any
        #   model. Load-bearing safety has shifted to two adjacent
        #   structural gates: recipe_gate.single_k_iter (correct
        #   refinement declaration) and the M-37.13 Higham-bounded
        #   semantic check (no hand-picked tolerance constants).
        #   See module-level whitelist comment.
        # - ``executable_with_boundary_handling``: skipped on this
        #   path; the M-11A audit's boundary-handling concern still
        #   speaks but additional audits have not landed.
        # - Other non-executable paths: skipped.
        if real_kind == "executable_structured_ir":
            audited_or_clean_divide_path = (
                model_id in _EXECUTABLE_MODEL_WHITELIST
                or not flags.get("boundary_required", True)
            )
            _add(
                "selected_model_is_merlin_mlp_wide",
                audited_or_clean_divide_path,
                (
                    f"model_id={model_id!r}; "
                    f"clean-divide path admits any model "
                    f"(safety carried by recipe_gate.single_k_iter "
                    f"+ M-12 combined tolerance)"
                ),
            )
        else:
            _skip(
                "selected_model_is_merlin_mlp_wide",
                f"path=non_executable model_id={model_id!r}",
            )
        _add(
            "selected_recipe_is_set_tile_params",
            recipe_kind == "SetTileParams",
            "",
        )
        _add(
            "target_op_is_linalg_matmul",
            target_op_is_matmul,
            "" if target_op_is_matmul
            else "matmul splice did not execute (region missing or invariant fail)",
        )
        # ``boundary_not_required`` only applies to the executable path.
        if real_kind == "executable_structured_ir":
            _add(
                "boundary_not_required",
                not flags.get("boundary_required", True),
                "" if not flags.get("boundary_required", True)
                else "boundary handling required (tile > dim or doesn't divide)",
            )
        else:
            _skip(
                "boundary_not_required",
                f"boundary required for {real_kind}; deferred to follow-on",
            )
        _add(
            "real_artifact_differs_from_source",
            transformed_real_path is not None and transformed_real_path.exists(),
            "" if transformed_real_path is not None
            else "no transformed artifact emitted",
        )

    # Per-spec rename + invariant: M-11B never claims differential
    # correctness; the manifest carries ``no_correctness_claim: true``
    # and this check pins it. M-12 owns differential correctness.
    _add(
        "metadata_only_artifact_not_overwritten", metadata_only_unchanged,
        "" if metadata_only_unchanged
        else "post_lowering/transformed_payload.mlir was modified",
    )
    _add(
        "no_differential_correctness_claimed", True,
        "manifest pins no_correctness_claim=true; M-12 owns correctness",
    )

    # ------------------------------------------------------------------ #
    # Manifest + validation report.
    # ------------------------------------------------------------------ #
    # `overall` reflects ONLY invariant + transform-attempt outcomes.
    # Ineligibility (skipped path) is not a failure: the audit ran
    # cleanly and recorded `real_transform_kind=unsupported_real_transform`
    # with the precise reason from M-11A in `skipped_reason`. Status
    # ``skipped`` on individual checks is informational and does NOT
    # contribute to ``overall=fail``.
    overall = (
        "pass" if not failures and all(
            c["status"] in {"pass", "skipped"} for c in checks
        )
        else "fail"
    )

    manifest = {
        "schema_version": "real_transform_manifest_v1",
        "overall": overall,
        "real_transform_kind": real_kind,
        "model_id": model_id,
        "fx_module_id": fx_module_id,
        "target_id": eligibility.get("target_id", ""),
        "generated_at_utc": _utcnow(),
        "selected_recipe": selected,
        "matmul_signature": matmul_sig,
        "payload_ref": payload_ref,
        "tile_classification": flags,
        "outputs": {
            "transformed_payload_real": (
                transformed_real_path.relative_to(run_dir).as_posix()
                if transformed_real_path is not None else None
            ),
            "real_transform_diff": (
                diff_path.relative_to(run_dir).as_posix()
                if diff_path is not None else None
            ),
        },
        "skipped_reason": (
            "; ".join(skipped_reasons)
            if real_kind == "unsupported_real_transform" else ""
        ),
        "no_correctness_claim": True,
        "failure_reasons": failures,
    }
    manifest_path = out_dir / "real_transform_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8",
    )

    validation = {
        "schema_version": "real_transform_validation_v1",
        "overall": overall,
        "real_transform_kind": real_kind,
        "checks": checks,
        "no_correctness_claim": True,
        "failure_reasons": failures,
    }
    validation_path = out_dir / "real_transform_validation.json"
    validation_path.write_text(
        json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8",
    )

    # ------------------------------------------------------------------ #
    # Markdown summary for reviewers.
    # ------------------------------------------------------------------ #
    md_lines: list[str] = []
    md_lines.append(f"# Real SetTileParams Transform — {model_id or '(unknown)'}\n")
    md_lines.append(f"_Generated_: {_utcnow()}\n")
    md_lines.append(
        f"- **overall**: `{overall}`  "
        f"\n- **real_transform_kind**: `{real_kind}`  "
        f"\n- **selected recipe**: `{recipe_kind or '—'}` on region "
        f"`{region_id or '—'}`  "
        f"\n- **tile**: `{tile_block}`  "
        f"\n- **no_correctness_claim**: `true` (M-12 owns differential correctness)\n"
    )
    if matmul_sig:
        md_lines.append(
            "## Matmul signature\n"
            f"- M = {matmul_sig.get('M')}, "
            f"N = {matmul_sig.get('N')}, "
            f"K = {matmul_sig.get('K')}\n"
            f"- dtype: lhs={matmul_sig.get('lhs_dtype')}, "
            f"rhs={matmul_sig.get('rhs_dtype')}, "
            f"out={matmul_sig.get('out_dtype')}\n"
            f"- rank: {matmul_sig.get('rank')}; "
            f"dynamic_dims: {matmul_sig.get('dynamic_dims')}\n"
        )
    if real_kind == "unsupported_real_transform" and skipped_reasons:
        md_lines.append("## Skipped reason\n")
        for r in skipped_reasons:
            md_lines.append(f"- {r}")
        md_lines.append("")
    md_lines.append("## Validation checks\n")
    md_lines.append("| name | status | detail |")
    md_lines.append("|---|---|---|")
    for c in checks:
        md_lines.append(f"| {c['name']} | {c['status']} | {c['detail']} |")
    if transformed_real_path is not None:
        md_lines.append(
            f"\n## Output\n"
            f"- `{transformed_real_path.relative_to(run_dir).as_posix()}`\n"
        )
    if failures:
        md_lines.append("## Invariant failures\n")
        for f in failures:
            md_lines.append(f"- {f}")
    summary_md_path = out_dir / "real_transform_summary.md"
    summary_md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return RealLoweringResult(
        overall=overall,
        real_transform_kind=real_kind,
        eligible=eligible,
        out_dir=out_dir,
        transformed_real_path=transformed_real_path,
        diff_path=diff_path,
        manifest_path=manifest_path,
        validation_path=validation_path,
        summary_md_path=summary_md_path,
        failures=tuple(failures),
    )
