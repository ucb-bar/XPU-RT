"""Post-Lowering Verification (Milestone 08).

Applies the M-07 lowering artifacts to a *copy* of Payload IR and runs
structural verification. This is the first stage that produces a
modified IR artifact (``transformed_payload.mlir``).

Hard invariants:

- ``01_payload_lowering/**/payload.mlir`` is **never** mutated. Every
  byte must round-trip identically across a complete M-08 invocation.
- ``transformed_payload.mlir`` lives **only** under
  ``03_recipe_planning/post_lowering/``. Writing it under
  ``01_payload_lowering/`` is a hard fail.
- M-08 makes no claim of full semantic equivalence. Structural
  obligations are marked ``partially_discharged_structural``;
  differential checks are recorded as ``pending`` for M-09 to discharge.
- ``CreateKernelContract`` recipes do **not** produce a
  ``transformed_payload.mlir``. They emit
  ``contract_structural_validation.json`` and leave the contract
  obligation as ``pending_kernel_contract_generation``.

The current MVP is *metadata-only*: ``SetTileParams`` injects a
``compgen.tile = [M, N, K]`` attribute on the matching ``linalg.matmul``
op (identified by its ``compgen.region_id``); ``FuseProducerConsumer``
injects ``compgen.fuse_producer`` / ``compgen.fuse_consumer`` /
``compgen.fuse_via_tensor`` markers on the consumer op. Real loop
tiling and real fusion land in later milestones; this milestone
guarantees the wiring + obligation tracking is correct.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from compgen.graph_compilation.hashing import sha256_tree

# --------------------------------------------------------------------------- #
# Result + entry point
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PostLoweringResult:
    overall: str  # "pass" | "fail"
    out_dir: Path
    transformed_payload_path: Path | None
    applied_manifest_path: Path | None
    structural_diff_path: Path | None
    contract_validation_path: Path | None
    verification_report_path: Path
    semantic_status_path: Path
    failures: tuple[str, ...]


def _utcnow() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> dict[str, Any]:
    obj: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return obj


# --------------------------------------------------------------------------- #
# Transform-script comment-header parser (M-07 emits structured headers)
# --------------------------------------------------------------------------- #


_HEADER_LINE_RE = re.compile(r"^//\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.+?)\s*$")


def _parse_transform_header(text: str) -> dict[str, str]:
    """Pull the leading ``// key: value`` block out of an M-07-emitted
    transform / contract artifact. Stops at the first non-comment line."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith("//"):
            break
        m = _HEADER_LINE_RE.match(stripped)
        if not m:
            continue
        out[m.group(1)] = m.group(2)
    return out


_TILE_RE = re.compile(r"M\s*=\s*(\d+)\s*:\s*i\d+.*?N\s*=\s*(\d+)\s*:\s*i\d+.*?K\s*=\s*(\d+)\s*:\s*i\d+", re.DOTALL)


def _parse_tile_from_transform(text: str) -> tuple[int, int, int] | None:
    m = _TILE_RE.search(text)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


_FUSE_RE = re.compile(
    r'producer\s*=\s*"(?P<producer>[^"]+)".*?'
    r'consumer\s*=\s*"(?P<consumer>[^"]+)".*?'
    r'via_tensor\s*=\s*"(?P<via_tensor>[^"]+)"',
    re.DOTALL,
)


def _parse_fuse_from_transform(text: str) -> dict[str, str] | None:
    m = _FUSE_RE.search(text)
    if not m:
        return None
    return {
        "producer": m.group("producer"),
        "consumer": m.group("consumer"),
        "via_tensor": m.group("via_tensor"),
    }


# --------------------------------------------------------------------------- #
# Payload-IR patcher: adds attributes to the inline `{...}` block of an op
# anchored by its ``compgen.region_id`` value.
# --------------------------------------------------------------------------- #


def _inject_attrs_on_region(
    mlir_text: str, region_id: str, new_attrs: dict[str, str],
) -> tuple[str, int]:
    """Find the (single) op carrying ``compgen.region_id = "<region_id>"``
    and append the given attributes to its inline attributes block.

    ``new_attrs`` values are emitted verbatim (caller is responsible for
    quoting strings, formatting i64, etc.). Returns ``(new_text, n_changed)``.
    """
    needle_re = re.compile(
        r'(\{[^{}]*compgen\.region_id\s*=\s*"' + re.escape(region_id) + r'"[^{}]*?)\}'
    )
    match_count = 0

    def _patch(m: re.Match[str]) -> str:
        nonlocal match_count
        match_count += 1
        head = m.group(1).rstrip()
        if not head.endswith(","):
            head = head + ","
        appended = ", ".join(f"{k} = {v}" for k, v in new_attrs.items())
        return head + " " + appended + "}"

    new_text = needle_re.sub(_patch, mlir_text, count=1)
    return new_text, match_count


def _format_tile_dense_array(M: int, N: int, K: int) -> str:
    """Emit an MLIR dense ``i64`` array attribute literal ``[M, N, K]``.

    We keep it as a plain bracketed list with explicit type tags so
    downstream parsers can recover ``int`` values cleanly.
    """
    return f"[{M} : i64, {N} : i64, {K} : i64]"


# --------------------------------------------------------------------------- #
# Region → payload_ref resolution
# --------------------------------------------------------------------------- #


def _find_region(region_map: dict[str, Any], region_id: str) -> dict[str, Any] | None:
    for r in region_map.get("regions", []):
        if r["region_id"] == region_id:
            return r  # type: ignore[no-any-return]
    return None


def _payload_ref_for_region(region_map: dict[str, Any], region_id: str) -> str:
    r = _find_region(region_map, region_id)
    if r is None:
        return ""
    for po in r.get("payload_ops", []):
        if po.get("payload_ref"):
            ref: str = po["payload_ref"]
            return ref
    return ""


# --------------------------------------------------------------------------- #
# Structural diff
# --------------------------------------------------------------------------- #


def _line_count_diff(a: str, b: str) -> int:
    """Count of lines that differ between two strings, excluding trailing
    newline. Cheap proxy for "how big is the change". Includes adds/removes."""
    al = a.splitlines()
    bl = b.splitlines()
    differing = 0
    for x, y in zip(al, bl, strict=False):
        if x != y:
            differing += 1
    differing += abs(len(al) - len(bl))
    return differing


# --------------------------------------------------------------------------- #
# Top-level entry point
# --------------------------------------------------------------------------- #


def run_post_lowering_verification(run_dir: Path) -> PostLoweringResult:
    """Apply M-07 lowering artifacts on a copy and emit M-08 reports.

    Reads:

    - ``03_recipe_planning/lowering_artifact_manifest.json``
    - ``03_recipe_planning/lowering_artifacts/transforms/*.mlir``
    - ``03_recipe_planning/lowering_artifacts/contracts/*.kernel_contract.mlir``
    - ``03_recipe_planning/verified_recipe.mlir``
    - ``03_recipe_planning/semantic_obligations.json``
    - ``02_graph_analysis/region_map.json`` and tensor_use_def_graph.json
    - ``01_payload_lowering/**/payload.mlir`` (read-only)

    Writes:

    - ``03_recipe_planning/post_lowering/transformed_payload.mlir``
      (only when at least one transform-like artifact applied)
    - ``03_recipe_planning/post_lowering/applied_transform_manifest.json``
      (same condition)
    - ``03_recipe_planning/post_lowering/structural_diff.json`` (same)
    - ``03_recipe_planning/post_lowering/contract_structural_validation.json``
      (only when at least one contract artifact present)
    - ``03_recipe_planning/post_lowering/post_lowering_verification_report.json``
    - ``03_recipe_planning/post_lowering/semantic_obligations_status.json``

    Source payload.mlir is NEVER mutated.
    """
    run_dir = Path(run_dir).resolve()
    rp = run_dir / "03_recipe_planning"
    if not rp.is_dir():
        raise FileNotFoundError(f"03_recipe_planning/ missing under {run_dir}")
    manifest_path = rp / "lowering_artifact_manifest.json"
    sem_json_path = rp / "semantic_obligations.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"M-08 requires M-07 to have run first; missing {manifest_path}"
        )
    if not sem_json_path.exists():
        raise FileNotFoundError(f"semantic_obligations.json missing: {sem_json_path}")

    pl_dir = run_dir / "01_payload_lowering"
    payload_pre_sha = sha256_tree(pl_dir)

    out_dir = rp / "post_lowering"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Wipe stale outputs so an old transformed_payload.mlir cannot
    # masquerade.
    for p in (
        "transformed_payload.mlir",
        "structural_diff.json",
        "applied_transform_manifest.json",
        "contract_structural_validation.json",
        "post_lowering_verification_report.json",
        "semantic_obligations_status.json",
    ):
        target = out_dir / p
        if target.exists():
            target.unlink()

    manifest = _read_json(manifest_path)
    sem = _read_json(sem_json_path)
    region_map = _read_json(run_dir / "02_graph_analysis" / "region_map.json")
    use_def = _read_json(run_dir / "02_graph_analysis" / "tensor_use_def_graph.json")
    obligations = {ob["id"]: dict(ob) for ob in sem.get("obligations", [])}

    failures: list[str] = []
    applied_records: list[dict[str, Any]] = []
    diff_records: list[dict[str, Any]] = []
    contract_records: list[dict[str, Any]] = []

    # The current MVP applies all transforms to a single payload module —
    # the one that the FIRST transform-like artifact's recipe op targets.
    # All canonical models in the suite have one selected recipe op, so
    # this is unambiguous.
    chosen_payload_rel: str | None = None
    chosen_payload_path: Path | None = None

    for art in manifest["artifacts"]:
        kind = art["artifact_kind"]
        recipe_kind = art["recipe_kind"]
        recipe_op_id = art["recipe_op_id"]
        artifact_path = run_dir / art["path"]
        if not artifact_path.exists():
            failures.append(f"missing lowering artifact: {art['path']}")
            continue
        text = artifact_path.read_text(encoding="utf-8")
        header = _parse_transform_header(text)
        cand = header.get("source_candidate", "")
        obl = header.get("semantic_obligation", "")
        refinement = header.get("declared_refinement", "")

        if kind == "transform_script":
            # Resolve target payload + region from header, then patch.
            region_id = header.get("region", "")
            if recipe_kind == "FuseProducerConsumer":
                # For fusion, the consumer is the natural anchor: that's
                # the op whose attributes we patch with the fuse markers.
                fuse = _parse_fuse_from_transform(text)
                if fuse is None:
                    failures.append(
                        f"{recipe_op_id}: fusion transform script malformed"
                    )
                    continue
                anchor_region = fuse["consumer"]
                payload_rel = _payload_ref_for_region(region_map, anchor_region)
                if not payload_rel:
                    failures.append(
                        f"{recipe_op_id}: cannot resolve payload_ref for consumer "
                        f"{anchor_region!r}"
                    )
                    continue
                payload_path = run_dir / payload_rel
                # Pin the payload module on the first transform that lands.
                if chosen_payload_rel is None:
                    chosen_payload_rel = payload_rel
                    chosen_payload_path = payload_path
                elif payload_rel != chosen_payload_rel:
                    failures.append(
                        f"{recipe_op_id}: cross-module fusion not supported in MVP "
                        f"({payload_rel} != {chosen_payload_rel})"
                    )
                    continue
                # Forbid opaque endpoints (M-07 already filtered, but
                # belt-and-suspenders).
                p_region = _find_region(region_map, fuse["producer"])
                c_region = _find_region(region_map, fuse["consumer"])
                if p_region is None or c_region is None:
                    failures.append(
                        f"{recipe_op_id}: producer/consumer missing in region_map"
                    )
                    continue
                # "Opaque" here is the M-04 sense: kind starts with
                # "opaque_". A region whose lead op is a func.call but
                # whose kind is e.g. "elementwise_relu" (because the
                # callee name resolves to a known elementwise family) is
                # legal for fusion under M-04's policy.
                p_kind = str(p_region.get("kind", ""))
                c_kind = str(c_region.get("kind", ""))
                if p_kind.startswith("opaque_") or c_kind.startswith("opaque_"):
                    failures.append(
                        f"{recipe_op_id}: refused to mark fusion across an "
                        f"opaque region (producer={p_kind!r}, consumer={c_kind!r})"
                    )
                    continue
                # Confirm via_tensor exists in tensor_use_def_graph.
                tensor_ids = {t["tensor_id"] for t in use_def.get("tensors", [])}
                if fuse["via_tensor"] not in tensor_ids:
                    failures.append(
                        f"{recipe_op_id}: via_tensor {fuse['via_tensor']!r} "
                        f"not in tensor_use_def_graph"
                    )
                    continue
                # Patch the consumer op.
                applied_records.append(
                    {
                        "recipe_op_id": recipe_op_id,
                        "recipe_kind": recipe_kind,
                        "anchor_region": anchor_region,
                        "producer": fuse["producer"],
                        "consumer": fuse["consumer"],
                        "via_tensor": fuse["via_tensor"],
                        "application_mode": "metadata_only_structural_mvp",
                        "status": "applied",
                        "source_candidate": cand,
                        "semantic_obligation": obl,
                    }
                )
                diff_records.append(
                    {
                        "recipe_op_id": recipe_op_id,
                        "kind": "annotation_added",
                        "anchor_region": anchor_region,
                        "attribute": "compgen.fuse_consumer",
                        "before": None,
                        "after": fuse["consumer"],
                    }
                )
                # Defer the actual injection until we have all transforms,
                # so we apply them in one pass over the file.

            elif recipe_kind == "SetTileParams":
                if not region_id:
                    failures.append(f"{recipe_op_id}: tile transform has no region")
                    continue
                # Reject opaque target.
                rec = _find_region(region_map, region_id)
                if rec is None:
                    failures.append(
                        f"{recipe_op_id}: region {region_id!r} not in region_map"
                    )
                    continue
                kind = str(rec.get("kind", ""))
                if kind.startswith("opaque_"):
                    failures.append(
                        f"{recipe_op_id}: refused to tile opaque region "
                        f"{region_id!r} (kind={kind!r})"
                    )
                    continue
                tile = _parse_tile_from_transform(text)
                if tile is None:
                    failures.append(
                        f"{recipe_op_id}: tile transform script missing M/N/K"
                    )
                    continue
                # Cross-check the tile against the verified recipe — the
                # script must not have drifted from the source candidate.
                vrecipe_tile = _verified_recipe_tile(rp, recipe_op_id)
                if vrecipe_tile is not None and vrecipe_tile != tile:
                    failures.append(
                        f"{recipe_op_id}: tile script {tile} disagrees with "
                        f"verified_recipe.mlir {vrecipe_tile}"
                    )
                    continue
                payload_rel = _payload_ref_for_region(region_map, region_id)
                if not payload_rel:
                    failures.append(
                        f"{recipe_op_id}: cannot resolve payload_ref for region "
                        f"{region_id!r}"
                    )
                    continue
                payload_path = run_dir / payload_rel
                if chosen_payload_rel is None:
                    chosen_payload_rel = payload_rel
                    chosen_payload_path = payload_path
                elif payload_rel != chosen_payload_rel:
                    failures.append(
                        f"{recipe_op_id}: cross-module tile not supported in MVP "
                        f"({payload_rel} != {chosen_payload_rel})"
                    )
                    continue
                applied_records.append(
                    {
                        "recipe_op_id": recipe_op_id,
                        "recipe_kind": recipe_kind,
                        "region": region_id,
                        "tile": {"M": tile[0], "N": tile[1], "K": tile[2]},
                        "application_mode": "metadata_only_structural_mvp",
                        "status": "applied",
                        "source_candidate": cand,
                        "semantic_obligation": obl,
                    }
                )
                diff_records.append(
                    {
                        "recipe_op_id": recipe_op_id,
                        "kind": "annotation_added",
                        "region": region_id,
                        "attribute": "compgen.tile",
                        "before": None,
                        "after": list(tile),
                    }
                )
            else:
                # Numerics / placement etc. — emit a generic marker on the region.
                if region_id:
                    applied_records.append(
                        {
                            "recipe_op_id": recipe_op_id,
                            "recipe_kind": recipe_kind,
                            "region": region_id,
                            "application_mode": "metadata_only_structural_mvp",
                            "status": "applied",
                            "source_candidate": cand,
                            "semantic_obligation": obl,
                        }
                    )
                    diff_records.append(
                        {
                            "recipe_op_id": recipe_op_id,
                            "kind": "annotation_added",
                            "region": region_id,
                            "attribute": "compgen.recipe_op",
                            "before": None,
                            "after": recipe_op_id,
                        }
                    )

        elif kind == "kernel_contract_draft":
            # Validate structurally; do NOT emit transformed_payload.mlir.
            cval, cfailures = _validate_kernel_contract(
                run_dir, art, region_map, obligations,
            )
            failures.extend(cfailures)
            contract_records.append(cval)

        else:
            failures.append(f"unknown artifact_kind: {kind!r}")

    # ------------------------------------------------------------------ #
    # Apply the queued transform records to a copy of the chosen payload.
    # We do all attribute injections in a single pass over the file.
    # ------------------------------------------------------------------ #
    transformed_payload_path: Path | None = None
    applied_manifest_path: Path | None = None
    structural_diff_path: Path | None = None

    transform_like = [
        r for r in applied_records if r.get("status") == "applied"
    ]
    if transform_like and chosen_payload_path is not None and not failures:
        original_text = chosen_payload_path.read_text(encoding="utf-8")
        new_text = original_text

        for rec in transform_like:
            kind = rec["recipe_kind"]
            if kind == "SetTileParams":
                M = rec["tile"]["M"]
                N = rec["tile"]["N"]
                K = rec["tile"]["K"]
                attrs = {
                    "compgen.tile": _format_tile_dense_array(M, N, K),
                    "compgen.recipe_op": f'"{rec["recipe_op_id"]}"',
                    "compgen.semantic_obligation": f'"{rec["semantic_obligation"]}"',
                }
                new_text, n = _inject_attrs_on_region(
                    new_text, rec["region"], attrs,
                )
                if n != 1:
                    failures.append(
                        f"{rec['recipe_op_id']}: expected 1 SetTileParams "
                        f"injection on region {rec['region']!r}, got {n}"
                    )
            elif kind == "FuseProducerConsumer":
                attrs = {
                    "compgen.fuse_producer": f'"{rec["producer"]}"',
                    "compgen.fuse_consumer": f'"{rec["consumer"]}"',
                    "compgen.fuse_via_tensor": f'"{rec["via_tensor"]}"',
                    "compgen.recipe_op": f'"{rec["recipe_op_id"]}"',
                    "compgen.semantic_obligation": f'"{rec["semantic_obligation"]}"',
                }
                new_text, n = _inject_attrs_on_region(
                    new_text, rec["anchor_region"], attrs,
                )
                if n != 1:
                    failures.append(
                        f"{rec['recipe_op_id']}: expected 1 FuseProducerConsumer "
                        f"injection on region {rec['anchor_region']!r}, got {n}"
                    )
            else:
                attrs = {
                    "compgen.recipe_op": f'"{rec["recipe_op_id"]}"',
                    "compgen.semantic_obligation": f'"{rec["semantic_obligation"]}"',
                }
                new_text, n = _inject_attrs_on_region(
                    new_text, rec.get("region", ""), attrs,
                )
                if n != 1:
                    failures.append(
                        f"{rec['recipe_op_id']}: expected 1 generic injection, got {n}"
                    )

        if not failures:
            transformed_payload_path = out_dir / "transformed_payload.mlir"
            transformed_payload_path.write_text(new_text, encoding="utf-8")
            transformed_sha = "sha256:" + hashlib.sha256(
                new_text.encode("utf-8")
            ).hexdigest()
            original_sha = "sha256:" + hashlib.sha256(
                original_text.encode("utf-8")
            ).hexdigest()
            changed_lines = _line_count_diff(original_text, new_text)

            applied_manifest = {
                "schema_version": "applied_transform_manifest_v1",
                "status": "pass",
                "model_id": manifest.get("model_id", ""),
                "target_id": manifest.get("target_id", ""),
                "generated_at_utc": _utcnow(),
                "source": {
                    "payload": chosen_payload_rel,
                    "payload_sha256_before": original_sha,
                    "verified_recipe": "03_recipe_planning/verified_recipe.mlir",
                    "transform_scripts": [
                        a["path"]
                        for a in manifest["artifacts"]
                        if a["artifact_kind"] == "transform_script"
                    ],
                },
                "outputs": {
                    "transformed_payload":
                        transformed_payload_path.relative_to(run_dir).as_posix(),
                    "transformed_payload_sha256": transformed_sha,
                },
                "applied": applied_records,
            }
            applied_manifest_path = out_dir / "applied_transform_manifest.json"
            applied_manifest_path.write_text(
                json.dumps(applied_manifest, indent=2, sort_keys=True),
                encoding="utf-8",
            )

            structural_diff = {
                "schema_version": "structural_diff_v1",
                "status": "pass",
                "summary": {
                    "source_payload_sha256": original_sha,
                    "transformed_payload_sha256": transformed_sha,
                    "changed_lines": changed_lines,
                    "semantic_change_claimed": False,
                },
                "diffs": diff_records,
            }
            structural_diff_path = out_dir / "structural_diff.json"
            structural_diff_path.write_text(
                json.dumps(structural_diff, indent=2, sort_keys=True),
                encoding="utf-8",
            )

    # Contract-only path: write contract_structural_validation.json
    contract_validation_path: Path | None = None
    if contract_records:
        contract_validation_path = out_dir / "contract_structural_validation.json"
        contract_validation_path.write_text(
            json.dumps(
                {
                    "schema_version": "contract_structural_validation_v1",
                    "status": "pass" if not failures else "fail",
                    "model_id": manifest.get("model_id", ""),
                    "target_id": manifest.get("target_id", ""),
                    "generated_at_utc": _utcnow(),
                    "validations": contract_records,
                },
                indent=2, sort_keys=True,
            ),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------ #
    # Belt-and-suspenders: source payload tree must be byte-identical.
    # ------------------------------------------------------------------ #
    payload_post_sha = sha256_tree(pl_dir)
    payload_unchanged = payload_pre_sha == payload_post_sha
    if not payload_unchanged:
        failures.append(
            f"01_payload_lowering/ tree was modified during M-08 "
            f"(pre={payload_pre_sha[:16]}..., post={payload_post_sha[:16]}...)"
        )

    # transformed_payload.mlir must NEVER live under 01_payload_lowering/
    leak = list(pl_dir.rglob("transformed_payload*"))
    if leak:
        failures.append(
            f"transformed_payload found under 01_payload_lowering/: "
            f"{[p.relative_to(run_dir).as_posix() for p in leak]}"
        )

    # ------------------------------------------------------------------ #
    # semantic_obligations_status.json
    # ------------------------------------------------------------------ #
    statuses: list[dict[str, Any]] = []
    for rec in applied_records:
        obl_id = rec["semantic_obligation"].lstrip("@")
        ob = obligations.get(obl_id)
        if ob is None:
            failures.append(
                f"{rec['recipe_op_id']}: obligation {obl_id!r} missing in "
                f"semantic_obligations.json"
            )
            continue
        refinement = ob["refinement"]
        if refinement in ("bit_equality", "tolerance_eps"):
            status = "partially_discharged_structural"
            remaining = ["differential_check"]
        elif refinement == "placement_obligation":
            status = "partially_discharged_structural"
            remaining = ["runtime_dispatch_check"]
        else:
            status = "pending_kernel_contract_generation"
            remaining = ["kernel_contract_generation"]
        statuses.append(
            {
                "obligation": obl_id,
                "recipe_op_id": rec["recipe_op_id"],
                "declared_refinement": refinement,
                "status": status,
                "remaining": remaining,
            }
        )
    for cval in contract_records:
        obl_id = cval["semantic_obligation"]
        ob = obligations.get(obl_id, {})
        statuses.append(
            {
                "obligation": obl_id,
                "recipe_op_id": cval["recipe_op_id"],
                "declared_refinement": ob.get("refinement", "contract_obligation"),
                "status": "pending_kernel_contract_generation",
                "remaining": ["kernel_contract_generation", "differential_check"],
            }
        )

    semantic_status_path = out_dir / "semantic_obligations_status.json"
    semantic_status_path.write_text(
        json.dumps(
            {
                "schema_version": "semantic_obligations_status_v1",
                "model_id": manifest.get("model_id", ""),
                "target_id": manifest.get("target_id", ""),
                "generated_at_utc": _utcnow(),
                "statuses": statuses,
            },
            indent=2, sort_keys=True,
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------ #
    # post_lowering_verification_report.json
    # ------------------------------------------------------------------ #
    checks: list[dict[str, Any]] = []

    def _add(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "status": "pass" if ok else "fail", "detail": detail})

    _add("source_payload_unchanged", payload_unchanged, "")
    if transform_like:
        _add(
            "transformed_payload_exists",
            transformed_payload_path is not None and transformed_payload_path.exists(),
            "",
        )
        _add(
            "transform_artifact_applied_once",
            len(applied_records) == len(transform_like),
            "",
        )
    else:
        _add(
            "no_transformed_payload_for_contract_only",
            transformed_payload_path is None,
            "",
        )
    _add(
        "transformed_payload_not_under_01_payload_lowering",
        not leak,
        "",
    )
    _add(
        "semantic_obligation_referenced",
        all(rec["semantic_obligation"] for rec in applied_records),
        "",
    )
    _add(
        "structural_obligation_partially_discharged",
        all(
            s["status"] == "partially_discharged_structural"
            for s in statuses
            if s["declared_refinement"] in ("bit_equality", "tolerance_eps")
        ),
        "",
    )
    _add(
        "no_full_differential_discharge_claimed",
        all(
            "differential_check" in s["remaining"] or
            s["declared_refinement"] == "contract_obligation"
            for s in statuses
        ),
        "",
    )

    overall = "pass" if (not failures and all(c["status"] == "pass" for c in checks)) else "fail"
    verification_report = {
        "schema_version": "post_lowering_verification_report_v1",
        "status": overall,
        "model_id": manifest.get("model_id", ""),
        "target_id": manifest.get("target_id", ""),
        "generated_at_utc": _utcnow(),
        "checks": checks,
        "semantic_status": statuses,
        "failure_reasons": failures,
        "source": {
            "payload_pre_sha256": payload_pre_sha,
            "payload_post_sha256": payload_post_sha,
        },
    }
    verification_report_path = out_dir / "post_lowering_verification_report.json"
    verification_report_path.write_text(
        json.dumps(verification_report, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return PostLoweringResult(
        overall=overall,
        out_dir=out_dir,
        transformed_payload_path=transformed_payload_path,
        applied_manifest_path=applied_manifest_path,
        structural_diff_path=structural_diff_path,
        contract_validation_path=contract_validation_path,
        verification_report_path=verification_report_path,
        semantic_status_path=semantic_status_path,
        failures=tuple(failures),
    )


# --------------------------------------------------------------------------- #
# Helpers used above
# --------------------------------------------------------------------------- #


def _verified_recipe_tile(
    rp: Path, recipe_op_id: str
) -> tuple[int, int, int] | None:
    """Pull the M/N/K tile from verified_recipe.mlir for a SetTileParams op."""
    text = (rp / "verified_recipe.mlir").read_text(encoding="utf-8")
    pat = re.compile(
        rf"recipe\.set_tile_params\s+@{re.escape(recipe_op_id)}.*?"
        r"tile\s*=\s*\{\s*K\s*=\s*(\d+)\s*:\s*i\d+\s*,\s*"
        r"M\s*=\s*(\d+)\s*:\s*i\d+\s*,\s*"
        r"N\s*=\s*(\d+)\s*:\s*i\d+\s*\}",
        re.DOTALL,
    )
    m = pat.search(text)
    if not m:
        return None
    K, M, N = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return M, N, K


def _validate_kernel_contract(
    run_dir: Path,
    art: dict[str, Any],
    region_map: dict[str, Any],
    obligations: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    """Structurally validate a kernel contract draft. Returns (record, failures)."""
    rp = run_dir / art["path"]
    text = rp.read_text(encoding="utf-8")
    header = _parse_transform_header(text)
    failures: list[str] = []

    region_match = re.search(r'region_id\s*=\s*"([^"]+)"', text)
    region_id = region_match.group(1) if region_match else ""
    payload_match = re.search(r'payload_ref\s*=\s*"([^"]+)"', text)
    payload_ref = payload_match.group(1) if payload_match else ""

    proof_stage_match = re.search(r'proof_stage\s*=\s*"([^"]+)"', text)
    proof_stage = proof_stage_match.group(1) if proof_stage_match else ""

    sem_match = re.search(r"sem\.obligation_ref\s+@([A-Za-z_][A-Za-z0-9_]*)", text)
    obl_id = sem_match.group(1) if sem_match else header.get("semantic_obligation", "")

    rec = _find_region(region_map, region_id)
    src_class = rec.get("source_classification", "") if rec else ""

    if not region_id:
        failures.append(f"contract {art['path']}: region_id missing")
    if rec is None:
        failures.append(f"contract {art['path']}: region {region_id!r} not in region_map")
    if rec is not None and src_class != "opaque_fallback":
        failures.append(
            f"contract {art['path']}: source_classification={src_class!r}, "
            f"expected opaque_fallback"
        )
    if not obl_id or obl_id not in obligations:
        failures.append(
            f"contract {art['path']}: semantic_obligation {obl_id!r} missing"
        )
    if proof_stage != "kernel_contract_generation":
        failures.append(
            f"contract {art['path']}: proof_stage={proof_stage!r}, "
            f"expected kernel_contract_generation"
        )
    if payload_ref and not (run_dir / payload_ref).exists():
        failures.append(
            f"contract {art['path']}: payload_ref {payload_ref!r} not on disk"
        )

    return (
        {
            "recipe_op_id": art["recipe_op_id"],
            "recipe_kind": art["recipe_kind"],
            "contract_path": art["path"],
            "region_id": region_id,
            "source_classification": src_class,
            "payload_ref": payload_ref,
            "proof_stage": proof_stage,
            "semantic_obligation": obl_id,
            "checks": {
                "contract_draft_exists": True,
                "references_semantic_obligation": bool(obl_id) and obl_id in obligations,
                "references_opaque_region": src_class == "opaque_fallback",
                "references_payload_ref": bool(payload_ref) and (
                    run_dir / payload_ref
                ).exists() if payload_ref else False,
                "proof_stage_is_kernel_contract_generation":
                    proof_stage == "kernel_contract_generation",
            },
            "failures": [f for f in failures if art["path"] in f],
        },
        failures,
    )
