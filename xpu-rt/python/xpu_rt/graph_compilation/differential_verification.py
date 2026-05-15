"""Differential / Reference Verification (Milestone 09).

Discharge obligations on the metadata-only post-lowering MVP. Given
that the current ``transformed_payload.mlir`` differs from the source
only by injected ``compgen.*`` attributes, this stage proves
*semantic-inert-by-metadata*: stripping every ``compgen.*`` attribute
from both files yields byte-identical text. It also re-checks Stage-0
golden references (when available) and validates contract drafts for
contract-only recipes.

What this stage does NOT prove (and must not claim):

- Real loop tiling correctness.
- Real fusion correctness.
- Differential correctness of any future real transform.
- Functional equivalence under runtime execution.

Those are tracked as ``real_transform_differential_check`` /
``real_fusion_differential_check`` / ``contract_differential_check``
under ``remaining`` in the obligation status file.

Hard invariants:

- ``01_payload_lowering/`` is read-only across the stage; the source
  payload tree must be byte-identical pre/post.
- Contract-only recipes never produce a ``transformed_payload.mlir``;
  finding one for a contract-only model fails the stage.
- Reports must explicitly carry ``no_real_transform_claimed: pass`` so
  any future false discharge claim shows up as an inverted check.
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
class DifferentialVerificationResult:
    overall: str  # "pass" | "fail"
    out_dir: Path
    mode: str  # "metadata_noop_mvp" | "contract_only_mvp"
    report_path: Path
    semantic_status_path: Path
    failures: tuple[str, ...]


def _utcnow() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> dict[str, Any]:
    obj: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return obj


# --------------------------------------------------------------------------- #
# compgen metadata stripper
# --------------------------------------------------------------------------- #


def _split_attr_entries(body: str) -> list[str]:
    """Split a single-line attribute-dict body into entries by top-level
    commas. Respects nested ``[]`` / ``{}`` and double-quoted strings.

    The body is the text *inside* the outermost ``{...}`` (without the
    braces). Returns one string per entry, with leading/trailing
    whitespace preserved per entry to support faithful round-trip.
    """
    out: list[str] = []
    depth_sq = 0  # [
    depth_cu = 0  # {
    depth_pa = 0  # (
    in_str = False
    escape = False
    start = 0
    for i, ch in enumerate(body):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "[":
            depth_sq += 1
        elif ch == "]":
            depth_sq -= 1
        elif ch == "{":
            depth_cu += 1
        elif ch == "}":
            depth_cu -= 1
        elif ch == "(":
            depth_pa += 1
        elif ch == ")":
            depth_pa -= 1
        elif ch == "," and depth_sq == 0 and depth_cu == 0 and depth_pa == 0:
            out.append(body[start:i])
            start = i + 1
    out.append(body[start:])
    return out


def _entry_key(entry: str) -> str:
    """Pull the attribute key out of an entry like ``compgen.tile = [...]``."""
    m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_.]*)\s*(?:=|$)", entry)
    return m.group(1) if m else ""


def _find_inline_attr_blocks(text: str) -> list[tuple[int, int]]:
    """Return (start, end) byte ranges for each single-line inline attribute
    block ``{...}``. Region bodies (multi-line ``{`` ... ``}``) are skipped:
    we only treat a ``{`` as the opener of an attribute block when its
    matching ``}`` lives on the same line *and* the body looks like a
    comma-separated key=value list (every top-level entry has an ``=``).
    """
    ranges: list[tuple[int, int]] = []
    n = len(text)
    i = 0
    in_str = False
    escape = False
    while i < n:
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
            i += 1
            continue
        if ch == "{":
            # Try to find the matching '}' on the same line.
            depth = 1
            j = i + 1
            same_line = True
            in_str2 = False
            esc2 = False
            while j < n:
                cj = text[j]
                if in_str2:
                    if esc2:
                        esc2 = False
                    elif cj == "\\":
                        esc2 = True
                    elif cj == '"':
                        in_str2 = False
                    j += 1
                    continue
                if cj == '"':
                    in_str2 = True
                    j += 1
                    continue
                if cj == "\n":
                    same_line = False
                    break
                if cj == "{":
                    depth += 1
                elif cj == "}":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            if same_line and depth == 0 and j < n and text[j] == "}":
                body = text[i + 1:j]
                # Heuristic for "this is an attribute dict": every
                # top-level entry must have an '=' before its first
                # top-level comma.
                entries = _split_attr_entries(body)
                non_empty = [e for e in entries if e.strip()]
                if non_empty and all("=" in e for e in non_empty):
                    ranges.append((i, j + 1))
                    i = j + 1
                    continue
            i += 1
            continue
        i += 1
    return ranges


def strip_compgen_metadata(text: str) -> str:
    """Return ``text`` with every ``compgen.*`` attribute removed from
    inline attribute dicts.

    Empty attribute blocks ``{}`` produced by the strip are collapsed
    along with any single space immediately preceding them. The rest of
    the file (including non-attribute ``{...}`` groups, region bodies,
    and ``//`` comments) is preserved byte-for-byte.
    """
    blocks = _find_inline_attr_blocks(text)
    if not blocks:
        return text
    out: list[str] = []
    cursor = 0
    for start, end in blocks:
        out.append(text[cursor:start])
        body = text[start + 1:end - 1]
        entries = _split_attr_entries(body)
        kept: list[str] = []
        for entry in entries:
            if not entry.strip():
                continue
            key = _entry_key(entry)
            if key.startswith("compgen."):
                continue
            kept.append(entry)
        if kept:
            # Reassemble with comma+space separators, trimming surrounding
            # whitespace inside each kept entry to avoid a leading/trailing
            # space drift that would break textual equality.
            new_body = ", ".join(e.strip() for e in kept)
            out.append("{" + new_body + "}")
        else:
            # Drop the empty {} and an immediately preceding space, if any,
            # to avoid leaving "linalg.matmul  ins(..." with a double space.
            tail = out[-1] if out else ""
            if tail.endswith(" "):
                out[-1] = tail[:-1]
            # Skip writing the empty block.
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


# --------------------------------------------------------------------------- #
# Golden reference check
# --------------------------------------------------------------------------- #


def _check_goldens(run_dir: Path) -> dict[str, Any]:
    gc = run_dir / "00_graph_capture"
    gi = gc / "golden_inputs.pt"
    go = gc / "golden_outputs.pt"
    record: dict[str, Any] = {
        "schema_version": "golden_reference_check_v1",
        "generated_at_utc": _utcnow(),
        "inputs_path": "00_graph_capture/golden_inputs.pt",
        "outputs_path": "00_graph_capture/golden_outputs.pt",
        "checks": [],
    }
    inputs_ok = gi.exists()
    outputs_ok = go.exists()
    record["checks"].append(
        {"name": "golden_inputs_present", "status": "pass" if inputs_ok else "skipped",
         "detail": "" if inputs_ok else "golden_inputs.pt not on disk"}
    )
    record["checks"].append(
        {"name": "golden_outputs_present", "status": "pass" if outputs_ok else "skipped",
         "detail": "" if outputs_ok else "golden_outputs.pt not on disk"}
    )

    inputs_loadable = False
    outputs_loadable = False
    inputs_meta: list[dict[str, Any]] = []
    outputs_meta: list[dict[str, Any]] = []
    load_failures: list[str] = []
    if inputs_ok or outputs_ok:
        try:
            import torch
        except ImportError:  # pragma: no cover - defensive only
            load_failures.append("torch unavailable for golden reload")
            torch = None  # type: ignore[assignment]
        else:
            def _meta_of(obj: Any) -> list[dict[str, Any]]:
                items = obj if isinstance(obj, (list, tuple)) else [obj]
                meta: list[dict[str, Any]] = []
                for x in items:
                    if isinstance(x, torch.Tensor):
                        meta.append(
                            {
                                "dtype": str(x.dtype),
                                "shape": list(x.shape),
                            }
                        )
                    else:
                        meta.append({"non_tensor_type": type(x).__name__})
                return meta

            if inputs_ok:
                try:
                    obj_in = torch.load(gi, weights_only=False)
                    inputs_meta = _meta_of(obj_in)
                    inputs_loadable = True
                except Exception as e:  # noqa: BLE001
                    load_failures.append(f"failed to load golden_inputs.pt: {e}")
            if outputs_ok:
                try:
                    obj_out = torch.load(go, weights_only=False)
                    outputs_meta = _meta_of(obj_out)
                    outputs_loadable = True
                except Exception as e:  # noqa: BLE001
                    load_failures.append(f"failed to load golden_outputs.pt: {e}")

    if inputs_ok:
        record["checks"].append(
            {
                "name": "golden_inputs_loadable",
                "status": "pass" if inputs_loadable else "fail",
                "detail": "",
            }
        )
    if outputs_ok:
        record["checks"].append(
            {
                "name": "golden_outputs_loadable",
                "status": "pass" if outputs_loadable else "fail",
                "detail": "",
            }
        )

    record["inputs_meta"] = inputs_meta
    record["outputs_meta"] = outputs_meta
    record["load_failures"] = load_failures

    # status semantics: pass when every present file is loadable; skipped
    # when neither is on disk; fail when any present file fails to load.
    if not inputs_ok and not outputs_ok:
        record["status"] = "skipped"
        record["skipped_reason"] = (
            "no Stage-0 golden artifacts on disk for this model"
        )
    elif load_failures:
        record["status"] = "fail"
    else:
        record["status"] = "pass"
    return record


# --------------------------------------------------------------------------- #
# Contract reference check (contract-only path)
# --------------------------------------------------------------------------- #


def _check_contracts(
    run_dir: Path, manifest: dict[str, Any], obligations: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    """Validate contract drafts referenced by the lowering manifest.

    Reads the ``contract_structural_validation.json`` for the
    structural verdict and re-verifies that each contract still
    references a known semantic obligation and a payload ref that
    exists on disk.
    """
    failures: list[str] = []
    rp = run_dir / "03_recipe_planning"
    contract_artifacts = [
        a for a in manifest["artifacts"]
        if a["artifact_kind"] == "kernel_contract_draft"
    ]
    pl_validation = rp / "post_lowering" / "contract_structural_validation.json"
    pl_validation_status = "missing"
    pl_validation_records: list[dict[str, Any]] = []
    if pl_validation.exists():
        body = _read_json(pl_validation)
        pl_validation_status = str(body.get("status", "missing"))
        pl_validation_records = list(body.get("validations", []))

    drafts: list[dict[str, Any]] = []
    for art in contract_artifacts:
        cpath = run_dir / art["path"]
        text = cpath.read_text(encoding="utf-8") if cpath.exists() else ""

        sem_match = re.search(r"sem\.obligation_ref\s+@([A-Za-z_][A-Za-z0-9_]*)", text)
        obl_id = sem_match.group(1) if sem_match else ""
        proof_match = re.search(r'proof_stage\s*=\s*"([^"]+)"', text)
        proof_stage = proof_match.group(1) if proof_match else ""
        region_match = re.search(r'region_id\s*=\s*"([^"]+)"', text)
        region_id = region_match.group(1) if region_match else ""

        contract_exists = cpath.exists()
        obl_ok = bool(obl_id) and obl_id in obligations
        proof_ok = proof_stage == "kernel_contract_generation"
        region_ok = bool(region_id)

        # Mirror the structural finding for this contract, if any.
        m08_record = next(
            (r for r in pl_validation_records if r.get("contract_path") == art["path"]),
            None,
        )

        drafts.append(
            {
                "recipe_op_id": art["recipe_op_id"],
                "recipe_kind": art["recipe_kind"],
                "contract_path": art["path"],
                "region_id": region_id,
                "semantic_obligation": obl_id,
                "proof_stage": proof_stage,
                "checks": {
                    "contract_draft_exists": contract_exists,
                    "references_known_obligation": obl_ok,
                    "proof_stage_is_kernel_contract_generation": proof_ok,
                    "references_region": region_ok,
                    "structural_validation_pass":
                        bool(m08_record)
                        and not m08_record.get("failures")
                        and pl_validation_status == "pass",
                },
                "evidence_status": "pending_kernel_contract_generation",
            }
        )

        if not contract_exists:
            failures.append(f"contract {art['path']}: file missing")
        if not obl_ok:
            failures.append(
                f"contract {art['path']}: semantic_obligation {obl_id!r} not in obligations"
            )
        if not proof_ok:
            failures.append(
                f"contract {art['path']}: proof_stage={proof_stage!r}, "
                f"expected kernel_contract_generation"
            )
        if not region_ok:
            failures.append(f"contract {art['path']}: region_id missing")

    record = {
        "schema_version": "contract_reference_check_v1",
        "status": "pass" if not failures else "fail",
        "model_id": manifest.get("model_id", ""),
        "target_id": manifest.get("target_id", ""),
        "generated_at_utc": _utcnow(),
        "structural_validation_source": pl_validation.relative_to(run_dir).as_posix()
        if pl_validation.exists() else None,
        "structural_validation_status": pl_validation_status,
        "drafts": drafts,
    }
    return record, failures


# --------------------------------------------------------------------------- #
# Top-level entry point
# --------------------------------------------------------------------------- #


def run_differential_verification(run_dir: Path) -> DifferentialVerificationResult:
    """Run differential / reference verification.

    Operates in two modes depending on the recipes in the lowering
    manifest:

    - **metadata_noop_mvp** when at least one transform-like artifact
      ran. Strips ``compgen.*`` from the source and transformed payloads
      and proves textual equality. The presence of
      ``transformed_payload.mlir`` is required.
    - **contract_only_mvp** when every artifact is a kernel-contract
      draft. ``transformed_payload.mlir`` must NOT exist; contract
      drafts are validated structurally instead.

    Source payload bytes (under ``01_payload_lowering/``) must be
    unchanged across the call. The check is wired in by hashing the
    tree before and after.
    """
    run_dir = Path(run_dir).resolve()
    rp = run_dir / "03_recipe_planning"
    if not rp.is_dir():
        raise FileNotFoundError(f"03_recipe_planning/ missing under {run_dir}")
    manifest_path = rp / "lowering_artifact_manifest.json"
    sem_json_path = rp / "semantic_obligations.json"
    pl_report_path = rp / "post_lowering" / "post_lowering_verification_report.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"M-09 requires M-07 to have run first; missing {manifest_path}"
        )
    if not pl_report_path.exists():
        raise FileNotFoundError(
            f"M-09 requires M-08 to have run first; missing {pl_report_path}"
        )
    if not sem_json_path.exists():
        raise FileNotFoundError(f"semantic_obligations.json missing: {sem_json_path}")

    pl_dir = run_dir / "01_payload_lowering"
    payload_pre_sha = sha256_tree(pl_dir)

    out_dir = rp / "differential_verification"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Wipe stale outputs.
    for p in (
        "metadata_noop_equivalence.json",
        "normalized_source_payload.mlir",
        "normalized_transformed_payload.mlir",
        "normalized_payload_diff.txt",
        "golden_reference_check.json",
        "contract_reference_check.json",
        "differential_verification_report.json",
        "semantic_obligations_status.json",
    ):
        target = out_dir / p
        if target.exists():
            target.unlink()

    manifest = _read_json(manifest_path)
    sem = _read_json(sem_json_path)
    pl_report = _read_json(pl_report_path)
    obligations = {ob["id"]: dict(ob) for ob in sem.get("obligations", [])}

    has_transform = any(
        a["artifact_kind"] == "transform_script" for a in manifest["artifacts"]
    )
    has_contract = any(
        a["artifact_kind"] == "kernel_contract_draft" for a in manifest["artifacts"]
    )
    mode = "metadata_noop_mvp" if has_transform else "contract_only_mvp"

    failures: list[str] = []
    checks: list[dict[str, Any]] = []

    def _add(name: str, ok: bool, detail: str = "") -> None:
        checks.append(
            {"name": name, "status": "pass" if ok else "fail", "detail": detail}
        )

    _add("source_payload_exists", any(pl_dir.rglob("payload.mlir")))
    _add(
        "post_lowering_verification_passed",
        pl_report.get("status") == "pass",
        "" if pl_report.get("status") == "pass" else f"got {pl_report.get('status')!r}",
    )

    metadata_noop_record: dict[str, Any] | None = None
    contract_record: dict[str, Any] | None = None
    transformed_payload_present = (
        rp / "post_lowering" / "transformed_payload.mlir"
    ).exists()

    if mode == "metadata_noop_mvp":
        # Transform-like models: prove metadata-only equivalence.
        if not transformed_payload_present:
            failures.append(
                "metadata_noop_mvp requires transformed_payload.mlir, but it "
                "is missing under 03_recipe_planning/post_lowering/"
            )
            _add("transformed_payload_exists", False, "missing")
        else:
            _add("transformed_payload_exists", True)

        applied_manifest_path = rp / "post_lowering" / "applied_transform_manifest.json"
        if not applied_manifest_path.exists():
            failures.append(
                "applied_transform_manifest.json missing; cannot resolve source payload"
            )
        else:
            applied = _read_json(applied_manifest_path)
            chosen_rel = applied["source"]["payload"]
            chosen_path = run_dir / chosen_rel
            transformed_path = rp / "post_lowering" / "transformed_payload.mlir"
            if not chosen_path.exists():
                failures.append(
                    f"source payload referenced by applied_transform_manifest "
                    f"missing: {chosen_rel}"
                )
            elif transformed_payload_present:
                source_text = chosen_path.read_text(encoding="utf-8")
                transformed_text = transformed_path.read_text(encoding="utf-8")
                normalized_source = strip_compgen_metadata(source_text)
                normalized_transformed = strip_compgen_metadata(transformed_text)
                (out_dir / "normalized_source_payload.mlir").write_text(
                    normalized_source, encoding="utf-8",
                )
                (out_dir / "normalized_transformed_payload.mlir").write_text(
                    normalized_transformed, encoding="utf-8",
                )
                # Empty diff file when equal; line-level diff otherwise.
                if normalized_source == normalized_transformed:
                    (out_dir / "normalized_payload_diff.txt").write_text(
                        "", encoding="utf-8",
                    )
                    diff_lines: list[str] = []
                else:
                    import difflib

                    diff_lines = list(
                        difflib.unified_diff(
                            normalized_source.splitlines(keepends=False),
                            normalized_transformed.splitlines(keepends=False),
                            fromfile="normalized_source_payload.mlir",
                            tofile="normalized_transformed_payload.mlir",
                            lineterm="",
                        )
                    )
                    (out_dir / "normalized_payload_diff.txt").write_text(
                        "\n".join(diff_lines) + "\n", encoding="utf-8",
                    )

                src_norm_sha = "sha256:" + hashlib.sha256(
                    normalized_source.encode("utf-8")
                ).hexdigest()
                tx_norm_sha = "sha256:" + hashlib.sha256(
                    normalized_transformed.encode("utf-8")
                ).hexdigest()

                metadata_noop_record = {
                    "schema_version": "metadata_noop_equivalence_v1",
                    "status": "pass" if not diff_lines else "fail",
                    "model_id": manifest.get("model_id", ""),
                    "target_id": manifest.get("target_id", ""),
                    "generated_at_utc": _utcnow(),
                    "source": {
                        "source_payload": chosen_rel,
                        "transformed_payload":
                            "03_recipe_planning/post_lowering/transformed_payload.mlir",
                    },
                    "normalized": {
                        "source_payload":
                            "03_recipe_planning/differential_verification/"
                            "normalized_source_payload.mlir",
                        "transformed_payload":
                            "03_recipe_planning/differential_verification/"
                            "normalized_transformed_payload.mlir",
                        "diff":
                            "03_recipe_planning/differential_verification/"
                            "normalized_payload_diff.txt",
                        "source_sha256": src_norm_sha,
                        "transformed_sha256": tx_norm_sha,
                    },
                    "checks": [
                        {
                            "name": "compgen_metadata_only_diff",
                            "status": "pass" if not diff_lines else "fail",
                            "detail": (
                                "" if not diff_lines
                                else f"{len(diff_lines)} unified-diff lines"
                            ),
                        },
                        {
                            "name": "normalized_source_and_transformed_byte_equal",
                            "status": "pass" if not diff_lines else "fail",
                            "detail": "",
                        },
                    ],
                }
                (out_dir / "metadata_noop_equivalence.json").write_text(
                    json.dumps(metadata_noop_record, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                if diff_lines:
                    failures.append(
                        f"normalized payloads differ "
                        f"({len(diff_lines)} unified-diff lines); see "
                        f"normalized_payload_diff.txt"
                    )
                _add(
                    "normalized_payloads_equal_after_stripping_compgen_metadata",
                    not diff_lines,
                )

    # Contract validation runs whenever any contract draft exists, but is
    # the *only* required check in contract-only mode.
    if has_contract:
        contract_record, contract_failures = _check_contracts(
            run_dir, manifest, obligations,
        )
        (out_dir / "contract_reference_check.json").write_text(
            json.dumps(contract_record, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        failures.extend(contract_failures)
        _add("contract_reference_check_pass", not contract_failures)

    if mode == "contract_only_mvp":
        if transformed_payload_present:
            failures.append(
                "contract-only model has transformed_payload.mlir, which "
                "must not exist for kernel-contract recipes"
            )
            _add("no_transformed_payload_for_contract_only", False, "exists")
        else:
            _add("no_transformed_payload_for_contract_only", True)

    # Golden re-check.
    golden = _check_goldens(run_dir)
    (out_dir / "golden_reference_check.json").write_text(
        json.dumps(golden, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if golden["status"] == "fail":
        failures.append("golden_reference_check failed; see report")
        _add("golden_reference_reproducible", False)
    elif golden["status"] == "skipped":
        _add(
            "golden_reference_reproducible", True,
            f"skipped: {golden.get('skipped_reason', '')}",
        )
    else:
        _add("golden_reference_reproducible", True)

    # ------------------------------------------------------------------ #
    # semantic_obligations_status.json — view
    # ------------------------------------------------------------------ #
    pl_statuses = (
        _read_json(rp / "post_lowering" / "semantic_obligations_status.json")
        .get("statuses", [])
    )
    statuses: list[dict[str, Any]] = []
    metadata_noop_pass = (
        metadata_noop_record is not None
        and metadata_noop_record["status"] == "pass"
    )
    contract_pass = contract_record is not None and contract_record["status"] == "pass"

    for prev in pl_statuses:
        refinement = prev.get("declared_refinement", "")
        rop = prev.get("recipe_op_id", "")
        oid = prev.get("obligation", "")
        if refinement in ("bit_equality", "tolerance_eps", "placement_obligation"):
            if metadata_noop_pass:
                status = "discharged_metadata_noop"
                discharged = ["structural_check", "metadata_noop_equivalence"]
                if refinement == "bit_equality":
                    remaining = ["real_transform_differential_check"]
                elif refinement == "tolerance_eps":
                    remaining = ["real_fusion_differential_check"]
                else:
                    remaining = ["runtime_dispatch_check"]
            else:
                status = "pending_metadata_noop_equivalence"
                discharged = ["structural_check"]
                remaining = [
                    "metadata_noop_equivalence",
                    "real_transform_differential_check",
                ]
        elif refinement == "contract_obligation":
            status = "pending_kernel_contract_generation"
            discharged = ["structural_check"]
            if contract_pass:
                discharged.append("contract_structural_validation")
            remaining = ["kernel_contract_generation", "contract_differential_check"]
        else:
            status = prev.get("status", "pending_kernel_contract_generation")
            discharged = ["structural_check"]
            remaining = ["kernel_contract_generation", "contract_differential_check"]
        statuses.append(
            {
                "obligation": oid,
                "recipe_op_id": rop,
                "declared_refinement": refinement,
                "status": status,
                "discharged": discharged,
                "remaining": remaining,
            }
        )

    semantic_status_path = out_dir / "semantic_obligations_status.json"
    semantic_status_path.write_text(
        json.dumps(
            {
                "schema_version": "semantic_obligations_status_v2",
                "model_id": manifest.get("model_id", ""),
                "target_id": manifest.get("target_id", ""),
                "stage_id": "differential_verification",
                "generated_at_utc": _utcnow(),
                "statuses": statuses,
            },
            indent=2, sort_keys=True,
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------ #
    # Final invariants + report
    # ------------------------------------------------------------------ #
    payload_post_sha = sha256_tree(pl_dir)
    payload_unchanged = payload_pre_sha == payload_post_sha
    if not payload_unchanged:
        failures.append(
            f"01_payload_lowering/ tree was modified during M-09 "
            f"(pre={payload_pre_sha[:16]}..., post={payload_post_sha[:16]}...)"
        )
    _add("source_payload_unchanged", payload_unchanged)

    # Reject any false discharge claims in the status output.
    forbidden = {
        "discharged_real_transform",
        "discharged_real_transform_differential_check",
        "fully_discharged",
        "discharged_real_fusion",
    }
    bad_claim = any(s["status"] in forbidden for s in statuses)
    _add("no_real_transform_claimed", not bad_claim)

    # Every transform-like obligation must keep at least one
    # ``real_*_differential_check`` in its remaining list.
    real_remaining_ok = True
    for s in statuses:
        if s["declared_refinement"] not in ("bit_equality", "tolerance_eps"):
            continue
        if s["status"] != "discharged_metadata_noop":
            continue
        rem = s.get("remaining", [])
        if not any(r.startswith("real_") for r in rem):
            real_remaining_ok = False
            break
    _add("real_transform_obligation_still_pending", real_remaining_ok)

    overall = (
        "pass" if not failures and all(c["status"] == "pass" for c in checks) else "fail"
    )

    report: dict[str, Any] = {
        "schema_version": "differential_verification_report_v1",
        "status": overall,
        "model_id": manifest.get("model_id", ""),
        "target_id": manifest.get("target_id", ""),
        "mode": mode,
        "generated_at_utc": _utcnow(),
        "source": {
            "source_payload": (
                _read_json(rp / "post_lowering" / "applied_transform_manifest.json")
                .get("source", {}).get("payload", "")
                if (rp / "post_lowering" / "applied_transform_manifest.json").exists()
                else ""
            ),
            "transformed_payload":
                "03_recipe_planning/post_lowering/transformed_payload.mlir"
                if transformed_payload_present else None,
            "semantic_obligations":
                "03_recipe_planning/semantic_obligations.mlir",
        },
        "checks": checks,
        "semantic_status": statuses,
        "failure_reasons": failures,
        "payload_pre_sha256": payload_pre_sha,
        "payload_post_sha256": payload_post_sha,
    }
    report_path = out_dir / "differential_verification_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return DifferentialVerificationResult(
        overall=overall,
        out_dir=out_dir,
        mode=mode,
        report_path=report_path,
        semantic_status_path=semantic_status_path,
        failures=tuple(failures),
    )
