#!/usr/bin/env python3
"""Wide Model Coverage Gate (Milestone 08.5).

Formalizes the wide-suite run into reproducible coverage artifacts.
Read-only against compiler core; this script only inspects an
already-finished ``run-suite`` output and the on-disk merlin source
inventory. It emits:

- ``model_inventory.json``                — every directory under
  ``--merlin-root``, classified by source/framework/admission_status.
- ``wide_suite_report.json``              — top-level pass/fail summary
  with explicit smoke / proxy / merlin separation.
- ``wide_suite_coverage_matrix.json``     — per-model row covering every
  pipeline stage with status + counts.
- ``wide_suite_failures.json``            — concrete failure reasons per
  model (empty list when the suite is clean).
- ``wide_suite_summary.md``               — reviewer-facing one-pager.
- ``audit_figures/`` (refreshed)          — payload coverage, candidate
  legality heatmap, region roofline scatter, refinement histogram.

Usage::

    python scripts/dev/build_wide_coverage_gate.py \\
        --suite-results results/graph_compilation/wide_post_lowering_suite \\
        --suite-yaml    configs/graph_compilation/wide_test_models.yaml \\
        --merlin-root   /scratch2/agustin/merlin/models
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


# --------------------------------------------------------------------------- #
# Merlin inventory
# --------------------------------------------------------------------------- #


_NN_MODULE_RE = re.compile(r"class\s+\w+\s*\([^)]*nn\.Module")


def _has_nn_module_definition(py_files: list[Path]) -> bool:
    for f in py_files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if _NN_MODULE_RE.search(text):
            return True
    return False


def _has_external_submodule(dir_path: Path) -> tuple[bool, str | None]:
    """A merlin model dir often contains a sibling submodule whose name is
    a CamelCase / hyphenated variant of the dir name. Detect that."""
    name_lower = dir_path.name.lower().replace("_", "")
    for child in dir_path.iterdir():
        if not child.is_dir():
            continue
        cn = child.name.lower().replace("_", "").replace("-", "")
        if cn == name_lower or cn.startswith(name_lower):
            # Check if it actually has source — empty submodule dir is
            # usually a sign of an unfetched git submodule.
            has_content = any(child.rglob("*.py"))
            return True, ("populated" if has_content else "unfetched")
    return False, None


def _config_for_merlin_dir(repo_root: Path, dir_name: str) -> Path | None:
    cfg = repo_root / "configs" / "models" / f"merlin_{dir_name}.yaml"
    if cfg.exists():
        return cfg
    return None


def inventory_merlin(merlin_root: Path, repo_root: Path) -> list[dict[str, Any]]:
    """Walk every direct subdirectory of ``merlin_root`` and classify it.

    Categories used:

    - ``admitted_pytorch`` — has ``nn.Module`` and a wired CompGen YAML
      under ``configs/models/merlin_<name>.yaml``.
    - ``needs_custom_loader`` — has ``nn.Module`` but no CompGen YAML
      yet (either user hasn't authored a wrapper or the model needs a
      bespoke factory).
    - ``external_dependency_missing`` — has a sibling submodule
      directory that's empty or unfetched (e.g. ``MiDaS/``,
      ``Depth-Anything-V2/``, ``TinyDepth/``).
    - ``onnx_only_pending_importer`` — only ``.onnx`` source available;
      our PyTorch / Dynamo / Fx pipeline cannot ingest it without an
      ONNX importer.
    - ``mlir_only`` — only ``.mlir`` artifacts (already lowered through
      another framework); not a candidate for this pipeline.
    - ``utility_or_subdir`` — directory has no model sources (e.g.
      ``compiled_models``, ``research``).
    """
    out: list[dict[str, Any]] = []
    for child in sorted(merlin_root.iterdir()):
        if not child.is_dir():
            continue
        py_files = sorted(child.glob("*.py"))
        onnx_files = sorted(child.glob("*.onnx"))
        mlir_files = sorted(child.glob("*.mlir"))
        has_nn = _has_nn_module_definition(py_files)
        has_subm, subm_state = _has_external_submodule(child)

        cfg = _config_for_merlin_dir(repo_root, child.name)

        if has_nn and cfg is not None:
            admission = "admitted_pytorch"
            reason = "has importable torch.nn.Module factory wired via CompGen YAML"
            framework = "pytorch"
        elif has_nn and has_subm and subm_state == "unfetched":
            admission = "external_dependency_missing"
            reason = (
                f"requires populated submodule under {child.name}/ "
                f"(currently {subm_state})"
            )
            framework = "pytorch"
        elif has_nn and has_subm:
            admission = "external_dependency_missing"
            reason = (
                f"depends on sibling source tree {subm_state} that is not on "
                f"the Python path; needs a custom loader / explicit submodule "
                f"setup"
            )
            framework = "pytorch"
        elif has_nn and not cfg:
            admission = "needs_custom_loader"
            reason = (
                "has importable torch.nn.Module but no CompGen YAML wrapper "
                "yet (admit by adding configs/models/merlin_<dir>.yaml)"
            )
            framework = "pytorch"
        elif onnx_files and not has_nn:
            admission = "onnx_only_pending_importer"
            reason = (
                "only ONNX artifacts available; current graph_compilation "
                "path requires PyTorch module or an ONNX importer"
            )
            framework = "onnx"
        elif mlir_files and not has_nn and not onnx_files:
            admission = "mlir_only"
            reason = (
                "only pre-lowered .mlir artifacts; not a source-level "
                "candidate for this pipeline"
            )
            framework = "mlir"
        else:
            admission = "utility_or_subdir"
            reason = "no model source detected (likely build artifacts / docs)"
            framework = "unknown"

        out.append(
            {
                "model_id": f"merlin_{child.name}",
                "source": "merlin",
                "path": str(child),
                "framework": framework,
                "admission_status": admission,
                "reason": reason,
                "config": str(cfg.relative_to(repo_root)) if cfg else None,
                "evidence": {
                    "py_files": [p.name for p in py_files],
                    "onnx_files": [p.name for p in onnx_files],
                    "mlir_files": [p.name for p in mlir_files],
                    "has_nn_module": has_nn,
                    "has_sibling_submodule": has_subm,
                    "submodule_state": subm_state,
                },
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Suite-side inventory: smoke / proxy / merlin
# --------------------------------------------------------------------------- #


_SMOKE_MODELS = frozenset({
    "tiny_mlp", "tiny_attention", "tiny_conv_block",
    "proxy_vlm", "proxy_vla", "custom_unsupported_op",
})


def _classify_compgen_model(model_id: str) -> str:
    if model_id.startswith("merlin_"):
        return "merlin"
    if model_id in _SMOKE_MODELS:
        return "smoke"
    if model_id.startswith("proxy_"):
        return "proxy"
    return "compgen_synthetic"


def inventory_compgen(suite_yaml: Path) -> list[dict[str, Any]]:
    raw = yaml.safe_load(Path(suite_yaml).read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    for entry in raw.get("models", []):
        mid = entry["id"]
        out.append(
            {
                "model_id": mid,
                "source": _classify_compgen_model(mid),
                "config": entry["config"],
                "role": entry.get("role", ""),
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Per-run artifact extraction (no compiler invocations)
# --------------------------------------------------------------------------- #


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        obj: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return obj
    except (OSError, json.JSONDecodeError):
        return None


def _stage_status(run_dir: Path, stage_id: str) -> str:
    """Look up a stage's status in the run manifest."""
    manifest = _read_json(run_dir / "run_manifest.json")
    if manifest is None:
        return "missing_manifest"
    for s in manifest.get("stages", []):
        if s.get("stage_id") == stage_id:
            status: str = s.get("status", "unknown")
            return status
    return "absent"


def per_model_row(run_dir: Path, model_id: str, source: str) -> dict[str, Any]:
    rp = run_dir / "03_recipe_planning"
    pl = rp / "post_lowering"
    pl_dir = run_dir / "01_payload_lowering"

    capture = _stage_status(run_dir, "graph_capture")
    payload = _stage_status(run_dir, "payload_lowering")
    graph_analysis = _stage_status(run_dir, "graph_analysis")
    recipe_planning = _stage_status(run_dir, "recipe_planning")

    accounting = _read_json(pl_dir / "fx_to_payload_accounting.json")
    summary = (accounting or {}).get("summary", {})

    region_map = _read_json(run_dir / "02_graph_analysis" / "region_map.json")
    regions = (region_map or {}).get("totals", {}).get("regions", 0)

    cas = _read_json(run_dir / "02_graph_analysis" / "candidate_actions.json")
    families: set[str] = set()
    if cas:
        for c in cas.get("candidates", []):
            families.add(c["kind"])

    sel = _read_json(rp / "candidate_selection.json")
    selected_kind = (sel or {}).get("candidate_kind")

    gate = _read_json(rp / "recipe_gate_verdict.json")
    gate_status = (gate or {}).get("status")
    refinement = ""
    if gate and gate.get("checked_recipe_ops"):
        refinement = gate["checked_recipe_ops"][0].get("declared_refinement", "")

    artifact_manifest = _read_json(rp / "lowering_artifact_manifest.json")
    artifact_kinds: list[str] = []
    if artifact_manifest:
        for a in artifact_manifest.get("artifacts", []):
            artifact_kinds.append(a["artifact_kind"])

    pl_report = _read_json(pl / "post_lowering_verification_report.json")
    post_lowering_status = (pl_report or {}).get("status", "absent")

    has_transformed = (pl / "transformed_payload.mlir").exists()
    has_contract_validation = (pl / "contract_structural_validation.json").exists()

    artifact_validation = _read_json(run_dir / "validation" / "artifact_validation.json")
    artifact_validator_overall = (artifact_validation or {}).get("overall", "missing")

    # Two honest measures:
    # - overall_strict: every stage including payload_lowering's strict gate
    #   reports "pass". A model can fall short here when its export_program
    #   path doesn't fully lower while the dynamo path or partial artifacts
    #   were enough for downstream stages to keep going.
    # - overall_pipeline_completed: capture + structural downstream stages
    #   all produced valid artifacts and the post-lowering verification
    #   itself passed. This reflects "the pipeline reached the end without
    #   aborting" without hiding strict sub-gate failures.
    overall_strict = "pass" if (
        capture == "pass"
        and payload == "pass"
        and graph_analysis == "pass"
        and recipe_planning == "pass"
        and gate_status == "pass"
        and post_lowering_status == "pass"
    ) else "fail"
    overall_pipeline_completed = "pass" if (
        capture == "pass"
        and graph_analysis == "pass"
        and recipe_planning == "pass"
        and gate_status == "pass"
        and post_lowering_status == "pass"
    ) else "fail"

    return {
        "model_id": model_id,
        "source": source,
        "run_dir": str(run_dir),
        "capture": capture,
        "payload_lowering": payload,
        "graph_analysis": graph_analysis,
        "recipe_planning": recipe_planning,
        "recipe_verification_gate": gate_status,
        "recipe_lowering": "pass" if artifact_manifest else "absent",
        "post_lowering_verification": post_lowering_status,
        "artifact_validator": artifact_validator_overall,
        "fx_call_function_nodes": summary.get("call_function_nodes", 0),
        "decomposed_structured": summary.get("decomposed_structured", 0),
        "opaque_fallback": summary.get("opaque_fallback", 0),
        "regions": regions,
        "candidate_families": sorted(families),
        "selected_candidate_kind": selected_kind,
        "declared_refinement": refinement,
        "lowering_artifact_kinds": sorted(set(artifact_kinds)),
        "has_transformed_payload": has_transformed,
        "has_contract_structural_validation": has_contract_validation,
        "overall_strict": overall_strict,
        "overall_pipeline_completed": overall_pipeline_completed,
        "overall": overall_pipeline_completed,
    }


# --------------------------------------------------------------------------- #
# Failure extraction
# --------------------------------------------------------------------------- #


def collect_failures(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Two failure lists:

    - ``pipeline_completion_failures`` — models that did NOT reach a passing
      ``post_lowering_verification`` (the strongest end-to-end gate this
      milestone applies). These are real, blocking failures.
    - ``strict_gate_warnings`` — models that completed the pipeline AND
      passed post-lowering verification, but had at least one stage
      report ``status: fail`` along the way (typically
      ``payload_lowering`` because the strict export_program path didn't
      fully lower while a usable partial artifact was emitted). These do
      not block; they are surfaced explicitly so they cannot be quietly
      conflated with full passes.
    """
    pipe_fail: list[dict[str, Any]] = []
    strict_warn: list[dict[str, Any]] = []
    stage_order = (
        "capture",
        "payload_lowering",
        "graph_analysis",
        "recipe_planning",
        "recipe_verification_gate",
        "recipe_lowering",
        "post_lowering_verification",
    )
    for r in rows:
        if r["overall_pipeline_completed"] != "pass":
            for stage_field in stage_order:
                if r[stage_field] != "pass":
                    pipe_fail.append(
                        {
                            "model_id": r["model_id"],
                            "source": r["source"],
                            "first_failed_stage": stage_field,
                            "stage_status": r[stage_field],
                            "run_dir": r["run_dir"],
                        }
                    )
                    break
            else:
                pipe_fail.append(
                    {
                        "model_id": r["model_id"],
                        "source": r["source"],
                        "first_failed_stage": "unknown",
                        "stage_status": "unknown",
                        "run_dir": r["run_dir"],
                    }
                )
        elif r["overall_strict"] != "pass":
            failing_stages = [s for s in stage_order if r[s] not in {"pass", "absent"}]
            strict_warn.append(
                {
                    "model_id": r["model_id"],
                    "source": r["source"],
                    "failing_stages": failing_stages,
                    "stage_statuses": {s: r[s] for s in failing_stages},
                    "note": (
                        "pipeline reached post-lowering verification with "
                        "status=pass, but a strict sub-gate reported "
                        "fail (typically payload_lowering: export_program "
                        "lowered=false while dynamo / partial artifacts "
                        "carried the run forward)."
                    ),
                    "run_dir": r["run_dir"],
                }
            )
    return {
        "pipeline_completion_failures": pipe_fail,
        "strict_gate_warnings": strict_warn,
    }


# --------------------------------------------------------------------------- #
# Markdown summary
# --------------------------------------------------------------------------- #


def _git_head(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root),
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "(unknown)"


def write_summary_md(
    *,
    out_dir: Path,
    repo_root: Path,
    inventory: list[dict[str, Any]],
    suite_inventory: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    failures: dict[str, list[dict[str, Any]]],
) -> Path:
    now = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    head = _git_head(repo_root)

    by_source: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_source.setdefault(r["source"], []).append(r)

    by_admission: dict[str, list[dict[str, Any]]] = {}
    for m in inventory:
        by_admission.setdefault(m["admission_status"], []).append(m)

    refinements: dict[str, int] = {}
    selected_kinds: dict[str, int] = {}
    for r in rows:
        if r["declared_refinement"]:
            refinements[r["declared_refinement"]] = refinements.get(
                r["declared_refinement"], 0
            ) + 1
        if r["selected_candidate_kind"]:
            selected_kinds[r["selected_candidate_kind"]] = selected_kinds.get(
                r["selected_candidate_kind"], 0
            ) + 1

    lines: list[str] = []
    lines.append("# Wide Model Coverage Gate (M-08.5)\n")
    lines.append(f"_Generated_: {now}  ")
    lines.append(f"_git HEAD_: `{head}`\n")

    lines.append("## 1. Pipeline coverage\n")
    lines.append(f"- **Models admitted**: {len(rows)}  ")
    n_pipe_pass = sum(1 for r in rows if r["overall_pipeline_completed"] == "pass")
    n_strict_pass = sum(1 for r in rows if r["overall_strict"] == "pass")
    lines.append(
        f"- **Pipeline-completion (capture + downstream + post-lowering "
        f"verification)**: {n_pipe_pass} / {len(rows)} pass  "
    )
    lines.append(
        f"- **Strict gate (every stage status=pass)**: "
        f"{n_strict_pass} / {len(rows)} pass\n"
    )
    lines.append("Per-source breakdown (admitted only):\n")
    lines.append("| source | n | pipeline-pass | strict-pass |")
    lines.append("|---|---:|---:|---:|")
    for s in sorted(by_source):
        sl = by_source[s]
        npipe = sum(1 for r in sl if r["overall_pipeline_completed"] == "pass")
        nstrict = sum(1 for r in sl if r["overall_strict"] == "pass")
        lines.append(f"| {s} | {len(sl)} | {npipe} | {nstrict} |")
    lines.append("")

    lines.append("## 2. Merlin inventory (every directory under `--merlin-root`)\n")
    lines.append("| admission_status | count | examples |")
    lines.append("|---|---:|---|")
    for status in sorted(by_admission):
        examples = ", ".join(m["model_id"] for m in by_admission[status][:5])
        lines.append(f"| {status} | {len(by_admission[status])} | {examples} |")
    lines.append("")
    lines.append("Concrete reasons for non-admission:\n")
    for status in sorted(by_admission):
        if status == "admitted_pytorch":
            continue
        for m in by_admission[status]:
            lines.append(
                f"- `{m['model_id']}` _(merlin)_ — **{m['admission_status']}**: "
                f"{m['reason']}"
            )
    lines.append("")

    lines.append("## 3. Diversity\n")
    lines.append(
        f"- Selected candidate kinds: " + ", ".join(
            f"`{k}` × {v}" for k, v in sorted(selected_kinds.items(), key=lambda kv: -kv[1])
        )
    )
    lines.append(
        f"- Declared refinement types: " + ", ".join(
            f"`{k}` × {v}" for k, v in sorted(refinements.items(), key=lambda kv: -kv[1])
        ) + "\n"
    )

    lines.append("## 4. Per-model coverage matrix\n")
    lines.append(
        "| model | source | pipeline | strict | selected | refinement | "
        "regions | call_fn | dec / opaq |"
    )
    lines.append("|---|---|---|---|---|---|---:|---:|---|")
    for r in rows:
        lines.append(
            f"| {r['model_id']} | {r['source']} | "
            f"{r['overall_pipeline_completed']} | {r['overall_strict']} | "
            f"{r['selected_candidate_kind'] or '—'} | "
            f"{r['declared_refinement'] or '—'} | {r['regions']} | "
            f"{r['fx_call_function_nodes']} | "
            f"{r['decomposed_structured']} / {r['opaque_fallback']} |"
        )
    lines.append("")

    pipe_fail = failures.get("pipeline_completion_failures", [])
    strict_warn = failures.get("strict_gate_warnings", [])
    if pipe_fail:
        lines.append("## 5a. Pipeline-completion failures\n")
        lines.append("| model | source | first_failed_stage | status |")
        lines.append("|---|---|---|---|")
        for f in pipe_fail:
            lines.append(
                f"| {f['model_id']} | {f['source']} | "
                f"{f['first_failed_stage']} | {f['stage_status']} |"
            )
        lines.append("")
    if strict_warn:
        lines.append(
            "## 5b. Strict-gate warnings (pipeline completed but a "
            "sub-gate reported fail)\n"
        )
        lines.append("| model | source | failing_stages |")
        lines.append("|---|---|---|")
        for w in strict_warn:
            lines.append(
                f"| {w['model_id']} | {w['source']} | "
                f"{', '.join(w['failing_stages'])} |"
            )
        lines.append("")
        lines.append(
            f"_Note_: {strict_warn[0]['note']}  These models are NOT "
            f"silently treated as full passes; they are recorded in "
            f"`wide_suite_failures.json::strict_gate_warnings`.\n"
        )

    lines.append("## 6. What this proves vs. what it does NOT\n")
    n_merlin_total = sum(1 for r in rows if r["source"] == "merlin")
    n_merlin_pipe = sum(
        1 for r in rows
        if r["source"] == "merlin" and r["overall_pipeline_completed"] == "pass"
    )
    n_merlin_strict = sum(
        1 for r in rows
        if r["source"] == "merlin" and r["overall_strict"] == "pass"
    )
    lines.append(
        "**Proves**: CompGen's current front-end loop captures, lowers, "
        "attributes, analyzes, constructs an action space, selects a candidate, "
        "verifies the recipe pre-lowering, emits lowering artifacts, and applies "
        "metadata-only structural transforms on a copy of Payload IR — across a "
        "wider set of PyTorch model families than the original 6-model smoke "
        f"suite. Real upstream merlin models in the suite: {n_merlin_total}; "
        f"pipeline-completed: {n_merlin_pipe}; strict-gate clean: "
        f"{n_merlin_strict}.\n"
    )
    lines.append(
        "**Does NOT prove**:\n"
        "- Real loop tiling performance.\n"
        "- Real fusion correctness.\n"
        "- Differential correctness after semantic transforms (M-09).\n"
        "- ONNX-only merlin models — they are intentionally classified "
        "`onnx_only_pending_importer` and excluded from the admitted set.\n"
        "- Frontier models (gemma2, llama32, qwen3-vl-8b, …) that need "
        "HF weight downloads — they are intentionally excluded from this run.\n"
        "- Calibrated target cost model.\n"
        "- Real LLM in the selection loop (greedy/llm-stub only).\n"
    )

    out = out_dir / "wide_suite_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


# --------------------------------------------------------------------------- #
# Audit-figure regeneration (delegates to the existing script)
# --------------------------------------------------------------------------- #


def render_audit_figures(suite_root: Path, out_dir: Path) -> None:
    import importlib.util

    script = (
        Path(__file__).resolve().parent / "render_graph_compilation_audit.py"
    )
    spec = importlib.util.spec_from_file_location("render_audit_module", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.render_audit(suite_root, out_dir)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def build_wide_coverage_gate(
    *,
    suite_results: Path,
    suite_yaml: Path,
    merlin_root: Path,
    repo_root: Path,
) -> dict[str, Path]:
    suite_results = Path(suite_results).resolve()
    if not suite_results.is_dir():
        raise FileNotFoundError(f"--suite-results does not exist: {suite_results}")

    merlin_inv = inventory_merlin(Path(merlin_root), repo_root)
    suite_inv = inventory_compgen(Path(suite_yaml))

    rows: list[dict[str, Any]] = []
    for entry in suite_inv:
        run_dir = suite_results / entry["model_id"]
        if not run_dir.is_dir():
            rows.append(
                {
                    "model_id": entry["model_id"],
                    "source": entry["source"],
                    "run_dir": str(run_dir),
                    "overall_pipeline_completed": "fail",
                    "overall_strict": "fail",
                    "overall": "fail",
                    "capture": "missing_run_dir",
                    "payload_lowering": "missing_run_dir",
                    "graph_analysis": "missing_run_dir",
                    "recipe_planning": "missing_run_dir",
                    "recipe_verification_gate": "missing_run_dir",
                    "recipe_lowering": "missing_run_dir",
                    "post_lowering_verification": "missing_run_dir",
                    "artifact_validator": "missing_run_dir",
                    "fx_call_function_nodes": 0,
                    "decomposed_structured": 0,
                    "opaque_fallback": 0,
                    "regions": 0,
                    "candidate_families": [],
                    "selected_candidate_kind": None,
                    "declared_refinement": "",
                    "lowering_artifact_kinds": [],
                    "has_transformed_payload": False,
                    "has_contract_structural_validation": False,
                }
            )
            continue
        rows.append(per_model_row(run_dir, entry["model_id"], entry["source"]))

    failures = collect_failures(rows)

    # ------------------------------------------------------------------ #
    # Emit artifacts.
    # ------------------------------------------------------------------ #
    out_dir = suite_results
    inv_path = out_dir / "model_inventory.json"
    inv_path.write_text(
        json.dumps(
            {
                "schema_version": "model_inventory_v1",
                "root_scanned": str(merlin_root),
                "generated_at_utc": datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "models": merlin_inv + [
                    {
                        "model_id": e["model_id"],
                        "source": e["source"],
                        "framework": "pytorch",
                        "admission_status": "admitted_pytorch",
                        "reason": "in suite YAML; pipeline target",
                        "config": e["config"],
                    }
                    for e in suite_inv
                    if not e["model_id"].startswith("merlin_")
                ],
            },
            indent=2, sort_keys=True,
        ),
        encoding="utf-8",
    )

    suite_report_path = out_dir / "wide_suite_report.json"

    def _pass_count(source: str, key: str) -> int:
        return sum(1 for r in rows if r["source"] == source and r[key] == "pass")

    other_pipe = sum(
        1 for r in rows
        if r["source"] not in {"smoke", "proxy", "merlin"}
        and r["overall_pipeline_completed"] == "pass"
    )
    other_strict = sum(
        1 for r in rows
        if r["source"] not in {"smoke", "proxy", "merlin"}
        and r["overall_strict"] == "pass"
    )
    suite_report_path.write_text(
        json.dumps(
            {
                "schema_version": "wide_suite_report_v2",
                "suite": str(suite_yaml),
                "stop_after": "post-lowering-verification",
                "selection_mode": "greedy",
                "generated_at_utc": datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "summary": {
                    "models_total_in_inventory": len(merlin_inv),
                    "models_admitted_in_suite": len(suite_inv),
                    "pipeline_completed": {
                        "passed": sum(
                            1 for r in rows
                            if r["overall_pipeline_completed"] == "pass"
                        ),
                        "failed": sum(
                            1 for r in rows
                            if r["overall_pipeline_completed"] != "pass"
                        ),
                        "smoke_passed": _pass_count("smoke", "overall_pipeline_completed"),
                        "proxy_passed": _pass_count("proxy", "overall_pipeline_completed"),
                        "real_merlin_pytorch_passed": _pass_count(
                            "merlin", "overall_pipeline_completed"
                        ),
                        "compgen_synthetic_passed": other_pipe,
                    },
                    "strict_gate": {
                        "passed": sum(1 for r in rows if r["overall_strict"] == "pass"),
                        "failed": sum(1 for r in rows if r["overall_strict"] != "pass"),
                        "smoke_passed": _pass_count("smoke", "overall_strict"),
                        "proxy_passed": _pass_count("proxy", "overall_strict"),
                        "real_merlin_pytorch_passed": _pass_count(
                            "merlin", "overall_strict"
                        ),
                        "compgen_synthetic_passed": other_strict,
                    },
                },
                "results": rows,
            },
            indent=2, sort_keys=True,
        ),
        encoding="utf-8",
    )

    matrix_path = out_dir / "wide_suite_coverage_matrix.json"
    matrix_path.write_text(
        json.dumps(
            {
                "schema_version": "wide_suite_coverage_matrix_v1",
                "rows": rows,
                "totals": {
                    "regions": sum(r["regions"] for r in rows),
                    "fx_call_function_nodes": sum(
                        r["fx_call_function_nodes"] for r in rows
                    ),
                    "transformed_payload_models": sum(
                        1 for r in rows if r["has_transformed_payload"]
                    ),
                    "contract_only_models": sum(
                        1 for r in rows if r["has_contract_structural_validation"]
                    ),
                    "distinct_selected_candidate_kinds": sorted(
                        {r["selected_candidate_kind"] for r in rows if r["selected_candidate_kind"]}
                    ),
                    "distinct_refinement_types": sorted(
                        {r["declared_refinement"] for r in rows if r["declared_refinement"]}
                    ),
                },
            },
            indent=2, sort_keys=True,
        ),
        encoding="utf-8",
    )

    failures_path = out_dir / "wide_suite_failures.json"
    failures_path.write_text(
        json.dumps(
            {
                "schema_version": "wide_suite_failures_v2",
                "pipeline_completion_failure_count": len(
                    failures["pipeline_completion_failures"]
                ),
                "strict_gate_warning_count": len(failures["strict_gate_warnings"]),
                "pipeline_completion_failures": failures["pipeline_completion_failures"],
                "strict_gate_warnings": failures["strict_gate_warnings"],
            },
            indent=2, sort_keys=True,
        ),
        encoding="utf-8",
    )

    summary_path = write_summary_md(
        out_dir=out_dir,
        repo_root=repo_root,
        inventory=merlin_inv,
        suite_inventory=suite_inv,
        rows=rows,
        failures=failures,
    )

    audit_dir = out_dir / "audit_figures"
    render_audit_figures(out_dir, audit_dir)

    return {
        "model_inventory": inv_path,
        "wide_suite_report": suite_report_path,
        "wide_suite_coverage_matrix": matrix_path,
        "wide_suite_failures": failures_path,
        "wide_suite_summary": summary_path,
        "audit_figures_dir": audit_dir,
    }


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite-results", required=True, type=Path,
        help="Path to a completed run-suite output (e.g. results/.../wide_post_lowering_suite).",
    )
    parser.add_argument(
        "--suite-yaml", required=True, type=Path,
        help="Path to the suite YAML used to drive the run.",
    )
    parser.add_argument(
        "--merlin-root", required=True, type=Path,
        help="Root of the merlin model inventory (e.g. /scratch2/agustin/merlin/models).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    artifacts = build_wide_coverage_gate(
        suite_results=args.suite_results,
        suite_yaml=args.suite_yaml,
        merlin_root=args.merlin_root,
        repo_root=repo_root,
    )
    print("wrote:")
    for name, path in artifacts.items():
        print(f"  {name:32s} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
