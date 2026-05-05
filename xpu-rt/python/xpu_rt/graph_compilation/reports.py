"""Aggregate reports emitted by Payload Lowering.

These are the *top-level* JSON files inside ``01_payload_lowering/``:

- ``lowering_summary.json`` — one-shot status + totals.
- ``payload_index.json`` — input-graph → ``payload.mlir`` map.
- ``opaque_calls.json`` — every opaque ``func.call`` across all modules.
- ``unsupported_ops.json`` — typed unsupported-op records (drives gap discovery).
- ``lowering_diagnostics.json`` — every diagnostic, tagged with module_id.
- ``canonical_pass_trace.json`` — deterministic pass-by-pass trace, llm_allowed=False.

Aggregates are computed as the sum of per-module results. The validator
re-checks that aggregate counts equal the sum of per-module counts —
catches stale or fabricated summaries.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from compgen.graph_compilation.artifacts import ArtifactRef
from compgen.graph_compilation.hashing import sha256_file

if TYPE_CHECKING:  # avoid circular import at runtime
    from compgen.graph_compilation.lower import ModuleLoweringResult


_LOWERING_API = "compgen.ir.payload.import_fx.FXImporter"


def emit_top_level_reports(
    *,
    run_dir: Path,
    results: list[ModuleLoweringResult],
    capture_report_path: Path,
    dynamo_summary_path: Path,
    target_id: str,
) -> list[ArtifactRef]:
    out_dir = run_dir / "01_payload_lowering"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- payload_index.json -----
    payload_index = {
        "schema_version": "payload_index_v1",
        "modules": [
            {
                "module_id": r.module_id,
                "input_kind": r.input_kind,
                "input_graph": r.input_graph_path,
                "input_graph_hash": r.input_graph_sha256,
                "payload_mlir": r.payload_mlir_path,
                "payload_mlir_sha256": r.payload_mlir_sha256,
                "lowering_report": _lowering_report_path(r),
                "status": r.status,
            }
            for r in results
        ],
    }
    payload_index_path = out_dir / "payload_index.json"
    payload_index_path.write_text(
        json.dumps(payload_index, indent=2, sort_keys=True), encoding="utf-8"
    )

    # ----- opaque_calls.json (top level) -----
    all_opaque = [o for r in results for o in r.opaque_calls]
    opaque_obj = {
        "schema_version": "opaque_calls_v1",
        "opaque_calls": all_opaque,
        "summary": _opaque_summary(all_opaque),
    }
    opaque_path = out_dir / "opaque_calls.json"
    opaque_path.write_text(json.dumps(opaque_obj, indent=2, sort_keys=True), encoding="utf-8")

    # ----- unsupported_ops.json (top level) -----
    all_unsupported = [u for r in results for u in r.unsupported_op_records]
    # Renumber unsupported_ids globally so they are unique.
    for new_idx, u in enumerate(all_unsupported):
        u["unsupported_id"] = f"unsupported_{new_idx:04d}"
    unsupported_obj = {
        "schema_version": "unsupported_ops_v1",
        "unsupported_ops": all_unsupported,
        "summary": _unsupported_summary(all_unsupported),
    }
    unsupported_path = out_dir / "unsupported_ops.json"
    unsupported_path.write_text(
        json.dumps(unsupported_obj, indent=2, sort_keys=True), encoding="utf-8"
    )

    # ----- lowering_diagnostics.json -----
    all_diags = [d for r in results for d in r.diagnostics]
    diag_obj = {
        "schema_version": "payload_lowering_diagnostics_v1",
        "diagnostics": all_diags,
        "summary": {
            "info": sum(1 for d in all_diags if d.get("level") == "info"),
            "warning": sum(1 for d in all_diags if d.get("level") == "warning"),
            "error": sum(1 for d in all_diags if d.get("level") == "error"),
        },
    }
    diag_path = out_dir / "lowering_diagnostics.json"
    diag_path.write_text(json.dumps(diag_obj, indent=2, sort_keys=True), encoding="utf-8")

    # ----- canonical_pass_trace.json -----
    passes: list[dict[str, Any]] = []
    for r in results:
        if r.input_kind == "exported_program" and r.status == "skipped":
            passes.append(
                {
                    "name": f"load_{r.module_id}",
                    "implementation": "torch.export.load",
                    "status": "skipped",
                    "input_hash": "",
                    "output_hash": "",
                    "changed_ir": False,
                }
            )
            continue
        passes.append(
            {
                "name": f"load_{r.module_id}",
                "implementation": (
                    "torch.load" if r.input_kind == "torch_dynamo_partition" else "torch.export.load"
                ),
                "status": "pass" if r.input_graph_sha256 else "skipped",
                "input_hash": r.input_graph_sha256,
                "output_hash": r.input_graph_sha256,
                "changed_ir": False,
            }
        )
        passes.append(
            {
                "name": f"fx_importer_import_graph_{r.module_id}",
                "implementation": "compgen.ir.payload.import_fx.FXImporter.import_graph",
                "status": r.status,
                "input_hash": r.input_graph_sha256,
                "output_hash": r.payload_mlir_sha256,
                "changed_ir": True,
            }
        )
        passes.append(
            {
                "name": f"xdsl_module_verify_{r.module_id}",
                "implementation": "xdsl.ir.Operation.verify",
                "status": r.module_verify_status,
                "input_hash": r.payload_mlir_sha256,
                "output_hash": r.payload_mlir_sha256,
                "changed_ir": False,
            }
        )
    trace_obj = {
        "schema_version": "canonical_pass_trace_v1",
        "stage_id": "payload_lowering",
        "llm_allowed": False,
        "passes": passes,
    }
    trace_path = out_dir / "canonical_pass_trace.json"
    trace_path.write_text(json.dumps(trace_obj, indent=2, sort_keys=True), encoding="utf-8")

    # ----- lowering_summary.json -----
    dynamo_results = [r for r in results if r.input_kind == "torch_dynamo_partition"]
    export_results = [r for r in results if r.input_kind == "exported_program"]
    non_skipped = [r for r in results if r.status != "skipped"]
    if not non_skipped:
        overall = "fail"
    elif all(r.status == "pass" for r in non_skipped):
        overall = "pass"
    elif any(r.status == "pass" for r in non_skipped):
        overall = "partial_success"
    else:
        overall = "fail"

    summary_obj = {
        "schema_version": "payload_lowering_summary_v1",
        "stage_id": "payload_lowering",
        "status": overall,
        "primary_capture": "torch_dynamo",
        "target_id": target_id,
        "inputs": {
            "capture_report_sha256": "sha256:" + sha256_file(capture_report_path),
            "dynamo_summary_sha256": "sha256:" + sha256_file(dynamo_summary_path),
        },
        "dynamo": {
            "attempted": True,
            "input_partition_count": len(dynamo_results),
            "lowered_partition_count": sum(1 for r in dynamo_results if r.status == "pass"),
            "failed_partition_count": sum(1 for r in dynamo_results if r.status == "fail"),
        },
        "torch_export": {
            "attempted": bool(export_results),
            "available": any(r.status != "skipped" for r in export_results),
            "status": _export_status(export_results),
            "lowered": any(r.status == "pass" for r in export_results),
        },
        "totals": {
            "payload_modules_total": sum(1 for r in non_skipped),
            "fx_nodes_total": sum(r.num_fx_nodes for r in non_skipped),
            "call_function_nodes_total": sum(r.num_call_function for r in non_skipped),
            "payload_ops_total": sum(r.payload_ops_total for r in non_skipped),
            "decomposed_ops_total": sum(r.decomposed_ops for r in non_skipped),
            "opaque_ops_total": sum(r.opaque_ops for r in non_skipped),
            "unsupported_ops_total": sum(r.unsupported_ops for r in non_skipped),
            "decomposition_coverage": _aggregate_coverage(non_skipped),
        },
        "outputs": {
            "payload_index": "01_payload_lowering/payload_index.json",
            "opaque_calls": "01_payload_lowering/opaque_calls.json",
            "unsupported_ops": "01_payload_lowering/unsupported_ops.json",
        },
        "llm_calls": 0,
    }
    summary_path = out_dir / "lowering_summary.json"
    summary_path.write_text(json.dumps(summary_obj, indent=2, sort_keys=True), encoding="utf-8")

    # Return ArtifactRefs for everything we wrote.
    refs: list[ArtifactRef] = []
    for p in (
        summary_path,
        payload_index_path,
        opaque_path,
        unsupported_path,
        diag_path,
        trace_path,
    ):
        refs.append(
            ArtifactRef(
                path=p.relative_to(run_dir).as_posix(),
                sha256=sha256_file(p),
                size_bytes=p.stat().st_size,
                kind="file",
            )
        )
    return refs


def _lowering_report_path(r: ModuleLoweringResult) -> str:
    """Where the per-module ``lowering_report.json`` lives, relative to run_dir."""
    if r.input_kind == "torch_dynamo_partition":
        return f"01_payload_lowering/dynamo_partitions/{r.module_id.removeprefix('dynamo_')}/lowering_report.json"
    if r.input_kind == "exported_program":
        return "01_payload_lowering/export_program/lowering_report.json"
    return ""


def _opaque_summary(opaque_calls: list[dict[str, Any]]) -> dict[str, Any]:
    by_target: dict[str, int] = {}
    for o in opaque_calls:
        t = o.get("fx_target", "")
        by_target[t] = by_target.get(t, 0) + 1
    return {"count": len(opaque_calls), "by_target": dict(sorted(by_target.items()))}


def _unsupported_summary(unsupported: list[dict[str, Any]]) -> dict[str, Any]:
    by_target: dict[str, int] = {}
    by_reason: dict[str, int] = {}
    for u in unsupported:
        t = u.get("fx_target", "")
        r = u.get("reason", "")
        by_target[t] = by_target.get(t, 0) + 1
        by_reason[r] = by_reason.get(r, 0) + 1
    return {
        "count": len(unsupported),
        "by_target": dict(sorted(by_target.items())),
        "by_reason": dict(sorted(by_reason.items())),
    }


def _export_status(export_results: list[ModuleLoweringResult]) -> str:
    if not export_results:
        return "skipped"
    r = export_results[0]
    if r.status == "skipped":
        return "skipped"
    return r.status


def _aggregate_coverage(results: list[ModuleLoweringResult]) -> float:
    total_decomposed = sum(r.decomposed_ops for r in results)
    total_opaque = sum(r.opaque_ops for r in results)
    if total_decomposed + total_opaque == 0:
        return 1.0
    return total_decomposed / (total_decomposed + total_opaque)
