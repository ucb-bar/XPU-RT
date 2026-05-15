"""Payload Lowering: thin wrapper around ``FXImporter``.

Consumes ``00_graph_capture/`` artifacts from a previous Graph Capture
run and lowers each Dynamo partition (and the optional
``exported_program.pt2``) into Payload/xDSL/MLIR via the existing
:class:`compgen.ir.payload.import_fx.FXImporter`.

This module **does not** rewrite the importer or invent new lowering
rules. When an op has no decomposition entry, ``FXImporter`` emits an
opaque ``func.call`` and records the diagnostic — and we surface that
in ``opaque_calls.json`` / ``unsupported_ops.json`` for the downstream
gap-discovery pass.

No LLM calls. The ``canonical_pass_trace.json`` carries
``llm_allowed: false``.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from compgen.graph_compilation.artifacts import ArtifactRef, StageRecord
from compgen.graph_compilation.hashing import sha256_file, sha256_tree
from compgen.ir.payload.import_fx import FXImporter, ImportDiagnostic

_LOWERING_API = "compgen.ir.payload.import_fx.FXImporter"


# --------------------------------------------------------------------------- #
# Per-module result
# --------------------------------------------------------------------------- #


@dataclass
class ModuleLoweringResult:
    """One lowered graph (Dynamo partition or exported program)."""

    module_id: str
    input_kind: str  # "torch_dynamo_partition" | "exported_program"
    status: str  # "pass" | "fail" | "skipped"
    input_graph_path: str  # relative to run_dir
    input_graph_sha256: str
    payload_mlir_path: str  # relative to run_dir
    payload_mlir_sha256: str
    payload_ops_total: int
    num_fx_nodes: int
    num_call_function: int
    decomposed_ops: int
    opaque_ops: int
    unsupported_ops: int
    decomposition_coverage: float
    module_verify_status: str  # "pass" | "fail" | "skipped"
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    opaque_calls: list[dict[str, Any]] = field(default_factory=list)
    unsupported_op_records: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    skip_reason: str | None = None
    artifact_refs: list[ArtifactRef] = field(default_factory=list)

    def to_report(self) -> dict[str, Any]:
        return {
            "schema_version": "payload_lowering_report_v1",
            "stage_id": "payload_lowering",
            "module_id": self.module_id,
            "input_kind": self.input_kind,
            "lowering_api": _LOWERING_API,
            "status": self.status,
            "input": {
                "graph_path": self.input_graph_path,
                "graph_hash": self.input_graph_sha256,
                "num_fx_nodes": self.num_fx_nodes,
                "num_call_function": self.num_call_function,
            },
            "output": {
                "payload_mlir": self.payload_mlir_path,
                "payload_mlir_sha256": self.payload_mlir_sha256,
                "payload_ops_total": self.payload_ops_total,
                "module_verify_status": self.module_verify_status,
            },
            "lowering": {
                "decomposed_ops": self.decomposed_ops,
                "opaque_ops": self.opaque_ops,
                "unsupported_ops": self.unsupported_ops,
                "decomposition_coverage": self.decomposition_coverage,
            },
            "diagnostics": self.diagnostics,
            "error": self.error,
            "skip_reason": self.skip_reason,
            "llm_calls": 0,
        }


# --------------------------------------------------------------------------- #
# Helpers — shape/dtype extraction from FX node meta
# --------------------------------------------------------------------------- #


def _shape_of(t: Any) -> list[Any]:
    if isinstance(t, torch.Tensor):
        return [int(s) if isinstance(s, int) else str(s) for s in t.shape]
    return []


def _dtype_of(t: Any) -> str:
    if isinstance(t, torch.Tensor):
        return str(t.dtype)
    return ""


def _node_signatures(node: Any) -> tuple[list[Any], list[str], list[Any], list[str]]:
    """Return ``(input_shapes, input_dtypes, output_shapes, output_dtypes)``.

    ``args`` may contain non-tensor values; those contribute empty
    entries so the caller still has a fixed-length list aligned with
    the call signature.
    """
    in_shapes: list[Any] = []
    in_dtypes: list[str] = []
    for a in node.args:
        if hasattr(a, "meta"):
            v = a.meta.get("val")
            in_shapes.append(_shape_of(v))
            in_dtypes.append(_dtype_of(v))
    val = node.meta.get("val") if hasattr(node, "meta") else None
    if isinstance(val, (tuple, list)):
        out_shapes = [_shape_of(x) for x in val]
        out_dtypes = [_dtype_of(x) for x in val]
    else:
        out_shapes = [_shape_of(val)]
        out_dtypes = [_dtype_of(val)]
    return in_shapes, in_dtypes, out_shapes, out_dtypes


# --------------------------------------------------------------------------- #
# Diagnostic + opaque/unsupported extraction
# --------------------------------------------------------------------------- #


def _diagnostic_to_dict(module_id: str, d: ImportDiagnostic) -> dict[str, Any]:
    return {
        "module_id": module_id,
        "fx_node": d.fx_node,
        "level": d.level,
        "message": _canonicalize_target_string(d.message),
    }


_HEX_ADDR_RE = re.compile(r" at 0x[0-9a-fA-F]+")


def _canonicalize_target_string(s: str) -> str:
    """Strip Python ``id()`` memory addresses from a target/diagnostic string.

    Dynamo records ``call_function`` targets as the live Python callable
    object — ``str(node.target)`` then yields things like
    ``<built-in method conv2d of type object at 0x7f3c...>``. The
    ``0x...`` part is non-canonical across reruns, which would make
    every downstream identity (opaque-call by_target keys, gap_action
    queue entries, etc.) drift run-to-run. We strip the address but
    keep the rest of the string intact for legibility.
    """
    return _HEX_ADDR_RE.sub("", s)


def _extract_opaque_calls(
    module_id: str,
    payload_mlir_rel: str,
    importer: FXImporter,
    fx_graph: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Walk importer diagnostics + the FX graph to produce opaque + unsupported records.

    The "Opaque:" diagnostic line carries the exact ``func.call``
    callee. We pair that with the originating FX node so the record
    surfaces shape/dtype signatures and a payload reference.
    """
    # Index FX nodes by name for cross-reference.
    nodes_by_name = {n.name: n for n in fx_graph.nodes}

    opaque_records: list[dict[str, Any]] = []
    unsupported_records: list[dict[str, Any]] = []
    unsupported_seq = 0

    for d in importer.diagnostics:
        # FXImporter formats opaque-fallback messages as
        #   "Opaque: <fx_target> -> func.call @<callee>"
        if not d.message.startswith("Opaque: "):
            continue
        body = d.message[len("Opaque: "):]
        if " -> func.call @" not in body:
            continue
        fx_target, _, callee = body.partition(" -> func.call @")
        fx_target = _canonicalize_target_string(fx_target.strip())
        callee = _canonicalize_target_string(callee.strip())
        node = nodes_by_name.get(d.fx_node)
        in_shapes, in_dtypes, out_shapes, out_dtypes = (
            _node_signatures(node) if node is not None else ([], [], [], [])
        )
        opaque_records.append(
            {
                "module_id": module_id,
                "fx_node": d.fx_node,
                "fx_target": fx_target.strip(),
                "callee": callee,
                "input_types": [
                    f"shape={s} dtype={t}" for s, t in zip(in_shapes, in_dtypes)
                ],
                "output_types": [
                    f"shape={s} dtype={t}" for s, t in zip(out_shapes, out_dtypes)
                ],
                "payload_ref": {
                    "payload_mlir": payload_mlir_rel,
                    "callee": callee,
                },
                "diagnostic": _canonicalize_target_string(d.message),
            }
        )

        unsupported_seq += 1
        unsupported_records.append(
            {
                "unsupported_id": f"unsupported_{unsupported_seq - 1:04d}",
                "module_id": module_id,
                "source_kind": "opaque_func_call",
                "fx_node": d.fx_node,
                "fx_target": fx_target.strip(),
                "callee": callee,
                "reason": "no_decomposition_rule",
                "shape_signature": {"inputs": in_shapes, "outputs": out_shapes},
                "dtype_signature": {"inputs": in_dtypes, "outputs": out_dtypes},
                "payload_ref": {
                    "payload_mlir": payload_mlir_rel,
                    "callee": callee,
                },
            }
        )

    return opaque_records, unsupported_records


# --------------------------------------------------------------------------- #
# Per-module lowering
# --------------------------------------------------------------------------- #


def _verify_module(module_op: Any) -> tuple[str, str | None]:
    try:
        module_op.verify()
        return "pass", None
    except Exception as exc:
        return "fail", f"{type(exc).__name__}: {exc}"


def _count_payload_ops(module_op: Any) -> int:
    """Count operations inside the module excluding the module itself."""
    count = 0
    for func in module_op.body.ops:
        for region in func.regions:
            for block in region.blocks:
                for _ in block.ops:
                    count += 1
    return count


def _write_per_module_artifacts(
    *,
    run_dir: Path,
    module_dir: Path,
    module_id: str,
    payload_mlir_rel: str,
    result: ModuleLoweringResult,
) -> list[ArtifactRef]:
    """Write the per-module ``payload.mlir`` was already written by the caller.
    Now write the four sibling JSON files (lowering_report, diagnostics,
    opaque_calls, unsupported_ops) and produce ArtifactRefs for all five.
    """
    refs: list[ArtifactRef] = []

    payload_path = run_dir / payload_mlir_rel
    refs.append(
        ArtifactRef(
            path=payload_mlir_rel,
            sha256=sha256_file(payload_path),
            size_bytes=payload_path.stat().st_size,
            kind="file",
        )
    )

    report_obj = result.to_report()
    report_path = module_dir / "lowering_report.json"
    report_path.write_text(json.dumps(report_obj, indent=2, sort_keys=True), encoding="utf-8")
    refs.append(
        ArtifactRef(
            path=report_path.relative_to(run_dir).as_posix(),
            sha256=sha256_file(report_path),
            size_bytes=report_path.stat().st_size,
            kind="file",
        )
    )

    diag_obj = {
        "schema_version": "payload_lowering_diagnostics_v1",
        "diagnostics": result.diagnostics,
        "summary": {
            "info": sum(1 for d in result.diagnostics if d["level"] == "info"),
            "warning": sum(1 for d in result.diagnostics if d["level"] == "warning"),
            "error": sum(1 for d in result.diagnostics if d["level"] == "error"),
        },
    }
    diag_path = module_dir / "diagnostics.json"
    diag_path.write_text(json.dumps(diag_obj, indent=2, sort_keys=True), encoding="utf-8")
    refs.append(
        ArtifactRef(
            path=diag_path.relative_to(run_dir).as_posix(),
            sha256=sha256_file(diag_path),
            size_bytes=diag_path.stat().st_size,
            kind="file",
        )
    )

    opaque_obj = {
        "schema_version": "opaque_calls_v1",
        "opaque_calls": result.opaque_calls,
        "summary": _opaque_summary(result.opaque_calls),
    }
    opaque_path = module_dir / "opaque_calls.json"
    opaque_path.write_text(json.dumps(opaque_obj, indent=2, sort_keys=True), encoding="utf-8")
    refs.append(
        ArtifactRef(
            path=opaque_path.relative_to(run_dir).as_posix(),
            sha256=sha256_file(opaque_path),
            size_bytes=opaque_path.stat().st_size,
            kind="file",
        )
    )

    unsupported_obj = {
        "schema_version": "unsupported_ops_v1",
        "unsupported_ops": result.unsupported_op_records,
        "summary": _unsupported_summary(result.unsupported_op_records),
    }
    unsupported_path = module_dir / "unsupported_ops.json"
    unsupported_path.write_text(
        json.dumps(unsupported_obj, indent=2, sort_keys=True), encoding="utf-8"
    )
    refs.append(
        ArtifactRef(
            path=unsupported_path.relative_to(run_dir).as_posix(),
            sha256=sha256_file(unsupported_path),
            size_bytes=unsupported_path.stat().st_size,
            kind="file",
        )
    )

    return refs


def _opaque_summary(opaque_calls: list[dict[str, Any]]) -> dict[str, Any]:
    by_target: dict[str, int] = {}
    for o in opaque_calls:
        t = o["fx_target"]
        by_target[t] = by_target.get(t, 0) + 1
    return {"count": len(opaque_calls), "by_target": dict(sorted(by_target.items()))}


def _unsupported_summary(unsupported: list[dict[str, Any]]) -> dict[str, Any]:
    by_target: dict[str, int] = {}
    by_reason: dict[str, int] = {}
    for u in unsupported:
        t = u["fx_target"]
        r = u["reason"]
        by_target[t] = by_target.get(t, 0) + 1
        by_reason[r] = by_reason.get(r, 0) + 1
    return {
        "count": len(unsupported),
        "by_target": dict(sorted(by_target.items())),
        "by_reason": dict(sorted(by_reason.items())),
    }


_DTYPE_TO_TORCH = {
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.int8": torch.int8,
    "torch.int16": torch.int16,
    "torch.int32": torch.int32,
    "torch.int64": torch.int64,
    "torch.bool": torch.bool,
    "torch.uint8": torch.uint8,
}


def _normalize_fx_meta(graph: Any, sidecar: dict[str, dict[str, Any]] | None = None) -> None:
    """Restore ``meta['val']`` so ``FXImporter`` can type the lowered IR.

    Two sources, in order:

    1. Live ``meta['example_value']`` if present (the freshly-captured
       case — used when lowering during the same process as capture).
    2. The on-disk meta sidecar produced by capture (see
       ``capture._extract_meta_sidecar``). After ``torch.save``/``torch.load``
       round-trips, the live FakeTensor metas are gone — but the sidecar
       carries shape+dtype, which is enough to fabricate a ``torch.empty``
       so that ``FXImporter._tensor_type_from_meta`` can read off the
       shape/dtype it needs.

    Only writes ``meta['val']`` when it is missing — never overwrites
    ``torch.export``'s richer metadata.
    """
    sidecar = sidecar or {}
    for node in graph.nodes:
        meta = getattr(node, "meta", None)
        if meta is None or meta.get("val") is not None:
            continue
        live = meta.get("example_value")
        if isinstance(live, torch.Tensor):
            meta["val"] = live
            continue
        info = sidecar.get(node.name)
        if not info:
            continue
        shape = info.get("shape", [])
        dtype_str = info.get("dtype", "torch.float32")
        # Coerce dynamic-shape strings to 1 — FXImporter falls back
        # gracefully on those today.
        static_shape = [s if isinstance(s, int) else 1 for s in shape]
        dtype = _DTYPE_TO_TORCH.get(dtype_str, torch.float32)
        meta["val"] = torch.empty(static_shape, dtype=dtype)


def _lower_one(
    *,
    run_dir: Path,
    module_id: str,
    input_kind: str,
    fx_carrier: Any,  # has .graph
    input_graph_path_rel: str,
    input_graph_sha256: str,
    out_dir_rel: str,  # e.g. "01_payload_lowering/dynamo_partitions/partition_000"
    meta_sidecar: dict[str, dict[str, Any]] | None = None,
    extension_registry: Any = None,  # ExtensionRegistry | None
) -> ModuleLoweringResult:
    """Lower one FX-carrying object via FXImporter; write payload.mlir."""
    out_dir = run_dir / out_dir_rel
    out_dir.mkdir(parents=True, exist_ok=True)
    payload_path = out_dir / "payload.mlir"
    payload_mlir_rel = (Path(out_dir_rel) / "payload.mlir").as_posix()

    importer = FXImporter()
    fx_graph = fx_carrier.graph
    _normalize_fx_meta(fx_graph, sidecar=meta_sidecar)

    # IR-level closure: rewrite the FX graph to inline registered
    # extensions BEFORE FXImporter sees it. The result is a payload.mlir
    # without opaque ``func.call`` for any closed target.
    substitution_diagnostics: list[dict[str, Any]] = []
    if extension_registry is not None and getattr(extension_registry, "entries", []):
        from compgen.graph_compilation.payload_substitution import apply_extensions

        sub_result = apply_extensions(fx_graph, extension_registry)
        for s in sub_result.substitutions:
            substitution_diagnostics.append(
                {
                    "module_id": module_id,
                    "fx_node": s["fx_node"],
                    "level": "info",
                    "message": (
                        f"Inlined extension {s['extension_id']!r} "
                        f"for {s['fx_target']!r} (registry-driven)"
                    ),
                }
            )
        for s in sub_result.skipped:
            substitution_diagnostics.append(
                {
                    "module_id": module_id,
                    "fx_node": s["fx_node"],
                    "level": "warning",
                    "message": (
                        f"Skipped extension {s['extension_id']!r} for {s['fx_target']!r}: "
                        f"{s['reason']}"
                    ),
                }
            )

    fx_nodes = list(fx_graph.nodes)
    num_fx_nodes = len(fx_nodes)
    num_call_function = sum(1 for n in fx_nodes if n.op == "call_function")

    try:
        module_op = importer.import_graph(fx_carrier)
    except Exception as exc:
        # Importer crashed before producing IR. We still write a
        # placeholder file so the validator's path-existence check
        # has something to point at, but the report status is fail.
        payload_path.write_text(
            f"; payload lowering failed: {type(exc).__name__}: {exc}\n", encoding="utf-8"
        )
        return ModuleLoweringResult(
            module_id=module_id,
            input_kind=input_kind,
            status="fail",
            input_graph_path=input_graph_path_rel,
            input_graph_sha256=input_graph_sha256,
            payload_mlir_path=payload_mlir_rel,
            payload_mlir_sha256=sha256_file(payload_path),
            payload_ops_total=0,
            num_fx_nodes=num_fx_nodes,
            num_call_function=num_call_function,
            decomposed_ops=0,
            opaque_ops=0,
            unsupported_ops=0,
            decomposition_coverage=0.0,
            module_verify_status="skipped",
            error=f"{type(exc).__name__}: {exc}",
        )

    # Serialize MLIR text.
    mlir_text = importer.get_ir_text(module_op)
    payload_path.write_text(mlir_text, encoding="utf-8")

    # Verify the module.
    verify_status, verify_error = _verify_module(module_op)

    # Diagnostics + opaque + unsupported.
    diags = [_diagnostic_to_dict(module_id, d) for d in importer.diagnostics]
    # Prepend any substitution diagnostics so the per-module log shows
    # the inlining decisions before importer-level diagnostics.
    diags = list(substitution_diagnostics) + diags
    if verify_error:
        diags.append({"module_id": module_id, "fx_node": "", "level": "error", "message": verify_error})
    opaque, unsupported = _extract_opaque_calls(module_id, payload_mlir_rel, importer, fx_graph)

    payload_ops = _count_payload_ops(module_op)

    overall_status = "pass" if verify_status == "pass" else "fail"

    return ModuleLoweringResult(
        module_id=module_id,
        input_kind=input_kind,
        status=overall_status,
        input_graph_path=input_graph_path_rel,
        input_graph_sha256=input_graph_sha256,
        payload_mlir_path=payload_mlir_rel,
        payload_mlir_sha256=sha256_file(payload_path),
        payload_ops_total=payload_ops,
        num_fx_nodes=num_fx_nodes,
        num_call_function=num_call_function,
        decomposed_ops=importer.decomposed_count,
        opaque_ops=importer.opaque_count,
        unsupported_ops=len(unsupported),
        decomposition_coverage=importer.decomposition_coverage,
        module_verify_status=verify_status,
        diagnostics=diags,
        opaque_calls=opaque,
        unsupported_op_records=unsupported,
        error=verify_error,
    )


# --------------------------------------------------------------------------- #
# Stage entry point
# --------------------------------------------------------------------------- #


def run_payload_lowering(
    run_dir: Path,
    *,
    target_id: str,
    extension_registry: Path | None = None,
) -> tuple[StageRecord, list[ModuleLoweringResult]]:
    """Lower every captured graph under ``run_dir/00_graph_capture/``.

    Returns the stage record and the per-module results so the caller
    (``run.py``) can emit the aggregate top-level reports.
    """
    started_at = _utcnow()
    run_dir = Path(run_dir).resolve()
    capture_dir = run_dir / "00_graph_capture"
    if not capture_dir.is_dir():
        raise FileNotFoundError(f"graph_capture directory missing: {capture_dir}")

    out_dir = run_dir / "01_payload_lowering"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load the extension registry once so every module's _lower_one
    # call gets the same ExtensionRegistry instance.
    registry_obj = None
    if extension_registry is not None:
        from compgen.graph_compilation.extension_registry import load_registry

        registry_obj = load_registry(Path(extension_registry))

    dynamo_summary_path = capture_dir / "dynamo_summary.json"
    capture_report_path = capture_dir / "capture_report.json"

    dynamo_summary = json.loads(dynamo_summary_path.read_text(encoding="utf-8"))

    results: list[ModuleLoweringResult] = []
    artifact_refs: list[ArtifactRef] = []

    # ------------------------------------------------------------------ #
    # 1. Dynamo partitions
    # ------------------------------------------------------------------ #
    for part in dynamo_summary.get("partitions", []):
        idx = part["index"]
        module_id = f"dynamo_partition_{idx:03d}"
        gm_rel = part["graphmodule_pt"]
        gm_path = run_dir / gm_rel
        if not gm_path.exists():
            results.append(_skipped_result(module_id, "torch_dynamo_partition", gm_rel,
                                           f"01_payload_lowering/dynamo_partitions/partition_{idx:03d}"))
            continue

        sidecar: dict[str, dict[str, Any]] = {}
        sidecar_rel = part.get("meta_sidecar")
        if sidecar_rel:
            sidecar_path = run_dir / sidecar_rel
            if sidecar_path.exists():
                sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))

        gm: Any = torch.load(gm_path, weights_only=False)
        result = _lower_one(
            run_dir=run_dir,
            module_id=module_id,
            input_kind="torch_dynamo_partition",
            fx_carrier=gm,
            input_graph_path_rel=gm_rel,
            input_graph_sha256=sha256_file(gm_path),
            out_dir_rel=f"01_payload_lowering/dynamo_partitions/partition_{idx:03d}",
            meta_sidecar=sidecar,
            extension_registry=registry_obj,
        )
        module_dir = run_dir / f"01_payload_lowering/dynamo_partitions/partition_{idx:03d}"
        result.artifact_refs = _write_per_module_artifacts(
            run_dir=run_dir,
            module_dir=module_dir,
            module_id=module_id,
            payload_mlir_rel=result.payload_mlir_path,
            result=result,
        )
        artifact_refs.extend(result.artifact_refs)
        results.append(result)

    # ------------------------------------------------------------------ #
    # 2. Optional exported_program
    # ------------------------------------------------------------------ #
    ep_path = capture_dir / "exported_program.pt2"
    ep_module_dir = out_dir / "export_program"
    ep_module_dir.mkdir(parents=True, exist_ok=True)
    if ep_path.exists():
        ep_load_error: str | None = None
        try:
            ep = torch.export.load(str(ep_path))
        except Exception as exc:
            ep = None
            ep_load_error = f"{type(exc).__name__}: {exc}"

        if ep is None:
            result = ModuleLoweringResult(
                module_id="export_program",
                input_kind="exported_program",
                status="fail",
                input_graph_path="00_graph_capture/exported_program.pt2",
                input_graph_sha256=sha256_file(ep_path),
                payload_mlir_path="01_payload_lowering/export_program/payload.mlir",
                payload_mlir_sha256="",
                payload_ops_total=0,
                num_fx_nodes=0,
                num_call_function=0,
                decomposed_ops=0,
                opaque_ops=0,
                unsupported_ops=0,
                decomposition_coverage=0.0,
                module_verify_status="skipped",
                error=ep_load_error,
            )
            payload_path = run_dir / result.payload_mlir_path
            payload_path.write_text(f"; ExportedProgram load failed: {ep_load_error}\n", encoding="utf-8")
            result.payload_mlir_sha256 = sha256_file(payload_path)
        else:
            result = _lower_one(
                run_dir=run_dir,
                module_id="export_program",
                input_kind="exported_program",
                fx_carrier=ep,
                input_graph_path_rel="00_graph_capture/exported_program.pt2",
                input_graph_sha256=sha256_file(ep_path),
                out_dir_rel="01_payload_lowering/export_program",
                extension_registry=registry_obj,
            )
        result.artifact_refs = _write_per_module_artifacts(
            run_dir=run_dir,
            module_dir=ep_module_dir,
            module_id=result.module_id,
            payload_mlir_rel=result.payload_mlir_path,
            result=result,
        )
        artifact_refs.extend(result.artifact_refs)
        results.append(result)
    else:
        # Write skipped marker so the per-module dir is honest.
        skip_report = {
            "schema_version": "payload_lowering_report_v1",
            "stage_id": "payload_lowering",
            "module_id": "export_program",
            "input_kind": "exported_program",
            "lowering_api": _LOWERING_API,
            "status": "skipped",
            "skip_reason": "exported_program.pt2 not available",
            "llm_calls": 0,
        }
        skip_path = ep_module_dir / "lowering_report.json"
        skip_path.write_text(json.dumps(skip_report, indent=2, sort_keys=True), encoding="utf-8")
        artifact_refs.append(
            ArtifactRef(
                path=skip_path.relative_to(run_dir).as_posix(),
                sha256=sha256_file(skip_path),
                size_bytes=skip_path.stat().st_size,
                kind="file",
            )
        )

    # ------------------------------------------------------------------ #
    # 3. Aggregate top-level reports (delegated to reports.py)
    # ------------------------------------------------------------------ #
    from compgen.graph_compilation.reports import emit_top_level_reports

    aggregate_refs = emit_top_level_reports(
        run_dir=run_dir,
        results=results,
        capture_report_path=capture_report_path,
        dynamo_summary_path=dynamo_summary_path,
        target_id=target_id,
    )
    artifact_refs.extend(aggregate_refs)

    # Structured Payload Attribution (02.5) — exact per-FX-node →
    # payload-op mapping, derived deterministically from the lowering
    # diagnostics + payload.mlir order. Runs BEFORE the coverage audit
    # so the latter can consume payload_attribution.json.
    from compgen.graph_compilation.payload_attribution import (
        build_payload_attribution,
    )

    attribution = build_payload_attribution(run_dir)
    artifact_refs.append(
        ArtifactRef(
            path=attribution.path.relative_to(run_dir).as_posix(),
            sha256=sha256_file(attribution.path),
            size_bytes=attribution.path.stat().st_size,
            kind="file",
        )
    )

    # Payload Coverage Audit — every FX node accounted for, dialect
    # coverage tallied, silent drops surfaced. This runs unconditionally
    # after lowering so the audit JSONs are part of the contract.
    from compgen.graph_compilation.payload_coverage import audit_payload_coverage

    pca = audit_payload_coverage(run_dir)
    for path in (
        pca.fx_to_payload_accounting_path,
        pca.dialect_coverage_path,
        pca.silent_drop_audit_path,
    ):
        artifact_refs.append(
            ArtifactRef(
                path=path.relative_to(run_dir).as_posix(),
                sha256=sha256_file(path),
                size_bytes=path.stat().st_size,
                kind="file",
            )
        )

    finished_at = _utcnow()

    # input_hash for this stage MUST equal graph_capture.output_hash —
    # that is precisely what R009 (the artifact-contract validator's
    # hash-chain rule) checks. graph_capture.output_hash is
    # ``sha256_tree(00_graph_capture/)`` and we recompute the same here.
    input_hash = sha256_tree(capture_dir)
    output_hash = sha256_tree(out_dir)

    # Stage status: pass if every non-skipped module passed; partial_success
    # if any module failed but at least one passed; fail if all non-skipped
    # failed. The graph_compilation manifest contract is {pass, fail, skipped}, so
    # partial_success is recorded in lowering_summary.status only.
    non_skipped = [r for r in results if r.status != "skipped"]
    if not non_skipped:
        manifest_status = "fail"
    elif all(r.status == "pass" for r in non_skipped):
        manifest_status = "pass"
    elif any(r.status == "pass" for r in non_skipped):
        # Treat partial as pass at the manifest level — gap discovery still
        # has real artifacts to look at — and surface partial_success in
        # the per-stage report. This keeps R007 satisfied (status=pass
        # implies ≥1 output) and lets validators downstream introspect.
        manifest_status = "pass"
    else:
        manifest_status = "fail"

    record = StageRecord(
        stage_id="payload_lowering",
        status=manifest_status,
        inputs=(
            ArtifactRef(
                path=capture_report_path.relative_to(run_dir).as_posix(),
                sha256=sha256_file(capture_report_path),
                size_bytes=capture_report_path.stat().st_size,
                kind="file",
            ),
            ArtifactRef(
                path=dynamo_summary_path.relative_to(run_dir).as_posix(),
                sha256=sha256_file(dynamo_summary_path),
                size_bytes=dynamo_summary_path.stat().st_size,
                kind="file",
            ),
        ),
        outputs=tuple(artifact_refs),
        report_path="01_payload_lowering/lowering_summary.json",
        input_hash=input_hash,
        output_hash=output_hash,
        llm_calls=0,
        started_at_utc=started_at,
        finished_at_utc=finished_at,
    )
    return record, results


def _skipped_result(module_id: str, input_kind: str, input_path: str, out_dir_rel: str) -> ModuleLoweringResult:
    return ModuleLoweringResult(
        module_id=module_id,
        input_kind=input_kind,
        status="skipped",
        input_graph_path=input_path,
        input_graph_sha256="",
        payload_mlir_path=str(Path(out_dir_rel) / "payload.mlir"),
        payload_mlir_sha256="",
        payload_ops_total=0,
        num_fx_nodes=0,
        num_call_function=0,
        decomposed_ops=0,
        opaque_ops=0,
        unsupported_ops=0,
        decomposition_coverage=0.0,
        module_verify_status="skipped",
        skip_reason="input GraphModule artifact missing",
    )


def copy_capture_run(source: Path, dest: Path) -> str:
    """Copy ``source/00_graph_capture/`` into ``dest/`` and return the source manifest sha256.

    Used by the ``lower --capture-run …`` CLI. We copy the capture
    artifacts so the new run's manifest can list them with their
    recomputed hashes.
    """
    src = Path(source).resolve()
    dst = Path(dest).resolve()
    if not (src / "00_graph_capture").is_dir():
        raise FileNotFoundError(f"capture run missing 00_graph_capture/: {src}")
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src / "00_graph_capture", dst / "00_graph_capture", dirs_exist_ok=True)
    manifest = src / "run_manifest.json"
    return sha256_file(manifest) if manifest.exists() else ""


def _utcnow() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
