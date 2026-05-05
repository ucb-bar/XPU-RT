"""graph_capture stage: real Stage 0 capture wrapper.

This module is a *thin orchestrator* around existing CompGen capture
infrastructure. It does not reimplement capture or diagnostics — it
calls:

- :func:`compgen.capture.torch_export.capture_dynamo_partitions` (primary)
- :func:`compgen.capture.torch_export.capture_frontend_artifact` (optional canonical)
- :func:`compgen.capture.dynamo_baseline.compile_baseline` (non-gating timing)

…and then materialises every result on disk under ``00_graph_capture/``.

Per graph_capture stage policy:

- TorchDynamo is the **primary** graph-discovery surface. A run is
  considered ``pass`` if Dynamo captured at least one partition, even
  if ``torch.export`` failed.
- ``torch.export`` is treated as an **optional canonical artifact**.
  When it succeeds, ``exported_program.pt2`` is the strongest evidence
  for replay; when it fails, the failure is recorded honestly in
  ``capture_report.json::torch_export.status = "fail"`` with the error
  message, and the run is still ``pass`` (sub-status
  ``partial_success``).

No LLM is consulted in this stage. ``capture_report.json::llm_calls``
is always 0.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from typing import Any

import torch
import torch.fx
import yaml

from compgen.capture.dynamo_baseline import compile_baseline
from compgen.capture.torch_export import (
    CaptureArtifact,
    capture_dynamo_partitions,
    capture_frontend_artifact,
)
from compgen.graph_compilation.artifacts import ArtifactRef, StageRecord
from compgen.graph_compilation.hashing import sha256_file, sha256_tree

# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModelConfig:
    schema_version: str
    model_id: str
    model_path: Path
    factory: str
    seed: int
    primary: str
    also_try_torch_export: bool
    fullgraph: bool
    run_default_decompositions: bool
    raw_path: Path
    raw_sha256: str
    # Set when loading a ``model_config_v1`` (admission) YAML — the
    # graph_compilation capture stage will route through
    # :mod:`compgen.graph_compilation.admission_bridge` instead of
    # importing a Python module from ``model_path``.
    admission_yaml: Path | None = None
    admission_slice_id: str | None = None

    @classmethod
    def load(cls, config_path: Path) -> ModelConfig:
        config_path = config_path.resolve()
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        sv = raw.get("schema_version")
        if sv == "graphcomp_model_config_v1":
            cap = raw.get("capture", {}) or {}
            return cls(
                schema_version=sv,
                model_id=raw["model_id"],
                model_path=Path(raw["model_path"]),
                factory=raw.get("factory", "get_model_and_inputs"),
                seed=int(raw.get("seed", 0)),
                primary=cap.get("primary", "torch_dynamo"),
                also_try_torch_export=bool(cap.get("also_try_torch_export", True)),
                fullgraph=bool(cap.get("fullgraph", False)),
                run_default_decompositions=bool(cap.get("run_default_decompositions", True)),
                raw_path=config_path,
                raw_sha256=sha256_file(config_path),
            )
        if sv == "model_config_v1":
            # Admission config — route through admission_bridge. The
            # ``model_path`` field is set to the YAML itself for the
            # input-hash chain; the capture step recognises
            # ``admission_yaml`` and uses the bridge factory.
            return cls(
                schema_version=sv,
                model_id=raw["model_id"],
                model_path=config_path,
                factory="__admission_bridge__",
                seed=0,
                primary="torch_dynamo",
                also_try_torch_export=False,
                fullgraph=False,
                run_default_decompositions=True,
                raw_path=config_path,
                raw_sha256=sha256_file(config_path),
                admission_yaml=config_path,
                admission_slice_id=None,
            )
        raise ValueError(f"unsupported model config schema_version: {sv!r}")


@dataclass(frozen=True)
class TargetConfig:
    schema_version: str
    target_id: str
    device_kind: str
    raw_path: Path
    raw_sha256: str

    @classmethod
    def load(cls, config_path: Path) -> TargetConfig:
        config_path = config_path.resolve()
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if raw.get("schema_version") != "graphcomp_target_config_v1":
            raise ValueError(f"unsupported target config schema_version: {raw.get('schema_version')!r}")
        return cls(
            schema_version=raw["schema_version"],
            target_id=raw["target_id"],
            device_kind=raw.get("device_kind", "cpu"),
            raw_path=config_path,
            raw_sha256=sha256_file(config_path),
        )


# --------------------------------------------------------------------------- #
# Model loader (deliberately separate from compgen.capture's loader so we own
# the seed / factory contract for graph compilation)
# --------------------------------------------------------------------------- #


def _load_model_factory(model_path: Path, factory: str) -> Any:
    abs_path = model_path.resolve()
    spec = importlib.util.spec_from_file_location(f"graphcomp_model_{abs_path.stem}", abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load model module from {abs_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, factory):
        raise AttributeError(f"{abs_path} has no factory {factory!r}")
    return getattr(module, factory)


# --------------------------------------------------------------------------- #
# Graph-hash + per-partition serialization
# --------------------------------------------------------------------------- #


def _node_summary(node: torch.fx.Node) -> dict[str, Any]:
    """Stable, JSON-friendly summary of one FX node.

    Excludes pointer-y kwargs values; we keep only a string representation
    so the JSON is reproducible across runs.
    """
    tensor_meta: dict[str, Any] = {}
    meta_obj = node.meta.get("tensor_meta") if hasattr(node, "meta") else None
    if meta_obj is not None:
        shape = getattr(meta_obj, "shape", None)
        dtype = getattr(meta_obj, "dtype", None)
        if shape is not None:
            tensor_meta["shape"] = [int(s) if isinstance(s, int) else str(s) for s in shape]
        if dtype is not None:
            tensor_meta["dtype"] = str(dtype)
    val = node.meta.get("val") if hasattr(node, "meta") else None
    if val is not None and isinstance(val, torch.Tensor):
        tensor_meta.setdefault("shape", [int(s) if isinstance(s, int) else str(s) for s in val.shape])
        tensor_meta.setdefault("dtype", str(val.dtype))
    return {
        "name": str(node.name),
        "op": str(node.op),
        "target": str(node.target),
        "args": [str(a) for a in node.args],
        "kwargs": {k: str(v) for k, v in node.kwargs.items()},
        "tensor_meta": tensor_meta,
    }


def _graph_summary(gm: torch.fx.GraphModule, model_id: str) -> dict[str, Any]:
    nodes = [_node_summary(n) for n in gm.graph.nodes]
    summary = {
        "num_nodes": len(nodes),
        "num_placeholders": sum(1 for n in nodes if n["op"] == "placeholder"),
        "num_call_function": sum(1 for n in nodes if n["op"] == "call_function"),
        "num_call_module": sum(1 for n in nodes if n["op"] == "call_module"),
        "num_outputs": sum(1 for n in nodes if n["op"] == "output"),
    }
    body = {
        "schema_version": "fx_graph_v1",
        "model_id": model_id,
        "nodes": nodes,
        "summary": summary,
    }
    serialized = json.dumps(body, sort_keys=True, separators=(",", ":"))
    body["graph_hash"] = "sha256:" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return body


def _extract_meta_sidecar(gm: torch.fx.GraphModule) -> dict[str, dict[str, Any]]:
    """Snapshot per-node tensor shape+dtype before pickling.

    ``torch.save`` strips Dynamo's ``meta['example_value']`` (the
    FakeTensor doesn't pickle cleanly), so without a sidecar the
    reloaded GraphModule has no usable type info and ``FXImporter``
    can't lower it. The sidecar is a tiny JSON keyed by node name —
    cheap to produce, sufficient for ``FXImporter._tensor_type_from_meta``
    after we materialize an ``empty`` tensor at load time.
    """
    sidecar: dict[str, dict[str, Any]] = {}
    for node in gm.graph.nodes:
        val = node.meta.get("val") or node.meta.get("example_value")
        if isinstance(val, torch.Tensor):
            sidecar[node.name] = {
                "shape": [int(s) if isinstance(s, int) else str(s) for s in val.shape],
                "dtype": str(val.dtype),
            }
        elif isinstance(val, (tuple, list)):
            # Multi-output op (e.g. native_layer_norm). Record the first
            # tensor's shape; the importer's tuple handling falls through
            # the per-result type lookup.
            for v in val:
                if isinstance(v, torch.Tensor):
                    sidecar[node.name] = {
                        "shape": [int(s) if isinstance(s, int) else str(s) for s in v.shape],
                        "dtype": str(v.dtype),
                    }
                    break
    return sidecar


def _write_partition(
    out_dir: Path, idx: int, gm: torch.fx.GraphModule, model_id: str
) -> tuple[Path, Path, Path, Path]:
    """Persist one Dynamo partition.

    Four artifacts per partition:

    - ``partition_NNN_graph.json`` — node-by-node summary for review/diffing.
    - ``partition_NNN_graph.py``   — readable FX text (humans).
    - ``partition_NNN_graphmodule.pt`` — pickled ``torch.fx.GraphModule``;
      this is the **machine-loadable** artifact Payload Lowering reloads
      with ``torch.load`` so it does not need to rerun Dynamo.
    - ``partition_NNN_meta.json`` — per-node shape/dtype sidecar.
      ``torch.save`` drops the Dynamo FakeTensor metadata; this sidecar
      preserves what FXImporter needs to type the lowered IR.

    Saving the GraphModule is the load-bearing piece of the artifact
    chain — without it, downstream stages either re-run the model
    (defeats determinism) or reconstruct FX from JSON (lossy).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"partition_{idx:03d}_graph.json"
    py_path = out_dir / f"partition_{idx:03d}_graph.py"
    pt_path = out_dir / f"partition_{idx:03d}_graphmodule.pt"
    meta_path = out_dir / f"partition_{idx:03d}_meta.json"
    summary = _graph_summary(gm, model_id)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    try:
        readable = gm.print_readable(print_output=False)
    except (TypeError, AttributeError):
        readable = repr(gm.graph)
    py_path.write_text(str(readable), encoding="utf-8")
    # Sidecar must be written BEFORE torch.save, while meta is still
    # populated on the live GraphModule.
    sidecar = _extract_meta_sidecar(gm)
    meta_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True), encoding="utf-8")
    torch.save(gm, pt_path)
    return json_path, py_path, pt_path, meta_path


# --------------------------------------------------------------------------- #
# Sub-stage helpers
# --------------------------------------------------------------------------- #


def _save_goldens(
    capture_dir: Path,
    model: torch.nn.Module,
    sample_inputs: tuple[torch.Tensor, ...],
) -> tuple[ArtifactRef, ArtifactRef, dict[str, Any]]:
    inputs_path = capture_dir / "golden_inputs.pt"
    outputs_path = capture_dir / "golden_outputs.pt"

    # Save inputs as a flat tuple. Use weights_only-compatible torch.save.
    torch.save(sample_inputs, inputs_path)

    with torch.no_grad():
        outputs = model(*sample_inputs)
    if isinstance(outputs, torch.Tensor):
        outputs_obj: Any = (outputs,)
    elif isinstance(outputs, (list, tuple)):
        outputs_obj = tuple(outputs)
    else:
        outputs_obj = outputs
    torch.save(outputs_obj, outputs_path)

    # Self-check: re-run eager with no_grad and compare.
    with torch.no_grad():
        outputs2 = model(*sample_inputs)
    if isinstance(outputs2, torch.Tensor):
        outs1 = (outputs,) if isinstance(outputs, torch.Tensor) else outputs_obj
        outs2 = (outputs2,)
    else:
        outs1 = outputs_obj
        outs2 = tuple(outputs2) if isinstance(outputs2, (list, tuple)) else outputs2
    max_abs = 0.0
    max_rel = 0.0
    for a, b in zip(outs1, outs2):
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            diff = (a - b).abs()
            max_abs = max(max_abs, float(diff.max().item()) if diff.numel() else 0.0)
            denom = b.abs().clamp_min(1e-12)
            rel = (diff / denom).max().item() if diff.numel() else 0.0
            max_rel = max(max_rel, float(rel))

    inputs_ref = ArtifactRef(
        path="00_graph_capture/golden_inputs.pt",
        sha256=sha256_file(inputs_path),
        size_bytes=inputs_path.stat().st_size,
        kind="file",
    )
    outputs_ref = ArtifactRef(
        path="00_graph_capture/golden_outputs.pt",
        sha256=sha256_file(outputs_path),
        size_bytes=outputs_path.stat().st_size,
        kind="file",
    )
    self_check = {
        "num_inputs": len(sample_inputs),
        "num_outputs": len(outs1) if isinstance(outs1, tuple) else 1,
        "max_abs_self_check_error": max_abs,
        "max_rel_self_check_error": max_rel,
    }
    return inputs_ref, outputs_ref, self_check


def _try_dynamo_capture(
    model: torch.nn.Module,
    sample_inputs: tuple[torch.Tensor, ...],
    *,
    fullgraph: bool,
) -> tuple[CaptureArtifact | None, str | None]:
    try:
        artifact = capture_dynamo_partitions(model, sample_inputs, fullgraph=fullgraph)
        return artifact, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _try_export_capture(
    model: torch.nn.Module,
    sample_inputs: tuple[torch.Tensor, ...],
    *,
    run_default_decompositions: bool,
) -> tuple[CaptureArtifact | None, str | None]:
    try:
        artifact = capture_frontend_artifact(
            model,
            sample_inputs,
            run_default_decompositions=run_default_decompositions,
        )
        return artifact, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _try_compile_baseline(
    model: torch.nn.Module,
    sample_inputs: tuple[torch.Tensor, ...],
) -> dict[str, Any]:
    try:
        report = compile_baseline(model, sample_inputs, num_warmup=1, num_runs=2)
        return {
            "schema_version": "compile_baseline_v1",
            "attempted": True,
            "status": "pass",
            "backend": report.backend,
            "latency_ms_p50": float(report.warm_run_ms),
            "latency_ms_p95": float(report.warm_run_ms),
            "cold_compile_ms": float(report.cold_compile_ms),
            "num_graph_breaks": int(report.num_graph_breaks),
            "compiled_op_fraction": float(report.compiled_op_fraction),
            "error": None,
        }
    except Exception as exc:
        return {
            "schema_version": "compile_baseline_v1",
            "attempted": True,
            "status": "fail",
            "backend": "inductor",
            "latency_ms_p50": 0.0,
            "latency_ms_p95": 0.0,
            "error": f"{type(exc).__name__}: {exc}",
        }


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #


def run_graph_capture(
    model_cfg: ModelConfig,
    target_cfg: TargetConfig,
    run_dir: Path,
) -> StageRecord:
    """Execute Stage 0 capture and return a graph_compilation artifact contract ``StageRecord``.

    Side effects: writes ``00_graph_capture/`` and the per-stage report. Does
    not write the run-level manifest or ledger — that is the caller's
    responsibility (``run.py``).
    """
    capture_dir = run_dir / "00_graph_capture"
    capture_dir.mkdir(parents=True, exist_ok=True)

    started_at = _utcnow()

    # Seed early so model factories can lean on it.
    torch.manual_seed(model_cfg.seed)

    if model_cfg.admission_yaml is not None:
        from compgen.graph_compilation.admission_bridge import (
            make_factory_from_admission_config,
        )

        factory = make_factory_from_admission_config(
            model_cfg.admission_yaml, slice_id=model_cfg.admission_slice_id
        )
    else:
        factory = _load_model_factory(model_cfg.model_path, model_cfg.factory)
    model, sample_inputs = factory()

    # input_hash for the stage = sha256 of (model_config + target_config + factory bytes).
    factory_sha = sha256_file(model_cfg.model_path.resolve())
    input_hash = hashlib.sha256(
        f"{model_cfg.raw_sha256}|{target_cfg.raw_sha256}|{factory_sha}|seed={model_cfg.seed}".encode()
    ).hexdigest()

    # 1. Goldens (eager, no_grad). Done first so we have ground truth before
    #    Dynamo / export do any rewriting.
    inputs_ref, outputs_ref, golden_self_check = _save_goldens(capture_dir, model, sample_inputs)
    artifacts: list[ArtifactRef] = [inputs_ref, outputs_ref]

    # 2. Dynamo primary capture.
    dynamo_artifact, dynamo_error = _try_dynamo_capture(
        model, sample_inputs, fullgraph=model_cfg.fullgraph
    )
    dynamo_summary: dict[str, Any]
    if dynamo_artifact is None:
        dynamo_summary = {
            "schema_version": "dynamo_summary_v1",
            "status": "fail",
            "partition_count": 0,
            "graph_break_count": 0,
            "fullgraph": model_cfg.fullgraph,
            "partitions": [],
            "warnings": [],
            "error": dynamo_error,
        }
    else:
        partitions_dir = capture_dir / "dynamo_partitions"
        partitions: list[dict[str, Any]] = []
        for idx, gm in enumerate(dynamo_artifact.graphs):
            json_path, py_path, pt_path, meta_path = _write_partition(
                partitions_dir, idx, gm, model_cfg.model_id
            )
            summary = _graph_summary(gm, model_cfg.model_id)
            partition_id = f"partition_{idx:03d}"
            partitions.append(
                {
                    "partition_id": partition_id,
                    "index": idx,
                    "graph_hash": summary["graph_hash"],
                    "num_nodes": summary["summary"]["num_nodes"],
                    "num_call_function": summary["summary"]["num_call_function"],
                    "num_placeholders": summary["summary"]["num_placeholders"],
                    "num_outputs": summary["summary"]["num_outputs"],
                    "graph_json": json_path.relative_to(run_dir).as_posix(),
                    "graph_py": py_path.relative_to(run_dir).as_posix(),
                    "graphmodule_pt": pt_path.relative_to(run_dir).as_posix(),
                    "graphmodule_pt_sha256": sha256_file(pt_path),
                    "meta_sidecar": meta_path.relative_to(run_dir).as_posix(),
                }
            )
            for p in (json_path, py_path, pt_path, meta_path):
                artifacts.append(
                    ArtifactRef(
                        path=p.relative_to(run_dir).as_posix(),
                        sha256=sha256_file(p),
                        size_bytes=p.stat().st_size,
                        kind="file",
                    )
                )
        dynamo_summary = {
            "schema_version": "dynamo_summary_v1",
            "status": "pass" if partitions else "fail",
            "partition_count": len(partitions),
            "graph_break_count": dynamo_artifact.graph_break_count,
            "fullgraph": model_cfg.fullgraph,
            "partitions": partitions,
            "warnings": list(dynamo_artifact.diagnostics.warnings),
            "error": None,
        }

    dynamo_summary_path = capture_dir / "dynamo_summary.json"
    dynamo_summary_path.write_text(json.dumps(dynamo_summary, indent=2, sort_keys=True), encoding="utf-8")
    artifacts.append(
        ArtifactRef(
            path="00_graph_capture/dynamo_summary.json",
            sha256=sha256_file(dynamo_summary_path),
            size_bytes=dynamo_summary_path.stat().st_size,
            kind="file",
        )
    )

    # 3. Optional: torch.export canonical artifact.
    export_section: dict[str, Any]
    if model_cfg.also_try_torch_export:
        export_artifact, export_error = _try_export_capture(
            model, sample_inputs, run_default_decompositions=model_cfg.run_default_decompositions
        )
        if export_artifact is not None and export_artifact.exported_program is not None:
            ep_path = capture_dir / "exported_program.pt2"
            torch.export.save(export_artifact.exported_program, str(ep_path))
            artifacts.append(
                ArtifactRef(
                    path="00_graph_capture/exported_program.pt2",
                    sha256=sha256_file(ep_path),
                    size_bytes=ep_path.stat().st_size,
                    kind="file",
                )
            )

            ep_graph: torch.fx.GraphModule = export_artifact.exported_program.graph_module
            export_summary = _graph_summary(ep_graph, model_cfg.model_id)
            export_json = capture_dir / "export_graph.json"
            export_json.write_text(json.dumps(export_summary, indent=2, sort_keys=True), encoding="utf-8")
            artifacts.append(
                ArtifactRef(
                    path="00_graph_capture/export_graph.json",
                    sha256=sha256_file(export_json),
                    size_bytes=export_json.stat().st_size,
                    kind="file",
                )
            )
            try:
                readable = ep_graph.print_readable(print_output=False)
            except (TypeError, AttributeError):
                readable = repr(ep_graph.graph)
            export_py = capture_dir / "export_graph_readable.py"
            export_py.write_text(str(readable), encoding="utf-8")
            artifacts.append(
                ArtifactRef(
                    path="00_graph_capture/export_graph_readable.py",
                    sha256=sha256_file(export_py),
                    size_bytes=export_py.stat().st_size,
                    kind="file",
                )
            )

            export_section = {
                "status": "pass",
                "exported_program_path": "00_graph_capture/exported_program.pt2",
                "num_ops": export_artifact.validation.num_ops,
                "round_trip_ok": export_artifact.validation.round_trip_ok,
                "warnings": list(export_artifact.validation.warnings),
                "graph_hash": export_summary["graph_hash"],
                "error": None,
            }
        else:
            export_section = {
                "status": "fail",
                "exported_program_path": None,
                "error": export_error or "torch.export returned no exported_program",
            }
    else:
        export_section = {
            "status": "skipped",
            "exported_program_path": None,
            "error": None,
            "reason": "also_try_torch_export=false",
        }

    # 4. graph_breaks.json — sourced from Dynamo diagnostics.
    diag = dynamo_artifact.diagnostics if dynamo_artifact is not None else None
    graph_breaks_obj = {
        "schema_version": "graph_breaks_v1",
        "graph_break_count": diag.graph_breaks.__len__() if diag else 0,
        "graph_breaks": [
            {"location": loc, "reason": reason} for (loc, reason) in (diag.graph_breaks if diag else [])
        ],
        "guard_failures": diag.guard_failures if diag else 0,
        "graph_count": diag.graph_count if diag else 0,
        "op_count": diag.op_count if diag else 0,
        "warnings": list(diag.warnings) if diag else [],
    }
    graph_breaks_path = capture_dir / "graph_breaks.json"
    graph_breaks_path.write_text(json.dumps(graph_breaks_obj, indent=2, sort_keys=True), encoding="utf-8")
    artifacts.append(
        ArtifactRef(
            path="00_graph_capture/graph_breaks.json",
            sha256=sha256_file(graph_breaks_path),
            size_bytes=graph_breaks_path.stat().st_size,
            kind="file",
        )
    )

    # 5. compile_baseline.json — non-gating.
    compile_baseline_obj = _try_compile_baseline(model, sample_inputs)
    compile_baseline_path = capture_dir / "compile_baseline.json"
    compile_baseline_path.write_text(
        json.dumps(compile_baseline_obj, indent=2, sort_keys=True), encoding="utf-8"
    )
    artifacts.append(
        ArtifactRef(
            path="00_graph_capture/compile_baseline.json",
            sha256=sha256_file(compile_baseline_path),
            size_bytes=compile_baseline_path.stat().st_size,
            kind="file",
        )
    )

    # 6. capture_report.json — the per-stage report.
    runtime_versions: dict[str, str] = {}
    if dynamo_artifact is not None and dynamo_artifact.runtime_versions:
        runtime_versions.update(dynamo_artifact.runtime_versions)

    unsupported_summary = {
        "unsupported_resolution_count": 0,
        "explicit_blackboxes": [],
        "synthesized_payload_translations": [],
    }
    if dynamo_artifact is not None and dynamo_artifact.unsupported_resolutions:
        unsupported_summary["unsupported_resolution_count"] = len(dynamo_artifact.unsupported_resolutions)
        unsupported_summary["explicit_blackboxes"] = list(dynamo_artifact.explicit_blackboxes)
        unsupported_summary["synthesized_payload_translations"] = sorted(
            dynamo_artifact.synthesized_payload_translations.keys()
        )

    if dynamo_summary["status"] == "pass" and export_section["status"] == "pass":
        overall_status = "pass"
    elif dynamo_summary["status"] == "pass":
        overall_status = "partial_success"
    else:
        overall_status = "fail"

    capture_report = {
        "schema_version": "capture_report_v1",
        "stage_id": "graph_capture",
        "status": overall_status,
        "capture_api": [
            "compgen.capture.torch_export.capture_dynamo_partitions",
            "compgen.capture.torch_export.capture_frontend_artifact",
        ],
        "model_id": model_cfg.model_id,
        "target_id": target_cfg.target_id,
        "seed": model_cfg.seed,
        "primary_capture": "torch_dynamo",
        "canonical_capture": "torch_dynamo_partitions",
        "torch_dynamo": {
            "status": dynamo_summary["status"],
            "partition_count": dynamo_summary["partition_count"],
            "graph_break_count": dynamo_summary["graph_break_count"],
            "fullgraph": model_cfg.fullgraph,
            "error": dynamo_summary["error"],
        },
        "torch_export": export_section,
        "golden": golden_self_check,
        "diagnostics": {
            "graph_break_count": graph_breaks_obj["graph_break_count"],
            "guard_failures": graph_breaks_obj["guard_failures"],
            "warnings": graph_breaks_obj["warnings"],
        },
        "unsupported_preparation": unsupported_summary,
        "runtime_versions": runtime_versions,
        "llm_calls": 0,
    }
    capture_report_path = capture_dir / "capture_report.json"
    capture_report_path.write_text(
        json.dumps(capture_report, indent=2, sort_keys=True), encoding="utf-8"
    )

    finished_at = _utcnow()
    output_hash = sha256_tree(capture_dir)

    # graph_compilation artifact contract manifest status: pass if Dynamo got at least one partition
    # (the contract is {pass, fail, skipped}). partial_success in the report
    # still maps to "pass" at the manifest level — the capture stage as a
    # whole succeeded; the optional canonical artifact is what was missing.
    manifest_status = "pass" if dynamo_summary["status"] == "pass" else "fail"

    return StageRecord(
        stage_id="graph_capture",
        status=manifest_status,
        inputs=(),
        outputs=tuple(artifacts),
        report_path="00_graph_capture/capture_report.json",
        input_hash=input_hash,
        output_hash=output_hash,
        llm_calls=0,
        started_at_utc=started_at,
        finished_at_utc=finished_at,
    )


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #


def _utcnow() -> str:
    # Avoid importing datetime above as it's only used here.
    from datetime import datetime

    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# Keep dataclass/lint happy when this module is imported as part of __all__.
__all__ = [
    "ModelConfig",
    "TargetConfig",
    "run_graph_capture",
]
