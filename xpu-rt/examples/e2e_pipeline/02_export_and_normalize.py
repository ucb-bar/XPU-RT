"""Phase 2: export + normalize. The PyTorch → CompGen boundary.

For `--workload` this script:
  1. Loads the workload (same loader as Phase 1).
  2. Calls `compgen.capture.capture_frontend_artifact(model, sample_inputs)`
     to produce the canonical CaptureArtifact.
  3. Saves the ExportedProgram to `exported_program.pt2` (the repo does not
     auto-save it).
  4. Calls `compgen.ir.payload.import_fx.fx_to_xdsl(ep, **strict_import_options)`
     to produce Payload IR; writes `payload.mlir` (via the importer's text
     printer) and `import_diagnostics.json`.
  5. Emits the summary artifacts: `exported_program_summary.json`,
     `shape_constraints.json`, `normalized_ops.json`.
  6. Emits `boundary_manifest.json` listing what crossed from PyTorch into
     CompGen and what was dropped.

Artifacts under
    user_perspective/artifacts/<workload>/stage_2_boundary/

This stage is where we **exit PyTorch**. After it, everything is
CompGen-owned xDSL IR.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

from user_perspective.models import smolvla_slice, gemma_decode_slice   # noqa: E402
from compgen.capture import capture_frontend_artifact                    # noqa: E402
from compgen.ir.payload.import_fx import FXImporter, fx_to_xdsl          # noqa: E402

log = logging.getLogger("phase2")

WORKLOADS = {
    "smolvla_slice": lambda: smolvla_slice.load("auto"),
    "gemma_decode_slice": lambda: gemma_decode_slice.load(),
}


def _serialize(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if is_dataclass(obj):
        return {k: _serialize(v) for k, v in asdict(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _serialize(v) for k, v in obj.items()}
    if isinstance(obj, set):
        return sorted(_serialize(x) for x in obj)
    return repr(obj)


def _op_inventory(ep: Any) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for node in ep.graph.nodes:
        if node.op == "call_function":
            counts[str(node.target)] += 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _input_output_spec(ep: Any) -> dict[str, Any]:
    inputs: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    for node in ep.graph.nodes:
        if node.op == "placeholder":
            val = node.meta.get("val")
            shape = list(val.shape) if hasattr(val, "shape") else None
            dtype = str(val.dtype) if hasattr(val, "dtype") else None
            inputs.append({"name": node.name, "shape": shape, "dtype": dtype})
        elif node.op == "output":
            for arg in node.args:
                for item in (arg if isinstance(arg, (list, tuple)) else [arg]):
                    meta = getattr(item, "meta", {}) if item is not None else {}
                    val = meta.get("val") if meta else None
                    outputs.append({
                        "producer": str(item) if item is not None else None,
                        "shape": list(val.shape) if hasattr(val, "shape") else None,
                        "dtype": str(val.dtype) if hasattr(val, "dtype") else None,
                    })
    return {"inputs": inputs, "outputs": outputs}


def run(workload: str) -> int:
    if workload not in WORKLOADS:
        raise SystemExit(f"unknown workload {workload!r}; pick from {sorted(WORKLOADS)}")

    out_dir = ROOT / "artifacts" / workload / "stage_2_boundary"
    out_dir.mkdir(parents=True, exist_ok=True)

    phase1_dir = ROOT / "artifacts" / workload / "stage_1_capture"
    model_source = None
    capture_mode = None
    if (phase1_dir / "manifest.json").exists():
        prior = json.loads((phase1_dir / "manifest.json").read_text())
        model_source = prior.get("model_source")
        capture_mode = prior.get("capture_mode")

    log.info("loading workload=%s", workload)
    bundle = WORKLOADS[workload]()
    model = bundle.model
    inputs = tuple(bundle.sample_inputs)

    # Guard: SmolVLA real path uses Dynamo partitioning, not torch.export.
    # If we're on the real SmolVLA source, torch.export will fail; we record
    # that and stop for this workload. The miniature fallback exports cleanly.
    if bundle.capture_mode != "torch_export":
        msg = (
            f"capture_mode={bundle.capture_mode!r} — this Phase 2 script targets "
            "torch.export. Rerun with a workload whose bundle reports torch_export, "
            "or extend 02_export_and_normalize.py to drive capture_dynamo_partitions."
        )
        (out_dir / "boundary_manifest.json").write_text(
            json.dumps({
                "workload": workload,
                "status": "skipped",
                "reason": msg,
                "model_source": bundle.source,
                "capture_mode": bundle.capture_mode,
            }, indent=2), encoding="utf-8",
        )
        log.error("%s", msg)
        return 2

    # --- capture ---
    log.info("capture_frontend_artifact(...)")
    artifact = capture_frontend_artifact(model, inputs)

    ep = artifact.exported_program
    assert ep is not None, "capture_frontend_artifact did not produce an ExportedProgram"

    # --- save exported program ---
    ep_path = out_dir / "exported_program.pt2"
    try:
        torch.export.save(ep, str(ep_path))
        ep_saved = True
        ep_save_error = None
    except Exception as exc:
        ep_saved = False
        ep_save_error = f"{type(exc).__name__}: {exc}"
        log.warning("torch.export.save failed: %s", ep_save_error)

    # --- exported program summary ---
    op_counts = _op_inventory(ep)
    io_spec = _input_output_spec(ep)
    shape_constraints = {
        "range_constraints": [
            {"symbol": rc.symbol, "minimum": rc.minimum, "maximum": rc.maximum}
            for rc in artifact.range_constraints
        ],
        "graph_signature": artifact.graph_signature,
        "module_call_graph": list(artifact.module_call_graph),
    }
    ep_summary = {
        "workload": workload,
        "validation": asdict(artifact.validation),
        "graph_break_count": artifact.graph_break_count,
        "analysis_success": artifact.analysis_success,
        "decomposition_targets_count": len(artifact.decomposition_targets),
        "decomposition_targets_sample": list(artifact.decomposition_targets[:20]),
        "num_ops_call_function": sum(op_counts.values()),
        "num_unique_ops": len(op_counts),
        "io": io_spec,
        "runtime_versions": dict(artifact.runtime_versions),
        "explicit_blackboxes": list(artifact.explicit_blackboxes),
        "unsupported_resolutions_count": len(artifact.unsupported_resolutions),
        "synthesized_payload_translations": sorted(artifact.synthesized_payload_translations.keys()),
    }
    (out_dir / "exported_program_summary.json").write_text(
        json.dumps(_serialize(ep_summary), indent=2, sort_keys=True), encoding="utf-8",
    )
    (out_dir / "shape_constraints.json").write_text(
        json.dumps(_serialize(shape_constraints), indent=2, sort_keys=True), encoding="utf-8",
    )
    (out_dir / "normalized_ops.json").write_text(
        json.dumps(op_counts, indent=2, sort_keys=True), encoding="utf-8",
    )
    log.info("exported program: ops=%d unique=%d graph_breaks=%d",
             ep_summary["num_ops_call_function"], ep_summary["num_unique_ops"],
             ep_summary["graph_break_count"])

    # --- fx_to_xdsl: the actual boundary ---
    log.info("fx_to_xdsl(...) — exiting PyTorch")
    strict = artifact.strict_import_options()
    importer = FXImporter(
        allow_opaque_fallback=strict["allow_opaque_fallback"],
        explicit_blackboxes=strict["explicit_blackboxes"],
        dynamic_decompositions=strict["dynamic_decompositions"],
    )
    module = importer.import_graph(ep)
    diagnostics = importer.diagnostics
    ir_text = importer.get_ir_text(module)

    (out_dir / "payload.mlir").write_text(ir_text, encoding="utf-8")
    (out_dir / "import_diagnostics.json").write_text(
        json.dumps(_serialize([d for d in diagnostics]), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    log.info("payload.mlir written (%d chars, %d diagnostics)", len(ir_text), len(diagnostics))

    # Quick MLIR-level inventory (regions, linalg.* ops)
    ir_stats = {
        "payload_mlir_bytes": len(ir_text.encode("utf-8")),
        "occurrences": {
            "linalg.matmul": ir_text.count("linalg.matmul"),
            "linalg.transpose": ir_text.count("linalg.transpose"),
            "linalg.generic": ir_text.count("linalg.generic"),
            "linalg.reduce": ir_text.count("linalg.reduce"),
            "arith.addf": ir_text.count("arith.addf"),
            "arith.mulf": ir_text.count("arith.mulf"),
            "tensor.insert": ir_text.count("tensor.insert"),
            "tensor.extract": ir_text.count("tensor.extract"),
            "func.call": ir_text.count("func.call"),
            "compgen.region_id": ir_text.count("compgen.region_id"),
        },
    }

    # --- boundary manifest ---
    boundary_manifest = {
        "workload": workload,
        "phase": "2_export_and_normalize",
        "status": "ok",
        "model_source": bundle.source,
        "capture_mode": bundle.capture_mode,
        "phase1_model_source": model_source,
        "phase1_capture_mode": capture_mode,
        "seed": 0,
        "exported_program_saved": ep_saved,
        "exported_program_save_error": ep_save_error,
        "exported_program_path": "exported_program.pt2" if ep_saved else None,
        "crossed_into_compgen": {
            "graph_topology": True,
            "input_output_shapes": True,
            "dtypes": True,
            "decomposition_provenance": True,
            "region_id_attributes": ir_stats["occurrences"]["compgen.region_id"] > 0,
            "range_constraints": bool(artifact.range_constraints),
            "explicit_blackboxes": bool(artifact.explicit_blackboxes),
        },
        "dropped_at_boundary": {
            "python_control_flow": True,
            "torch_autograd_graph": True,
            "nn_module_hierarchy": True,
            "torch_dynamo_guards": True,
        },
        "import_diagnostic_levels": dict(Counter(d.level for d in diagnostics)),
        "ir_stats": ir_stats,
        "artifacts": {
            "exported_program": "exported_program.pt2",
            "exported_program_summary": "exported_program_summary.json",
            "shape_constraints": "shape_constraints.json",
            "normalized_ops": "normalized_ops.json",
            "payload_mlir": "payload.mlir",
            "import_diagnostics": "import_diagnostics.json",
        },
    }
    (out_dir / "boundary_manifest.json").write_text(
        json.dumps(_serialize(boundary_manifest), indent=2, sort_keys=True), encoding="utf-8",
    )

    print(f"\nPhase 2 complete for {workload}")
    print(f"  ops in exported_program : {ep_summary['num_ops_call_function']} "
          f"(unique: {ep_summary['num_unique_ops']})")
    print(f"  diagnostics on import   : {boundary_manifest['import_diagnostic_levels']}")
    print(f"  payload.mlir size       : {ir_stats['payload_mlir_bytes']:,} bytes")
    print(f"  region_id attrs         : {ir_stats['occurrences']['compgen.region_id']}")
    print(f"  func.call (opaque) count: {ir_stats['occurrences']['func.call']}")
    print(f"  artifacts under {out_dir.relative_to(ROOT)}/")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--workload", required=True, choices=sorted(WORKLOADS.keys()))
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return run(args.workload)


if __name__ == "__main__":
    raise SystemExit(main())
