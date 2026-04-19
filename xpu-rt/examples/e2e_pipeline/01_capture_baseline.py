"""Phase 1: capture + PyTorch-owned baselines for one workload.

For `--workload {smolvla_slice, gemma_decode_slice}` this script:
  1. Loads the workload model + deterministic sample_inputs (seed 0).
  2. Runs an eager baseline: times per-iteration wall-clock, peak memory.
  3. Tries `torch.compile` (inductor) and records the same metrics, plus
     the number of graph breaks and compiled-op fraction (via
     `compgen.capture.compile_baseline` + `collect_diagnostics`).
  4. Saves golden inputs/outputs and a manifest.

Artifacts written under
    user_perspective/artifacts/<workload>/stage_1_capture/
        golden_inputs.pt
        golden_outputs.pt
        compile_baseline.json
        graph_breaks.json
        manifest.json

This stage is entirely PyTorch-owned. No CompGen IR is produced here.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT.parent))          # import user_perspective.*
sys.path.insert(0, str(ROOT))                 # import models.*

from user_perspective.models import smolvla_slice, gemma_decode_slice  # noqa: E402
from compgen.capture import collect_diagnostics, compile_baseline       # noqa: E402

log = logging.getLogger("phase1")

WORKLOADS = {
    "smolvla_slice": lambda: smolvla_slice.load("auto"),
    "gemma_decode_slice": lambda: gemma_decode_slice.load(),
}


@dataclass
class EagerResult:
    median_ms: float
    p10_ms: float
    p90_ms: float
    iterations: int
    warmup: int
    peak_python_bytes: int
    peak_cuda_bytes: int | None


def _to_cpu(x: Any) -> Any:
    return x.detach().cpu() if isinstance(x, torch.Tensor) else x


def _flatten_output(out: Any) -> list[torch.Tensor]:
    if isinstance(out, torch.Tensor):
        return [out.detach().cpu()]
    if isinstance(out, (list, tuple)):
        flat: list[torch.Tensor] = []
        for item in out:
            flat.extend(_flatten_output(item))
        return flat
    if isinstance(out, dict):
        flat: list[torch.Tensor] = []
        for _, v in sorted(out.items()):
            flat.extend(_flatten_output(v))
        return flat
    # Unknown type; represent as empty so downstream scripts can still run.
    return []


def _eager_timings(model: torch.nn.Module, inputs: tuple[Any, ...],
                   warmup: int, iterations: int) -> EagerResult:
    model.eval()
    tracemalloc.start()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for _ in range(warmup):
            model(*inputs)

    ts: list[float] = []
    with torch.no_grad():
        for _ in range(iterations):
            t0 = time.perf_counter()
            model(*inputs)
            ts.append((time.perf_counter() - t0) * 1000.0)

    peak_cuda = None
    if torch.cuda.is_available():
        peak_cuda = int(torch.cuda.max_memory_allocated())
    _, peak_python = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    ts.sort()
    return EagerResult(
        median_ms=ts[len(ts) // 2],
        p10_ms=ts[max(0, int(0.1 * len(ts)) - 1)],
        p90_ms=ts[min(len(ts) - 1, int(0.9 * len(ts)))],
        iterations=iterations,
        warmup=warmup,
        peak_python_bytes=int(peak_python),
        peak_cuda_bytes=peak_cuda,
    )


def _serialize(obj: Any) -> Any:
    """JSON-safe serializer that coerces arbitrary objects to strings."""
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


def run(workload: str, iters: int = 10, warmup: int = 3) -> int:
    if workload not in WORKLOADS:
        raise SystemExit(f"unknown workload {workload!r}; pick from {sorted(WORKLOADS)}")

    out_dir = ROOT / "artifacts" / workload / "stage_1_capture"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- load ---
    log.info("loading workload=%s", workload)
    bundle = WORKLOADS[workload]()
    model = bundle.model
    inputs = tuple(_to_cpu(x) for x in bundle.sample_inputs)
    sample_shapes = [list(t.shape) for t in inputs if isinstance(t, torch.Tensor)]
    sample_dtypes = [str(t.dtype) for t in inputs if isinstance(t, torch.Tensor)]

    # --- golden I/O ---
    with torch.no_grad():
        golden_out = model(*inputs)
    golden_out_flat = _flatten_output(golden_out)

    torch.save(inputs, out_dir / "golden_inputs.pt")
    torch.save(golden_out, out_dir / "golden_outputs.pt")
    log.info("golden saved: %d input tensors, %d output tensors", len(inputs), len(golden_out_flat))

    # --- eager timings ---
    gc.collect()
    eager = _eager_timings(model, inputs, warmup=warmup, iterations=iters)
    log.info("eager median=%.3fms p10=%.3fms p90=%.3fms", eager.median_ms, eager.p10_ms, eager.p90_ms)

    # --- torch.compile baseline ---
    compile_report: dict[str, Any] | None = None
    compile_error: str | None = None
    try:
        gc.collect()
        rep = compile_baseline(model, inputs, backend="inductor",
                               num_warmup=warmup, num_runs=iters)
        compile_report = _serialize(rep)
        log.info("compile cold=%.1fms warm=%.3fms breaks=%d op_frac=%.2f",
                 rep.cold_compile_ms, rep.warm_run_ms, rep.num_graph_breaks,
                 rep.compiled_op_fraction)
    except Exception as exc:
        compile_error = f"{type(exc).__name__}: {exc}"
        log.warning("torch.compile failed: %s", compile_error)

    # --- dynamo diagnostics (graph breaks, guards) ---
    dyn_error: str | None = None
    try:
        gc.collect()
        report = collect_diagnostics(model, inputs)
        dyn_payload = _serialize({
            "graph_breaks": [{"location": loc, "reason": reason}
                             for (loc, reason) in report.graph_breaks],
            "graph_break_count": len(report.graph_breaks),
            "guard_observations": [g for g in report.guard_observations],
            "graph_count": report.graph_count,
            "op_count": report.op_count,
            "op_coverage_sample": dict(list(report.op_coverage.items())[:25]),
            "op_coverage_total": len(report.op_coverage),
            "warnings": report.warnings,
        })
    except Exception as exc:
        dyn_error = f"{type(exc).__name__}: {exc}"
        dyn_payload = {"error": dyn_error}
    (out_dir / "graph_breaks.json").write_text(
        json.dumps(dyn_payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    # --- write compile_baseline.json ---
    baseline_payload = {
        "workload": workload,
        "eager": asdict(eager),
        "torch_compile": compile_report,
        "torch_compile_error": compile_error,
        "sample_input_shapes": sample_shapes,
        "sample_input_dtypes": sample_dtypes,
    }
    (out_dir / "compile_baseline.json").write_text(
        json.dumps(baseline_payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    # --- manifest ---
    manifest = {
        "workload": workload,
        "phase": "1_capture_baseline",
        "model_source": bundle.source,
        "capture_mode": bundle.capture_mode,
        "notes": bundle.notes,
        "num_params": bundle.extra.get("param_count"),
        "num_cams": getattr(bundle, "num_cams", None),
        "sample_input_count": len(inputs),
        "sample_input_shapes": sample_shapes,
        "sample_input_dtypes": sample_dtypes,
        "output_count": len(golden_out_flat),
        "output_shapes": [list(t.shape) for t in golden_out_flat],
        "seed": 0,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "artifacts": {
            "golden_inputs": "golden_inputs.pt",
            "golden_outputs": "golden_outputs.pt",
            "compile_baseline": "compile_baseline.json",
            "graph_breaks": "graph_breaks.json",
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(f"\nPhase 1 complete for {workload}")
    print(f"  source={bundle.source}  capture_mode={bundle.capture_mode}  params={bundle.extra.get('param_count')}")
    print(f"  eager median={eager.median_ms:.3f}ms  peak_python={eager.peak_python_bytes/1e6:.1f} MB")
    if compile_report:
        print(f"  torch.compile warm={compile_report['warm_run_ms']:.3f}ms  breaks={compile_report['num_graph_breaks']}")
    else:
        print(f"  torch.compile: FAILED ({compile_error})")
    print(f"  artifacts under {out_dir.relative_to(ROOT)}/")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--workload", required=True, choices=sorted(WORKLOADS.keys()))
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return run(args.workload, iters=args.iters, warmup=args.warmup)


if __name__ == "__main__":
    raise SystemExit(main())
