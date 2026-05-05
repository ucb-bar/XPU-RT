"""Stress-test runner for the M-19 → M-22.1 kernel-evidence pipeline.

Runs CompGen end-to-end on a configurable model list with kernels ON,
then aggregates a per-model summary that exposes whether the analytical
predictions, post-hoc utilization derivations, and real profiler
measurements all line up.

This is a STRESS TEST — its job is to surface where the system breaks
down honestly, not to claim everything works. The output table makes
it easy to spot:

- Models that didn't even reach M-19 (capture / lowering / strict-gate
  blocked them).
- Models that have M-19/M-20 measurements but NO M-21 cross-reference
  (analytical model skipped them).
- Models where M-22's measured bottleneck disagrees with M-21's
  analytical bottleneck (calibration insight).
- Models where M-22.1 surfaces real CUDA kernel time vs the M-19
  cuda.Event launch-overhead-inflated number.

Usage:
    .venv/bin/python scripts/dev/run_kernel_stress_suite.py \
        --models tiny_mlp,tiny_attention,tiny_conv_block,merlin_mlp_wide,\\
                 proxy_vla,proxy_vlm,custom_unsupported_op \
        --target configs/targets/host_cpu.yaml \
        --out /tmp/kernel_stress

The script always uses ``COMPGEN_RUN_KERNELS=1``. It does NOT enable
``COMPGEN_CALIBRATE_PROFILER`` (M-18) or
``COMPGEN_CALIBRATE_CANDIDATES`` (M-18.3) — those are independent and
add noise to the stress-test signal we care about (real compiled
kernel evidence).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_json_or_none(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _run_one(
    model_id: str, target_path: Path, out_root: Path,
) -> dict[str, Any]:
    """Run the pipeline on one model with kernels ON. Returns a
    per-model summary dict the aggregator consumes."""
    model_yaml = REPO_ROOT / "configs" / "models" / f"{model_id}.yaml"
    if not model_yaml.exists():
        return {
            "model_id": model_id, "status": "model_yaml_missing",
            "model_yaml_tried": str(model_yaml),
        }
    out_dir = out_root / model_id
    if out_dir.exists():
        # subprocess will refuse to overwrite — clean it.
        import shutil
        shutil.rmtree(out_dir)

    env = os.environ.copy()
    env["COMPGEN_RUN_KERNELS"] = "1"
    env.pop("COMPGEN_CALIBRATE_PROFILER", None)
    env.pop("COMPGEN_CALIBRATE_CANDIDATES", None)

    proc = subprocess.run(
        [
            sys.executable, "-m", "compgen.graph_compilation", "run",
            "--model", str(model_yaml),
            "--target", str(target_path),
            "--out", str(out_dir),
            "--stop-after", "agent-decision-request",
            "--selection-mode", "greedy",
        ],
        cwd=REPO_ROOT, capture_output=True, text=True, env=env,
        timeout=600,
    )

    summary: dict[str, Any] = {
        "model_id": model_id,
        "out_dir": str(out_dir),
        "pipeline_returncode": proc.returncode,
    }

    # Capture stage statuses.
    cap = _read_json_or_none(
        out_dir / "00_graph_capture" / "capture_report.json"
    )
    summary["capture_status"] = (
        (cap or {}).get("status") or "missing"
    )

    lowering = _read_json_or_none(
        out_dir / "01_payload_lowering" / "lowering_summary.json"
    )
    summary["lowering_status"] = (
        (lowering or {}).get("status") or "missing"
    )

    sg = _read_json_or_none(
        out_dir / "01_payload_lowering" / f"{model_id}_strict_gate_report.json"
    )
    summary["strict_gate_status"] = (sg or {}).get("status") or "missing"
    summary["strict_gate_root_cause"] = (
        ((sg or {}).get("root_cause") or {}).get("category") or "none"
    )

    # M-21 analytical (always-on).
    ac = _read_json_or_none(
        out_dir / "02_graph_analysis" / "analytical_cost"
        / "per_candidate_analytical_cost.json"
    )
    summary["m21_overall"] = (ac or {}).get("overall") or "missing"
    summary["m21_candidates_modeled"] = (
        ((ac or {}).get("summary") or {}).get("candidates_modeled") or 0
    )
    summary["m21_candidates_total"] = (
        ((ac or {}).get("summary") or {}).get("candidates_total") or 0
    )

    # M-20 region compiled differential.
    m20 = _read_json_or_none(
        out_dir / "02_graph_analysis" / "kernel_execution"
        / "region_compiled_differential_report.json"
    )
    if m20 is not None:
        summary["m20_status"] = m20.get("status") or m20.get("overall") or "?"
        summary["m20_regions"] = len(m20.get("regions", []) or [])
        gpu_compiled = sum(
            1 for r in m20.get("regions", []) or []
            if (r.get("gpu") or {}).get("compile_status") == "compiled"
        )
        cpu_compiled = sum(
            1 for r in m20.get("regions", []) or []
            if (r.get("cpu") or {}).get("compile_status") == "compiled"
        )
        summary["m20_gpu_compiled"] = gpu_compiled
        summary["m20_cpu_compiled"] = cpu_compiled
        gpu_uss = [
            float((r.get("gpu") or {}).get("measured_us_per_iter") or 0)
            for r in m20.get("regions", []) or []
            if (r.get("gpu") or {}).get("compile_status") == "compiled"
        ]
        summary["m20_gpu_mean_us"] = (
            sum(gpu_uss) / len(gpu_uss) if gpu_uss else None
        )
    else:
        summary["m20_status"] = "missing"
        summary["m20_regions"] = 0
        summary["m20_gpu_compiled"] = 0
        summary["m20_cpu_compiled"] = 0
        summary["m20_gpu_mean_us"] = None

    # M-22 compiled bottleneck.
    cb = _read_json_or_none(
        out_dir / "02_graph_analysis" / "compiled_bottleneck"
        / "compiled_bottleneck_report.json"
    )
    if cb is not None:
        summary["m22_overall"] = cb.get("overall") or "?"
        summary["m22_kernel_calibration_status"] = (
            cb.get("kernel_calibration_status") or "?"
        )
        s = cb.get("summary", {}) or {}
        summary["m22_regions_with_evidence"] = s.get(
            "regions_with_evidence") or 0
        summary["m22_non_opaque_total"] = s.get(
            "non_opaque_total_regions") or 0
        agree = s.get("agreement_with_analytical", {}) or {}
        summary["m22_agreement_count"] = agree.get(
            "agreement_count") or 0
        summary["m22_disagreement_count"] = agree.get(
            "disagreement_count") or 0
    else:
        summary["m22_overall"] = "missing"
        summary["m22_kernel_calibration_status"] = "missing"
        summary["m22_regions_with_evidence"] = 0
        summary["m22_non_opaque_total"] = 0
        summary["m22_agreement_count"] = 0
        summary["m22_disagreement_count"] = 0

    # M-22.1 profiler evidence.
    pe = _read_json_or_none(
        out_dir / "02_graph_analysis" / "profiler_evidence"
        / "profiler_evidence_report.json"
    )
    if pe is not None:
        summary["m22_1_overall"] = pe.get("overall") or "?"
        s = pe.get("summary", {}) or {}
        summary["m22_1_gpu_collected"] = s.get(
            "gpu_collected_count") or 0
        summary["m22_1_cpu_collected"] = s.get(
            "cpu_collected_count") or 0
        summary["m22_1_region_count"] = s.get("region_count") or 0
        # Real per-region GPU stats.
        gpu_uss = []
        for r in pe.get("regions", []) or []:
            gpu = r.get("gpu") or {}
            us = gpu.get("self_cuda_us_per_iter")
            if us is not None and float(us) > 0:
                gpu_uss.append(float(us))
        summary["m22_1_gpu_mean_self_us"] = (
            sum(gpu_uss) / len(gpu_uss) if gpu_uss else None
        )
        # Launch-overhead ratio: M-19 cuda.Event time vs profiler self_cuda time.
        if (summary.get("m20_gpu_mean_us") is not None
                and summary["m22_1_gpu_mean_self_us"] is not None
                and summary["m22_1_gpu_mean_self_us"] > 0):
            summary["launch_overhead_ratio"] = (
                summary["m20_gpu_mean_us"]
                / summary["m22_1_gpu_mean_self_us"]
            )
        else:
            summary["launch_overhead_ratio"] = None
        summary["m22_1_perf_available"] = (
            pe.get("perf_availability", {}) or {}).get("available", False)
    else:
        summary["m22_1_overall"] = "missing"
        summary["m22_1_gpu_collected"] = 0
        summary["m22_1_cpu_collected"] = 0
        summary["m22_1_region_count"] = 0
        summary["m22_1_gpu_mean_self_us"] = None
        summary["launch_overhead_ratio"] = None
        summary["m22_1_perf_available"] = False

    # M-23 compiled fusion.
    cf = _read_json_or_none(
        out_dir / "02_graph_analysis" / "compiled_fusion"
        / "compiled_fusion_differential_report.json"
    )
    if cf is not None:
        summary["m23_overall"] = cf.get("overall") or "?"
        s = cf.get("summary", {}) or {}
        summary["m23_bit_eq"] = s.get("bit_equality_count") or 0
        summary["m23_tol"] = s.get("tolerance_eps_count") or 0
        summary["m23_fail"] = s.get("fail_outside_tolerance_count") or 0
        summary["m23_gpu_compiled"] = s.get("gpu_compiled") or False
        summary["m23_cpu_compiled"] = s.get("cpu_compiled") or False
    else:
        summary["m23_overall"] = "missing"
        summary["m23_bit_eq"] = 0
        summary["m23_tol"] = 0
        summary["m23_fail"] = 0
        summary["m23_gpu_compiled"] = False
        summary["m23_cpu_compiled"] = False

    # Deep audit.
    try:
        from audit_kernel_pipeline import audit_run_dir
    except ImportError:
        # Add scripts/dev to path then retry.
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent))
        from audit_kernel_pipeline import audit_run_dir
    audit = audit_run_dir(out_dir)
    summary["audit_overall"] = audit["overall"]
    summary["audit_fail_count"] = audit["fail_count"]
    summary["audit_failed_checks"] = sorted(
        name for name, c in audit["checks"].items()
        if c.get("status") == "fail"
    )
    summary["audit_full"] = audit["checks"]

    return summary


def _write_aggregate(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "schema_version": "kernel_stress_suite_v1",
        "model_count": len(rows),
        "models_with_m20_compiled": sum(
            1 for r in rows if (r.get("m20_gpu_compiled") or 0) > 0
            or (r.get("m20_cpu_compiled") or 0) > 0
        ),
        "models_with_m22_evidence": sum(
            1 for r in rows
            if (r.get("m22_regions_with_evidence") or 0) > 0
        ),
        "models_with_m22_1_gpu": sum(
            1 for r in rows if (r.get("m22_1_gpu_collected") or 0) > 0
        ),
        "models_with_m22_disagreement": sum(
            1 for r in rows if (r.get("m22_disagreement_count") or 0) > 0
        ),
        "rows": rows,
    }
    out_path.write_text(
        json.dumps(body, indent=2, sort_keys=True), encoding="utf-8",
    )


def _print_table(rows: list[dict[str, Any]]) -> None:
    cols = [
        ("model_id", 22),
        ("capture", 8),
        ("lower", 14),
        ("strict", 8),
        ("m21", 6),
        ("m20_g", 6),
        ("m20_c", 6),
        ("m22", 11),
        ("ag/dis", 8),
        ("m22.1_g", 8),
        ("m23", 8),
        ("m23_eq", 7),
        ("audit", 12),
    ]
    header = " ".join(f"{c[0]:<{c[1]}}" for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        m23_overall = r.get("m23_overall", "?")
        audit_str = (
            r.get("audit_overall", "?") + (
                f"({r.get('audit_fail_count', 0)})"
                if r.get("audit_fail_count", 0) > 0 else ""
            )
        )
        cells = [
            (r["model_id"][:21] or "?", 22),
            (r.get("capture_status", "?")[:7], 8),
            (r.get("lowering_status", "?")[:13], 14),
            (r.get("strict_gate_status", "?")[:7], 8),
            (
                f"{r.get('m21_candidates_modeled', 0)}/"
                f"{r.get('m21_candidates_total', 0)}",
                6,
            ),
            (str(r.get("m20_gpu_compiled", 0)), 6),
            (str(r.get("m20_cpu_compiled", 0)), 6),
            (r.get("m22_kernel_calibration_status", "?")[:10], 11),
            (
                f"{r.get('m22_agreement_count', 0)}/"
                f"{r.get('m22_disagreement_count', 0)}",
                8,
            ),
            (str(r.get("m22_1_gpu_collected", 0)), 8),
            (m23_overall[:7], 8),
            (
                f"{r.get('m23_bit_eq', 0)}/{r.get('m23_fail', 0)}"
                if m23_overall == "pass" else "—",
                7,
            ),
            (audit_str[:11], 12),
        ]
        print(" ".join(f"{c[0]:<{c[1]}}" for c in cells))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", default=(
            "tiny_mlp,tiny_attention,tiny_conv_block,"
            "merlin_mlp_wide,proxy_vla,proxy_vlm,"
            "custom_unsupported_op,graph_break_mlp"
        ),
        help="comma-separated list of model IDs",
    )
    parser.add_argument(
        "--target",
        default=str(REPO_ROOT / "configs" / "targets" / "host_cpu.yaml"),
    )
    parser.add_argument(
        "--out", default="/tmp/kernel_stress",
        help="output root for per-model run dirs + aggregate",
    )
    args = parser.parse_args()
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    rows: list[dict[str, Any]] = []
    print(f"running {len(models)} models with COMPGEN_RUN_KERNELS=1")
    for m in models:
        print(f"  → {m}")
        try:
            row = _run_one(m, Path(args.target), out_root)
        except subprocess.TimeoutExpired:
            row = {"model_id": m, "status": "timeout"}
        except Exception as exc:  # noqa: BLE001
            row = {"model_id": m, "status": "exception",
                   "exception": f"{type(exc).__name__}: {exc}"}
        rows.append(row)

    aggregate_path = out_root / "kernel_stress_summary.json"
    _write_aggregate(rows, aggregate_path)
    print(f"\naggregate written: {aggregate_path}")
    print()
    _print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
