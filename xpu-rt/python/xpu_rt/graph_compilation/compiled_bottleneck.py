"""M-22 Compiled Bottleneck Analysis.

Per-region MEASURED bottleneck classification. Cross-references M-19/M-20
real compiled-kernel measurements with M-21's analytical flops/bytes
predictions to produce per-region:

- ``achieved_compute_gflops``    = flops / measured_time_s
- ``achieved_bandwidth_gb_s``    = bytes_moved / measured_time_s
- ``compute_utilization``        = achieved_compute_gflops / peak_compute_gflops
- ``bandwidth_utilization``      = achieved_bandwidth_gb_s / peak_bandwidth_gb_s
- ``measured_bottleneck``        = whichever utilization is higher

These are deterministic post-hoc derivations: same M-19/M-20 + M-21 +
target YAML inputs produce byte-identical output across reruns.

The measured bottleneck is then compared with M-21's analytical
``bottleneck_resource`` prediction; agreements / disagreements are
counted, and per-region disagreements are surfaced explicitly so the
agent (and the paper) can see where the analytical model was wrong.

M-22 layers an additive ``compiled_evidence`` block per region onto
``02_graph_analysis/readiness/hardware_resource_report.json`` (same
pattern M-21 uses on cost_preview_v2). M-17.1's
``calibration_status="not_profiler_calibrated"`` field is deliberately
left untouched (it documents the deterministic baseline). A NEW
top-level field ``kernel_calibration_status`` carries the M-22 verdict:

- ``not_kernel_calibrated``     — no compiled measurements available
- ``partial_kernel_calibration`` — some regions have evidence
- ``kernel_calibrated``         — every non-opaque region has evidence

Hard non-goals:

- No new measurement machinery (M-22.1 follow-up may add torch.profiler
  CUDA activities + linux perf for cache fractions; this MVP derives
  utilization from M-19/M-20's measured time + M-21's analytical
  flops/bytes).
- No compiler-core imports.
- Cache fractions / occupancy / launch-overhead breakdown are explicitly
  ``not_collected`` (cache_evidence: not_collected).
- region_map / candidate_actions / cost_preview_v2 / llm_graph_view /
  M-21 analytical_cost / M-19 / M-20 reports stay byte-identical.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _utcnow() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# --------------------------------------------------------------------------- #
# Pure-function utilization derivation
# --------------------------------------------------------------------------- #


def derive_utilization(
    *,
    flops: int,
    bytes_moved: int,
    measured_us: float,
    peak_compute_gflops: float,
    peak_bandwidth_gb_s: float,
) -> dict[str, Any]:
    """Pure function: from analytical flops/bytes + measured time +
    target peaks, derive achieved compute and bandwidth utilization.
    Same inputs → byte-identical output.

    Returns ``measured_bottleneck`` = "compute" if compute_utilization
    >= bandwidth_utilization else "memory". Empty / non-positive
    inputs return None for the affected fields and ``unknown`` for
    the bottleneck."""
    if measured_us <= 0:
        return {
            "achieved_compute_gflops": None,
            "achieved_bandwidth_gb_s": None,
            "compute_utilization": None,
            "bandwidth_utilization": None,
            "measured_bottleneck": "unknown",
            "skip_reason": "non_positive_measured_time",
        }
    measured_s = measured_us * 1e-6

    achieved_compute_gflops = (flops / 1e9) / measured_s if flops > 0 else 0.0
    achieved_bw_gb_s = (
        (bytes_moved / 1e9) / measured_s if bytes_moved > 0 else 0.0
    )

    compute_util = (
        achieved_compute_gflops / peak_compute_gflops
        if peak_compute_gflops > 0 else None
    )
    bw_util = (
        achieved_bw_gb_s / peak_bandwidth_gb_s
        if peak_bandwidth_gb_s > 0 else None
    )

    # Classify bottleneck by whichever utilization is closer to peak.
    if compute_util is None and bw_util is None:
        bn = "unknown"
    elif compute_util is None:
        bn = "memory"
    elif bw_util is None:
        bn = "compute"
    elif compute_util >= bw_util:
        bn = "compute"
    else:
        bn = "memory"

    return {
        "achieved_compute_gflops": achieved_compute_gflops,
        "achieved_bandwidth_gb_s": achieved_bw_gb_s,
        "compute_utilization": compute_util,
        "bandwidth_utilization": bw_util,
        "measured_bottleneck": bn,
    }


# --------------------------------------------------------------------------- #
# M-19/M-20 measurement loader
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _CompiledMeasurement:
    region_id: str
    candidate_id: str
    matmul_shape: tuple[int, int, int]
    tile: tuple[int, int, int]
    gpu_us: float | None
    cpu_us: float | None
    source: str  # "m19" | "m20"


def _load_compiled_measurements(
    run_dir: Path,
) -> list[_CompiledMeasurement]:
    """Best-effort: load M-19 single-region run + M-20 per-region fan-out.
    Per region, M-20 wins (it's the more recent / more complete view).
    Returns one measurement per region with at least one compiled track."""
    base = run_dir / "02_graph_analysis" / "kernel_execution"
    if not base.is_dir():
        return []

    by_region: dict[str, _CompiledMeasurement] = {}

    # M-19 single-region selected candidate.
    m19_gpu = _read_json(base / "compiled_kernel_run_gpu.json")
    m19_cpu = _read_json(base / "compiled_kernel_run_cpu.json")
    if m19_gpu and m19_gpu.get("compile_status") == "compiled":
        rid = m19_gpu.get("region_id") or ""
        if rid:
            sh = m19_gpu.get("matmul_shape") or {}
            t = m19_gpu.get("tile") or {}
            cpu_us = None
            if (m19_cpu
                    and m19_cpu.get("compile_status") == "compiled"
                    and m19_cpu.get("region_id") == rid):
                cpu_us = m19_cpu.get("measured_us_per_iter")
            by_region[rid] = _CompiledMeasurement(
                region_id=rid,
                candidate_id=str(m19_gpu.get("candidate_id") or ""),
                matmul_shape=(int(sh.get("M") or 0),
                              int(sh.get("N") or 0),
                              int(sh.get("K") or 0)),
                tile=(int(t.get("M") or 0),
                      int(t.get("N") or 0),
                      int(t.get("K") or 0)),
                gpu_us=float(m19_gpu.get("measured_us_per_iter") or 0.0)
                       if m19_gpu.get("measured_us_per_iter") is not None
                       else None,
                cpu_us=float(cpu_us) if cpu_us is not None else None,
                source="m19",
            )

    # M-20 per-region fan-out (overrides M-19 since it covers all regions).
    m20 = _read_json(base / "region_compiled_differential_report.json")
    if m20 is not None:
        for r in m20.get("regions", []) or []:
            rid = r.get("region_id") or ""
            if not rid:
                continue
            sh = r.get("matmul_shape") or {}
            t = r.get("tile") or {}
            gpu = r.get("gpu") or {}
            cpu = r.get("cpu") or {}
            gpu_us = (
                float(gpu.get("measured_us_per_iter") or 0.0)
                if (gpu.get("compile_status") == "compiled"
                    and gpu.get("measured_us_per_iter") is not None)
                else None
            )
            cpu_us = (
                float(cpu.get("measured_us_per_iter") or 0.0)
                if (cpu.get("compile_status") == "compiled"
                    and cpu.get("measured_us_per_iter") is not None)
                else None
            )
            if gpu_us is None and cpu_us is None:
                continue
            by_region[rid] = _CompiledMeasurement(
                region_id=rid,
                candidate_id=str(r.get("candidate_id") or ""),
                matmul_shape=(int(sh.get("M") or 0),
                              int(sh.get("N") or 0),
                              int(sh.get("K") or 0)),
                tile=(int(t.get("M") or 0),
                      int(t.get("N") or 0),
                      int(t.get("K") or 0)),
                gpu_us=gpu_us, cpu_us=cpu_us,
                source="m20",
            )

    return list(by_region.values())


# --------------------------------------------------------------------------- #
# M-21 analytical lookup
# --------------------------------------------------------------------------- #


def _index_m21_by_candidate(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Index M-21 per-candidate analytical entries by candidate_id.
    Only ok-modeled entries are kept (we need flops/bytes/predicted)."""
    p = (
        run_dir / "02_graph_analysis" / "analytical_cost"
        / "per_candidate_analytical_cost.json"
    )
    doc = _read_json(p)
    if doc is None:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for c in doc.get("candidates", []) or []:
        if c.get("model_status") != "ok":
            continue
        cid = c.get("candidate_id")
        if cid:
            out[str(cid)] = c
    return out


# --------------------------------------------------------------------------- #
# Target-spec loader (mirrors analytical_cost.py)
# --------------------------------------------------------------------------- #


def _load_target_yaml(target_id: str, repo_root: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        return {}
    candidates = [
        repo_root / "configs" / "targets" / f"{target_id}.yaml",
        repo_root / "configs" / "targets" / "host_cpu.yaml",
    ]
    for c in candidates:
        if c.exists():
            try:
                return yaml.safe_load(c.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError:
                continue
    return {}


# --------------------------------------------------------------------------- #
# hardware_resource_report overlay
# --------------------------------------------------------------------------- #


def _apply_hardware_resource_overlay(
    *,
    run_dir: Path,
    evidence_by_region: dict[str, dict[str, Any]],
    kernel_calibration_status: str,
) -> None:
    """Layer ``compiled_evidence`` per region onto
    hardware_resource_report.json AND add a top-level
    ``kernel_calibration_status`` field. Additive only — M-17.1's
    existing fields stay untouched."""
    p = (
        run_dir / "02_graph_analysis" / "readiness"
        / "hardware_resource_report.json"
    )
    if not p.exists():
        return
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return

    for region in doc.get("regions", []) or []:
        rid = region.get("region_id") or ""
        ev = evidence_by_region.get(rid)
        if ev is not None:
            region["compiled_evidence"] = ev

    doc["kernel_calibration_status"] = kernel_calibration_status

    p.write_text(
        json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# Top-level entry point
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CompiledBottleneckResult:
    overall: str          # "ok" | "no_measurements" | "not_run"
    out_dir: Path
    report_path: Path
    summary_md_path: Path
    region_count_with_evidence: int
    region_count_total: int
    kernel_calibration_status: str


def run_compiled_bottleneck(
    run_dir: Path, *, repo_root: Path | None = None,
) -> CompiledBottleneckResult:
    """Build the M-22 deterministic compiled-bottleneck analysis.
    Best-effort; never raises."""
    run_dir = Path(run_dir).resolve()
    repo_root = repo_root or Path(__file__).resolve().parents[3]

    ga = run_dir / "02_graph_analysis"
    out_dir = ga / "compiled_bottleneck"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "compiled_bottleneck_report.json"
    summary_md_path = out_dir / "compiled_bottleneck_summary.md"

    measurements = _load_compiled_measurements(run_dir)
    m21_by_cid = _index_m21_by_candidate(run_dir)

    cap = _read_json(run_dir / "00_graph_capture" / "capture_report.json")
    target_id = (cap or {}).get("target_id", "host_cpu")
    target = _load_target_yaml(target_id, repo_root)
    peak_compute = float(target.get("peak_compute_gflops", 0.0) or 0.0)
    peak_bw = float(target.get("peak_bandwidth_gb_s", 0.0) or 0.0)

    # Read M-17.1 hardware_resource_report to count total non-opaque regions
    # for the calibration-coverage classification.
    hrr = _read_json(
        ga / "readiness" / "hardware_resource_report.json"
    )
    non_opaque_total = 0
    if hrr is not None:
        for r in hrr.get("regions", []) or []:
            if not r.get("is_opaque"):
                non_opaque_total += 1

    if not measurements:
        body = {
            "schema_version": "compiled_bottleneck_report_v1",
            "overall": "no_measurements",
            "model_kind": "post_hoc_utilization_v1",
            "deterministic": True,
            "target_id": target_id,
            "model_inputs_used": {
                "peak_compute_gflops": peak_compute,
                "peak_bandwidth_gb_s": peak_bw,
            },
            "regions": [],
            "summary": {
                "regions_with_evidence": 0,
                "non_opaque_total_regions": non_opaque_total,
                "kernel_calibration_status": "not_kernel_calibrated",
                "agreement_with_analytical": {
                    "agreement_count": 0,
                    "disagreement_count": 0,
                    "disagreements": [],
                },
            },
            "kernel_calibration_status": "not_kernel_calibrated",
            "known_limitations": [
                "post-hoc derivation: utilization from M-19/M-20 measured "
                "time × M-21 analytical flops/bytes",
                "cache_evidence not collected (Nsight / linux perf integration "
                "is M-22.1 follow-up)",
                "no occupancy / register-pressure model",
                "no launch-overhead breakdown",
                "fp32 matmul only",
            ],
            "generated_at_utc": _utcnow(),
        }
        report_path.write_text(
            json.dumps(body, indent=2, sort_keys=True), encoding="utf-8",
        )
        summary_md_path.write_text(
            "# Compiled Bottleneck (M-22) — no_measurements\n\n"
            "No M-19 / M-20 compiled-kernel measurements found. "
            "Set `COMPGEN_RUN_KERNELS=1` to enable.\n",
            encoding="utf-8",
        )
        return CompiledBottleneckResult(
            overall="no_measurements", out_dir=out_dir,
            report_path=report_path, summary_md_path=summary_md_path,
            region_count_with_evidence=0,
            region_count_total=non_opaque_total,
            kernel_calibration_status="not_kernel_calibrated",
        )

    regions_out: list[dict[str, Any]] = []
    evidence_by_region: dict[str, dict[str, Any]] = {}
    agreement_count = 0
    disagreement_count = 0
    disagreements: list[dict[str, Any]] = []

    for m in measurements:
        m21 = m21_by_cid.get(m.candidate_id)
        if m21 is None:
            regions_out.append({
                "region_id": m.region_id,
                "candidate_id": m.candidate_id,
                "model_status": "skipped",
                "skip_reason": (
                    "no M-21 analytical entry for this candidate"
                ),
            })
            continue

        flops = int((m21.get("compute") or {}).get("flops") or 0)
        bytes_moved = int(
            (m21.get("memory") or {}).get("total_bytes_moved") or 0
        )
        analytical_bn = m21.get("bottleneck_resource") or "unknown"

        gpu_block: dict[str, Any] | None = None
        cpu_block: dict[str, Any] | None = None
        if m.gpu_us is not None:
            gpu_block = {
                "measured_us_per_iter": m.gpu_us,
                **derive_utilization(
                    flops=flops,
                    bytes_moved=bytes_moved,
                    measured_us=m.gpu_us,
                    peak_compute_gflops=peak_compute,
                    peak_bandwidth_gb_s=peak_bw,
                ),
            }
        if m.cpu_us is not None:
            cpu_block = {
                "measured_us_per_iter": m.cpu_us,
                **derive_utilization(
                    flops=flops,
                    bytes_moved=bytes_moved,
                    measured_us=m.cpu_us,
                    peak_compute_gflops=peak_compute,
                    peak_bandwidth_gb_s=peak_bw,
                ),
            }

        # Pick the canonical "measured_bottleneck" for cross-reference.
        # Convention: prefer GPU when available (real accelerator path).
        # Fallback CPU. The per-track classifications are still recorded.
        canonical_track = None
        canonical_bn = "unknown"
        if gpu_block is not None and gpu_block.get("measured_bottleneck"):
            canonical_track = "gpu"
            canonical_bn = gpu_block["measured_bottleneck"]
        elif cpu_block is not None and cpu_block.get("measured_bottleneck"):
            canonical_track = "cpu"
            canonical_bn = cpu_block["measured_bottleneck"]

        agrees = (
            canonical_bn == analytical_bn
            and canonical_bn in ("compute", "memory")
        )
        if canonical_bn in ("compute", "memory"):
            if agrees:
                agreement_count += 1
            else:
                disagreement_count += 1
                disagreements.append({
                    "region_id": m.region_id,
                    "analytical_bottleneck": analytical_bn,
                    "measured_bottleneck": canonical_bn,
                    "canonical_track": canonical_track,
                })

        evidence_block = {
            "candidate_id": m.candidate_id,
            "matmul_shape": {"M": m.matmul_shape[0],
                             "N": m.matmul_shape[1],
                             "K": m.matmul_shape[2]},
            "tile": {"M": m.tile[0], "N": m.tile[1], "K": m.tile[2]},
            "analytical_flops": flops,
            "analytical_bytes_moved": bytes_moved,
            "analytical_bottleneck": analytical_bn,
            "measured_bottleneck": canonical_bn,
            "canonical_track": canonical_track,
            "bottleneck_classification_agreement": agrees,
            "gpu": gpu_block,
            "cpu": cpu_block,
            "cache_evidence": "not_collected",
            "source": m.source,
        }

        regions_out.append({
            **evidence_block,
            "region_id": m.region_id,
            "model_status": "ok",
        })
        evidence_by_region[m.region_id] = evidence_block

    # Determine kernel_calibration_status.
    n_evidence = len(evidence_by_region)
    if n_evidence == 0:
        kc_status = "not_kernel_calibrated"
    elif (non_opaque_total > 0 and n_evidence >= non_opaque_total):
        kc_status = "kernel_calibrated"
    else:
        kc_status = "partial_kernel_calibration"

    summary = {
        "regions_with_evidence": n_evidence,
        "non_opaque_total_regions": non_opaque_total,
        "kernel_calibration_status": kc_status,
        "agreement_with_analytical": {
            "agreement_count": agreement_count,
            "disagreement_count": disagreement_count,
            "disagreements": disagreements,
        },
    }

    body = {
        "schema_version": "compiled_bottleneck_report_v1",
        "overall": "ok" if n_evidence > 0 else "no_measurements",
        "model_kind": "post_hoc_utilization_v1",
        "deterministic": True,
        "target_id": target_id,
        "model_inputs_used": {
            "peak_compute_gflops": peak_compute,
            "peak_bandwidth_gb_s": peak_bw,
        },
        "regions": regions_out,
        "summary": summary,
        "kernel_calibration_status": kc_status,
        "known_limitations": [
            "post-hoc derivation: utilization from M-19/M-20 measured "
            "time × M-21 analytical flops/bytes",
            "cache_evidence not collected (Nsight / linux perf integration "
            "is M-22.1 follow-up)",
            "no occupancy / register-pressure model",
            "no launch-overhead breakdown",
            "fp32 matmul only",
            "canonical_track prefers GPU when available, CPU fallback",
        ],
        "generated_at_utc": _utcnow(),
    }
    report_path.write_text(
        json.dumps(body, indent=2, sort_keys=True), encoding="utf-8",
    )

    md_lines = [
        f"# Compiled Bottleneck (M-22) — overall=ok  status={kc_status}\n",
        f"- target: `{target_id}`",
        f"- peak_compute_gflops: {peak_compute}",
        f"- peak_bandwidth_gb_s: {peak_bw}",
        f"- regions with evidence: {n_evidence} / {non_opaque_total}",
        f"- bottleneck agreement: {agreement_count} match, "
        f"{disagreement_count} disagree",
        "",
        "| region | analytical | measured | agree? | gpu_us | gpu_compute_util | "
        "gpu_bw_util | cpu_us |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in regions_out:
        if r.get("model_status") != "ok":
            continue
        gpu = r.get("gpu") or {}
        cpu = r.get("cpu") or {}

        def _fmt(v: Any, prec: int = 4) -> str:
            if v is None:
                return "—"
            try:
                return f"{float(v):.{prec}f}"
            except (TypeError, ValueError):
                return str(v)

        md_lines.append(
            f"| `{r['region_id']}` | `{r.get('analytical_bottleneck')}` "
            f"| `{r.get('measured_bottleneck')}` "
            f"| {'✓' if r.get('bottleneck_classification_agreement') else '✗'} "
            f"| {_fmt(gpu.get('measured_us_per_iter'), 2)} "
            f"| {_fmt(gpu.get('compute_utilization'))} "
            f"| {_fmt(gpu.get('bandwidth_utilization'))} "
            f"| {_fmt(cpu.get('measured_us_per_iter'), 2)} |"
        )
    if disagreements:
        md_lines.append("")
        md_lines.append("## Bottleneck disagreements\n")
        for d in disagreements:
            md_lines.append(
                f"- `{d['region_id']}`: analytical=`{d['analytical_bottleneck']}` "
                f"vs measured=`{d['measured_bottleneck']}` "
                f"(track=`{d['canonical_track']}`)"
            )
    summary_md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # Layer onto hardware_resource_report.
    if evidence_by_region:
        _apply_hardware_resource_overlay(
            run_dir=run_dir,
            evidence_by_region=evidence_by_region,
            kernel_calibration_status=kc_status,
        )

    return CompiledBottleneckResult(
        overall="ok" if n_evidence > 0 else "no_measurements",
        out_dir=out_dir, report_path=report_path,
        summary_md_path=summary_md_path,
        region_count_with_evidence=n_evidence,
        region_count_total=non_opaque_total,
        kernel_calibration_status=kc_status,
    )
