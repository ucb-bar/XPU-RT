"""Real stress harness for the M-19 → M-23 kernel pipeline.

Goes BEYOND the basic stress runner. Dimensions exercised:

- Model breadth: every model in canonical-6 + a wide-suite sample
  (12+ total).
- Config matrix: at least three configurations per model:
    1. ``kernels``         (COMPGEN_RUN_KERNELS=1)
    2. ``full_optins``     (kernels + M-18 + M-18.3)
    3. ``no_kernels``      (default; baseline)
- Repeat factor: each (model × config) is run N times so the harness
  can surface non-determinism in measurement-bearing artifacts and
  honest variance in M-22 / M-22.1 / M-23 timings.
- Per-stage wall-clock instrumented via the ledger's start/finish
  events.
- Artifact-tree size (file count + total bytes) per run.
- M-22 / M-22.1 / M-23 outcome variance across reruns.
- Agent-surface stability: ``agent_decision_request.candidate_ids_allowed``
  identical across reruns of the same (model, config)?
- M-15B retry-needed count per (model, config).

Aggregate output:
- ``stress_extensive_summary.json`` with full per-(model, config, run)
  rows + cross-config / cross-run aggregates.
- A wide markdown table summarising honest signals + outliers.

Usage:
    .venv/bin/python scripts/dev/run_kernel_stress_extensive.py \\
        --models tiny_mlp,tiny_attention,tiny_conv_block,merlin_mlp_wide,\\
                 proxy_vla,proxy_vlm,custom_unsupported_op,\\
                 graph_break_mlp,residual_branch,merlin_dronet \\
        --configs kernels,full_optins,no_kernels \\
        --runs 2 \\
        --out /tmp/kernel_stress_extensive
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


# Map config name → env-var dict.
_CONFIGS: dict[str, dict[str, str]] = {
    "no_kernels": {},
    "kernels": {"COMPGEN_RUN_KERNELS": "1"},
    "full_optins": {
        "COMPGEN_RUN_KERNELS": "1",
        "COMPGEN_CALIBRATE_PROFILER": "1",
        "COMPGEN_CALIBRATE_CANDIDATES": "1",
    },
}


def _utcnow() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _tree_size(root: Path) -> tuple[int, int]:
    """Return (file_count, total_bytes) under root."""
    files = 0
    total = 0
    for p in root.rglob("*"):
        if p.is_file():
            files += 1
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return files, total


def _per_stage_wall_times(run_dir: Path) -> dict[str, float]:
    """From stage_ledger.jsonl, compute per-stage wall time using
    paired start/finish events. Returns {stage_id: seconds}."""
    events = _read_jsonl(run_dir / "stage_ledger.jsonl")
    starts: dict[str, str] = {}
    out: dict[str, float] = {}
    for e in events:
        sid = e.get("stage_id") or ""
        ev = e.get("event") or ""
        ts = e.get("timestamp_utc") or ""
        if not sid or not ts:
            continue
        if ev == "start":
            starts[sid] = ts
        elif ev == "finish" and sid in starts:
            try:
                t0 = datetime.strptime(starts[sid], "%Y-%m-%dT%H:%M:%SZ")
                t1 = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
                out[sid] = (t1 - t0).total_seconds()
            except ValueError:
                pass
    return out


def _candidate_ids_allowed(run_dir: Path) -> list[str]:
    for p in (
        run_dir / "03_recipe_planning" / "agent_decision"
        / "agent_decision_request.json",
        run_dir / "agent_decision_request.json",
    ):
        d = _read_json(p)
        if d is not None:
            return list(d.get("candidate_ids_allowed", []) or [])
    return []


def _m15b_retry(run_dir: Path) -> dict[str, Any]:
    p = (
        run_dir / "03_recipe_planning" / "downstream_retry"
        / "downstream_retry_request.json"
    )
    d = _read_json(p)
    if d is None:
        return {"retry_needed": False}
    return {
        "retry_needed": True,
        "failed_check": d.get("failed_check"),
        "failed_stage": d.get("failed_stage"),
    }


def _m22_summary(run_dir: Path) -> dict[str, Any]:
    p = (
        run_dir / "02_graph_analysis" / "compiled_bottleneck"
        / "compiled_bottleneck_report.json"
    )
    d = _read_json(p)
    if d is None:
        return {"present": False}
    s = d.get("summary", {}) or {}
    a = s.get("agreement_with_analytical", {}) or {}
    return {
        "present": True,
        "overall": d.get("overall"),
        "kernel_calibration_status": d.get("kernel_calibration_status"),
        "agreement_count": a.get("agreement_count", 0),
        "disagreement_count": a.get("disagreement_count", 0),
        "regions_with_evidence": s.get("regions_with_evidence", 0),
    }


def _m221_summary(run_dir: Path) -> dict[str, Any]:
    p = (
        run_dir / "02_graph_analysis" / "profiler_evidence"
        / "profiler_evidence_report.json"
    )
    d = _read_json(p)
    if d is None:
        return {"present": False}
    s = d.get("summary", {}) or {}
    # collect per-region self_cuda_us measurements for variance analysis.
    gpu_uss = []
    for r in d.get("regions", []) or []:
        gpu = r.get("gpu") or {}
        us = gpu.get("self_cuda_us_per_iter")
        if us is not None and float(us) > 0:
            gpu_uss.append(float(us))
    return {
        "present": True,
        "gpu_collected": s.get("gpu_collected_count", 0),
        "cpu_collected": s.get("cpu_collected_count", 0),
        "region_count": s.get("region_count", 0),
        "gpu_us_per_region": gpu_uss,
        "gpu_mean_us": (
            statistics.mean(gpu_uss) if gpu_uss else None
        ),
    }


def _m23_summary(run_dir: Path) -> dict[str, Any]:
    p = (
        run_dir / "02_graph_analysis" / "compiled_fusion"
        / "compiled_fusion_differential_report.json"
    )
    d = _read_json(p)
    if d is None:
        return {"present": False}
    s = d.get("summary", {}) or {}
    return {
        "present": True,
        "overall": d.get("overall"),
        "case_count": s.get("case_count", 0),
        "bit_equality_count": s.get("bit_equality_count", 0),
        "fail_count": s.get("fail_outside_tolerance_count", 0),
    }


def _hashable_artifact_signature(run_dir: Path) -> dict[str, str]:
    """Build a signature dict mapping rel-path → sha256 for canonical
    artifacts that SHOULD be deterministic (modulo timestamps).
    Stripping timestamps is per-file; here we just snapshot raw SHAs
    for cross-run inspection."""
    out: dict[str, str] = {}
    for rel in (
        "02_graph_analysis/region_map.json",
        "02_graph_analysis/candidate_actions.json",
        "02_graph_analysis/cost_preview_v2.json",
        "02_graph_analysis/llm_graph_view.json",
    ):
        p = run_dir / rel
        if p.exists():
            out[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def _agent_guidance_signature(run_dir: Path) -> dict[str, Any] | None:
    """The agent_guidance block is a deterministic constant function
    of the milestone version. Hash it for cross-run/cross-model
    comparison."""
    for p in (
        run_dir / "03_recipe_planning" / "agent_decision"
        / "agent_decision_request.json",
        run_dir / "agent_decision_request.json",
    ):
        d = _read_json(p)
        if d is not None:
            g = d.get("agent_guidance") or {}
            return {
                "guidance_version": g.get("guidance_version"),
                "sha256": hashlib.sha256(
                    json.dumps(g, sort_keys=True).encode("utf-8")
                ).hexdigest(),
            }
    return None


def _run_one(
    *, model: str, config: str, run_idx: int, out_root: Path,
) -> dict[str, Any]:
    """Run pipeline once. Returns a per-run row."""
    config_env = _CONFIGS[config]
    out_dir = out_root / model / f"{config}__r{run_idx}"
    if out_dir.exists():
        shutil.rmtree(out_dir)

    env = os.environ.copy()
    # Strip prior opt-ins that aren't in this config.
    for v in ("COMPGEN_RUN_KERNELS", "COMPGEN_CALIBRATE_PROFILER",
              "COMPGEN_CALIBRATE_CANDIDATES"):
        env.pop(v, None)
    env.update(config_env)

    model_yaml = REPO_ROOT / "configs" / "models" / f"{model}.yaml"
    if not model_yaml.exists():
        return {
            "model": model, "config": config, "run_idx": run_idx,
            "status": "model_yaml_missing",
        }

    t_start = time.perf_counter()
    proc = subprocess.run(
        [
            sys.executable, "-m", "compgen.graph_compilation", "run",
            "--model", str(model_yaml),
            "--target", str(REPO_ROOT / "configs" / "targets" / "host_cpu.yaml"),
            "--out", str(out_dir),
            "--stop-after", "agent-decision-request",
            "--selection-mode", "greedy",
        ],
        cwd=REPO_ROOT, capture_output=True, text=True, env=env,
        timeout=600,
    )
    wall_time_s = time.perf_counter() - t_start

    file_count, total_bytes = _tree_size(out_dir) if out_dir.exists() else (0, 0)

    row: dict[str, Any] = {
        "model": model, "config": config, "run_idx": run_idx,
        "out_dir": str(out_dir),
        "pipeline_returncode": proc.returncode,
        "wall_time_s": wall_time_s,
        "file_count": file_count,
        "total_bytes": total_bytes,
    }

    if not out_dir.exists():
        row["status"] = "pipeline_did_not_produce_run_dir"
        row["stderr_tail"] = (proc.stderr or "")[-400:]
        return row

    # Per-stage wall times from ledger.
    row["stage_wall_times_s"] = _per_stage_wall_times(out_dir)
    row["ledger_event_count"] = len(_read_jsonl(out_dir / "stage_ledger.jsonl"))

    # Capture / lower / strict_gate.
    cap = _read_json(out_dir / "00_graph_capture" / "capture_report.json")
    row["capture_status"] = (cap or {}).get("status") or "missing"
    lowering = _read_json(
        out_dir / "01_payload_lowering" / "lowering_summary.json"
    )
    row["lowering_status"] = (lowering or {}).get("status") or "missing"
    sg = _read_json(
        out_dir / "01_payload_lowering"
        / f"{model}_strict_gate_report.json"
    )
    row["strict_gate_status"] = (sg or {}).get("status") or "missing"

    # M-15B retry?
    row["m15b"] = _m15b_retry(out_dir)

    # Agent surface.
    row["candidate_ids_allowed_count"] = len(_candidate_ids_allowed(out_dir))
    row["agent_guidance"] = _agent_guidance_signature(out_dir)
    row["canonical_artifact_shas"] = _hashable_artifact_signature(out_dir)

    # M-21.
    ac = _read_json(
        out_dir / "02_graph_analysis" / "analytical_cost"
        / "per_candidate_analytical_cost.json"
    )
    if ac is not None:
        s = ac.get("summary", {}) or {}
        row["m21"] = {
            "overall": ac.get("overall"),
            "modeled": s.get("candidates_modeled", 0),
            "total": s.get("candidates_total", 0),
        }
    else:
        row["m21"] = {"overall": "missing"}

    # M-22 / M-22.1 / M-23.
    row["m22"] = _m22_summary(out_dir)
    row["m22_1"] = _m221_summary(out_dir)
    row["m23"] = _m23_summary(out_dir)

    row["status"] = (
        "ok" if proc.returncode == 0 else (
            "retry_required" if row["m15b"]["retry_needed"]
            else f"nonzero_returncode_{proc.returncode}"
        )
    )
    return row


def _aggregate_per_model_config(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Group rows by (model, config), compute cross-run aggregates."""
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        if r.get("status") == "model_yaml_missing":
            continue
        by_key.setdefault((r["model"], r["config"]), []).append(r)

    out: list[dict[str, Any]] = []
    for (model, config), grp in sorted(by_key.items()):
        wall_times = [r["wall_time_s"] for r in grp if r.get("wall_time_s")]
        ret_codes = sorted({r.get("pipeline_returncode") for r in grp})
        cids_lists = [
            tuple(_candidate_ids_allowed(Path(r["out_dir"])))
            for r in grp
        ]
        cids_stable = len(set(cids_lists)) == 1
        m21_modeled = [
            (r.get("m21") or {}).get("modeled", 0) for r in grp
        ]
        m21_stable = len(set(m21_modeled)) == 1
        # M-22.1 GPU per-region timings — capture variance.
        gpu_means = [
            (r.get("m22_1") or {}).get("gpu_mean_us")
            for r in grp
        ]
        gpu_means_clean = [m for m in gpu_means if m is not None]
        gpu_variance = (
            statistics.stdev(gpu_means_clean)
            if len(gpu_means_clean) >= 2 else None
        )
        canonical_sha_lists = [
            tuple(sorted(r.get("canonical_artifact_shas", {}).items()))
            for r in grp
        ]
        canonical_stable = len(set(canonical_sha_lists)) == 1
        ag_sig = [
            (r.get("agent_guidance") or {}).get("sha256")
            for r in grp
        ]
        ag_stable = len(set(ag_sig)) == 1

        out.append({
            "model": model, "config": config, "n_runs": len(grp),
            "wall_time_min_s": min(wall_times) if wall_times else None,
            "wall_time_max_s": max(wall_times) if wall_times else None,
            "wall_time_mean_s": (
                statistics.mean(wall_times) if wall_times else None
            ),
            "pipeline_returncodes": ret_codes,
            "candidate_ids_allowed_stable": cids_stable,
            "m21_modeled_stable": m21_stable,
            "canonical_artifacts_stable": canonical_stable,
            "agent_guidance_stable": ag_stable,
            "m22_1_gpu_mean_us_per_run": gpu_means,
            "m22_1_gpu_mean_us_stdev_across_runs": gpu_variance,
            "first_row_status": grp[0].get("status"),
            "m15b_retry_count": sum(
                1 for r in grp if r["m15b"].get("retry_needed")
            ),
        })
    return out


def _print_per_model_table(agg: list[dict[str, Any]]) -> None:
    cols = [
        ("model", 24), ("config", 13), ("runs", 5),
        ("wall_min", 9), ("wall_max", 9), ("wall_mean", 9),
        ("retcode", 8), ("cids", 6), ("m21", 6), ("can", 6),
        ("ag", 6), ("m22.1_var", 12),
    ]
    header = " ".join(f"{c[0]:<{c[1]}}" for c in cols)
    print(header)
    print("-" * len(header))
    for r in agg:
        cells = [
            (r["model"][:23], 24),
            (r["config"][:12], 13),
            (str(r["n_runs"]), 5),
            (
                f"{r['wall_time_min_s']:.1f}s"
                if r["wall_time_min_s"] else "—",
                9,
            ),
            (
                f"{r['wall_time_max_s']:.1f}s"
                if r["wall_time_max_s"] else "—",
                9,
            ),
            (
                f"{r['wall_time_mean_s']:.1f}s"
                if r["wall_time_mean_s"] else "—",
                9,
            ),
            (
                "/".join(str(c) for c in r["pipeline_returncodes"])[:7],
                8,
            ),
            ("Y" if r["candidate_ids_allowed_stable"] else "DRIFT", 6),
            ("Y" if r["m21_modeled_stable"] else "DRIFT", 6),
            ("Y" if r["canonical_artifacts_stable"] else "DRIFT", 6),
            ("Y" if r["agent_guidance_stable"] else "DRIFT", 6),
            (
                f"{r['m22_1_gpu_mean_us_stdev_across_runs']:.2f}us"
                if r["m22_1_gpu_mean_us_stdev_across_runs"] is not None
                else "—",
                12,
            ),
        ]
        print(" ".join(f"{c[0]:<{c[1]}}" for c in cells))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--models", default=(
            "tiny_mlp,tiny_attention,tiny_conv_block,merlin_mlp_wide,"
            "proxy_vla,proxy_vlm,custom_unsupported_op,"
            "graph_break_mlp,residual_branch,merlin_dronet"
        ),
    )
    p.add_argument(
        "--configs", default="kernels,full_optins,no_kernels",
        help="comma-separated subset of " + ",".join(_CONFIGS),
    )
    p.add_argument("--runs", type=int, default=2)
    p.add_argument("--out", default="/tmp/kernel_stress_extensive")
    args = p.parse_args()

    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    for c in configs:
        if c not in _CONFIGS:
            raise SystemExit(f"unknown config: {c!r}")

    rows: list[dict[str, Any]] = []
    n_total = len(models) * len(configs) * args.runs
    n_done = 0
    t0 = time.perf_counter()
    print(f"running {n_total} pipeline invocations "
          f"({len(models)} models × {len(configs)} configs × "
          f"{args.runs} runs)")
    for m in models:
        for c in configs:
            for r in range(args.runs):
                n_done += 1
                elapsed = time.perf_counter() - t0
                print(
                    f"  [{n_done:>3}/{n_total}] {m} / {c} / "
                    f"r{r} (elapsed={elapsed:.0f}s)"
                )
                try:
                    rows.append(_run_one(
                        model=m, config=c, run_idx=r, out_root=out_root,
                    ))
                except subprocess.TimeoutExpired:
                    rows.append({
                        "model": m, "config": c, "run_idx": r,
                        "status": "timeout",
                    })
                except Exception as exc:  # noqa: BLE001
                    rows.append({
                        "model": m, "config": c, "run_idx": r,
                        "status": "exception",
                        "exception": f"{type(exc).__name__}: {exc}",
                    })

    agg = _aggregate_per_model_config(rows)

    body = {
        "schema_version": "stress_extensive_summary_v1",
        "generated_at_utc": _utcnow(),
        "n_models": len(models),
        "n_configs": len(configs),
        "n_runs_per_combo": args.runs,
        "n_total_runs": n_total,
        "total_wall_time_s": time.perf_counter() - t0,
        "rows": rows,
        "aggregate_per_model_config": agg,
        "honest_summary": {
            "models_with_pipeline_failure": sorted({
                r["model"] for r in rows
                if r.get("pipeline_returncode") not in (0, None)
                and not r.get("m15b", {}).get("retry_needed")
            }),
            "models_with_m15b_retry": sorted({
                r["model"] for r in rows
                if r.get("m15b", {}).get("retry_needed")
            }),
            "configs_with_drift": sorted({
                (r["model"], r["config"])
                for r in agg
                if not (r["candidate_ids_allowed_stable"]
                        and r["m21_modeled_stable"]
                        and r["canonical_artifacts_stable"]
                        and r["agent_guidance_stable"])
            }),
        },
    }
    summary_path = out_root / "stress_extensive_summary.json"
    summary_path.write_text(
        json.dumps(body, indent=2, sort_keys=True), encoding="utf-8",
    )
    print(f"\nsummary written: {summary_path}")
    print(f"total wall time: {body['total_wall_time_s']:.0f}s\n")
    _print_per_model_table(agg)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
