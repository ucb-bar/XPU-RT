"""Render an honest markdown findings report from a stress-extensive
summary JSON. Reads the JSON, surfaces real outliers / drifts /
failures, writes a report.

Usage:
    .venv/bin/python scripts/dev/render_stress_findings.py \\
        --summary /tmp/stress_extensive/stress_extensive_summary.json \\
        --out tmp/stress_findings.md
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _read(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    body = _read(args.summary)

    rows = body["rows"]
    agg = body["aggregate_per_model_config"]
    honest = body.get("honest_summary", {}) or {}

    def _fmt_pct(numer: int, denom: int) -> str:
        return f"{numer}/{denom} ({100.0 * numer / max(denom, 1):.0f}%)"

    n_runs = body["n_total_runs"]
    n_ok = sum(1 for r in rows if r.get("status") == "ok")
    n_retry = sum(
        1 for r in rows
        if r.get("m15b", {}).get("retry_needed")
    )
    n_fail = sum(
        1 for r in rows
        if r.get("status") not in ("ok",)
        and not r.get("m15b", {}).get("retry_needed")
    )

    # Per-stage timing distribution.
    stage_times: dict[str, list[float]] = {}
    for r in rows:
        for sid, t in (r.get("stage_wall_times_s") or {}).items():
            stage_times.setdefault(sid, []).append(t)
    stage_summary = {
        sid: {
            "mean_s": statistics.mean(ts),
            "max_s": max(ts),
            "min_s": min(ts),
            "n": len(ts),
        }
        for sid, ts in stage_times.items()
    }

    # M-22.1 GPU variance across reruns of the same (model, config).
    drift_rows = [
        r for r in agg
        if not (r["candidate_ids_allowed_stable"]
                and r["m21_modeled_stable"]
                and r["canonical_artifacts_stable"]
                and r["agent_guidance_stable"])
    ]

    # Slowest run.
    runs_with_time = [r for r in rows if r.get("wall_time_s") is not None]
    slowest = sorted(
        runs_with_time, key=lambda r: -float(r.get("wall_time_s") or 0),
    )[:5]

    # Largest artifact tree.
    largest = sorted(
        runs_with_time, key=lambda r: -int(r.get("total_bytes") or 0),
    )[:5]

    # M-22 agreement / disagreement aggregated.
    m22_agreement = 0
    m22_disagreement = 0
    for r in rows:
        m22 = r.get("m22") or {}
        if m22.get("present"):
            m22_agreement += int(m22.get("agreement_count", 0))
            m22_disagreement += int(m22.get("disagreement_count", 0))

    lines: list[str] = []
    lines.append("# Extensive Stress Test — Findings\n")
    lines.append(
        f"**Generated**: {body.get('generated_at_utc', '?')}  "
        f"  **Total wall**: {body.get('total_wall_time_s', 0):.0f}s\n"
    )
    lines.append(
        f"**Matrix**: {body['n_models']} models × {body['n_configs']} "
        f"configs × {body['n_runs_per_combo']} runs = "
        f"{n_runs} pipeline invocations.\n"
    )
    lines.append("## Outcomes\n")
    lines.append(f"- ok: {_fmt_pct(n_ok, n_runs)}")
    lines.append(f"- M-15B retry-required (honest, not a bug): "
                 f"{_fmt_pct(n_retry, n_runs)}")
    lines.append(f"- Pipeline failures (other than M-15B): "
                 f"{_fmt_pct(n_fail, n_runs)}")
    lines.append("")

    lines.append("## Per-stage wall time distribution\n")
    lines.append("| stage | n | mean_s | min_s | max_s |")
    lines.append("|---|---|---|---|---|")
    for sid in ("graph_capture", "payload_lowering", "graph_analysis",
                "recipe_planning"):
        s = stage_summary.get(sid)
        if s is None:
            continue
        lines.append(
            f"| {sid} | {s['n']} | {s['mean_s']:.2f} | "
            f"{s['min_s']:.2f} | {s['max_s']:.2f} |"
        )
    lines.append("")

    lines.append("## Cross-run determinism (per model × config)\n")
    lines.append(
        "Y = identical across reruns. DRIFT = different bytes "
        "between reruns (a real find).\n"
    )
    lines.append(
        "| model | config | runs | retcode | cids | m21 | canon | "
        "agent_guid | m22.1 stdev |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|---|---|"
    )
    for r in agg:
        cells = [
            r["model"],
            r["config"],
            str(r["n_runs"]),
            "/".join(str(c) for c in r["pipeline_returncodes"]),
            "Y" if r["candidate_ids_allowed_stable"] else "DRIFT",
            "Y" if r["m21_modeled_stable"] else "DRIFT",
            "Y" if r["canonical_artifacts_stable"] else "DRIFT",
            "Y" if r["agent_guidance_stable"] else "DRIFT",
            (
                f"{r['m22_1_gpu_mean_us_stdev_across_runs']:.2f}us"
                if r["m22_1_gpu_mean_us_stdev_across_runs"] is not None
                else "—"
            ),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    if drift_rows:
        lines.append(
            "**Drift surfaces** found in:\n"
        )
        for r in drift_rows:
            lines.append(
                f"- `{r['model']}` / `{r['config']}`: "
                f"cids_stable={r['candidate_ids_allowed_stable']}, "
                f"m21_stable={r['m21_modeled_stable']}, "
                f"canon_stable={r['canonical_artifacts_stable']}, "
                f"ag_stable={r['agent_guidance_stable']}"
            )
        lines.append("")
    else:
        lines.append(
            "**No drift across reruns** for any (model, config) — "
            "canonical artifacts, M-21 modeled count, and "
            "agent_guidance signature were byte-identical on "
            "every rerun.\n"
        )

    lines.append("## M-15B retry surface\n")
    lines.append(
        f"Models that hit retry on at least one config: "
        f"{honest.get('models_with_m15b_retry', [])}\n"
    )
    retry_breakdown: dict[str, int] = {}
    for r in rows:
        if r.get("m15b", {}).get("retry_needed"):
            check = r["m15b"].get("failed_check") or "?"
            retry_breakdown[check] = retry_breakdown.get(check, 0) + 1
    if retry_breakdown:
        lines.append("Retry counts by failed_check:")
        for k, v in sorted(retry_breakdown.items()):
            lines.append(f"- `{k}`: {v}")
        lines.append("")

    lines.append(
        f"## M-22 measured-vs-analytical agreement (across all runs)\n"
    )
    total_m22 = m22_agreement + m22_disagreement
    if total_m22 > 0:
        lines.append(
            f"- Agreements: {m22_agreement} regions"
        )
        lines.append(
            f"- Disagreements: {m22_disagreement} regions"
        )
        lines.append(
            f"- Agreement rate: "
            f"{100.0 * m22_agreement / total_m22:.0f}%\n"
        )

    lines.append("## Slowest pipeline invocations\n")
    lines.append("| model | config | run | wall_s |")
    lines.append("|---|---|---|---|")
    for r in slowest:
        lines.append(
            f"| {r['model']} | {r['config']} | "
            f"r{r.get('run_idx', '?')} | "
            f"{r.get('wall_time_s', 0):.1f} |"
        )
    lines.append("")

    lines.append("## Largest artifact trees\n")
    lines.append("| model | config | run | files | bytes |")
    lines.append("|---|---|---|---|---|")
    for r in largest:
        b = int(r.get("total_bytes") or 0)
        lines.append(
            f"| {r['model']} | {r['config']} | "
            f"r{r.get('run_idx', '?')} | "
            f"{r.get('file_count', 0)} | "
            f"{b/1024/1024:.1f}MB |"
        )
    lines.append("")

    lines.append("## Configuration drift (cross-config differences)\n")
    lines.append(
        "For each model, compare M-21 candidates_modeled, M-22 "
        "evidence count, M-23 case count across configs. Big "
        "differences mean the opt-in matrix shifts what the agent "
        "sees.\n"
    )
    by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for r in rows:
        m = r.get("model")
        c = r.get("config")
        if m and c:
            by_model.setdefault(m, {})[c] = r
    lines.append(
        "| model | config | m21_modeled | m22_evidence | m22_agree | m23_cases |"
    )
    lines.append(
        "|---|---|---|---|---|---|"
    )
    for m, by_cfg in sorted(by_model.items()):
        for c in ("no_kernels", "kernels", "full_optins"):
            r = by_cfg.get(c)
            if r is None:
                continue
            m21 = r.get("m21") or {}
            m22 = r.get("m22") or {}
            m23 = r.get("m23") or {}
            lines.append(
                f"| {m} | {c} | "
                f"{m21.get('modeled', 0)} | "
                f"{m22.get('regions_with_evidence', 0)} | "
                f"{m22.get('agreement_count', 0)} | "
                f"{m23.get('case_count', 0)} |"
            )
    lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
