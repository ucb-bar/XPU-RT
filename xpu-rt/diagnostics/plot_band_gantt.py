"""Band-aware Gantt renderer (Phase A4).

Takes a scheduled fixture and the workload JSON it was generated from,
draws a Gantt where:
  - each periodic network has dashed vertical lines at every period
    boundary (k * P_N for instance k),
  - each instance band [R_k, D_k] is a faint colored rectangle on the
    network's lane(s),
  - any dispatch whose finish exceeds its band's D_k is rendered as a
    RED rectangle (deadline overrun),
  - any dispatch whose start precedes its band's R_k is rendered with a
    red hatched border (release violation — should never happen but
    surfaces it).

Uses `band_invariant.check_band_invariant` for the per-op classification.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .band_invariant import (
    BandReport,
    check_band_invariant,
    _periodic_metadata,
    _parse_instance,
)


# Colorblind-friendly palette: distinct hues for each network. Falls
# back to gray for unknown networks.
_PALETTE = {
    "mlp_control": "#1f77b4",  # blue
    "dronet": "#2ca02c",       # green
    "yolov8_nano": "#ff7f0e",  # orange
    "yolov8": "#ff7f0e",
}


def _color_for(network: str) -> str:
    return _PALETTE.get(network, "#94a3b8")


def render_band_gantt(fixture: Dict[str, Any],
                       workload_data: Dict[str, Any],
                       out_path: str,
                       *,
                       title: Optional[str] = None,
                       solver: str = "",
                       show_bands: bool = True,
                       show_period_lines: bool = True,
                       ) -> Dict[str, Any]:
    """Render a band-aware Gantt to `out_path` (PNG).

    Returns a small dict {n_dispatches, n_lanes, makespan,
    n_release_violations, n_deadline_violations}.

    The fixture's start_time / duration units must match the workload's
    period / window_duration units (typically milliseconds).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    dispatches = fixture.get("dispatches", {})
    if not dispatches:
        raise RuntimeError("no dispatches in fixture")

    # Phase D3 measured-cycles guard. Two thresholds:
    #   - ALWAYS print stderr if any zero-duration op is present
    #     (legitimate IR ops like reshape/broadcast/concat are
    #     genuinely zero-cost in the profile DB, so this is a
    #     diagnostic-warning level signal).
    #   - HARD ERROR if >30% of dispatches are zero-duration — that
    #     ratio is the bookkeeping-fiction signature: a manual edit
    #     dropped real-cost ops or the profile DB failed to populate.
    #     The pipeline must fix this before a Gantt becomes a published
    #     artifact.
    zero_dur = []
    for name, entry in dispatches.items():
        dur = float(entry.get("duration", 0.0))
        if dur < 1e-9:
            if entry.get("synthetic_zero_cost") is True:
                continue
            zero_dur.append(name)
    n_zero = len(zero_dur)
    n_total = len(dispatches)
    if n_zero:
        import sys
        print(f"[band-gantt:D3-guard] {solver}: {n_zero}/{n_total} "
              f"({100.0*n_zero/n_total:.1f}%) dispatches have duration=0. "
              f"First 3: {zero_dur[:3]}.",
              file=sys.stderr)
    # Threshold rationale: yolov8/dronet/mlp_control workloads have
    # ~30-40% legitimately-zero ops (reshape, broadcast, quant
    # scale/shift, concat — the IR decomposes these into separate
    # dispatches but the profile DB reports 0 cycles because they fold
    # away on the lowered code). 50% is high enough to allow these,
    # low enough to catch "deleted half the dispatches" bookkeeping.
    if n_total > 0 and n_zero / n_total > 0.50:
        raise RuntimeError(
            f"render guard (Phase D3): {n_zero}/{n_total} "
            f"({100.0*n_zero/n_total:.1f}%) dispatches have duration=0. "
            f"That's above the 50% bookkeeping-fiction threshold. "
            f"Either the profile DB failed to populate, or a manual "
            f"edit dropped real-cost ops. Fix upstream before rendering."
        )

    periodic_bases, nonperiodic_bases = _periodic_metadata(workload_data)

    # Pre-compute per-op band classification using the same check the
    # audit uses, so the Gantt and the CSV agree.
    report = check_band_invariant(fixture, workload_data, solver=solver)
    violation_index = {v.dispatch: v for v in report.violations}

    # Lanes by hardware_target.
    lanes = sorted({d["hardware_target"] for d in dispatches.values()})
    lane_idx = {lane: i for i, lane in enumerate(lanes)}

    items: List[Tuple[str, Dict[str, Any], str, int]] = []  # (name, entry, net, inst)
    for name, entry in dispatches.items():
        job_name = entry.get("job_name") or ""
        base_job = re.sub(r"\d+$", "", job_name) if job_name else ""
        if base_job in periodic_bases:
            base, inst = _parse_instance(
                name, job_name, periodic_bases, nonperiodic_bases
            )
            if base not in periodic_bases:
                base, inst = base_job, 0
        else:
            base, inst = _parse_instance(
                name, job_name, periodic_bases, nonperiodic_bases
            )
        items.append((name, entry, base, inst))

    # Figure layout: tall enough to give each lane a clean band.
    fig_h = max(3.5, 0.5 * len(lanes) + 2.0)
    fig, ax = plt.subplots(figsize=(16, fig_h))
    BAR_H = 0.7

    makespan = max(
        float(e.get("start_time", 0.0)) + float(e.get("duration", 0.0))
        for e in dispatches.values()
    )

    # Draw instance bands BEHIND the bars (light fill).
    if show_bands:
        for net, (n_inst, period, window, start_t) in periodic_bases.items():
            color = _color_for(net)
            for k in range(int(n_inst)):
                R_k = start_t + k * period
                D_k = R_k + window
                # Fill spanning all lanes — keeps the rendering simple
                # and surfaces overruns regardless of which lane the
                # late op landed on.
                ax.axvspan(R_k, D_k, alpha=0.05, color=color, zorder=0)

    # Period boundary lines.
    if show_period_lines:
        for net, (n_inst, period, window, start_t) in periodic_bases.items():
            color = _color_for(net)
            for k in range(int(n_inst) + 1):
                x = start_t + k * period
                ax.axvline(x, color=color, linestyle="--",
                            linewidth=0.5, alpha=0.4, zorder=0)

    # Draw bars.
    handles_seen: Dict[str, mpatches.Patch] = {}
    for name, entry, net, inst in items:
        lane = lane_idx[entry["hardware_target"]]
        s = float(entry.get("start_time", 0.0))
        w = float(entry.get("duration", 0.0))
        color = _color_for(net)

        v = violation_index.get(name)
        if v and v.is_deadline_violation:
            # Red fill, thick red border. The size is what was scheduled.
            ax.broken_barh([(s, w)], (lane - BAR_H/2, BAR_H),
                            facecolors="#fee2e2",   # red-100
                            edgecolors="#dc2626",   # red-600
                            linewidth=1.2)
        elif v and v.is_release_violation:
            ax.broken_barh([(s, w)], (lane - BAR_H/2, BAR_H),
                            facecolors=color,
                            edgecolors="#dc2626",
                            linewidth=1.2,
                            hatch="////")
        else:
            ax.broken_barh([(s, w)], (lane - BAR_H/2, BAR_H),
                            facecolors=color,
                            edgecolors="black",
                            linewidth=0.15)

        if net not in handles_seen:
            handles_seen[net] = mpatches.Patch(color=color, label=net)

    # Style.
    ax.set_yticks(range(len(lanes)))
    ax.set_yticklabels(lanes)
    ax.set_xlabel("Time (ms)")
    ax.set_xlim(0, makespan * 1.02)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.axvline(makespan, color="black", linestyle="--", linewidth=0.6, alpha=0.6)

    n_rel = report.n_release_violations
    n_dl = report.n_deadline_violations
    subtitle = (f"{len(dispatches)} dispatches, makespan {makespan:.1f} ms"
                f", deadline misses {n_dl}, release violations {n_rel}")
    ax.set_title((title or f"{solver}: {Path(out_path).stem}") + "  —  " + subtitle)

    # Legend: network swatches + violation markers.
    handles = list(handles_seen.values())
    if n_dl > 0:
        handles.append(mpatches.Patch(facecolor="#fee2e2",
                                       edgecolor="#dc2626",
                                       linewidth=1.2,
                                       label="deadline overrun"))
    if n_rel > 0:
        handles.append(mpatches.Patch(facecolor="white",
                                       edgecolor="#dc2626",
                                       linewidth=1.2, hatch="////",
                                       label="release violation"))
    ax.legend(handles=handles, loc="upper right", fontsize=7,
                framealpha=0.9, ncol=min(4, len(handles)))

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "n_dispatches": len(dispatches),
        "n_lanes": len(lanes),
        "makespan": makespan,
        "n_release_violations": n_rel,
        "n_deadline_violations": n_dl,
    }


def main() -> int:
    """CLI: render a band-aware Gantt from (fixture, workload) → PNG."""
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--fixture", required=True)
    p.add_argument("--workload", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--title", default=None)
    p.add_argument("--solver", default="")
    args = p.parse_args()
    fixture = json.loads(Path(args.fixture).read_text())
    workload = json.loads(Path(args.workload).read_text())
    summary = render_band_gantt(fixture, workload, args.out,
                                 title=args.title, solver=args.solver)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
