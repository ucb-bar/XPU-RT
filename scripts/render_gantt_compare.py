#!/usr/bin/env python3
"""The measured Gantt rows on their own, one above the other, drawn by the composite's own routine:
XPU-RT CP-SAT, XPU-RT greedy (same spec, same board) and ROS 2 vanilla (8 OS-scheduled cores),
each from the board trace named in its sidecar (schedules/measured_gantt_<row>_metrics.json).

    scripts/render_gantt_compare.py [--out results/codesign_feedback/refined/gantt_compare_v2.png]
"""
from __future__ import annotations
import argparse, json, os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "sims", "scripts"))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt   # noqa: E402
import showdown_gatecourse as S   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", default=["xpu:XPU-RT · CP-SAT:xpu", "xpu2:XPU-RT · greedy:xpu2", "ros:ROS 2 vanilla:ros"],
                    help="<row>:<label>:<colour key xpu|xpu2|ros>, where <row> is a path to a "
                         "measured_gantt_*.json (as scripts/make_measured_gantt_pair.py writes under "
                         "results/codesign_feedback/refined/<build>/) or a bare name resolved against "
                         "schedules/measured_gantt_<name>.json")
    ap.add_argument("--out", default=os.path.join(REPO, "results/codesign_feedback/refined/gantt_compare_v2.png"))
    ap.add_argument("--dpi", type=int, default=220)
    a = ap.parse_args()
    cols = {"xpu": S.C_XPU, "xpu2": S.C_XPU2, "ros": S.C_ROS}
    rows, paths, prov = [], [], []
    for r in a.rows:
        name, label, ck = r.split(":", 2) if r.count(":") == 2 else r.rsplit(":", 2)
        label, ck = (label, ck) if ck in cols else (f"{label}:{ck}", "xpu")
        # a row is either a path to a Gantt build (any prefix, any directory) or a bare name in the
        # original schedules/ location, so a build made for one figure can be redrawn on its own
        p = name if name.endswith(".json") else os.path.join(REPO, f"schedules/measured_gantt_{name}.json")
        if not os.path.isabs(p):
            p = os.path.join(REPO, p)
        rows.append((json.load(open(p)), label, cols[ck], "ros" if ck == "ros" else "xpu")); paths.append(p)
        m = json.load(open(p.replace(".json", "_metrics.json")))
        prov.append(f"{label}: {os.path.basename(os.path.dirname(m['source']))}/{os.path.basename(m['source'])}  chain median {m['chain_ms_median']:.0f} ms")
    fig, ax = plt.subplots(figsize=(11, 3.1 * len(rows) + 1.0))
    S.draw_combined_gantt(ax, rows, paths)
    ax.set_title(ax.get_title().replace(" — ", "\n").replace(" · ", "\n"), fontsize=9, loc="left")   # one measured fact per line
    fig.text(0.01, 0.005, "board traces: " + "   |   ".join(prov), fontsize=7.5, color="#555")
    fig.savefig(a.out, dpi=a.dpi, bbox_inches="tight"); fig.savefig(a.out.replace(".png", ".pdf"), bbox_inches="tight")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
