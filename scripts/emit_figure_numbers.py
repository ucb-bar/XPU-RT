#!/usr/bin/env python3
"""Turn a figure's own metrics JSON into LaTeX macros, so a caption cannot drift.

WHY. `fig_schedule_evolution.png`'s caption states four makespans and four miss counts,
read from the artifacts. There are seven `schedule_evolution*.png` variants in results/, produced
from different loop runs and different lever sets, and a hand-typed caption matches exactly
one of them: panel 3 as "four dispatches past their deadlines at 34 ms" is true of one
variant, while the auto-loop variant of the same figure reads 2 missed / 36 ms. Neither the
figure nor a typed caption records which variant it describes.

`compose_schedule_evolution.py` already writes `<out>_metrics.json` with the per-panel
numbers it drew. This turns that into `\\newcommand`s, the paper `\\input`s the result,
and the caption then quotes the figure it is printed beside, by construction.

Macro names use spelled-out ordinals because a TeX command name cannot contain digits:

    \\schedEvoMakespanOne   \\schedEvoMissesOne   \\schedEvoTitleOne
    \\schedEvoProvenance    -- what produced the numbers

Usage:
  scripts/emit_figure_numbers.py \\
      --metrics results/codesign_feedback/schedule_evolution_auto_metrics.json \\
      --prefix schedEvo --out results/codesign_feedback/schedule_evolution_numbers.tex
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

ORDINALS = ("One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight")


def tex_escape(s: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    # The composer writes real unicode (an em dash, a middot); pdflatex under the
    # paper's preamble handles those, but a stray one in a macro is worth normalising.
    return s.replace("—", "---").replace("·", r"$\cdot$")


#: LaTeX macro names cannot carry digits or punctuation, so a cell/solver/metric key has
#: to become CamelCase words. Kept as a table rather than a transform so a new metric is a
#: one-line addition and the caption author can read what exists.
_METRIC_WORD = {
    "total_instance_misses": "Misses",
    "median_worst_lateness_ms": "WorstLate",
    "median_makespan_ms": "Makespan",
    "workloads_with_zero_misses": "Cleared",
    "median_instance_misses": "MedMisses",
}


def _camel(s: str) -> str:
    return "".join(w[:1].upper() + w[1:] for w in str(s).replace("-", "_").split("_")
                   if w)


def emit_grid(m, prefix, metrics_path, figure) -> tuple[list[str], list[str]]:
    """Macros for a cell x solver figure, e.g. `\\ablCpsatBMisses`.

    THE SHAPE IS DIFFERENT, not just bigger. A schedule-evolution sidecar is a LIST of
    panels each describing one schedule (title, driver, makespan, misses). An ablation
    sidecar is a GRID: each panel is one metric measured over cells x solver arms. Feeding
    the grid through the panel emitter produced macros that were syntactically fine and
    entirely blank -- `\\ablTitleOne` empty, `\\ablMissesOne` "??" -- which is the worst
    outcome, because a caption then quotes nothing and reads as though it quoted something.

    Ordinals are not used here: a caption wants to say "cell B under CP-SAT", and
    `\\ablCpsatBMisses` says that where `\\ablMissesTwo` does not.
    """
    lines, report = [], []
    solvers = m.get("solvers") or []
    for p in m.get("panels") or []:
        metric = str(p.get("metric") or "")
        word = _METRIC_WORD.get(metric, _camel(metric))
        for solver in solvers:
            for cell, v in sorted(((p.get("values") or {}).get(solver) or {}).items()):
                name = f"{prefix}{_camel(solver)}{cell}{word}"
                val = (f"{v:.1f}" if isinstance(v, float) else str(v))
                lines.append(f"\\newcommand{{\\{name}}}{{{val}}}")
                report.append(f"  \\{name} = {val}")
        lines.append("")
    # the stratum, which any honest caption has to state
    for key, word in (("n_workloads_shown", "NShown"),
                      ("n_workloads_total", "NTotal"),
                      ("stratum", "Stratum")):
        if m.get(key) is not None:
            lines.append(f"\\newcommand{{\\{prefix}{word}}}"
                         f"{{{tex_escape(str(m[key]))}}}")
    lines.append(f"\\newcommand{{\\{prefix}Solvers}}"
                 f"{{{tex_escape(', '.join(str(x) for x in solvers))}}}")
    return lines, report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True,
                    help="the <figure>_metrics.json written by compose_schedule_evolution")
    ap.add_argument("--prefix", default="schedEvo",
                    help="macro prefix; must be letters only (TeX command names)")
    ap.add_argument("--out", required=True, help="path of the .tex to write")
    ap.add_argument("--figure", default=None,
                    help="the figure file these numbers belong to, for provenance")
    a = ap.parse_args()

    if not a.prefix.isalpha():
        print(f"--prefix must be letters only, got {a.prefix!r}", file=sys.stderr)
        return 2
    m = json.load(open(a.metrics))
    panels = m.get("panels") or []
    if not panels:
        print(f"{a.metrics}: no panels", file=sys.stderr)
        return 2
    # WHICH SHAPE IS THIS? A grid sidecar (ablation: cells x solver arms) carries
    # `values` on each panel; a panel sidecar (schedule evolution) carries `title` and
    # `makespan_ms`. Guessing wrong emits blank macros rather than failing, so the
    # dispatch is explicit.
    is_grid = any(isinstance(p.get("values"), dict) for p in panels)
    if not is_grid and len(panels) > len(ORDINALS):
        print(f"{a.metrics}: {len(panels)} panels exceeds the {len(ORDINALS)} "
              f"ordinals this emits", file=sys.stderr)
        return 2

    lines = [
        "% GENERATED by scripts/emit_figure_numbers.py -- do not edit.",
        f"% source: {os.path.relpath(a.metrics)}",
        f"% figure: {a.figure or m.get('figure') or '(unstated)'}",
        f"% written: {datetime.datetime.now().isoformat(timespec='seconds')}",
        "%",
        "% Every number a caption quotes about this figure should come from here, so",
        "% the two cannot disagree. Seven schedule_evolution variants existed and the",
        "% hand-typed caption matched one of them.",
        "",
    ]
    if is_grid:
        glines, greport = emit_grid(m, a.prefix, a.metrics, a.figure)
        lines += glines
        prov = (f"generated from {os.path.basename(a.metrics)} by "
                f"scripts/emit\\_figure\\_numbers.py")
        lines.append(f"\\newcommand{{\\{a.prefix}Provenance}}{{{prov}}}")
        lines.append("")
        out = a.out if os.path.isabs(a.out) else os.path.abspath(a.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        open(out, "w").write("\n".join(lines))
        print(f"wrote {a.out}  (grid: {len(panels)} metric(s) x "
              f"{len(m.get('solvers') or [])} arm(s))")
        for r in greport:
            print(r)
        return 0

    for i, p in enumerate(panels):
        o = ORDINALS[i]
        mk = p.get("makespan_ms")
        misses = p.get("displayed_instance_deadline_misses")
        by = p.get("missed_instances_by_network") or {}
        lines += [
            f"\\newcommand{{\\{a.prefix}Title{o}}}{{{tex_escape(str(p.get('title','')))}}}",
            f"\\newcommand{{\\{a.prefix}Driver{o}}}{{{tex_escape(str(p.get('driver','')))}}}",
            f"\\newcommand{{\\{a.prefix}Costs{o}}}{{{tex_escape(str(p.get('cost_source','')))}}}",
            f"\\newcommand{{\\{a.prefix}Makespan{o}}}{{{mk:.1f}}}"
            if isinstance(mk, (int, float)) else
            f"\\newcommand{{\\{a.prefix}Makespan{o}}}{{??}}",
            f"\\newcommand{{\\{a.prefix}Misses{o}}}{{{misses}}}"
            if misses is not None else
            f"\\newcommand{{\\{a.prefix}Misses{o}}}{{??}}",
            f"\\newcommand{{\\{a.prefix}MissesBy{o}}}{{"
            + tex_escape(", ".join(f"{k}: {v}" for k, v in sorted(by.items())) or "none")
            + "}",
            "",
        ]
    lines.append(f"\\newcommand{{\\{a.prefix}PanelCount}}{{{len(panels)}}}")
    prov = (f"generated from {os.path.basename(a.metrics)} by "
            f"scripts/emit\\_figure\\_numbers.py")
    lines.append(f"\\newcommand{{\\{a.prefix}Provenance}}{{{prov}}}")
    lines.append("")

    out = a.out if os.path.isabs(a.out) else os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write("\n".join(lines))
    print(f"wrote {a.out}  ({len(panels)} panels)")
    for i, p in enumerate(panels):
        print(f"  {ORDINALS[i]:>5}: {p.get('makespan_ms', 0):.1f} ms, "
              f"{p.get('displayed_instance_deadline_misses')} miss(es)  "
              f"{p.get('title','')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
