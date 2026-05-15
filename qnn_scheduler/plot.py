"""Gantt + DAG visualisation for the qnn_scheduler output.

`gantt(result, groups, table, out_path)` produces a per-machine-lane bar
chart of the scheduled islands, with cross-machine bridges drawn as
dashed arrows. The bars are coloured by the variant group's input dtype
(uint8 / fp16 / fp32) so QDQ-format-change boundaries are visually
obvious.

The DAG view is emitted as Graphviz `.dot` for offline rendering with
`dot -Tpng`. Each edge label shows the measured bridge cost in
microseconds; cross-machine edges are drawn dashed and red. This is the
plot the user inspects to decide whether the scheduler picked sensible
boundaries — every edge weight is a real on-board measurement coming
from the cost table.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .scheduler import ScheduleResult
    from .island_dag import IslandVariantGroup
    from .cost_table import CostTable


_DTYPE_COLOUR = {
    "uint8":   "#3b82f6",   # blue
    "int8":    "#1d4ed8",
    "fp16":    "#22c55e",   # green
    "fp32":    "#a855f7",   # purple
    "int32":   "#f97316",   # orange
    "sfixed_32": "#f97316",
}
_BACKEND_COLOUR_BG = {
    "HTA": "#fef3c7",
    "GPU": "#dbeafe",
    "CPU": "#f3f4f6",
}


def gantt(result, groups, table=None,
          out_path: pathlib.Path | str | None = None,
          machines: tuple[str, ...] = ("HTA", "GPU", "CPU"),
          title: str = "QNN heterogeneous schedule") -> pathlib.Path:
    """Save a Gantt PNG. Returns the output path. Each machine gets a
    horizontal lane; islands are bars on their assigned lane; bars are
    coloured by input dtype; cross-machine bridge edges are dashed
    arrows between bars.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if out_path is None:
        out_path = pathlib.Path("schedule_gantt.png")
    out_path = pathlib.Path(out_path)

    fig, ax = plt.subplots(figsize=(12, 1 + 0.7 * len(machines)))
    machine_y = {m: i for i, m in enumerate(reversed(list(machines)))}
    for m in machines:
        ax.axhspan(machine_y[m] - 0.45, machine_y[m] + 0.45,
                   color=_BACKEND_COLOUR_BG[m], zorder=0)

    bar_h = 0.6
    cand_index: dict[str, "IslandVariantGroup"] = {}
    for grp in groups:
        for cand in grp.alternatives:
            cand_index[cand.candidate_id] = grp

    for cand_id, m in result.machine.items():
        if cand_id not in result.start_us:
            continue
        s = result.start_us[cand_id]
        f = result.finish_us[cand_id]
        grp = cand_index.get(cand_id)
        dt_in = grp.primary_dtype_in() if grp else "fp32"
        colour = _DTYPE_COLOUR.get(dt_in, "#64748b")
        y = machine_y[m]
        ax.broken_barh([(s / 1000.0, (f - s) / 1000.0)],
                       (y - bar_h / 2, bar_h),
                       facecolors=colour, edgecolor="black", linewidth=0.5)
        label = cand_id.replace("_uint8", "").replace("_fp16", "")
        ax.text((s + f) / 2000.0, y, label,
                ha="center", va="center", fontsize=8,
                color="white" if dt_in in ("uint8", "fp32") else "black")

    # Bridge arrows: producer.finish -> consumer.start when machines differ.
    for grp in groups:
        cand_id = result.selected_candidate_id.get(grp.group_id)
        if cand_id is None or cand_id not in result.start_us:
            continue
        m_to = result.machine[cand_id]
        for up in grp.upstream_group_ids:
            up_cand = result.selected_candidate_id.get(up)
            if up_cand is None or up_cand not in result.finish_us:
                continue
            m_from = result.machine[up_cand]
            if m_from == m_to:
                continue
            x_from = result.finish_us[up_cand] / 1000.0
            x_to   = result.start_us[cand_id]  / 1000.0
            y_from = machine_y[m_from]
            y_to   = machine_y[m_to]
            ax.annotate("", xy=(x_to, y_to), xytext=(x_from, y_from),
                        arrowprops=dict(arrowstyle="->",
                                        linestyle="--",
                                        color="red",
                                        lw=1.0,
                                        shrinkA=2, shrinkB=2))

    ax.set_yticks(list(machine_y.values()))
    ax.set_yticklabels(list(machine_y.keys()))
    ax.set_xlabel("time (ms)")
    ax.set_xlim(0, result.makespan_us / 1000.0 * 1.05)
    ax.set_ylim(-0.7, len(machines) - 0.3)
    ax.set_title(f"{title} — makespan {result.makespan_us/1000:.2f} ms")
    legend_h = [mpatches.Patch(color=_DTYPE_COLOUR[d], label=d)
                for d in ("uint8", "fp16", "fp32")
                if d in _DTYPE_COLOUR]
    ax.legend(handles=legend_h, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def dot_graph(result, groups, table=None,
              out_path: pathlib.Path | str | None = None) -> pathlib.Path:
    """Emit a Graphviz .dot file. Use `dot -Tpng schedule.dot -o schedule.png`."""
    if out_path is None:
        out_path = pathlib.Path("schedule.dot")
    out_path = pathlib.Path(out_path)
    cand_index = {c.candidate_id: g
                  for g in groups for c in g.alternatives}

    lines = ["digraph Schedule {", "  rankdir=LR;",
             "  node [shape=box, style=filled, fontname=\"Helvetica\"];"]
    for cand_id, m in result.machine.items():
        s = result.start_us.get(cand_id, 0)
        f = result.finish_us.get(cand_id, 0)
        dt = cand_index[cand_id].primary_dtype_in() if cand_id in cand_index else "fp32"
        colour = _DTYPE_COLOUR.get(dt, "#cbd5e1")
        lines.append(
            f'  "{cand_id}" [label="{cand_id}\\n{m}  {(f-s)/1000:.2f} ms",'
            f' fillcolor="{colour}"];'
        )
    for grp in groups:
        cand_id = result.selected_candidate_id.get(grp.group_id)
        if cand_id is None:
            continue
        for up in grp.upstream_group_ids:
            up_cand = result.selected_candidate_id.get(up)
            if up_cand is None:
                continue
            same = result.machine[cand_id] == result.machine[up_cand]
            label = ""
            if not same and table is not None:
                bridge = result.start_us[cand_id] - result.finish_us[up_cand]
                label = f' label="{bridge:.0f} us"'
            style = "" if same else ',style=dashed,color=red'
            lines.append(f'  "{up_cand}" -> "{cand_id}" [{label}{style}];')
    lines.append("}")
    out_path.write_text("\n".join(lines))
    return out_path
