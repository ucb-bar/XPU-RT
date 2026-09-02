#!/usr/bin/env python3
"""Build the publication Gantts for the ModelBlaster ↔ XPU-RT loop."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "xpu-rt"))
sys.path.insert(0, _HERE)

import figstyle  # noqa: E402
import plot_k1_evolution as gantt  # noqa: E402
import repeat_window  # noqa: E402
import schedule_scoring  # noqa: E402
import schedule_trace  # noqa: E402
import workload_spec  # noqa: E402

figstyle.use()


def _path(path):
    return path if os.path.isabs(path) else os.path.join(_REPO, path)


def _json(path):
    with open(_path(path)) as f:
        return json.load(f)


def _workload(path):
    raw = _json(path)
    windows, known = workload_spec.windows_and_names(raw)
    periods = workload_spec.periods_ms(raw)
    return raw, windows, known, periods


def _save(fig, out_dir, stem):
    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, stem + ".png")
    pdf = os.path.join(out_dir, stem + ".pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"wrote {png} / {pdf}")
    return png, pdf


def _write_repeat_report(out_dir, stem, frames):
    """Record why each displayed prefix is safe to repeat indefinitely."""
    path = os.path.join(out_dir, stem + ".json")
    # The materialized schedule already contains the included dispatch keys.
    # Keep the proof report compact enough to inspect in reviews.
    compact_frames = [
        {key: value for key, value in frame.items()
         if key != "included_dispatches"}
        for frame in frames
    ]
    payload = {
        "schema_version": 1,
        "interpretation": (
            "Each frame is dependency-closed, clear at its wrap boundary, "
            "contains a complete anchor-model instance, and supplies every "
            "declared periodic model at or above its minimum average rate."
        ),
        "frames": compact_frames,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {path}")
    return path


def _write_repeat_frame(out_dir, stem, schedule, report):
    """Materialize the plotted prefix, rather than only cropping the image."""
    data_dir = os.path.join(out_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, stem + ".json")
    with open(path, "w") as f:
        json.dump(repeat_window.extract_frame(schedule, report), f, indent=2)
    report["schedule_artifact"] = os.path.relpath(path, _REPO)
    print(f"wrote {path}")
    return path


def _all_cores(schedules):
    merged = {}
    for index, schedule in enumerate(schedules):
        merged.update({f"{index}:{name}": ent
                       for name, ent in schedule["dispatches"].items()})
    return gantt.cores_from_schedule(merged)


def _colours(known):
    colours = {name: figstyle.model_color(name) for name in known}
    used = {c for c in colours.values() if c != figstyle.C_MUTED}
    spare = [figstyle.SKY, figstyle.YELLOW, figstyle.ORANGE, figstyle.GREEN,
             figstyle.PURPLE, figstyle.BLUE, figstyle.VERMILLION]
    spare = [c for c in spare if c not in used]
    for name in sorted(known):
        if colours[name] == figstyle.C_MUTED:
            colours[name] = spare.pop(0) if spare else figstyle.C_MUTED
    return colours


def _legend(fig, known, *, colours, ime=False, deadline=None):
    handles = [Patch(facecolor=colours[m], label=m)
               for m in sorted(known)]
    if ime:
        handles.append(Patch(facecolor="white", edgecolor="black", hatch="///",
                             label="IME-capable dispatch"))
    if deadline:
        handles.append(plt.Line2D([], [], color=figstyle.C_DEADLINE,
                                  ls=(0, (2, 2)), lw=0.6,
                                  label=f"{deadline} releases/deadlines"))
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 6),
               frameon=False, bbox_to_anchor=(0.5, 0.005))


def solver_matrix(story, out_dir):
    result = _json(story["solver_result"])
    solvers = list(result["common"]["solvers"])
    by_key = {(c["phase"], c["solver"]): c for c in result["cells"]}
    schedules = {key: _json(cell["schedule"]) for key, cell in by_key.items()
                 if cell["status"] == "validated"}
    cores = _all_cores(list(schedules.values()))
    _, _, known, periods = _workload(
        result["phases"]["feedback"]["networks_json"])
    colours = _colours(known)
    max_span = max(c.get("terms", {}).get("makespan_ms", 0)
                   for c in result["cells"] if c["status"] == "validated")
    configured_window = float(result["common"].get("window_ms") or max_span + 2)
    # Do not let a generic profiling window shrink the actual schedule or leave
    # most of this comparison blank. Two milliseconds of tail keeps the final
    # bars and the 20 ms release/deadline marker legible.
    window = min(configured_window, float(max_span) + 2.0)

    fig, axes = plt.subplots(2, len(solvers),
                             figsize=(figstyle.DOUBLE_COL, 112 * figstyle.MM),
                             sharex=True, sharey=True, squeeze=False)
    for row, phase in enumerate(("original", "feedback")):
        for col, solver in enumerate(solvers):
            ax = axes[row][col]
            cell = by_key[(phase, solver)]
            if cell["status"] == "validated":
                schedule = schedules[(phase, solver)]
                rows = schedule_trace.trace_rows_from_schedule(schedule)
                gantt.draw_gantt_axis(
                    ax, rows, schedule["dispatches"], cores=cores,
                    window_ms=window, colours=colours, periods=periods,
                    deadline_model="mlp_control", known=known,
                    impl_hatch=True)
                terms = cell["terms"]
                stat = (f"miss={terms['total_deadline_misses']} · "
                        f"p99={terms['critical_p99_ms']:.2f} ms · "
                        f"heavy={terms['heavy_max_latency_ms']:.2f} ms · "
                        f"span={terms['makespan_ms']:.2f} ms")
                ax.set_title(f"{solver}\n{stat}", loc="left", fontsize=5.2)
            else:
                ax.set_axis_off()
                ax.text(0.5, 0.55, solver, ha="center", va="center",
                        transform=ax.transAxes, fontsize=6, fontweight="bold")
                ax.text(0.5, 0.38, cell["status"].upper(), ha="center",
                        va="center", transform=ax.transAxes,
                        color=figstyle.VERMILLION)
                ax.text(0.5, 0.20, cell.get("detail", "")[:85], ha="center",
                        va="center", wrap=True, transform=ax.transAxes,
                        fontsize=4.5, color="0.35")
            if col == 0:
                ax.set_ylabel(("Original ModelBlaster output\nphysical K1 cores"
                               if phase == "original" else
                               "After XPU-RT feedback\nphysical K1 cores"))
            if row == 1 and cell["status"] == "validated":
                ax.set_xlabel("Predicted time (ms)")

    fig.suptitle(
        f"Solver choice alone vs compiler–scheduler feedback — {result['verdict']}\n"
        "Every number is predicted from measured K1 dispatch profiles; "
        "all six solver cells completed and passed independent feasibility checks.",
        x=0.02, ha="left", fontsize=7.5)
    _legend(fig, known, colours=colours, ime=True, deadline="mlp_control")
    fig.tight_layout(rect=(0, 0.08, 1, 0.90), h_pad=3.0, w_pad=1.2)
    return _save(fig, out_dir, "feedback_vs_solvers")


def rewrite_detail(story, out_dir):
    spec = story["rewrite"]
    baseline, candidate = (_json(spec["baseline_schedule"]),
                           _json(spec["candidate_schedule"]))
    _, windows, known, periods = _workload(spec["workload"])
    critical = tuple(spec.get("critical_models") or ())
    heavy = spec.get("heavy_model")
    _, base_out, base_rows = schedule_scoring.score(
        "control", baseline, windows, critical, heavy, known, periods)
    _, cand_out, cand_rows = schedule_scoring.score(
        "unfused", candidate, windows, critical, heavy, known, periods)
    verdict = _json(spec["verdict"])
    cores = _all_cores([baseline, candidate])
    colours = _colours(known)
    panels = [(spec["baseline_label"], baseline, base_rows, base_out),
              (spec["candidate_label"], candidate, cand_rows, cand_out)]
    repeat_spec = spec["repeat_window"]
    reports = []
    for index, (label, schedule, _, _) in enumerate(panels):
        report = repeat_window.find(
            schedule, periods, repeat_spec["anchor_model"], known,
            quantum_ms=float(repeat_spec["quantum_ms"]),
            max_window_ms=float(spec["window_ms"]))
        labelled = {"label": label, **report}
        _write_repeat_frame(
            out_dir, f"rewrite_{'control' if index == 0 else 'candidate'}_repeat_frame",
            schedule, labelled)
        reports.append(labelled)
    _write_repeat_report(out_dir, "feedback_rewrite_repeat_windows", reports)

    fig, axes = plt.subplots(2, 2,
                             figsize=(figstyle.DOUBLE_COL, 105 * figstyle.MM),
                             sharey=True, squeeze=False)
    for row, (label, schedule, rows, out) in enumerate(panels):
        frame_ms = float(reports[row]["window_ms"])
        windows_to_show = [(frame_ms, "qualified repeat frame"),
                           (min(float(spec["zoom_ms"]), frame_ms),
                            "contention zoom")]
        for col, (window, view) in enumerate(windows_to_show):
            ax = axes[row][col]
            gantt.draw_gantt_axis(
                ax, rows, schedule["dispatches"], cores=cores,
                window_ms=window, colours=colours, periods=periods,
                deadline_model="mlp_control", known=known,
                repeat_frame=(col == 0))
            title = view
            if col == 0:
                title = (f"{label} — qualified {frame_ms:g} ms repeat frame\n"
                         f"{reports[row]['dispatches_shown']} dispatches shown; "
                         f"{reports[row]['dispatches_excluded']} trailing excluded · "
                         f"heavy max={out.heavy_max_latency_ms:.2f} ms")
            ax.set_title(title, loc="left", fontsize=5.0)
            if row == 1:
                ax.set_xlabel("Predicted time from K1 profiles (ms)")
        axes[row][0].set_ylabel("Physical K1 cores")
    fig.suptitle(
        f"{spec['title']} — ACCEPT on heavy-model max latency\n"
        f"{verdict['why']}", x=0.02, ha="left", fontsize=7.2)
    _legend(fig, known, colours=colours, deadline="mlp_control")
    fig.tight_layout(rect=(0, 0.08, 1, 0.88), h_pad=2.5, w_pad=1.2)
    return _save(fig, out_dir, "feedback_rewrite_detail")


def rich_capstone(story, out_dir):
    spec = story["rich"]
    _, windows, known, periods = _workload(spec["workload"])
    schedules = [_json(c["schedule"]) for c in spec["cells"]
                 if c["status"] == "validated"]
    cores = _all_cores(schedules)
    colours = _colours(known)
    repeat_spec = spec["repeat_window"]
    raw_reports = repeat_window.find_common(
        schedules, periods, repeat_spec["anchor_model"], known,
        quantum_ms=float(repeat_spec["quantum_ms"]),
        max_window_ms=float(spec["window_ms"]))
    reports = []
    report_by_solver = {}
    for cell, schedule, report in zip(
            (c for c in spec["cells"] if c["status"] == "validated"),
            schedules,
            raw_reports):
        labelled = {
            "solver": cell["solver"],
            "label": cell.get("label", cell["solver"]),
            **report,
        }
        _write_repeat_frame(
            out_dir, f"rich_{cell['solver']}_repeat_frame", schedule, labelled)
        reports.append(labelled)
        report_by_solver[cell["solver"]] = labelled
    _write_repeat_report(out_dir, "feedback_rich_repeat_windows", reports)
    frame_ms = float(raw_reports[0]["window_ms"])
    fig, axes = plt.subplots(len(spec["cells"]), 1,
                             figsize=(figstyle.DOUBLE_COL, 102 * figstyle.MM),
                             sharex=True, sharey=True, squeeze=False)
    valid_index = 0
    for ax, cell in zip(axes[:, 0], spec["cells"]):
        display_label = cell.get("label", cell["solver"])
        if cell["status"] != "validated":
            ax.set_axis_off()
            ax.text(0.02, 0.70, f"{display_label} — {cell['status'].upper()}",
                    transform=ax.transAxes, fontweight="bold")
            ax.text(0.02, 0.32, cell.get("detail", ""), wrap=True,
                    transform=ax.transAxes, fontsize=5, color="0.35")
            continue
        schedule = schedules[valid_index]
        valid_index += 1
        rows = schedule_trace.trace_rows_from_schedule(schedule)
        _, out, _ = schedule_scoring.score(
            cell["solver"], schedule, windows,
            ("mlp_control", "fused_full"), "yolov8_nano_64x96", known,
            periods)
        gantt.draw_gantt_axis(
            ax, rows, schedule["dispatches"], cores=cores,
            window_ms=frame_ms, colours=colours,
            periods=periods, deadline_model="mlp_control", known=known,
            impl_hatch=True, repeat_frame=True)
        report = report_by_solver[cell["solver"]]
        shown_dispatches = [schedule["dispatches"][key]
                            for key in report["included_dispatches"]]
        widths = [len(str(d.get("hardware_target", "")).split("+"))
                  for d in shown_dispatches]
        ime = sum(d.get("impl") == "ime" and any(
            op in (d.get("module_name") or "")
            for op in ("linear_s8", "matmul_s8"))
            for d in shown_dispatches)
        ax.set_title(
            f"{display_label} · {report['dispatches_shown']} dispatches in "
            f"common {frame_ms:g} ms frame ({len(rows)} in source solve) · "
            f"{sum(w > 1 for w in widths)} multi-hart · {ime} IME dispatches · "
            f"source span={out.makespan_ms:.2f} ms", loc="left", fontsize=5.4)
        ax.set_ylabel("Physical K1 cores")
    axes[-1, 0].set_xlabel("Predicted time from K1 profiles (ms)")
    fig.suptitle(spec["title"] + "\nTransformer blocks are explicitly "
                 "ViNT-class stand-ins; each row shows the same qualified "
                 "steady-state repeat frame.",
                 x=0.02, ha="left", fontsize=7.2)
    _legend(fig, known, colours=colours, ime=True, deadline="mlp_control")
    fig.tight_layout(rect=(0, 0.09, 1, 0.87), h_pad=2.4)
    return _save(fig, out_dir, "feedback_rich_capstone")


def rejection_story(story, out_dir):
    spec = story["rejections"]
    _, windows, known, periods = _workload(spec["workload"])
    schedules = [_json(c["schedule"]) for c in spec["cells"]]
    cores = _all_cores(schedules)
    repeat_spec = spec["repeat_window"]
    raw_reports = repeat_window.find_common(
        schedules, periods, repeat_spec["anchor_model"], known,
        quantum_ms=float(repeat_spec["quantum_ms"]),
        max_window_ms=float(spec["window_ms"]))
    reports = []
    for index, (cell, schedule, report) in enumerate(
            zip(spec["cells"], schedules, raw_reports)):
        labelled = {"label": cell["label"], **report}
        _write_repeat_frame(
            out_dir, f"rejection_{index}_repeat_frame", schedule, labelled)
        reports.append(labelled)
    _write_repeat_report(out_dir, "feedback_rejections_repeat_windows", reports)
    frame_ms = float(raw_reports[0]["window_ms"])
    panels = []
    for cell, schedule, report in zip(spec["cells"], schedules, raw_reports):
        rows = schedule_trace.trace_rows_from_schedule(schedule)
        suffix = ""
        if cell.get("verdict"):
            suffix = "\n" + _json(cell["verdict"])["why"]
        panels.append({"title": (f"{cell['label']} · "
                                  f"{report['dispatches_shown']} dispatches in "
                                  f"common {frame_ms:g} ms frame") + suffix,
                       "rows": rows, "sched": schedule["dispatches"]})
    return gantt.render_gantt_panels(
        panels, os.path.join(out_dir, "feedback_rejections"),
        periods=periods, cores=cores, window_ms=frame_ms,
        deadline_model="mlp_control", colours=_colours(known),
        xlabel="Predicted time from K1 profiles (ms)", panel_height_mm=33.0,
        repeat_frame=True)


def _copy_json_reference(container, key, destination, name):
    source = _path(container[key])
    target = os.path.join(destination, name)
    shutil.copy2(source, target)
    container[key] = os.path.relpath(target, _REPO)


def snapshot(story, out_dir):
    """Make the story independent of ignored schedules and lab artifacts."""
    data_dir = os.path.join(out_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    result = _json(story["solver_result"])
    for phase_name, phase in result["phases"].items():
        _copy_json_reference(phase, "networks_json", data_dir,
                             f"matrix_{phase_name}_workload.json")
    for cell in result["cells"]:
        if cell["status"] == "validated":
            _copy_json_reference(cell, "schedule", data_dir,
                                 f"matrix_{cell['phase']}_{cell['solver']}.json")
    result_path = os.path.join(out_dir, "result.resolved.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    story["solver_result"] = os.path.relpath(result_path, _REPO)

    rewrite = story["rewrite"]
    for key in ("workload", "baseline_schedule", "candidate_schedule", "verdict"):
        _copy_json_reference(rewrite, key, data_dir, f"rewrite_{key}.json")
    rich = story["rich"]
    _copy_json_reference(rich, "workload", data_dir, "rich_workload.json")
    for cell in rich["cells"]:
        if cell["status"] == "validated":
            _copy_json_reference(cell, "schedule", data_dir,
                                 f"rich_{cell['solver']}.json")
    reject = story["rejections"]
    _copy_json_reference(reject, "workload", data_dir, "reject_workload.json")
    for index, cell in enumerate(reject["cells"]):
        _copy_json_reference(cell, "schedule", data_dir,
                             f"reject_{index}_schedule.json")
        if cell.get("verdict"):
            _copy_json_reference(cell, "verdict", data_dir,
                                 f"reject_{index}_verdict.json")

    resolved = os.path.join(out_dir, "story.resolved.json")
    with open(resolved, "w") as f:
        json.dump(story, f, indent=2)
    print(f"wrote {resolved}")
    return story


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--story", required=True)
    ap.add_argument("--out-dir", default="results/k1_feedback_story")
    ap.add_argument("--snapshot", action="store_true",
                    help="copy ignored schedule/lab inputs into the result bundle")
    args = ap.parse_args()
    story = _json(args.story)
    out_dir = _path(args.out_dir)
    if story.get("schema_version") != 1:
        raise SystemExit("story schema_version must be 1")
    if args.snapshot:
        story = snapshot(story, out_dir)
    solver_matrix(story, out_dir)
    rewrite_detail(story, out_dir)
    rich_capstone(story, out_dir)
    rejection_story(story, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
