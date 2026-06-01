# XPU-RT — Claude Code guidance

This file documents the **scheduler-reasoning workflow** added on top of the
`xpurt` scheduling stack. It is additive; it does not change how the rest of the
repo is built or tested.

## Producing and reasoning about a schedule

```bash
# 1. schedule a benchmark (emits schedule + metrics + a SchedulerReport).
#    --solver {milp,greedy,greedy_periodic,decomposed};  --solver milp picks a
#    registry algorithm via --scheduler {mosek,heft,peft,edf,cpsat,milp_*,...}.
python3 scripts/run_xpurt_schedule.py \
  --networks-json data/toplevel/networks_mlp10_dronet20_yolov8_firesim_static_q31profile.json \
  --solver decomposed
#    -> schedules/scheduled_<base>_<tag>_report.json   (schema v2, per-dispatch list)

# 2. diagnose it: deadline? granularity? what to try (rebalance/coarsen/finer)?
python3 xpu-rt/advisor.py --report <report.json> --deadline-us <N> --gantt

# 3. visualize the schedule in-terminal (deadline marker, shaded late dispatches)
python3 xpu-rt/plot_gantt.py --report <report.json> --deadline-us <N>

# 4. profile multiple schedulers on one benchmark and compare
python3 scripts/profile_schedulers.py --networks-json <...> \
  --schedulers decomposed,heft,peft,edf,fifo,round_robin --deadline-us <N>
```

Slash-command equivalents live in `.claude/skills/`: `/diagnose-schedule`,
`/sweep-schedulers`, `/compare-runs`.

## Key modules (reuse, don't duplicate)

- `xpu-rt/profiling.py` — `SchedulerReport` (schema v2: `dispatches` list with
  `target`, `start/finish/duration_us`, `deps`, `feasible_targets`).
- `xpu-rt/advisor.py` — `advise_schedule(report, deadline_us=..., workload=...)`:
  deadline verdict, bottleneck, granularity, and rebalance/coarsen/finer recs
  with a projected-makespan check. Composes `fusion_advisor` + `dag_analysis` +
  `metrics`; never proposes an infeasible placement.
- `xpu-rt/plot_gantt.py` — `render_terminal_gantt(report, deadline_us=...)` (ASCII)
  plus the existing PNG renderers.
- `xpu-rt/metrics.py`, `xpu-rt/fusion_advisor.py`, `xpu-rt/postmortem.py`,
  `xpu-rt/schedulers.py` (registry: `available_schedulers()` / `get_scheduler()`).

## Conventions

- Run tools with `python3`. Tests: `cd xpu-rt && python3 -m unittest discover -s tests`.
- **Scoped edits:** for a targeted change, copy `.claude/edit_policy.example.json`
  to `.claude/edit_policy.json` and set `allow`/`deny`; the
  `deny_forbidden_edits` PreToolUse hook then blocks out-of-scope edits. Without
  that file the hook is inert.
- modelblaster lives in the `zephyr-chipyard-sw` submodule
  (`git submodule update --init zephyr-chipyard-sw`).
