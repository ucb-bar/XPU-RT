---
name: diagnose-schedule
description: Reason about an XPU-RT SchedulerReport — are we meeting the deadline, is dispatch granularity too fine/coarse, and what to try if we're missing. Renders a terminal Gantt and runs the deadline-aware advisor. Use to analyze a schedule before changing it.
---

# diagnose-schedule

Turn a `SchedulerReport` JSON (schema v2, with the per-dispatch list) into an
actionable diagnosis. Reports are emitted by `scripts/run_xpurt_schedule.py` as
`schedules/scheduled_<...>_report.json`.

## Steps

1. **Render the Gantt** so the schedule, deadline line, and late dispatches are
   visible in-terminal:
   ```bash
   python3 xpu-rt/plot_gantt.py --report <report.json> --deadline-us <N> --width 80
   ```

2. **Run the advisor** (deadline verdict + bottleneck + granularity + remedies +
   projected makespan):
   ```bash
   python3 xpu-rt/advisor.py --report <report.json> --deadline-us <N> --gantt
   # or structured: ... --json --emit advice.json
   ```

3. **Explain to the user, grounded in the numbers:**
   - **Deadline:** met or missed, by how much.
   - **Bottleneck:** busiest backend vs idle backends.
   - **Granularity:** the verdict (too_fine / balanced / coarse) and the bucket
     counts — many tiny dispatches ⇒ consider coarser (fuse); one heavy serial
     op while a backend is idle ⇒ consider finer (split).
   - **What to try, and why:** walk the advisor's `rebalance` / `coarsen` /
     `finer` recs with their confidence, and whether the projected makespan
     closes the deadline gap.

4. **Offer next action:** a scoped patch (scope edits via the edit_policy, see
   below), rerun `run_xpurt_schedule.py`, then `/compare-runs`.

## Rules

- The advisor only proposes moving an op to a backend it is *feasible* on
  (`feasible_targets`); trust that — don't suggest illegal placements.
- Don't claim a remedy will work beyond the advisor's projected-makespan check;
  if the top recs don't close the gap, say so.
- To make a scoped change, copy `.claude/edit_policy.example.json` to
  `.claude/edit_policy.json` and set `allow` (the hook then blocks out-of-scope edits).
