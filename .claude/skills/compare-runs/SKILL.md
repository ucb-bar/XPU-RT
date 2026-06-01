---
name: compare-runs
description: Compare two scheduler runs (two SchedulerReports, or predicted-vs-actual via postmortem) and report makespan/deadline/granularity deltas honestly, including tradeoffs. Use after a change to check it actually helped.
---

# compare-runs

Compare two runs and reason about the tradeoff — never report a one-sided
"it got better" when one axis regressed.

## Modes

**A. Two schedules (e.g. baseline vs new scheduler/patch):**
```bash
python3 xpu-rt/advisor.py --report <baseline_report.json> --deadline-us <N> | head -4
python3 xpu-rt/advisor.py --report <new_report.json>      --deadline-us <N> | head -4
```
Compare makespan, deadline verdict, bottleneck, and granularity between the two;
state what improved and what regressed.

**B. Predicted vs actual (after a hardware/sim run):**
```bash
python3 -c "import sys; sys.path.insert(0,'xpu-rt'); import postmortem, json; \
print(json.dumps(postmortem.compare_trace('<xpurt_trace.csv>', '<report.json>'), indent=2))"
```
Report makespan predicted-vs-actual delta, RMS/p99 prediction error, and the top
outlier dispatches.

## Rules

- Name the metric(s) that improved AND the metric(s) that regressed; don't claim
  a universal win.
- If comparing schedulers, tie the verdict to the deadline and the user's
  objective, not just raw makespan.
