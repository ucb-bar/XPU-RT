---
name: close-loop
description: Drive one full predicted→measured→re-advise iteration of the XPU-RT ⇄ ModelBlaster granularity loop on the FireSim 3-model workload — diagnose the baseline, propose a candidate bundle (Contracts 1 & 2), hand off to ModelBlaster for measured FireSim runs, and re-advise on the actual numbers. Use to close one round of axis-A/B/C iteration end-to-end.
---

# close-loop

Orchestrate the iterative scheduling-improvement loop end-to-end:

```
baseline ──► advise ──► propose bundle ──► [ModelBlaster: realize + FireSim] ──► re-advise on measured
```

The XPU-RT side (this skill) does prediction, bundle proposal, and
measured re-diagnosis. The ModelBlaster side (`/realize-and-run`)
realizes axis-C fusion hints, builds the bundle ELFs, runs FireSim,
and emits measured `SchedulerReport`s for the loop-back.

See `docs/iterative_firesim_loop.md` for the contracts (Contract 1 =
`xpurt.candidate_bundle/v1`, Contract 2 =
`modelblaster.fusion_hints/v1`).

## Steps

1. **Produce the predicted baseline + candidate bundle** in one
   command:
   ```bash
   bash scripts/demo_iterate_firesim.sh
   ```
   This calls `scripts/iterate_firesim.py` (axes A/B predicted) and
   `scripts/granularity_loop.py` (axis C predicted, emits fusion or
   split hint). Outputs land in `artifacts/iterate/`:
   - `firesim_batch.json` — Contract 1 candidate set.
   - `granularity_hint.json` — Contract 2 (fusion or split) hint.
   - `before_after_gantt.png` — predicted-only before/after.
   - `report.md` — human-readable summary.

2. **Brief the user** on what the predicted analysis says: which
   candidate beats the baseline, what axis it's on, what the
   advisor diagnosed (granularity / bottleneck / rebalance). Quote
   the deadline used and the projected makespan.

3. **Hand off to ModelBlaster** for the measured run. In a session
   rooted at `/scratch2/agustin/ModelBlaster`, invoke
   `/realize-and-run` with:
   - the `firesim_batch.json` path,
   - the `granularity_hint.json` path (for axis-C realization),
   - the target output dir (typically `artifacts/bundle/`).
   ModelBlaster builds the candidate ELFs, queues them under one
   `FIRESIM_QUEUE=1` infrasetup, and emits per-candidate
   `xpurt_trace.csv` + `measured_report.json`.

4. **Re-advise on measured numbers** once the traces are back. For
   each candidate:
   ```bash
   python3 xpu-rt/advisor.py \
       --report /scratch2/agustin/ModelBlaster/artifacts/bundle/<id>/measured_report.json \
       --deadline-us <N> --gantt
   python3 xpu-rt/plot_gantt.py \
       --trace  /scratch2/agustin/ModelBlaster/artifacts/bundle/<id>/xpurt_trace.csv \
       --out    artifacts/iterate/measured_gantt_<id>.png
   ```
   The advisor's measured verdict may differ from the predicted one
   when launch overhead, contention, or actual cycle counts diverge
   from the profile model.

5. **Decide the next round.** If the measured winner meets the
   deadline, stop and report the round-N result. Otherwise call
   `xpu-rt/bundle.py:propose_bundle` with the measured report to
   build a Round-(N+1) `firesim_batch.json` and loop to Step 3.
   Cap at 3 rounds before pausing to discuss with the user.

## Rules

- **The predicted Δ is not the measured Δ** for axis-C: the cost
  model has no per-dispatch launch overhead, so fusion's predicted
  payoff is bounded by removed cross-device transitions. The real
  signal comes from FireSim — quote both when reporting.
- **Surface PASS/FAIL per candidate, not just makespan.** A faster
  candidate that fails ModelBlaster's `MODELBLASTER_VERIFY` is a
  regression, not a winner.
- **Don't propose hints the advisor didn't generate.** `bundle.py`'s
  `propose_bundle` and `fusion_hints_from_diagnosis` are the source
  of truth; hand-tweaking the JSON drifts the demo away from the
  reasoning chain.
- **Use the same workload** for the whole loop unless explicitly
  switching — re-running with a different `networks_*.json` between
  rounds invalidates the predicted/measured comparison.
- **Loop cap at 3 rounds** without user check-in; FireSim infrasetup
  is ~15–30 min per bundle and runaway iteration burns shared
  resources.
