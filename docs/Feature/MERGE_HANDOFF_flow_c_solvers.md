# Handoff: merging `flow-c-qnn-qrb5165`'s solver work into `dev`

Branch `merge/flow-c-solvers`. Five of the eight conflicts are resolved here
and should not need revisiting. **Three are NOT resolved** — they are parked at
dev's side as placeholders so the branch is committable, and taking this branch
as-is would drop the new solvers. They need the author of `528129a4`.

## Resolved (5) — reasoning, so they are not re-litigated

| file | resolution | why |
|---|---|---|
| `data/banks/hardware_bank.json` | **union** | neither side is a superset: dev has 202 unique keys, flow-c 14 (`configs.spike_rv`) |
| `data/banks/model_bank.json` | **union** | dev 65 unique, flow-c 4 (spike `max_copies` / `variant_note`) |
| `.gitignore` | **dev's** | dev's is 170 lines against flow-c's 17; flow-c's only additions are three `plots/*.png` paths already covered by dev's `/plots/` rule (checked with `git check-ignore`) |
| `scripts/repro_mlp_dronet_yolo_spike.sh` | **flow-c's** | dev reduced it to a shim forwarding to `repro_workload.sh`, but that target has **zero `--solver` support**, so the shim would break the new solvers |
| `README.md` | **flow-c's text, dev's paths** | flow-c's new solver list (`greedy_reserved`, `auto`) is the newer content, but it links `docs/mlp_dronet_yolo_qnn_reproduction.md`, which moved to `docs/Qualcomm/` — kept as a live link |

## NOT resolved (3) — and why a merge tool cannot do it

The two sides are not near-copies. flow-c branched before three subsystems
landed on dev and then rewrote the same files for the new solvers, so the
conflict is between two designs, not two edits:

| file | dev lines | flow-c lines | differing | dev subsystems flow-c has **zero** refs to |
|---|---:|---:|---:|---|
| `scripts/run_xpurt_schedule.py` | 1184 | 1085 | 1107 | `contention_model` (8), `freshness_weight` (5), `emit_feedback` (3), `configure_contention` (1) |
| `xpu-rt/greedy_scheduler.py` | 902 | 968 | 422 | `configure_contention` (4), `contention_model` (3) |
| `xpu-rt/scheduler.py` | 1320 | 961 | 770 | `freshness_weight` (3) |

Neither side can be taken whole: flow-c's drops all of the above, dev's drops
the four new solvers, the CP-SAT backend and the selectable MILP backend.

Mechanical union fails too, three ways:

1. **It does not parse.** The refinement loop was re-indented into a
   per-candidate loop for `auto`, so ours and theirs interleave mid-expression
   and a union splices a `sum(` that is never closed.
2. **It would bypass the contention hook.** `greedy_scheduler.py` hunks 2 and 4
   are dev computing `min_dur` inline through `_duration()` — which is where
   dev's contention multiplier is applied — against flow-c extracting it to
   `_min_durations(workload, machine_combinations)`. Taking the helper silently
   removes contention from the greedy path.
3. **It may double-apply a constraint.** `greedy_scheduler.py` hunk 3 is 13
   lines of dev's "wait for any overlapping combination holding a running op".
   flow-c deleted it and still references `combinations_overlap` six times to
   dev's seven, so it was probably restructured rather than dropped — but
   whether it now lives elsewhere is a question about their design.

## One regression to watch for, whichever way it is resolved

flow-c's rewrite of `run_xpurt_schedule.py` **drops the purely-periodic
makespan reporting**:

```python
if effective_restrict_makespan_to_nonperiodic and iter_makespan <= 0.0:
    print(f"Final greedy makespan: {iter_makespan_all:.2f} ms "
          f"(... over all operations, this workload has no non-periodic work)")
```

Without it a workload of nothing but periodic networks prints `0.00 ms`, because
`C_max` tracks only non-periodic operations and that set is empty. This matters
beyond cosmetics: `sweep_random_workloads.py` **parses that line**, so it would
ingest a zero as a real measurement rather than failing. The companion
constraint fix in `xpu-rt/scheduler.py` (bounding `C_max` over all operations
when `not non_periodic_ops_exist`, without which MOSEK raises `SolverError`) is
present on both sides and is not at risk.

That is one bug found by chance in a 1107-line divergence. It is not evidence
it is the only one.

## Suggested route

Merge `dev` into `flow-c-qnn-qrb5165` from the flow-c side, or rebase the
solver work onto current `dev`. The open question — whether the contention and
freshness hooks belong inside the new per-candidate loop or outside it — is a
design decision in the new code.
