# Telemetry → Calibration Feedback

XPU-RT runs in a closed loop: compile → deploy → measure → recalibrate
→ recompile. The telemetry-feedback path turns on-board measurements
collected by `xpurt_scheduler_runner` into typed dispatch hints that
update the compiler's promotion ladder and recipe cache.

## Data flow

```
runtime/tools/xpurt_scheduler_runner.c
                │
                │ writes  --telemetry_jsonl=<path>
                ▼
         per-dispatch JSONL records
         (start_us, end_us, machine, status)
                │
                ▼
xpu_rt.scheduler.streaming_feedback (daemon mode)
                │
                ▼
xpu_rt.scheduler.feedback.derive_dispatch_hints(workload, t, alpha, telemetry)
                │
                │ returns FeedbackHints:
                │   - pin_target          (force op to a specific cluster)
                │   - prefer_finer        (split fused ops)
                │   - prefer_coarser      (fuse small ops)
                │   - fuse_hint           (cross-cluster cache-state penalty observed)
                │   - skipped_in_model    (op failed deadline; mark skip_allowed=True)
                │
                ▼
xpu_rt.promotion.calibration_bridge (Phase F follow-up)
                │
                ├─► update recipe_cache[recipe_id].calibration_status
                │       not_collected → cuda_collected → perf_collected (M-22)
                │
                ├─► trigger M-15B retry detector when tile-size mismatch
                │       crosses analytical-vs-measured threshold
                │
                └─► append caveat_ledger entry when calibration_delta is large
                        ("analytical_cost_model_disagrees_with_board_measurement")
```

## Live feedback during scheduling

```python
from xpu_rt.scheduler.feedback import derive_dispatch_hints

hints = derive_dispatch_hints(
    workload=wl,
    t_solved=t,
    alpha_solved=alpha,
    telemetry_jsonl="run.telemetry.jsonl",
    run_id="run-2026-05-15-T0001",
)
for h in hints:
    print(h.kind, h.op_id, h.detail)
```

`feedback.py` deliberately stays solver-agnostic — it consumes telemetry
JSONL + the solved `(t, alpha)` arrays and emits typed `Hint` objects.
This makes it usable both online (during a single scheduling round) and
offline (post-deployment analysis).

## Streaming daemon

For real-time replanning, `streaming_feedback.py` runs as a host-side
daemon that tails the telemetry JSONL stream and drips hints into a
queue the next compilation round consumes.

```bash
python -m xpu_rt.scheduler.streaming_feedback \
    --telemetry run.telemetry.jsonl \
    --workload workload.json \
    --output xpurt_feedback.json \
    --watch
```

## What we promise vs what's coming

| Today | Coming (Phase F+) |
|---|---|
| Hints derive cleanly, typed, byte-stable across reruns. | `xpu_rt.promotion.calibration_bridge` consumes them automatically. |
| Hints flow into the next `heterogeneous_loop` round via JSON. | Closed loop through the `xpu_rt.solve` registry — recipe cache updates without a manual round-trip. |
| `xpu_rt.audit.caveat_ledger` accepts manual entries. | Calibration deltas auto-write typed caveats when above threshold. |

## See also

- [Solver-backend integration](solver-backend.md) — where the calibrated
  cost feeds back into.
- `architecture/promotion-and-memory.md` — the promotion gate ladder this
  feedback path eventually plugs into.
- `xpu-rt/python/xpu_rt/scheduler/feedback.py` — hint definitions and
  derivation rules.
- `xpu-rt/python/xpu_rt/scheduler/test_feedback_derivation.py` — 8 unit
  tests covering each hint kind.
