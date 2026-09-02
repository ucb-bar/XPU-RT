# K1 feedback story

This bundle is the reproducible evidence for the ModelBlaster ↔ XPU-RT
feedback section in `docs/Feature/the_loop.md`.

## Headline result

On the four-model, 55-operation solver matrix, XPU-RT feedback exposes measured
multi-hart ModelBlaster implementations. The resulting Greedy schedule is
accepted against every validated original-output schedule by the repository's
nine-term objective:

- versus original Greedy: critical p99 8.00 → 4.89 ms (term 4), while FFN
  max latency also falls 16.55 → 11.56 ms;
- versus original CP-SAT: critical p99 10.83 → 4.89 ms;
- versus original MOSEK: critical p99 20.00 → 4.89 ms (term 4).

All six matrix solves completed without time limits and independently pass the
physical-core, dependency, target, and implementation feasibility gates. This
is the intended result: compiler–scheduler feedback beats Greedy and also beats
using CP-SAT or MOSEK alone on the unchanged ModelBlaster graph.

## Files

- `result.json`: hashes, gates, solver statuses, objective terms, and verdicts.
- `data/original_xpurt_feedback.json`: the original Greedy schedule's raw
  XPU-RT `prefer_finer` hints; the benchmark verifies its hash, transformed
  targets, and source-schedule hash.
- `workload_changes` in the result is the complete semantic input diff; the
  evaluator rejects any change not declared by the feedback transformation.
- `experiment.resolved.json`: self-contained matrix inputs.
- `story.resolved.json`: self-contained figure inputs.
- `data/`: exact workload, schedule, and verdict fixtures.
- `feedback_vs_solvers.*`: original versus feedback-expanded solver matrix.
- `feedback_rewrite_detail.*`: accepted YOLO graph rewrite.
- `feedback_*_repeat_windows.json`: machine-readable proof that every displayed
  steady-state frame is dependency-closed, boundary-clear, and frequency-safe.
- `data/*_repeat_frame.json`: executable schedule prefixes marked
  `repeat_indefinitely`; these remove the misleading trailing DroNet-only work.
- `feedback_rich_capstone.*`: independently feasible Greedy and corrected
  CP-SAT schedules for the five-network, 217-operation workload, shown over a
  common 60 ms repeat frame with physical multi-hart spans and IME dispatches.
- `rich_mosek_resource_exhaustion.json`: evidence from three no-time-limit
  rich MOSEK attempts. No incumbent was plotted or ranked; the last attempt
  was stopped at 89.1 GiB RSS with swap full to protect the host.
- `feedback_rejections.*`: DroNet split negative controls.

All placements are predicted schedules using measured K1 dispatch profiles;
they are not presented as observed board traces. See `docs/Feature/the_loop.md` for the
interpretation, limitations, and the reusable `scripts/extract_repeat_window.py`
postprocessor.
