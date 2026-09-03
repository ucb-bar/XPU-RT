# Solvers

`xpu-rt/schedulers.py` is a flat registry. Every entry shares one signature —

```python
scheduler(workload, **kwargs) -> (t, alpha, fused_workload, fusion_map)
```

— where `(t, alpha)` feed straight into `postprocessing.output_scheduled_json`
and `schedule_validation.validate_schedule`, so any of them substitutes into
the entry points with no further plumbing.

```bash
.venv/bin/python examples/solvers/compare_solvers.py
```

## Two axes, and they are not the same axis

`scripts/run_xpurt_schedule.py` takes both, and confusing them is easy:

| flag | picks | values |
|---|---|---|
| `--solver` | the **strategy** | `milp`, `greedy`, `greedy_periodic`, `decomposed` |
| `--scheduler` | which **registry entry** the `milp` strategy calls | any registered name |

`--scheduler heft --solver greedy` silently ignores `--scheduler`. The greedy
strategy has its own list scheduler and never consults the registry.

## What is registered

| name | family | notes |
|---|---|---|
| `mosek`, `milp_gurobi`, `milp_highs`, `milp_scip`, `milp_cbc` | CVXPY MILP | exact formulation, different back ends. MOSEK and Gurobi need licences |
| `cpsat`, `cpsat_memory` | OR-Tools CP-SAT | `_memory` adds buffer-residency constraints |
| `heft`, `peft` | list scheduling, upward rank | the usual heterogeneous baselines |
| `critical_path`, `edf`, `fifo`, `round_robin`, `random_list` | list scheduling | `edf` is the one to beat when deadlines matter |
| `min_min`, `max_min`, `fastest_device` | greedy assignment | |
| `simulated_annealing` | metaheuristic | slow; seeded from `random_seed` in the spec |

`available_schedulers()` is the authority; this table is a description of it
and can fall behind.

## MOSEK is a bounded upper bound at this size

The **monolithic MILP does not converge** on a real multi-network workload.
What converges is `scripts/mosek_decompose_by_network.py`, and its own header
says the result is a bound rather than the optimum.

So a table reading "MOSEK 4.1 ms, greedy 4.4 ms" invites the conclusion that
4.1 is optimal and greedy is 7% off it. Neither half of that follows. Label the
MOSEK arm a bounded upper bound in any figure that carries it.

## Makespan is term 7 of 9

Ranking solvers by makespan is the easiest way to pick the wrong one.
`xpu-rt/candidate_objective.py` ranks lexicographically:

1. hard deadline misses
2. max lateness
3. frequency compliance
4. p99 response
5. heavy-model max latency
6. heavy throughput
7. **makespan**
8. energy / utilisation
9. standalone kernel cycles

Its two worked examples are cases where makespan gets it backwards — a split
5% slower in total cycles that lets DroNet meet 30 Hz is a win; a fusion 10%
faster in isolation that creates an 8 ms non-preemptible dispatch and breaks a
100 Hz MLP is a loss.

To adjudicate two candidates:

```bash
python scripts/compare_candidates.py \
    --baseline-schedule A.json --candidate-schedule B.json \
    --windows-from data/toplevel/<spec>.json
```

It refuses to compare terms until `pdb_hash` differs (proving the two solves
read different measured costs) and the per-model instance counts match
(proving they scheduled the same amount of work). A tie is a **rejection**.

## When the choice does not matter

On an uncontended workload every heuristic returns the same schedule. On
`networks_k1_mb` — 28 dispatches with slack — greedy, HEFT, PEFT, EDF,
critical-path, min-min and FIFO all produce 8.0013 ms and zero misses.

That is the expected answer, not a broken comparison. Heuristics separate when
the machine is contended, so compare them on a saturated spec
(`networks_k1_mb_3model_12hz.json`) and expect it to take minutes rather than
seconds.

## Compaction

`schedulers.get_scheduler()` wraps every entry in `_wrap_with_compaction`, so
compaction applies uniformly regardless of which solver produced the schedule.
`compaction_enabled()` reads the environment; a solver comparison run with it
on for one arm and off for another is comparing two different things.

## Related

* [`the_loop.md`](the_loop.md) — where scheduling sits in the cycle
* [`workload_specs.md`](workload_specs.md) — what the solver is given
* `examples/solvers/compare_solvers.py` — runnable
