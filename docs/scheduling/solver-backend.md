# Solver-Backend Integration

The XPU-RT two-cluster CVXPY scheduler is registered as a first-class
`xpu_rt.solve` backend. Every solver call in the compiler — placement,
memory planning, makespan scheduling, semantic proofs — now flows through
the same `SolverRequest` / `SolverResponse` envelope and probe-aware
routing layer.

## What changed

| Before | After |
|---|---|
| `xpu_rt.scheduler.scheduler.schedule(workload, ...)` — free function, cvxpy + MOSEK directly. | `xpu_rt.scheduler.solve_makespan(workload, ...)` — typed envelope, registry-routed. |
| No probe — missing cvxpy raises `ImportError` at call time. | Typed `SolverStatus.BLOCKED` when cvxpy / MOSEK unavailable. |
| No audit trail — caller has to roll its own. | Every call carries `formulation_hash`, `selected_backend`, `time_ms`, `objective_value`. |
| Not discoverable by `compgen-solver-planning` MCP skill. | Discovered alongside placement / memory backends. |
| MOSEK license discovery only used for memory planning. | Shared `ensure_mosek_license_env()` path; same `mosek.lic` works for both. |

The original `xpu_rt.scheduler.scheduler.schedule()` stays as the direct
path for callers that already wire it up.

## Architecture

```
SolverRequest(problem_kind=MAKESPAN_SCHEDULE, formulation={signature}, metadata={workload, kwargs})
                                │
                                ▼
                  xpu_rt.solve.routing.choose_backend
                                │
                                ▼
                CvxpyMakespanBackend.solve(request)
                  │
                  ├── unpack workload + kwargs from metadata
                  ├── call xpu_rt.scheduler.scheduler.schedule(workload, **kwargs)
                  │   (CVXPY → MOSEK when licensed, HiGHS/SCIP/GLPK fallback)
                  └── pack (t, alpha, fused, meta) into typed SolverResponse
                                │
                                ▼
                        SolverResponse(status, formulation_hash,
                                       objective_value, solution, time_ms, ...)
```

## Backend registration

Auto-registered in `xpu_rt.solve.backend_registry.default_registry`:

```python
from xpu_rt.solve.backend_registry import default_registry
from xpu_rt.solve.solver_types import SolverBackendName

reg = default_registry()
probe = reg.probe(SolverBackendName.CVXPY_MAKESPAN)
print(probe.availability.value)      # "available" only when cvxpy AND MOSEK are present
print(probe.version)                  # cvxpy version string
print(probe.supports)                 # ('preferred_solver:MOSEK', 'milp', 'makespan_schedule', 'CLARABEL', 'HIGHS', 'MOSEK', ...)
```

The probe:

1. Calls `xpu_rt.solve.backends.mosek_backend.ensure_mosek_license_env()`
   so the repo-local `mosek.lic` is auto-discovered exactly as it is for
   memory planning.
2. Confirms cvxpy is importable and runs a tiny LP to validate the install.
3. **Requires MOSEK be in `cvxpy.installed_solvers()`**. XPU-RT's
   `scheduler.py` hard-codes `solver=cp.MOSEK` at every
   `cvxpy.Problem.solve()` site (4 occurrences) — that's a deliberate
   choice from the original XPU-RT design, since MOSEK's interior-point
   solver produced the reference scheduling results. If MOSEK is missing,
   the probe returns `LICENSE_MISSING` (mosek package present, license
   not registered) or `IMPORT_MISSING` (mosek package absent), and the
   registry routes the call to `BLOCKED` rather than silently letting
   cvxpy fall back to a different solver that would produce subtly
   different schedules.
4. When MOSEK is present, the `supports` tuple opens with the literal
   string `"preferred_solver:MOSEK"` so probe output (and the
   `compgen-solver-planning` MCP skill) make the choice visually obvious.

To install MOSEK with the repo's license:

```bash
uv pip install -e ".[solve-mosek]"
# mosek.lic at the repo root is auto-discovered; no env var needed
```

## Routing table

```python
ROUTING_TABLE[SolverProblemKind.MAKESPAN_SCHEDULE] = (SolverBackendName.CVXPY_MAKESPAN,)
```

Single-backend route. The CVXPY problem is dispatched internally to
MOSEK when licensed (matching the original XPU-RT default), and falls
back to HiGHS / SCIP / GLPK otherwise. The architecture guard refuses to
route any non-makespan kind to this backend.

## Carrier convention

CVXPY's `Workload` carries numpy arrays and dict-of-tuple structures that
aren't trivially JSON-serialisable. The envelope keeps
`SolverRequest.formulation` JSON-friendly (op / machine counts, hashes of
the cost tensors, scheduler kwargs) so `formulation_hash` is byte-stable;
the live `Workload` object travels through two reserved metadata keys:

```python
METADATA_WORKLOAD_KEY = "__cvxpy_makespan_workload"
METADATA_KWARGS_KEY   = "__cvxpy_makespan_kwargs"
```

The `solve_makespan` shim does this wrapping for you; you should never
need to touch the keys directly.

## Usage

```python
from xpu_rt.scheduler import solve_makespan

t, alpha, fused, response = solve_makespan(
    workload,
    time_limit=30.0,
    fusion_threshold=0.5,
    target_diversity_weight=10.0,
)

# Typed status, audit-friendly:
print(response.status.value)              # "optimal" | "feasible" | "timeout" | "infeasible" | "blocked" | "error"
print(response.selected_backend.value)    # "cvxpy_makespan"
print(response.formulation_hash)          # byte-stable across reruns
print(response.objective_value)           # makespan
print(response.time_ms)                   # wall-clock spent in the solver
```

## When to use which API

| Use `solve_makespan()` (envelope) when... | Use `schedule()` (direct) when... |
|---|---|
| You want typed status, formulation_hash, license-aware fallback. | You're inside the original XPU-RT codepath and don't need the envelope. |
| You're calling from a CompGen pipeline stage that already speaks `SolverRequest`/`SolverResponse`. | You need the raw fused `Workload` object back (the envelope drops it). |
| You want the call discoverable by the `compgen-solver-planning` MCP skill. | You're explicitly opting out of the registry — e.g. testing the MILP without probe overhead. |

## See also

- [Two-cluster scheduler](two-cluster-scheduler.md) — the underlying MILP.
- [Overview](overview.md) — how the scheduling stack composes with the
  compiler pipeline.
- `xpu-rt/python/xpu_rt/solve/backends/cvxpy_makespan_backend.py` — the
  full backend implementation.
- `xpu-rt/python/xpu_rt/scheduler/bridge.py` — the `solve_makespan` shim.
- `xpu-rt/tests/solve/test_cvxpy_makespan_backend.py` — registry +
  routing + architecture-guard tests.
