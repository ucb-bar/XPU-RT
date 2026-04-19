"""Cost-model gate — CPSatSolver feasibility + objective check.

Wraps :func:`compgen.solve.contracts.extract_solver_problem` +
:class:`compgen.solve.backends.cp_sat.CPSatSolver`. The gate context
requires either a prebuilt ``SolverProblem`` (``ctx["problem"]``) or
both an xDSL module (``ctx["module"]``) and a ``TargetProfile``
(``ctx["target"]``).

Returns ``accepted`` when the solver reports ``feasible=True``. Also
compares the solved ``objective_value`` against a baseline
(``ctx["baseline_cost"]`` — optional) and rejects if the new plan is
worse by more than ``ctx["tolerance_ratio"]`` (default 1.0 — any
regression rejects).
"""

from __future__ import annotations

from typing import Any


def cost_model_gate(proposal: dict[str, Any], **ctx: Any) -> dict[str, Any]:
    timeout_ms = int(ctx.get("timeout_ms", 10_000))
    baseline_cost = ctx.get("baseline_cost")
    tolerance_ratio = float(ctx.get("tolerance_ratio", 1.0))

    try:
        from compgen.solve.backends.cp_sat import CPSatSolver
        from compgen.solve.contracts import extract_solver_problem
    except ImportError as e:   # pragma: no cover
        return {
            "status": "deferred",
            "details": {"reason": f"solve backend unavailable: {e}"},
        }

    problem = ctx.get("problem")
    if problem is None:
        module = ctx.get("module")
        target = ctx.get("target")
        if module is None or target is None:
            return {
                "status": "deferred",
                "details": {
                    "reason": (
                        "cost_model_gate requires either ctx.problem OR "
                        "ctx.module + ctx.target"
                    )
                },
            }
        try:
            problem = extract_solver_problem(module, target)
        except Exception as e:   # noqa: BLE001
            return {
                "status": "rejected",
                "details": {
                    "reason": "extract_solver_problem failed",
                    "error": f"{type(e).__name__}: {e}",
                },
            }

    solver = CPSatSolver(timeout_ms=timeout_ms)
    try:
        result = solver.solve(problem)
    except Exception as e:   # noqa: BLE001
        return {
            "status": "rejected",
            "details": {
                "reason": "CPSatSolver.solve raised",
                "error": f"{type(e).__name__}: {e}",
            },
        }

    if not getattr(result, "feasible", False):
        return {
            "status": "rejected",
            "details": {
                "reason": "solver returned infeasible",
                "solve_time_ms": getattr(result, "solve_time_ms", 0.0),
            },
        }

    # Extract objective value from placement sub-result if present.
    objective = None
    placement = getattr(result, "placement", None)
    if placement is not None:
        objective = getattr(placement, "objective_value", None)

    details: dict[str, Any] = {
        "feasible": True,
        "objective_value": objective,
        "solve_time_ms": getattr(result, "solve_time_ms", 0.0),
    }

    if baseline_cost is not None and objective is not None:
        # Reject only if the candidate is meaningfully worse than baseline
        # (candidate_cost > baseline_cost * tolerance_ratio). Cheaper =
        # better, so candidate_cost <= baseline_cost * tolerance_ratio is
        # accepted.
        limit = float(baseline_cost) * tolerance_ratio
        if objective > limit:
            return {
                "status": "rejected",
                "details": {
                    **details,
                    "reason": "objective regressed vs baseline",
                    "baseline_cost": baseline_cost,
                    "tolerance_ratio": tolerance_ratio,
                    "limit": limit,
                },
            }

    return {"status": "accepted", "details": details}


__all__ = ["cost_model_gate"]
