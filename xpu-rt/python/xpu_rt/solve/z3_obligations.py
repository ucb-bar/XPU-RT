"""Z3 local proof harness for semantic obligations.

. Three obligation kinds:

* ``tile_index_bounds(dim, tile, iter)`` — proves boundary-aware
  tiled slicing stays in bounds for all integer iterations.
* ``copy_identity(in_arr, out_arr, lo, hi)`` — proves an inserted
  copy is value-preserving for ``i ∈ [lo, hi)`` (bounded universal
  quantifier).
* ``shape_predicate_implication(applies_when, precondition)`` —
  proves a promoted recipe's ``applies_when`` predicates imply a
  contract precondition (e.g. ``K mod 16 == 0 ⇒ K mod 8 == 0``).

The harness uses LIA / arrays in Z3; obligations are bounded for
tractability. Status mapping:

* ``solver.check() == unsat`` (negation unsat) → ``proved``
* ``solver.check() == sat`` (negation sat → original invalid) →
  ``sat_counterexample`` with a concrete model
* ``solver.check() == unknown`` due to timeout → ``timeout``

Callers pass a ``SolverRequest`` whose ``formulation`` describes
the obligation; :func:`solve_request` is the entry point used by
:class:`Z3Backend`.
"""

from __future__ import annotations

import time
from typing import Any

from compgen.solve.solver_types import (
    BackendAvailabilityStatus,
    BackendProbeResult,
    SolverBackendName,
    SolverProblemKind,
    SolverRequest,
    SolverResponse,
    SolverStatus,
)

__all__ = [
    "prove_tile_index_bounds",
    "prove_copy_identity",
    "prove_shape_predicate_implication",
    "solve_request",
    "OBLIGATION_KIND_TILE_INDEX_BOUNDS",
    "OBLIGATION_KIND_COPY_IDENTITY",
    "OBLIGATION_KIND_SHAPE_PREDICATE_IMPLICATION",
]


OBLIGATION_KIND_TILE_INDEX_BOUNDS = "tile_index_bounds"
OBLIGATION_KIND_COPY_IDENTITY = "copy_identity"
OBLIGATION_KIND_SHAPE_PREDICATE_IMPLICATION = "shape_predicate_implication"


def _build_response(
    request: SolverRequest,
    *,
    probe: BackendProbeResult,
    status: SolverStatus,
    time_ms: float,
    counterexample: dict[str, Any] | None = None,
    detail: str | None = None,
    objective_value: float | None = None,
    solution: Any = None,
) -> SolverResponse:
    return SolverResponse(
        problem_id=request.problem_id,
        problem_kind=request.problem_kind,
        selected_backend=SolverBackendName.Z3,
        backend_availability=probe.availability,
        status=status,
        formulation_hash=request.formulation_hash,
        time_ms=time_ms,
        objective_value=objective_value,
        solution=solution,
        counterexample=counterexample,
        infeasibility_reason=detail,
    )


def _extract_model(model: Any, names: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    import z3

    for n in names:
        # model maps decls; we need to find by name
        for decl in model.decls():
            if decl.name() == n:
                val = model[decl]
                if val is None:
                    continue
                if z3.is_int_value(val):
                    out[n] = val.as_long()
                elif z3.is_true(val):
                    out[n] = True
                elif z3.is_false(val):
                    out[n] = False
                else:
                    out[n] = str(val)
    return out


def prove_tile_index_bounds(
    *,
    dim_max: int = 1024,
    tile: int,
    timeout_ms: int = 5000,
    use_safe_len: bool = True,
) -> tuple[SolverStatus, dict[str, Any] | None, str]:
    """Prove tile index math stays in bounds.

    Assumes ``dim ∈ [1, dim_max]``, ``iter ∈ [0, ceil(dim/tile))``.
    Proves ``start >= 0 ∧ len > 0 ∧ len <= tile ∧ start+len <= dim``
    for ``start = iter*tile`` and either:

    * ``use_safe_len=True``: ``len = If(start+tile <= dim, tile, dim-start)``
      (correct boundary-aware formula).
    * ``use_safe_len=False``: ``len = tile`` (the negative-control
      case; should produce a counterexample).
    """

    import z3

    if tile <= 0:
        return SolverStatus.ERROR, None, "tile must be positive"

    solver = z3.Solver()
    solver.set(timeout=int(timeout_ms))
    dim = z3.Int("dim")
    iter_v = z3.Int("iter")
    solver.add(dim >= 1, dim <= dim_max)
    solver.add(tile >= 1)
    solver.add(iter_v >= 0)
    solver.add(iter_v * tile < dim)

    start = iter_v * tile
    if use_safe_len:
        ln = z3.If(start + tile <= dim, z3.IntVal(tile), dim - start)
    else:
        ln = z3.IntVal(tile)

    conclusion = z3.And(start >= 0, ln > 0, ln <= tile, start + ln <= dim)
    solver.add(z3.Not(conclusion))
    res = solver.check()
    if res == z3.unsat:
        return SolverStatus.PROVED, None, ""
    if res == z3.sat:
        model = solver.model()
        cex = _extract_model(model, ["dim", "iter"])
        return SolverStatus.SAT_COUNTEREXAMPLE, cex, "tile bounds violated"
    return SolverStatus.TIMEOUT, None, "z3 unknown / timeout"


def prove_copy_identity(
    *,
    lo: int,
    hi: int,
    perturb: int = 0,
    timeout_ms: int = 5000,
) -> tuple[SolverStatus, dict[str, Any] | None, str]:
    """Prove ``out[i] == in[i] + perturb`` for ``i ∈ [lo, hi)``.

    For ``perturb=0`` this is the identity-copy obligation; non-zero
    perturb is a deliberate negative control.
    """

    import z3

    if hi <= lo:
        return SolverStatus.ERROR, None, "empty range"

    solver = z3.Solver()
    solver.set(timeout=int(timeout_ms))
    in_arr = z3.Array("in_arr", z3.IntSort(), z3.IntSort())
    out_arr = z3.Array("out_arr", z3.IntSort(), z3.IntSort())
    i = z3.Int("i")
    # Universally quantified over bounded range.
    body = z3.Implies(
        z3.And(i >= lo, i < hi),
        out_arr[i] == in_arr[i] + perturb,
    )
    solver.add(z3.ForAll([i], body))
    # Now: prove that out == in on [lo, hi)
    j = z3.Int("j")
    solver.add(j >= lo, j < hi)
    solver.add(out_arr[j] != in_arr[j])
    res = solver.check()
    if res == z3.unsat:
        return SolverStatus.PROVED, None, ""
    if res == z3.sat:
        model = solver.model()
        cex = _extract_model(model, ["j"])
        return SolverStatus.SAT_COUNTEREXAMPLE, cex, "copy not identity"
    return SolverStatus.TIMEOUT, None, "z3 unknown / timeout"


def _compile_predicate(z3_mod, env: dict[str, Any], pred: dict[str, Any]) -> Any:
    op = pred.get("op")
    if op == "divisible_by":
        var = env[pred["var"]]
        k = int(pred["k"])
        if k <= 0:
            raise ValueError("divisible_by k must be positive")
        return var % k == 0
    if op == "equal":
        return env[pred["a"]] == _coerce(z3_mod, env, pred["b"])
    if op == "le":
        return env[pred["a"]] <= _coerce(z3_mod, env, pred["b"])
    if op == "ge":
        return env[pred["a"]] >= _coerce(z3_mod, env, pred["b"])
    if op == "in_set":
        values = pred["values"]
        if not values:
            raise ValueError("in_set requires at least one value")
        var = env[pred["var"]]
        return z3_mod.Or([var == int(v) for v in values])
    if op == "and":
        return z3_mod.And([_compile_predicate(z3_mod, env, p) for p in pred["terms"]])
    if op == "or":
        return z3_mod.Or([_compile_predicate(z3_mod, env, p) for p in pred["terms"]])
    raise ValueError(f"unknown predicate op: {op!r}")


def _coerce(z3_mod, env: dict[str, Any], value: Any) -> Any:
    if isinstance(value, str) and value in env:
        return env[value]
    if isinstance(value, (int, bool)):
        return int(value)
    if isinstance(value, str):
        # treat as int literal if possible
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(f"unbound symbol or non-int literal: {value!r}") from exc
    raise ValueError(f"unsupported predicate value: {value!r}")


def prove_shape_predicate_implication(
    *,
    variables: dict[str, dict[str, int]],
    applies_when: list[dict[str, Any]],
    precondition: dict[str, Any],
    timeout_ms: int = 5000,
) -> tuple[SolverStatus, dict[str, Any] | None, str]:
    """Prove ``applies_when ⇒ precondition``.

    ``variables`` maps variable name to ``{"min": ..., "max": ...}``
    integer bounds. Free variables in either side must appear in
    ``variables``.
    """

    import z3

    if not variables:
        return SolverStatus.UNSUPPORTED, None, "no variables declared"

    solver = z3.Solver()
    solver.set(timeout=int(timeout_ms))
    env: dict[str, Any] = {}
    for name, bounds in variables.items():
        v = z3.Int(name)
        if "min" in bounds:
            solver.add(v >= int(bounds["min"]))
        if "max" in bounds:
            solver.add(v <= int(bounds["max"]))
        env[name] = v

    try:
        if applies_when:
            premise = z3.And([_compile_predicate(z3, env, p) for p in applies_when])
            solver.add(premise)
        conclusion = _compile_predicate(z3, env, precondition)
    except ValueError as exc:
        return SolverStatus.ERROR, None, str(exc)

    solver.add(z3.Not(conclusion))
    res = solver.check()
    if res == z3.unsat:
        return SolverStatus.PROVED, None, ""
    if res == z3.sat:
        model = solver.model()
        cex = _extract_model(model, list(variables.keys()))
        return SolverStatus.SAT_COUNTEREXAMPLE, cex, "implication does not hold"
    return SolverStatus.TIMEOUT, None, "z3 unknown / timeout"


def solve_request(request: SolverRequest, *, probe: BackendProbeResult) -> SolverResponse:
    """Entry point used by :class:`Z3Backend.solve`."""

    if probe.availability is not BackendAvailabilityStatus.AVAILABLE:
        return _build_response(
            request,
            probe=probe,
            status=SolverStatus.BLOCKED,
            time_ms=0.0,
            detail=f"z3 unavailable: {probe.detail}",
        )

    formulation = request.formulation or {}
    obligation = formulation.get("obligation_kind") if isinstance(formulation, dict) else None
    params = formulation.get("params", {}) if isinstance(formulation, dict) else {}
    timeout_ms = min(request.time_budget_ms, params.get("timeout_ms", request.time_budget_ms))

    t0 = time.perf_counter()
    try:
        if obligation == OBLIGATION_KIND_TILE_INDEX_BOUNDS:
            status, cex, detail = prove_tile_index_bounds(
                dim_max=int(params.get("dim_max", 1024)),
                tile=int(params["tile"]),
                use_safe_len=bool(params.get("use_safe_len", True)),
                timeout_ms=timeout_ms,
            )
        elif obligation == OBLIGATION_KIND_COPY_IDENTITY:
            status, cex, detail = prove_copy_identity(
                lo=int(params["lo"]),
                hi=int(params["hi"]),
                perturb=int(params.get("perturb", 0)),
                timeout_ms=timeout_ms,
            )
        elif obligation == OBLIGATION_KIND_SHAPE_PREDICATE_IMPLICATION:
            status, cex, detail = prove_shape_predicate_implication(
                variables=dict(params.get("variables", {})),
                applies_when=list(params.get("applies_when", [])),
                precondition=dict(params["precondition"]),
                timeout_ms=timeout_ms,
            )
        else:
            return _build_response(
                request,
                probe=probe,
                status=SolverStatus.UNSUPPORTED,
                time_ms=(time.perf_counter() - t0) * 1000.0,
                detail=f"unknown obligation_kind: {obligation!r}",
            )
    except (ValueError, KeyError) as exc:
        return _build_response(
            request,
            probe=probe,
            status=SolverStatus.ERROR,
            time_ms=(time.perf_counter() - t0) * 1000.0,
            detail=f"malformed obligation: {exc}",
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return _build_response(
        request,
        probe=probe,
        status=status,
        time_ms=elapsed_ms,
        counterexample=cex,
        detail=detail or None,
        solution={"obligation_kind": obligation},
    )
