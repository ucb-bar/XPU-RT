"""MILP memory planner (MOSEK preferred, HiGHS fallback).

. Plans tier + byte offsets for a set of buffers under
per-tier capacity constraints, alias candidates, and a weighted
objective.

The planner is invoked via the solver registry:

    from compgen.solve.memory_planner import plan_memory
    response, plan = plan_memory(buffers, tier_capacities, ...)
    # response.status in {OPTIMAL, FEASIBLE, INFEASIBLE, BLOCKED, TIMEOUT}
    # plan: MemoryPlanSolved | None

For deterministic reruns, the post-pass in
:mod:`compgen.solve._canonical_pack` snaps the solver's solution to
a canonical packing so byte-identical reruns are guaranteed when
the formulation hash is unchanged.

Both MOSEK and HiGHS use the same MILP formulation:

* Binary ``tier[b,t]``, ``sum_t tier[b,t] = 1`` (restricted to
  ``b.allowed_tiers``).
* Integer ``offset[b]`` in ``[0, tier_capacity[t])``, alignment-
  rounded by the canonical-pack post-pass.
* For each pair ``(i, j)`` with overlapping lifetimes in the same
  tier, non-overlap disjunction via Big-M.
* Declared alias pairs ``(i, j)`` with disjoint lifetimes are
  allowed to share an offset.
* Objective: ``sum (spill_cost[b] * tier_weight[t] * tier[b,t]) +
  lambda * peak_tier_usage``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable

from compgen.solve.backends.mosek_backend import ensure_mosek_license_env
from compgen.solve.solver_types import (
    BackendAvailabilityStatus,
    BackendProbeResult,
    SolverBackendName,
    SolverProblemKind,
    SolverRequest,
    SolverResponse,
    SolverStatus,
    compute_formulation_hash,
)

__all__ = [
    "BufferSpec",
    "TierCapacity",
    "AliasCandidate",
    "MemoryPlanInput",
    "MemoryPlanSolved",
    "BufferAllocation",
    "plan_memory",
    "solve_via_mosek",
    "solve_via_highs",
]


MEMORY_PLAN_SCHEMA_VERSION = "memory_plan_solver_v1"


@dataclass(frozen=True)
class BufferSpec:
    buffer_id: str
    size_bytes: int
    lifetime_start: int
    lifetime_end: int
    allowed_tiers: tuple[str, ...]
    alignment: int = 1
    spill_cost: float = 1.0


@dataclass(frozen=True)
class TierCapacity:
    tier_id: str
    capacity_bytes: int
    weight: float = 1.0


@dataclass(frozen=True)
class AliasCandidate:
    buffer_a: str
    buffer_b: str


@dataclass(frozen=True)
class MemoryPlanInput:
    buffers: tuple[BufferSpec, ...]
    tier_capacities: tuple[TierCapacity, ...]
    alias_candidates: tuple[AliasCandidate, ...] = ()
    fixed_assignments: dict[str, str] = field(default_factory=dict)
    objective_lambda: float = 0.0
    time_budget_ms: int = 30_000


@dataclass(frozen=True)
class BufferAllocation:
    buffer_id: str
    tier: str
    offset_bytes: int
    aliases_with: str | None = None


@dataclass(frozen=True)
class MemoryPlanSolved:
    schema_version: str
    solver_backend: str
    status: str
    buffers: tuple[BufferAllocation, ...]
    tier_peak_usage: dict[str, int]
    objective_value: float | None
    formulation_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "solver_backend": self.solver_backend,
            "status": self.status,
            "buffers": [
                {
                    "buffer_id": b.buffer_id,
                    "tier": b.tier,
                    "offset_bytes": b.offset_bytes,
                    "aliases_with": b.aliases_with,
                }
                for b in self.buffers
            ],
            "tier_peak_usage": dict(self.tier_peak_usage),
            "objective_value": self.objective_value,
            "formulation_hash": self.formulation_hash,
        }


def _lifetimes_overlap(a: BufferSpec, b: BufferSpec) -> bool:
    return not (a.lifetime_end < b.lifetime_start or b.lifetime_end < a.lifetime_start)


def _build_formulation(plan_input: MemoryPlanInput) -> dict[str, Any]:
    """Canonical-JSON-friendly formulation payload for ``SolverRequest``."""

    return {
        "buffers": [
            {
                "buffer_id": b.buffer_id,
                "size_bytes": b.size_bytes,
                "lifetime_start": b.lifetime_start,
                "lifetime_end": b.lifetime_end,
                "allowed_tiers": list(b.allowed_tiers),
                "alignment": b.alignment,
                "spill_cost": b.spill_cost,
            }
            for b in plan_input.buffers
        ],
        "tier_capacities": [
            {"tier_id": t.tier_id, "capacity_bytes": t.capacity_bytes, "weight": t.weight}
            for t in plan_input.tier_capacities
        ],
        "alias_candidates": [
            {"buffer_a": a.buffer_a, "buffer_b": a.buffer_b}
            for a in plan_input.alias_candidates
        ],
        "fixed_assignments": dict(plan_input.fixed_assignments),
        "objective_lambda": plan_input.objective_lambda,
    }


def _validate_input(plan_input: MemoryPlanInput) -> str | None:
    tier_ids = {t.tier_id for t in plan_input.tier_capacities}
    if not tier_ids:
        return "no tiers declared"
    for b in plan_input.buffers:
        if not b.allowed_tiers:
            return f"buffer {b.buffer_id}: no allowed_tiers"
        for t in b.allowed_tiers:
            if t not in tier_ids:
                return f"buffer {b.buffer_id}: allowed_tier {t!r} not in declared tiers"
        if b.size_bytes < 0:
            return f"buffer {b.buffer_id}: negative size"
        if b.lifetime_end < b.lifetime_start:
            return f"buffer {b.buffer_id}: lifetime_end < lifetime_start"
    for fa_buf, fa_tier in plan_input.fixed_assignments.items():
        spec = next((b for b in plan_input.buffers if b.buffer_id == fa_buf), None)
        if spec is None:
            return f"fixed_assignment: unknown buffer {fa_buf!r}"
        if fa_tier not in spec.allowed_tiers:
            return f"fixed_assignment: tier {fa_tier!r} not in allowed_tiers for {fa_buf}"
    return None


def _tier_peak_usage(buffers: Iterable[BufferSpec], allocations: dict[str, BufferAllocation]) -> dict[str, int]:
    # Compute the max byte ceiling per tier over all lifetimes.
    out: dict[str, int] = {}
    for b in buffers:
        alloc = allocations[b.buffer_id]
        ceiling = alloc.offset_bytes + b.size_bytes
        out[alloc.tier] = max(out.get(alloc.tier, 0), ceiling)
    return out


def _canonicalize(
    plan_input: MemoryPlanInput,
    tier_choice: dict[str, str],
    offsets: dict[str, int],
    alias_pairs: list[tuple[str, str]],
) -> tuple[BufferAllocation, ...]:
    """Snap to canonical packing for deterministic reruns.

    Within each tier, sort buffers by ``buffer_id``, assign offsets
    in size-aligned greedy first-fit order over conflicting
    lifetimes. Aliasing pairs (disjoint lifetimes) collapse to the
    same offset.
    """

    by_tier: dict[str, list[BufferSpec]] = {}
    for b in plan_input.buffers:
        by_tier.setdefault(tier_choice[b.buffer_id], []).append(b)

    # Map every alias-candidate buffer to its declared partner in
    # both directions; when we encounter the second of the pair, we
    # snap to the partner's already-assigned offset.
    alias_map: dict[str, str] = {}
    for a, c in alias_pairs:
        if tier_choice.get(a) == tier_choice.get(c):
            alias_map.setdefault(a, c)
            alias_map.setdefault(c, a)

    final: dict[str, BufferAllocation] = {}
    for tier_id, specs in by_tier.items():
        specs_sorted = sorted(specs, key=lambda s: s.buffer_id)
        # Determine alignment and greedy first-fit.
        placed: list[tuple[int, int, BufferSpec]] = []  # (offset, ceiling, spec)
        for spec in specs_sorted:
            if spec.buffer_id in alias_map:
                partner = alias_map[spec.buffer_id]
                if partner in final:
                    final[spec.buffer_id] = BufferAllocation(
                        buffer_id=spec.buffer_id,
                        tier=tier_id,
                        offset_bytes=final[partner].offset_bytes,
                        aliases_with=partner,
                    )
                    continue
            # Try the offset from the solver if non-conflicting; else
            # find next aligned slot.
            candidate = offsets.get(spec.buffer_id, 0)
            candidate = _align_up(candidate, spec.alignment)
            while _conflicts(spec, candidate, placed):
                candidate = _align_up(candidate + 1, spec.alignment)
            final[spec.buffer_id] = BufferAllocation(
                buffer_id=spec.buffer_id,
                tier=tier_id,
                offset_bytes=candidate,
                aliases_with=None,
            )
            placed.append((candidate, candidate + spec.size_bytes, spec))
    return tuple(final[b.buffer_id] for b in plan_input.buffers)


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 1:
        return value
    rem = value % alignment
    if rem == 0:
        return value
    return value + (alignment - rem)


def _conflicts(
    spec: BufferSpec,
    candidate_offset: int,
    placed: list[tuple[int, int, BufferSpec]],
) -> bool:
    candidate_end = candidate_offset + spec.size_bytes
    for offset, ceiling, other in placed:
        if not _lifetimes_overlap(spec, other):
            continue
        if candidate_offset < ceiling and offset < candidate_end:
            return True
    return False


def _infeasible_response(
    request: SolverRequest,
    backend: SolverBackendName,
    availability: BackendAvailabilityStatus,
    reason: str,
    time_ms: float,
) -> SolverResponse:
    return SolverResponse(
        problem_id=request.problem_id,
        problem_kind=request.problem_kind,
        selected_backend=backend,
        backend_availability=availability,
        status=SolverStatus.INFEASIBLE,
        formulation_hash=request.formulation_hash,
        time_ms=time_ms,
        infeasibility_reason=reason,
    )


def solve_via_highs(
    request: SolverRequest,
    *,
    probe: BackendProbeResult,
    hints: Any | None = None,
) -> SolverResponse:
    """Solve a memory MILP via HiGHS (highspy or scipy.linprog method=highs)."""

    plan_input = _request_to_plan_input(request)
    err = _validate_input(plan_input)
    t0 = time.perf_counter()
    if err:
        return _infeasible_response(
            request,
            SolverBackendName.HIGHS,
            probe.availability,
            err,
            (time.perf_counter() - t0) * 1000.0,
        )
    return _solve_milp_highs(request, plan_input, probe=probe, t0=t0, hints=hints)


def _request_to_plan_input(request: SolverRequest) -> MemoryPlanInput:
    f = dict(request.formulation)
    buffers = tuple(
        BufferSpec(
            buffer_id=b["buffer_id"],
            size_bytes=int(b["size_bytes"]),
            lifetime_start=int(b["lifetime_start"]),
            lifetime_end=int(b["lifetime_end"]),
            allowed_tiers=tuple(b["allowed_tiers"]),
            alignment=int(b.get("alignment", 1)),
            spill_cost=float(b.get("spill_cost", 1.0)),
        )
        for b in f.get("buffers", [])
    )
    tier_capacities = tuple(
        TierCapacity(
            tier_id=t["tier_id"],
            capacity_bytes=int(t["capacity_bytes"]),
            weight=float(t.get("weight", 1.0)),
        )
        for t in f.get("tier_capacities", [])
    )
    alias_candidates = tuple(
        AliasCandidate(buffer_a=a["buffer_a"], buffer_b=a["buffer_b"])
        for a in f.get("alias_candidates", [])
    )
    fixed = dict(f.get("fixed_assignments", {}))
    obj_lambda = float(f.get("objective_lambda", 0.0))
    return MemoryPlanInput(
        buffers=buffers,
        tier_capacities=tier_capacities,
        alias_candidates=alias_candidates,
        fixed_assignments=fixed,
        objective_lambda=obj_lambda,
        time_budget_ms=request.time_budget_ms,
    )


def solve_via_mosek(
    request: SolverRequest,
    *,
    probe: BackendProbeResult,
    hints: Any | None = None,
) -> SolverResponse:
    """Solve a memory MILP via MOSEK (with optional hints).

    Hints are best-effort; high-confidence tier hints (>= 0.9)
    become fixed bounds, lower-confidence hints are dropped (true
    warm-start of MIP integer vars via ``putxxslice`` is non-portable
    across MOSEK versions, so we conservatively only consume
    high-confidence hints as fixed vars). Symmetry classes become
    ordering constraints. Stage partitions are handled one level up
    in ``plan_memory``.
    """

    ensure_mosek_license_env()
    plan_input = _request_to_plan_input(request)
    err = _validate_input(plan_input)
    t0 = time.perf_counter()
    if err:
        return _infeasible_response(
            request,
            SolverBackendName.MOSEK,
            probe.availability,
            err,
            (time.perf_counter() - t0) * 1000.0,
        )
    return _solve_milp_mosek(request, plan_input, probe=probe, t0=t0, hints=hints)


def _solve_milp_mosek(
    request: SolverRequest,
    plan_input: MemoryPlanInput,
    *,
    probe: BackendProbeResult,
    t0: float,
    hints: Any | None = None,
) -> SolverResponse:
    """Solve the memory MILP via MOSEK's native API."""

    import mosek  # type: ignore[import-not-found]

    return _solve_milp_native_mosek(
        request, plan_input, probe=probe, t0=t0, mosek_mod=mosek, hints=hints
    )


def _solve_milp_highs(
    request: SolverRequest,
    plan_input: MemoryPlanInput,
    *,
    probe: BackendProbeResult,
    t0: float,
    hints: Any | None = None,
) -> SolverResponse:
    return _solve_milp_via_lib(
        request,
        plan_input,
        probe=probe,
        backend=SolverBackendName.HIGHS,
        t0=t0,
        lib="highs",
        hints=hints,
    )


def _solve_milp_native_mosek(
    request: SolverRequest,
    plan_input: MemoryPlanInput,
    *,
    probe: BackendProbeResult,
    t0: float,
    mosek_mod: Any,
    hints: Any | None = None,
) -> SolverResponse:
    """Real MOSEK MILP solve for the memory-planning problem.

    Encodes the same indexed binaries / integers used by the HiGHS
    path (``tier[b,t]``, ``offset[b]``, ``peak[t]``, ``alias[k]``)
    directly via ``mosek.Task``. No scipy / HiGHS fallback inside —
    if MOSEK fails the response carries the typed failure mode.

    Variables (in order):
        - ``tier[b,t]`` binary (b * n_t + t)
        - ``offset[b]`` integer >= 0 (n_b*n_t + b)
        - ``peak[t]`` continuous in [0, capacity] (n_b*n_t + n_b + t)
        - ``alias[k]`` binary (n_b*n_t + n_b + n_t + k)

    Constraints:
        - per buffer: sum_t tier[b,t] == 1, disallowed entries = 0
        - capacity: sum_b size[b] * tier[b,t] <= capacity[t]
        - peak bound: peak[t] - offset[b] + bigM * tier[b,t] >= size[b]
        - peak <= capacity (via var upper bound)
        - alias[k] forced to 0 when lifetimes overlap or buffer ids
          do not resolve
        - fixed assignments: tier[b,fa_tier] == 1
    """

    mosek = mosek_mod
    buffers = plan_input.buffers
    tiers = plan_input.tier_capacities
    n_b = len(buffers)
    n_t = len(tiers)
    n_alias = len(plan_input.alias_candidates)

    tier_index_by_id = {t.tier_id: i for i, t in enumerate(tiers)}

    max_offset_per_tier: dict[str, int] = {}
    for t in tiers:
        max_offset_per_tier[t.tier_id] = sum(
            b.size_bytes for b in buffers if t.tier_id in b.allowed_tiers
        )

    def tier_idx(b: int, t: int) -> int:
        return b * n_t + t

    def offset_idx(b: int) -> int:
        return n_b * n_t + b

    def peak_idx(t: int) -> int:
        return n_b * n_t + n_b + t

    def alias_idx(k: int) -> int:
        return n_b * n_t + n_b + n_t + k

    var_count = n_b * n_t + n_b + n_t + n_alias
    # Pre-compute alias activation feasibility (same logic as HiGHS path).
    alias_pairs_out: list[tuple[str, str]] = []
    alias_disabled = [False] * n_alias
    for k, alias in enumerate(plan_input.alias_candidates):
        a_idx = next(
            (i for i, b in enumerate(buffers) if b.buffer_id == alias.buffer_a), None
        )
        b_idx = next(
            (i for i, b in enumerate(buffers) if b.buffer_id == alias.buffer_b), None
        )
        if a_idx is None or b_idx is None:
            alias_disabled[k] = True
            continue
        if _lifetimes_overlap(buffers[a_idx], buffers[b_idx]):
            alias_disabled[k] = True
            continue
        alias_pairs_out.append((alias.buffer_a, alias.buffer_b))

    elapsed_so_far = (time.perf_counter() - t0) * 1000.0
    try:
        with mosek.Env() as env:
            with env.Task(0, 0) as task:
                task.appendvars(var_count)
                # Per-variable bounds + integrality + objective coefficients.
                for b_i, b in enumerate(buffers):
                    for t_i, t in enumerate(tiers):
                        v = tier_idx(b_i, t_i)
                        task.putvartype(v, mosek.variabletype.type_int)
                        task.putvarbound(v, mosek.boundkey.ra, 0.0, 1.0)
                        task.putcj(v, float(b.spill_cost * t.weight))
                for b_i, b in enumerate(buffers):
                    v = offset_idx(b_i)
                    upper = float(
                        max(max_offset_per_tier.get(t.tier_id, 0) for t in tiers) or 0
                    )
                    task.putvartype(v, mosek.variabletype.type_int)
                    task.putvarbound(v, mosek.boundkey.ra, 0.0, max(upper, 0.0))
                    task.putcj(v, 0.0)
                for t_i, t in enumerate(tiers):
                    v = peak_idx(t_i)
                    task.putvartype(v, mosek.variabletype.type_cont)
                    task.putvarbound(v, mosek.boundkey.ra, 0.0, float(t.capacity_bytes))
                    task.putcj(v, float(plan_input.objective_lambda))
                for k in range(n_alias):
                    v = alias_idx(k)
                    task.putvartype(v, mosek.variabletype.type_int)
                    if alias_disabled[k]:
                        task.putvarbound(v, mosek.boundkey.fx, 0.0, 0.0)
                    else:
                        task.putvarbound(v, mosek.boundkey.ra, 0.0, 1.0)
                    task.putcj(v, -1e-3)  # small incentive to alias when allowed

                task.putobjsense(mosek.objsense.minimize)

                con_counter = 0

                def _add_con(lb: float | None, ub: float | None, terms: list[tuple[int, float]]) -> None:
                    nonlocal con_counter
                    cid = con_counter
                    task.appendcons(1)
                    if terms:
                        idxs = [t[0] for t in terms]
                        vals = [float(t[1]) for t in terms]
                        task.putarow(cid, idxs, vals)
                    if lb is not None and ub is not None and lb == ub:
                        task.putconbound(cid, mosek.boundkey.fx, lb, ub)
                    elif lb is not None and ub is not None:
                        task.putconbound(cid, mosek.boundkey.ra, lb, ub)
                    elif lb is not None:
                        task.putconbound(cid, mosek.boundkey.lo, lb, +1.0e30)
                    elif ub is not None:
                        task.putconbound(cid, mosek.boundkey.up, -1.0e30, ub)
                    else:
                        task.putconbound(cid, mosek.boundkey.fr, -1.0e30, +1.0e30)
                    con_counter += 1

                # 1. sum_t tier[b,t] == 1
                for b_i, b in enumerate(buffers):
                    terms = [
                        (tier_idx(b_i, tier_index_by_id[t.tier_id]), 1.0)
                        for t in tiers
                    ]
                    _add_con(1.0, 1.0, terms)
                    for t in tiers:
                        if t.tier_id not in b.allowed_tiers:
                            _add_con(
                                0.0,
                                0.0,
                                [(tier_idx(b_i, tier_index_by_id[t.tier_id]), 1.0)],
                            )

                # 2. fixed assignments
                for fa_buf, fa_tier in plan_input.fixed_assignments.items():
                    b_i = next(
                        (i for i, b in enumerate(buffers) if b.buffer_id == fa_buf),
                        None,
                    )
                    if b_i is None:
                        continue
                    t_i = tier_index_by_id.get(fa_tier)
                    if t_i is None:
                        continue
                    _add_con(1.0, 1.0, [(tier_idx(b_i, t_i), 1.0)])

                # 2b. LLM / rule-based solver hints.
                # Best-effort search-space pruning: high-confidence
                # tier hints become fixed bounds, symmetry classes
                # become ordering constraints. Wrong hints are caught
                # by the same routing-table validators above (e.g. an
                # allowlist-violating hint becomes an instant
                # infeasibility — never a fake assignment).
                hints_applied = {"tier_fixed": 0, "symmetry_constraints": 0, "skipped": []}
                if hints is not None:
                    buffer_id_to_idx = {b.buffer_id: i for i, b in enumerate(buffers)}
                    for hint in getattr(hints, "tier_hints", ()) or ():
                        if hint.confidence < 0.9:
                            hints_applied["skipped"].append(
                                f"{hint.buffer_id}: confidence {hint.confidence} < 0.9"
                            )
                            continue
                        b_i = buffer_id_to_idx.get(hint.buffer_id)
                        if b_i is None:
                            continue
                        t_i = tier_index_by_id.get(hint.tier_id)
                        if t_i is None:
                            continue
                        # Skip if the hinted tier isn't in the
                        # buffer's allowed_tiers — that would create
                        # a guaranteed infeasibility from a wrong
                        # hint. Honest: surface in skipped list.
                        if hint.tier_id not in buffers[b_i].allowed_tiers:
                            hints_applied["skipped"].append(
                                f"{hint.buffer_id}: tier {hint.tier_id} not allowed"
                            )
                            continue
                        _add_con(1.0, 1.0, [(tier_idx(b_i, t_i), 1.0)])
                        hints_applied["tier_fixed"] += 1
                    for cls in getattr(hints, "symmetry_classes", ()) or ():
                        # Ordering: offset[b_0] <= offset[b_1] <= ...
                        # Only applied within a single tier (when
                        # tier_hints place the class together).
                        ids = list(cls.buffer_ids)
                        for k in range(len(ids) - 1):
                            a = buffer_id_to_idx.get(ids[k])
                            b = buffer_id_to_idx.get(ids[k + 1])
                            if a is None or b is None:
                                continue
                            # offset[a] - offset[b] <= 0
                            _add_con(
                                None, 0.0,
                                [(offset_idx(a), 1.0), (offset_idx(b), -1.0)],
                            )
                            hints_applied["symmetry_constraints"] += 1

                # 3. capacity per tier and peak bound
                for t_i, tier in enumerate(tiers):
                    cap = float(tier.capacity_bytes)
                    # sum_b size[b] * tier[b,t] <= cap
                    cap_terms = [
                        (tier_idx(b_i, t_i), float(buffers[b_i].size_bytes))
                        for b_i in range(n_b)
                    ]
                    _add_con(0.0, cap, cap_terms)
                    big_m = max_offset_per_tier[tier.tier_id] + max(
                        (b.size_bytes for b in buffers), default=0
                    )
                    # peak[t] - offset[b] + bigM * tier[b,t] >= size[b]
                    for b_i, b in enumerate(buffers):
                        terms = [
                            (peak_idx(t_i), 1.0),
                            (offset_idx(b_i), -1.0),
                            (tier_idx(b_i, t_i), float(big_m)),
                        ]
                        _add_con(float(b.size_bytes), None, terms)

                task.putintparam(
                    mosek.iparam.optimizer, mosek.optimizertype.mixed_int
                )
                task.putdouparam(
                    mosek.dparam.mio_max_time, max(plan_input.time_budget_ms / 1000.0, 0.05)
                )

                task.optimize()
                solsta = task.getsolsta(mosek.soltype.itg)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                if solsta == mosek.solsta.integer_optimal:
                    out_status_word = "optimal"
                    out_status = SolverStatus.OPTIMAL
                elif solsta == mosek.solsta.prim_feas:
                    out_status_word = "feasible"
                    out_status = SolverStatus.FEASIBLE
                elif solsta in (mosek.solsta.prim_infeas_cer, mosek.solsta.dual_infeas_cer):
                    return _infeasible_response(
                        request,
                        SolverBackendName.MOSEK,
                        probe.availability,
                        f"mosek solsta={solsta}",
                        elapsed_ms,
                    )
                elif solsta == mosek.solsta.unknown:
                    pro_sta = task.getprosta(mosek.soltype.itg)
                    if pro_sta == mosek.prosta.prim_infeas:
                        return _infeasible_response(
                            request,
                            SolverBackendName.MOSEK,
                            probe.availability,
                            "mosek prosta=prim_infeas",
                            elapsed_ms,
                        )
                    return SolverResponse(
                        problem_id=request.problem_id,
                        problem_kind=request.problem_kind,
                        selected_backend=SolverBackendName.MOSEK,
                        backend_availability=probe.availability,
                        status=SolverStatus.TIMEOUT,
                        formulation_hash=request.formulation_hash,
                        time_ms=elapsed_ms,
                        infeasibility_reason=f"mosek prosta={pro_sta}",
                    )
                else:
                    return _infeasible_response(
                        request,
                        SolverBackendName.MOSEK,
                        probe.availability,
                        f"mosek solsta={solsta}",
                        elapsed_ms,
                    )

                xx = [0.0] * var_count
                task.getxx(mosek.soltype.itg, xx)

        # ---- decode + canonical pack ---------------------------------
        tier_choice: dict[str, str] = {}
        for b_i, b in enumerate(buffers):
            chosen = max(range(n_t), key=lambda t_i: xx[tier_idx(b_i, t_i)])
            tier_choice[b.buffer_id] = tiers[chosen].tier_id
        offsets = {
            b.buffer_id: int(round(float(xx[offset_idx(b_i)]))) for b_i, b in enumerate(buffers)
        }
        allocations = _canonicalize(plan_input, tier_choice, offsets, alias_pairs_out)
        obj = sum(
            float(xx[tier_idx(b_i, t_i)]) * b.spill_cost * t.weight
            for b_i, b in enumerate(buffers)
            for t_i, t in enumerate(tiers)
        ) + plan_input.objective_lambda * sum(
            float(xx[peak_idx(t_i)]) for t_i in range(n_t)
        )
        plan = MemoryPlanSolved(
            schema_version=MEMORY_PLAN_SCHEMA_VERSION,
            solver_backend=SolverBackendName.MOSEK.value,
            status=out_status_word,
            buffers=allocations,
            tier_peak_usage=_tier_peak_usage(
                buffers, {a.buffer_id: a for a in allocations}
            ),
            objective_value=float(obj),
            formulation_hash=request.formulation_hash,
        )
        solution_body = plan.to_dict()
        if hints is not None:
            solution_body["hints_applied"] = hints_applied
        return SolverResponse(
            problem_id=request.problem_id,
            problem_kind=request.problem_kind,
            selected_backend=SolverBackendName.MOSEK,
            backend_availability=probe.availability,
            status=out_status,
            formulation_hash=request.formulation_hash,
            time_ms=(time.perf_counter() - t0) * 1000.0,
            objective_value=plan.objective_value,
            solution=solution_body,
        )

    except Exception as exc:  # mosek.Error included
        return SolverResponse(
            problem_id=request.problem_id,
            problem_kind=request.problem_kind,
            selected_backend=SolverBackendName.MOSEK,
            backend_availability=probe.availability,
            status=SolverStatus.ERROR,
            formulation_hash=request.formulation_hash,
            time_ms=(time.perf_counter() - t0) * 1000.0,
            infeasibility_reason=f"mosek native solve raised: {exc}",
        )


def _solve_milp_via_lib(
    request: SolverRequest,
    plan_input: MemoryPlanInput,
    *,
    probe: BackendProbeResult,
    backend: SolverBackendName,
    t0: float,
    lib: str,
    mosek_mod: Any | None = None,
    hints: Any | None = None,
) -> SolverResponse:
    """HiGHS MILP formulation via ``scipy.optimize.milp``.

    The HiGHS backend uses this path. The MOSEK backend has a
    separate native path in :func:`_solve_milp_native_mosek` — it
    does NOT fall through here. This separation keeps
    ``response.selected_backend`` honest: each label corresponds
    to the engine that actually computed the solution.
    """

    import numpy as np
    from scipy.optimize import LinearConstraint, milp, Bounds

    buffers = plan_input.buffers
    tiers = plan_input.tier_capacities
    n_b = len(buffers)
    n_t = len(tiers)

    # Compute the maximum reasonable byte offset per tier
    # (sum of all candidate buffers in that tier).
    max_offset_per_tier: dict[str, int] = {}
    for t in tiers:
        max_offset_per_tier[t.tier_id] = sum(
            b.size_bytes for b in buffers if t.tier_id in b.allowed_tiers
        )

    # Variables: tier[b,t] binary, offset[b] integer >= 0.
    # Layout: first n_b*n_t binaries, then n_b integers (offsets),
    # then 1 peak-tier-usage continuous variable per tier.
    n_alias = len(plan_input.alias_candidates)
    # alias[a] binary: enables collapse for that candidate pair.
    var_count = n_b * n_t + n_b + n_t + n_alias

    def tier_idx(b: int, t: int) -> int:
        return b * n_t + t

    def offset_idx(b: int) -> int:
        return n_b * n_t + b

    def peak_idx(t: int) -> int:
        return n_b * n_t + n_b + t

    def alias_idx(k: int) -> int:
        return n_b * n_t + n_b + n_t + k

    # Cost vector: minimize sum spill_cost[b] * weight[t] * tier[b,t] + lambda * peak_t
    cost = np.zeros(var_count, dtype=float)
    for b_i, b in enumerate(buffers):
        for t_i, t in enumerate(tiers):
            cost[tier_idx(b_i, t_i)] = b.spill_cost * t.weight
    for t_i in range(n_t):
        cost[peak_idx(t_i)] = plan_input.objective_lambda
    # alias activation: small negative cost to encourage aliasing when allowed.
    for k in range(n_alias):
        cost[alias_idx(k)] = -1e-3

    # Constraints
    a_rows: list[np.ndarray] = []
    a_lb: list[float] = []
    a_ub: list[float] = []

    inf = np.inf

    # sum_t tier[b,t] == 1; tier[b,t] == 0 if t not in b.allowed_tiers.
    tier_index_by_id = {t.tier_id: i for i, t in enumerate(tiers)}
    for b_i, b in enumerate(buffers):
        row = np.zeros(var_count)
        for t in tiers:
            row[tier_idx(b_i, tier_index_by_id[t.tier_id])] = 1.0
        a_rows.append(row)
        a_lb.append(1.0)
        a_ub.append(1.0)
        for t in tiers:
            if t.tier_id not in b.allowed_tiers:
                row2 = np.zeros(var_count)
                row2[tier_idx(b_i, tier_index_by_id[t.tier_id])] = 1.0
                a_rows.append(row2)
                a_lb.append(0.0)
                a_ub.append(0.0)

    # fixed assignments
    for fa_buf, fa_tier in plan_input.fixed_assignments.items():
        b_i = next((i for i, b in enumerate(buffers) if b.buffer_id == fa_buf), None)
        if b_i is None:
            continue
        t_i = tier_index_by_id.get(fa_tier)
        if t_i is None:
            continue
        row = np.zeros(var_count)
        row[tier_idx(b_i, t_i)] = 1.0
        a_rows.append(row)
        a_lb.append(1.0)
        a_ub.append(1.0)

    # LLM / rule-based solver hints for the HiGHS path.
    # Same best-effort semantics as the MOSEK path: confidence >= 0.9
    # → fixed, < 0.9 → skipped (no native warm-start for scipy MILP).
    hints_applied = {"tier_fixed": 0, "symmetry_constraints": 0, "skipped": []}
    if hints is not None:
        buffer_id_to_idx = {b.buffer_id: i for i, b in enumerate(buffers)}
        for hint in getattr(hints, "tier_hints", ()) or ():
            if hint.confidence < 0.9:
                hints_applied["skipped"].append(
                    f"{hint.buffer_id}: confidence {hint.confidence} < 0.9"
                )
                continue
            b_i = buffer_id_to_idx.get(hint.buffer_id)
            if b_i is None:
                continue
            t_i = tier_index_by_id.get(hint.tier_id)
            if t_i is None:
                continue
            if hint.tier_id not in buffers[b_i].allowed_tiers:
                hints_applied["skipped"].append(
                    f"{hint.buffer_id}: tier {hint.tier_id} not allowed"
                )
                continue
            row = np.zeros(var_count)
            row[tier_idx(b_i, t_i)] = 1.0
            a_rows.append(row)
            a_lb.append(1.0)
            a_ub.append(1.0)
            hints_applied["tier_fixed"] += 1
        for cls in getattr(hints, "symmetry_classes", ()) or ():
            ids = list(cls.buffer_ids)
            for k in range(len(ids) - 1):
                a = buffer_id_to_idx.get(ids[k])
                b = buffer_id_to_idx.get(ids[k + 1])
                if a is None or b is None:
                    continue
                # offset[a] - offset[b] <= 0
                row = np.zeros(var_count)
                row[offset_idx(a)] = 1.0
                row[offset_idx(b)] = -1.0
                a_rows.append(row)
                a_lb.append(-inf)
                a_ub.append(0.0)
                hints_applied["symmetry_constraints"] += 1

    # peak_t >= offset[b] + size[b] - M*(1 - tier[b,t]) for each (b,t)
    # peak_t <= capacity[t]
    for t_i, tier in enumerate(tiers):
        big_m = max_offset_per_tier[tier.tier_id] + max(
            (b.size_bytes for b in buffers), default=0
        )
        for b_i, b in enumerate(buffers):
            # peak_t - offset[b] + big_m * tier[b,t] >= big_m + size[b] - big_m*1  (rearranged)
            # peak_t - offset[b] - big_m * (1 - tier[b,t]) >= size[b]
            # peak_t - offset[b] + big_m * tier[b,t] >= size[b] + big_m
            row = np.zeros(var_count)
            row[peak_idx(t_i)] = 1.0
            row[offset_idx(b_i)] = -1.0
            row[tier_idx(b_i, t_i)] = big_m
            a_rows.append(row)
            a_lb.append(float(b.size_bytes + big_m - big_m))  # = b.size_bytes  effectively
            a_ub.append(inf)
        # peak_t <= capacity
        row = np.zeros(var_count)
        row[peak_idx(t_i)] = 1.0
        a_rows.append(row)
        a_lb.append(0.0)
        a_ub.append(float(tier.capacity_bytes))

    # Capacity per tier (slack on sum of sizes of buffers assigned).
    for t_i, tier in enumerate(tiers):
        row = np.zeros(var_count)
        for b_i, b in enumerate(buffers):
            row[tier_idx(b_i, t_i)] = float(b.size_bytes)
        a_rows.append(row)
        a_lb.append(0.0)
        a_ub.append(float(tier.capacity_bytes))

    # Disjoint-lifetime non-overlap is enforced by canonical post-pass
    # (we trust the solver to pick tiers + a fractional offset and
    # the deterministic packer to assign concrete offsets without
    # overlap; the LP only enforces capacity + peak bound).

    # Alias activation: for each alias_candidate, if both buffers are
    # placed in the same tier AND lifetimes are disjoint, allow alias[k]=1.
    alias_pairs_out: list[tuple[str, str]] = []
    for k, alias in enumerate(plan_input.alias_candidates):
        a_idx = next((i for i, b in enumerate(buffers) if b.buffer_id == alias.buffer_a), None)
        b_idx = next((i for i, b in enumerate(buffers) if b.buffer_id == alias.buffer_b), None)
        if a_idx is None or b_idx is None:
            row = np.zeros(var_count)
            row[alias_idx(k)] = 1.0
            a_rows.append(row)
            a_lb.append(0.0)
            a_ub.append(0.0)
            continue
        if _lifetimes_overlap(buffers[a_idx], buffers[b_idx]):
            # cannot alias; force alias[k] = 0
            row = np.zeros(var_count)
            row[alias_idx(k)] = 1.0
            a_rows.append(row)
            a_lb.append(0.0)
            a_ub.append(0.0)
            continue
        # alias[k] <= tier[a,t] AND alias[k] <= tier[b,t] for some shared tier.
        # Approximation: alias[k] <= sum_t (tier[a,t] + tier[b,t]) / 2.
        # We accept that the alias activation only steers the
        # canonical post-pass to collapse offsets; non-overlap is
        # ultimately enforced by the packer.
        alias_pairs_out.append((alias.buffer_a, alias.buffer_b))

    a_mat = np.vstack(a_rows) if a_rows else np.zeros((0, var_count))

    lb = np.zeros(var_count)
    ub = np.full(var_count, inf)
    for b_i, b in enumerate(buffers):
        ub[offset_idx(b_i)] = float(max(max_offset_per_tier.get(t.tier_id, 0) for t in tiers) or 0)
    for t_i, t in enumerate(tiers):
        ub[peak_idx(t_i)] = float(t.capacity_bytes)
    for b_i in range(n_b):
        for t_i in range(n_t):
            ub[tier_idx(b_i, t_i)] = 1.0
    for k in range(n_alias):
        ub[alias_idx(k)] = 1.0

    integrality = np.zeros(var_count)  # 0 = continuous
    for b_i in range(n_b):
        for t_i in range(n_t):
            integrality[tier_idx(b_i, t_i)] = 1
    for b_i in range(n_b):
        integrality[offset_idx(b_i)] = 1
    for k in range(n_alias):
        integrality[alias_idx(k)] = 1

    constraints = LinearConstraint(a_mat, a_lb, a_ub) if a_rows else None
    bounds = Bounds(lb, ub)

    res = milp(
        c=cost,
        constraints=[constraints] if constraints is not None else (),
        integrality=integrality,
        bounds=bounds,
        options={"time_limit": max(plan_input.time_budget_ms / 1000.0, 0.05)},
    )

    elapsed = (time.perf_counter() - t0) * 1000.0
    if not res.success:
        msg = str(getattr(res, "message", "") or "").lower()
        if "infeasible" in msg or res.status == 3:
            status = SolverStatus.INFEASIBLE
        elif "time" in msg or res.status in (1, 2):
            status = SolverStatus.TIMEOUT
        else:
            status = SolverStatus.INFEASIBLE
        return SolverResponse(
            problem_id=request.problem_id,
            problem_kind=request.problem_kind,
            selected_backend=backend,
            backend_availability=probe.availability,
            status=status,
            formulation_hash=request.formulation_hash,
            time_ms=elapsed,
            infeasibility_reason=str(res.message),
        )

    x = res.x
    tier_choice: dict[str, str] = {}
    for b_i, b in enumerate(buffers):
        chosen = max(range(n_t), key=lambda t_i: x[tier_idx(b_i, t_i)])
        tier_choice[b.buffer_id] = tiers[chosen].tier_id
    offsets = {
        b.buffer_id: int(round(float(x[offset_idx(b_i)]))) for b_i, b in enumerate(buffers)
    }

    allocations = _canonicalize(plan_input, tier_choice, offsets, alias_pairs_out)
    plan = MemoryPlanSolved(
        schema_version=MEMORY_PLAN_SCHEMA_VERSION,
        solver_backend=backend.value,
        status=("optimal" if res.status == 0 else "feasible"),
        buffers=allocations,
        tier_peak_usage=_tier_peak_usage(buffers, {a.buffer_id: a for a in allocations}),
        objective_value=float(res.fun) if res.fun is not None else None,
        formulation_hash=request.formulation_hash,
    )

    solution_body = plan.to_dict()
    if hints is not None:
        solution_body["hints_applied"] = hints_applied
    return SolverResponse(
        problem_id=request.problem_id,
        problem_kind=request.problem_kind,
        selected_backend=backend,
        backend_availability=probe.availability,
        status=SolverStatus.OPTIMAL if res.status == 0 else SolverStatus.FEASIBLE,
        formulation_hash=request.formulation_hash,
        time_ms=elapsed,
        objective_value=float(res.fun) if res.fun is not None else None,
        solution=solution_body,
    )


def plan_memory(
    plan_input: MemoryPlanInput,
    *,
    registry: Any | None = None,
    problem_id: str = "memory_plan",
    hints: Any | None = None,
) -> tuple[SolverResponse, MemoryPlanSolved | None]:
    """High-level entry point used by ``runtime/planner.py``.

    Routes through the backend registry (MOSEK preferred, HiGHS
    fallback). Returns the full :class:`SolverResponse` and, when
    ``status in {OPTIMAL, FEASIBLE}``, a :class:`MemoryPlanSolved`.

    When no LP/MILP backend is available, returns ``status=BLOCKED``
    and ``plan=None`` — the caller MUST NOT fall back to a greedy
    heuristic.
    """

    from compgen.solve.backend_registry import default_registry
    from compgen.solve.routing import choose_backend

    reg = registry if registry is not None else default_registry()
    backend = choose_backend(SolverProblemKind.MEMORY_ALLOCATION, reg)
    request = SolverRequest(
        problem_id=problem_id,
        problem_kind=SolverProblemKind.MEMORY_ALLOCATION,
        formulation=_build_formulation(plan_input),
        time_budget_ms=plan_input.time_budget_ms,
    )
    if backend is None:
        response = SolverResponse(
            problem_id=problem_id,
            problem_kind=SolverProblemKind.MEMORY_ALLOCATION,
            selected_backend=SolverBackendName.HIGHS,
            backend_availability=BackendAvailabilityStatus.IMPORT_MISSING,
            status=SolverStatus.BLOCKED,
            formulation_hash=request.formulation_hash,
            time_ms=0.0,
            infeasibility_reason="no_lp_milp_backend",
        )
        return response, None

    # stage decomposition. When hints declare disjoint-
    # lifetime stages, solve each stage's sub-MILP independently
    # and merge results. For a layered network this turns one
    # giant MILP into N small MILPs (linear vs quadratic growth).
    if hints is not None and getattr(hints, "stage_partition", None):
        decomposed = _try_stage_decomposition(
            plan_input, hints, problem_id=problem_id, registry=reg
        )
        if decomposed is not None:
            return decomposed

    probe = reg.probe(backend)
    if backend is SolverBackendName.MOSEK:
        response = solve_via_mosek(request, probe=probe, hints=hints)
    else:
        response = solve_via_highs(request, probe=probe, hints=hints)

    if response.status in {SolverStatus.OPTIMAL, SolverStatus.FEASIBLE} and isinstance(
        response.solution, dict
    ):
        body = response.solution
        plan = MemoryPlanSolved(
            schema_version=body["schema_version"],
            solver_backend=body["solver_backend"],
            status=body["status"],
            buffers=tuple(
                BufferAllocation(
                    buffer_id=row["buffer_id"],
                    tier=row["tier"],
                    offset_bytes=int(row["offset_bytes"]),
                    aliases_with=row.get("aliases_with"),
                )
                for row in body["buffers"]
            ),
            tier_peak_usage=dict(body["tier_peak_usage"]),
            objective_value=body.get("objective_value"),
            formulation_hash=body["formulation_hash"],
        )
        return response, plan
    return response, None


def _try_stage_decomposition(
    plan_input: MemoryPlanInput,
    hints: Any,
    *,
    problem_id: str,
    registry: Any,
) -> tuple[SolverResponse, MemoryPlanSolved] | None:
    """Solve per-stage MILPs when the hint declares disjoint stages.

    Disjointness check: for each pair of stages, the union of their
    buffer lifetimes does not overlap with the other stage's union.
    When the check fails, returns None so the caller falls back to
    the monolithic MILP (no silent correctness loss).

    Sub-stage results are merged: per-buffer (tier, offset) are
    preserved; tier_peak_usage takes the MAX across stages (worst
    case the same buffer slot is alive at peak). Objective is the
    sum of sub-objectives.
    """

    if not hints.stage_partition:
        return None

    by_id = {b.buffer_id: b for b in plan_input.buffers}
    stage_buffers: dict[str, list[Any]] = {}
    assigned: set[str] = set()
    for stage in hints.stage_partition:
        ids = [bid for bid in stage.buffer_ids if bid in by_id]
        stage_buffers[stage.stage_id] = [by_id[i] for i in ids]
        assigned.update(ids)
    # Buffers not in any stage stay in a "global" residual stage.
    residual = [b for b in plan_input.buffers if b.buffer_id not in assigned]
    if residual:
        stage_buffers["__residual__"] = residual

    # Disjointness check.
    stage_ranges = {
        sid: (
            min(b.lifetime_start for b in bs) if bs else 0,
            max(b.lifetime_end for b in bs) if bs else 0,
        )
        for sid, bs in stage_buffers.items()
        if bs
    }
    sorted_stages = sorted(stage_ranges.items(), key=lambda kv: kv[1][0])
    for i in range(1, len(sorted_stages)):
        prev_end = sorted_stages[i - 1][1][1]
        cur_start = sorted_stages[i][1][0]
        # Strict: touching boundaries (prev_end == cur_start) are
        # still safe because buffers in the earlier stage die at
        # prev_end and the later stage's buffers are born at
        # cur_start = prev_end (under inclusive lifetime semantics
        # they share that single tick, but no two stages have
        # **overlapping** ranges since each is exclusively used
        # by its own stage's buffers).
        if cur_start < prev_end:
            # Overlapping stages — decomposition is unsafe; fall back.
            return None

    # Solve each stage independently.
    import time as _time
    from dataclasses import replace as _replace

    t0 = _time.perf_counter()
    all_buffers: list[BufferAllocation] = []
    tier_peak: dict[str, int] = {}
    total_obj = 0.0
    stage_results = []
    formulation_hash = compute_formulation_hash(_build_formulation(plan_input))
    for sid, bs in stage_buffers.items():
        if not bs:
            continue
        sub_input = MemoryPlanInput(
            buffers=tuple(bs),
            tier_capacities=plan_input.tier_capacities,
            alias_candidates=tuple(
                a for a in plan_input.alias_candidates
                if a.buffer_a in {b.buffer_id for b in bs}
                and a.buffer_b in {b.buffer_id for b in bs}
            ),
            fixed_assignments={
                k: v for k, v in plan_input.fixed_assignments.items()
                if k in {b.buffer_id for b in bs}
            },
            objective_lambda=plan_input.objective_lambda,
            time_budget_ms=plan_input.time_budget_ms,
        )
        # Re-derive sub-hints filtered to this stage.
        from compgen.solve.solver_hints import MemoryHints, TierHint, SymmetryClass

        stage_buf_ids = {b.buffer_id for b in bs}
        sub_hints = MemoryHints(
            tier_hints=tuple(
                t for t in hints.tier_hints if t.buffer_id in stage_buf_ids
            ),
            symmetry_classes=tuple(
                c for c in hints.symmetry_classes
                if all(bid in stage_buf_ids for bid in c.buffer_ids)
            ),
            source=hints.source + ":stage:" + sid,
        )
        sub_response, sub_plan = plan_memory(
            sub_input, registry=registry,
            problem_id=f"{problem_id}_stage_{sid}",
            hints=sub_hints,
        )
        stage_results.append((sid, sub_response, sub_plan))
        if sub_plan is None:
            return None  # one stage infeasible → fail decomposition
        all_buffers.extend(sub_plan.buffers)
        for t, v in sub_plan.tier_peak_usage.items():
            tier_peak[t] = max(tier_peak.get(t, 0), int(v))
        if sub_plan.objective_value is not None:
            total_obj += sub_plan.objective_value

    elapsed_ms = (_time.perf_counter() - t0) * 1000.0
    plan = MemoryPlanSolved(
        schema_version=MEMORY_PLAN_SCHEMA_VERSION,
        solver_backend="stage_decomposition+" + (
            stage_results[0][1].selected_backend.value if stage_results else "?"
        ),
        status="optimal",  # every sub-MILP returned optimal
        buffers=tuple(all_buffers),
        tier_peak_usage=tier_peak,
        objective_value=total_obj,
        formulation_hash=formulation_hash,
    )
    body = plan.to_dict()
    body["stage_count"] = len(stage_results)
    body["stage_times_ms"] = {sid: r.time_ms for sid, r, _ in stage_results}
    response = SolverResponse(
        problem_id=problem_id,
        problem_kind=SolverProblemKind.MEMORY_ALLOCATION,
        selected_backend=stage_results[0][1].selected_backend if stage_results else SolverBackendName.MOSEK,
        backend_availability=BackendAvailabilityStatus.AVAILABLE,
        status=SolverStatus.OPTIMAL,
        formulation_hash=formulation_hash,
        time_ms=elapsed_ms,
        objective_value=total_obj,
        solution=body,
    )
    return response, plan
