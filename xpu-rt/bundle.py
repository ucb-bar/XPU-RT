"""Candidate-bundle proposer + fusion-hint contract for the iterative loop.

From a baseline SchedulerReport + advisor Diagnosis, propose a *bundle* of
candidate configurations to try, spanning three axes:

  A. scheduler/placement  — swap --solver / --scheduler          (realizable_by: xpurt)
  B. profiler/backend      — reassign hardware.profile_hw          (realizable_by: xpurt)
  C. granularity/fusion    — fuse op groups (advisor coarsen recs) (realizable_by: modelblaster)

Each candidate is self-describing so xpurt can run A/B itself (predicted, fast),
while C is emitted as a fusion-hint payload that the ModelBlaster session
realizes (re-extract/re-gen kernels on spike, re-profile on FireSim). Bundling
lets the MB session batch many candidates into one expensive FireSim session.

Pure-python: this module only proposes; the driver (scripts/iterate_firesim.py)
runs the candidates.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# Curated axis-A scheduler candidates as (solver, scheduler) pairs. solver in
# {milp,greedy,greedy_periodic,decomposed}; scheduler is the registry algorithm
# used only when solver=="milp".
DEFAULT_SCHEDULERS: List[Dict[str, Optional[str]]] = [
    # Exact ILP solvers first — these are the preferred schedulers when
    # the time budget allows. MOSEK is the commercial cvxpy backend;
    # CPSAT is Google OR-Tools' constraint-programming solver. Both
    # give provably-optimal schedules on the demo workload's size.
    {"solver": "milp", "scheduler": "mosek"},
    {"solver": "milp", "scheduler": "cpsat"},
    # Polynomial-time heuristics — fast fallback when ILP is over
    # budget; they're typically within a few % of optimal on real
    # workloads.
    {"solver": "milp", "scheduler": "heft"},
    {"solver": "milp", "scheduler": "peft"},
    {"solver": "milp", "scheduler": "edf"},
    # Greedy variants — sanity baselines that don't use the ILP
    # registry at all. greedy_periodic explicitly partitions periodic
    # ops by release time so it always appears in the comparison row.
    {"solver": "greedy", "scheduler": None},
    {"solver": "greedy_periodic", "scheduler": None},
    {"solver": "decomposed", "scheduler": None},
]


def _parse_name(name: str):
    """('mlp_control0_dispatch_7') -> (root 'mlp_control', local_dispatch_id 7).

    The network *root* (instance index stripped) is shared across periodic
    instances, and the local dispatch id indexes the model's own dispatch graph
    — which is what the ModelBlaster fusion contract references (not the global
    flattened SchedulerReport id).
    """
    name = str(name)
    for sep in ("_dispatch_", "$dispatch_"):
        if sep in name:
            pre, post = name.split(sep, 1)
            root = pre.rstrip("0123456789") or pre
            local = int(post) if post.isdigit() else None
            return root, local
    return (name.split("_")[0] if name else "unknown"), None


def backend_assignments(available: List[str], machines=("cpu_p", "cpu_e")) -> List[Dict[str, str]]:
    """Enumerate profile_hw assignments to try: each backend applied homogeneously
    to all machines, plus the heterogeneous pairing of the first two backends."""
    out: List[Dict[str, str]] = []
    for hw in available:
        out.append({m: hw for m in machines})
    if len(available) >= 2 and len(machines) >= 2:
        het = {machines[0]: available[0], machines[1]: available[1]}
        if het not in out:
            out.append(het)
    return out


def fusion_hints_from_diagnosis(report: Dict[str, Any], diag: Any) -> Dict[str, Any]:
    """Derive the fusion-hint contract from the report's tiny-dispatch chains.

    Groups, per network, maximal runs of adjacent sub-1k-us dispatches connected
    by a dependency edge — i.e. small pre/post ops that fusing would collapse.
    The ModelBlaster session consumes `networks[*].fuse_groups` (lists of op ids
    in that network's dispatch graph) to re-extract a coarser graph.

    Schema:
      {"contract": "modelblaster.fusion_hints/v1",
       "reason": str,
       "networks": [{"network": str, "fuse_groups": [[op_id, ...], ...],
                     "n_tiny": int}]}
    """
    dispatches = report.get("dispatches") or []
    TINY = 1_000.0
    # per-dispatch metadata indexed by GLOBAL id (used to follow dep edges); each
    # carries its network root + LOCAL dispatch id (the fusion contract's unit).
    meta = {}
    for d in dispatches:
        if "id" not in d:
            continue
        root, local = _parse_name(d.get("name", ""))
        meta[d["id"]] = {"root": root, "local": local,
                         "tiny": float(d.get("duration_us", 0.0)) < TINY,
                         "deps": d.get("deps", []), "start": d.get("start_us", 0.0)}

    # group dispatches by network root
    per_net: Dict[str, List[int]] = {}
    for gid, m in meta.items():
        per_net.setdefault(m["root"], []).append(gid)

    networks_out: List[Dict[str, Any]] = []
    for net, gids in per_net.items():
        tiny = {g for g in gids if meta[g]["tiny"]}
        if len(tiny) < 2:
            continue
        # maximal chains of tiny dispatches linked by dep edges (within this net)
        local_groups: set = set()
        n_tiny_local: set = set()
        used: set = set()
        for g in sorted(gids, key=lambda x: meta[x]["start"]):
            if g not in tiny or g in used:
                continue
            chain = [g]
            used.add(g)
            changed = True
            while changed:
                changed = False
                tail = chain[-1]
                for j in gids:
                    if j in tiny and j not in used and tail in meta[j]["deps"]:
                        chain.append(j)
                        used.add(j)
                        changed = True
                        break
            locals_ = [meta[c]["local"] for c in chain if meta[c]["local"] is not None]
            n_tiny_local.update(l for l in (meta[c]["local"] for c in chain) if l is not None)
            if len(locals_) >= 2:
                local_groups.add(tuple(locals_))   # dedupe identical groups across instances
        if local_groups:
            networks_out.append({
                "network": net,
                "fuse_groups": sorted([list(g) for g in local_groups]),
                "n_tiny": len(n_tiny_local),
            })

    return {
        "contract": "modelblaster.fusion_hints/v1",
        "reason": (f"granularity verdict '{getattr(diag, 'granularity_verdict', '?')}': "
                   "fuse adjacent sub-1k-us dispatch chains to cut launch/transition overhead"),
        "networks": networks_out,
    }


def propose_bundle(report: Dict[str, Any], diag: Any, *,
                   baseline: Dict[str, Any],
                   available_backends: List[str],
                   schedulers: Optional[List[Dict[str, Optional[str]]]] = None) -> Dict[str, Any]:
    """Build a bundle of candidate configs to try, given a baseline run.

    `baseline` describes the current config: {solver, scheduler, profile_hw,
    makespan_us, meets_deadline}. Candidates exclude the baseline's own config.
    """
    schedulers = schedulers or DEFAULT_SCHEDULERS
    candidates: List[Dict[str, Any]] = []
    n = 0

    # Axis A: scheduler/placement swaps (xpurt-realizable)
    for sc in schedulers:
        if sc.get("solver") == baseline.get("solver") and sc.get("scheduler") == baseline.get("scheduler"):
            continue
        n += 1
        label = sc["solver"] + (f"/{sc['scheduler']}" if sc.get("scheduler") else "")
        candidates.append({
            "id": f"A{n}", "axis": "scheduler", "realizable_by": "xpurt",
            "solver": sc["solver"], "scheduler": sc.get("scheduler"),
            "profile_hw": baseline.get("profile_hw"),
            "rationale": f"try {label} placement vs baseline "
                         f"{baseline.get('solver')}",
        })

    # Axis B: backend/profiler reassignments (xpurt-realizable)
    for i, ph in enumerate(backend_assignments(available_backends), 1):
        if ph == baseline.get("profile_hw"):
            continue
        candidates.append({
            "id": f"B{i}", "axis": "backend", "realizable_by": "xpurt",
            "solver": baseline.get("solver"), "scheduler": baseline.get("scheduler"),
            "profile_hw": ph,
            "rationale": f"run on profile_hw={ph} (axis-B backend comparison)",
        })

    # Axis C: granularity/fusion (modelblaster-realizable) — only if too_fine
    if getattr(diag, "granularity_verdict", "") == "too_fine":
        hints = fusion_hints_from_diagnosis(report, diag)
        if hints["networks"]:
            candidates.append({
                "id": "C1", "axis": "fusion", "realizable_by": "modelblaster",
                "solver": baseline.get("solver"), "scheduler": baseline.get("scheduler"),
                "profile_hw": baseline.get("profile_hw"),
                "hints": hints,
                "rationale": "advisor flagged too_fine; fuse tiny dispatch chains "
                             "(ModelBlaster re-extract/re-profile needed)",
            })

    return {
        "contract": "xpurt.candidate_bundle/v1",
        "deadline_us": getattr(diag, "deadline_us", None),
        "baseline": baseline,
        "candidates": candidates,
        "counts": {
            "xpurt": sum(1 for c in candidates if c["realizable_by"] == "xpurt"),
            "modelblaster": sum(1 for c in candidates if c["realizable_by"] == "modelblaster"),
        },
    }
