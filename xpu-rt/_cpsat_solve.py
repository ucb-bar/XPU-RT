"""CP-SAT solve step. Runs under an interpreter that has ortools; talks JSON.

Kept deliberately dependency-free apart from ortools so it can run in a bare
venv: see cpsat_scheduler.py for why it is a separate process at all.
"""
import json
import os
import sys
import time

from ortools.sat.python import cp_model


def main(inp, outp):
    m = json.load(open(inp))
    n, n_combos = m["n"], m["n_combos"]
    dur, pred = m["dur"], m["pred"]
    conflict, first_machine, transfer = m["conflict"], m["first_machine"], m["transfer"]
    # Machines each combination occupies. Falls back to the first machine only
    # if an older payload omits it.
    combo_machines = m.get("combo_machines") or [[c] for c in first_machine]
    min_start, max_end, periodic = m["min_start"], m["max_end"], m["periodic"]

    horizon = sum(max(d for d in row if d >= 0) if any(d >= 0 for d in row) else 0
                  for row in dur)
    horizon = max(horizon, max((v for v in max_end if v >= 0), default=0)) + 1

    # ------------------------------------------------------------------
    # Redundant bounds.
    #
    # The formulation below is *implied* by the model already: it adds no
    # solutions and removes none. It exists because AddNoOverlap and the
    # reified precedence/duration constraints are invisible to the linear
    # relaxation CP-SAT prunes against, so the reported objective bound stayed
    # near the critical path of a single operation while the incumbent sat 17x
    # higher (5.07 ms against 85.42 on control_mix_gempair). Everything here is
    # a linear statement in the combination literals, which the relaxation can
    # actually use.
    #
    # XPURT_CPSAT_BOUNDS selects which to apply -- "0"/"none" for the original
    # model, or a comma list of prec,dur,tail,load -- so each can be measured
    # on its own.
    _b = os.environ.get("XPURT_CPSAT_BOUNDS")
    if _b is None:
        BOUNDS = {"prec", "dur", "tail", "load"}
    elif _b.strip().lower() in ("0", "none", ""):
        BOUNDS = set()
    else:
        BOUNDS = {x.strip() for x in _b.split(",") if x.strip()}

    succ = [[] for _ in range(n)]
    for i in range(n):
        for p in pred[i]:
            succ[p].append(i)
    indeg = [len(pred[i]) for i in range(n)]
    stack = [i for i in range(n) if indeg[i] == 0]
    topo = []
    while stack:
        u = stack.pop()
        topo.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                stack.append(v)
    # A cycle would make the head/tail recurrences meaningless, and a wrong
    # lower bound is far worse than a weak one: drop the path bounds instead.
    acyclic = len(topo) == n
    if not acyclic:
        BOUNDS -= {"tail"}

    usable_c = [[c for c in range(n_combos) if dur[i][c] >= 0] for i in range(n)]
    min_dur = [min((dur[i][c] for c in usable_c[i]), default=0) for i in range(n)]

    # Cheapest transfer over any pair of combinations the two ops could take:
    # a valid lower bound on the edge's real cost, whatever gets assigned.
    tt_edge = {}
    for i in range(n):
        for p in pred[i]:
            if usable_c[p] and usable_c[i]:
                tt_edge[(p, i)] = min(
                    transfer[first_machine[cp]][first_machine[c]]
                    for cp in usable_c[p] for c in usable_c[i])
            else:
                tt_edge[(p, i)] = 0

    targets = ([i for i in range(n) if not periodic[i]]
               if m["restrict_to_nonperiodic"] else list(range(n)))
    if not targets:
        targets = list(range(n))
    is_target = [False] * n
    for i in targets:
        is_target[i] = True

    # head[i]: earliest instant op i can start, from min_start and the longest
    # chain of minimum durations reaching it.
    head = list(min_start)
    if acyclic:
        for u in topo:
            h = min_start[u]
            for p in pred[u]:
                h = max(h, head[p] + min_dur[p] + tt_edge[(p, u)])
            head[u] = h

    # tail[i]: minimum work that must still follow op i before every target
    # downstream of it has finished. None means "no target is reachable", in
    # which case cmax says nothing about this op and it gets no constraint.
    tail = [None] * n
    if acyclic:
        for u in reversed(topo):
            best = 0 if is_target[u] else None
            for v in succ[u]:
                if tail[v] is None:
                    continue
                cand = tt_edge[(u, v)] + min_dur[v] + tail[v]
                if best is None or cand > best:
                    best = cand
            tail[u] = best

    # Static critical-path bound, also the yardstick for deciding which
    # periodic operations are certain to land inside [0, cmax].
    lb_cp = 0
    for i in range(n):
        if tail[i] is not None:
            lb_cp = max(lb_cp, head[i] + min_dur[i] + tail[i])

    model = cp_model.CpModel()
    start, end, combo_lit, dur_var = [], [], [], []
    for i in range(n):
        # head[i] >= min_start[i] by construction; it folds in the precedence
        # chain so the domain starts where the operation actually could.
        lo = min(head[i], horizon) if "prec" in BOUNDS else min_start[i]
        hi = max_end[i] if max_end[i] >= 0 else horizon
        s = model.NewIntVar(lo, horizon, f"s{i}")
        e = model.NewIntVar(0, horizon, f"e{i}")
        d = model.NewIntVar(0, horizon, f"d{i}")
        # A boolean per (op, usable combination): exactly one is true, and it
        # selects that combination's duration.
        lits = []
        for c in range(n_combos):
            if dur[i][c] < 0:
                lits.append(None)
                continue
            b = model.NewBoolVar(f"a{i}_{c}")
            lits.append(b)
            model.Add(d == dur[i][c]).OnlyEnforceIf(b)
        usable = [b for b in lits if b is not None]
        if not usable:                      # unusable everywhere: pin to zero
            b = model.NewBoolVar(f"a{i}_0")
            lits[0] = b
            model.Add(b == 1)
            model.Add(d == 0)
            usable = [b]
        model.AddExactlyOne(usable)
        # The reified `d == dur[i][c] if b` above is exact but opaque to the
        # relaxation. Exactly one literal is true, so the duration is equally
        # a plain linear function of them -- and that form the relaxation can
        # propagate, which is what makes the machine-load bound below bite.
        if "dur" in BOUNDS:
            terms = [dur[i][c] * lits[c] for c in range(n_combos)
                     if lits[c] is not None and dur[i][c] >= 0]
            if terms:
                model.Add(d == sum(terms))
        model.Add(e == s + d)
        if max_end[i] >= 0:
            model.Add(e <= hi)
        start.append(s); end.append(e); dur_var.append(d); combo_lit.append(lits)

    # No-overlap per machine: one optional interval per (op, combination),
    # present exactly when that combination is chosen. This is the piece the
    # MILP has to spell out as O(N^2) big-M ordering rows.
    n_machines = len(transfer)
    per_machine = [[] for _ in range(n_machines)]
    for i in range(n):
        for c in range(n_combos):
            b = combo_lit[i][c]
            if b is None:
                continue
            iv = model.NewOptionalIntervalVar(start[i], dur_var[i], end[i], b,
                                              f"iv{i}_{c}")
            # One interval per machine the combination actually occupies.
            #
            # This used to walk the conflict row and file the interval under
            # the FIRST conflicting combination's first machine, then break.
            # With sibling-core combinations -- ['CPU_P#0'], ['CPU_P#0',
            # 'CPU_P#1'], ['CPU_P#1'] -- combination 2 conflicts first with
            # combination 1, whose first machine is CPU_P#0, so ALL THREE
            # combinations landed on machine 0's list and machine 1's list
            # stayed empty. The single resulting AddNoOverlap then forbade
            # ['CPU_P#0'] and ['CPU_P#1'] from running at the same time, which
            # is precisely the two-hart parallelism the pair configurations
            # exist to use: CP-SAT was solving a model where a gemmini or rvv
            # pair is serialised. It answered that model correctly (85.42 ms on
            # control_mix_gempair against heft_edf's 60.07) and rejected a
            # correct schedule handed to it as a hint, reporting it "complete,
            # but infeasible".
            for mi in combo_machines[c]:
                per_machine[mi].append(iv)
    for machine_intervals in per_machine:
        if len(machine_intervals) > 1:
            model.AddNoOverlap(machine_intervals)

    # Precedence, including the transfer cost between the predecessor's
    # combination and this op's.
    for i in range(n):
        for p in pred[i]:
            # The per-(combination, combination) rows below are all guarded by
            # OnlyEnforceIf, so until every pair is decided the relaxation sees
            # no precedence at all. Charging the cheapest transfer any pair
            # could incur is valid whatever gets chosen, and states the edge
            # unconditionally.
            if "prec" in BOUNDS:
                model.Add(start[i] >= end[p] + tt_edge[(p, i)])
            for c in range(n_combos):
                bi = combo_lit[i][c]
                if bi is None:
                    continue
                for cp_ in range(n_combos):
                    bp = combo_lit[p][cp_]
                    if bp is None:
                        continue
                    tt = transfer[first_machine[cp_]][first_machine[c]]
                    model.Add(start[i] >= end[p] + tt).OnlyEnforceIf([bi, bp])

    cmax = model.NewIntVar(0, horizon, "cmax")
    model.AddMaxEquality(cmax, [end[i] for i in targets])

    if "tail" in BOUNDS:
        # cmax is a max over target ends only, so an operation constrains it
        # only through a path to some target. tail[i] is the least work left on
        # the longest such path, hence cmax >= end[i] + tail[i]. For a target
        # itself tail is 0 and this restates AddMaxEquality; upstream it is new
        # information, and it is the only thing tying a non-target operation's
        # end to the objective.
        model.Add(cmax >= lb_cp)
        for i in range(n):
            if tail[i]:
                model.Add(cmax >= end[i] + tail[i])

    if "load" in BOUNDS:
        # Per-machine total work. AddNoOverlap already forbids two operations
        # sharing a machine from overlapping, but the relaxation cannot see
        # that this forces their durations to *sum* inside the schedule. Stated
        # as a linear row over the combination literals it can.
        #
        # Only operations certain to finish by cmax may be counted. Target ops
        # qualify by definition. A periodic op qualifies when its deadline is
        # at or below the static critical-path bound, since cmax >= lb_cp >=
        # max_end[j] then makes its completion by cmax unconditional. Counting
        # every periodic op instead would be plain wrong: their windows run far
        # past cmax (128 ms of periodic work against a 33 ms objective on
        # control_mix_quad), and the bound would exceed the true optimum.
        counted = [is_target[i] or (0 <= max_end[i] <= lb_cp) for i in range(n)]
        for mi in range(n_machines):
            terms, floors = [], []
            for i in range(n):
                if not counted[i]:
                    continue
                for c in range(n_combos):
                    b = combo_lit[i][c]
                    if b is None or dur[i][c] <= 0:
                        continue
                    if mi in combo_machines[c]:
                        terms.append(dur[i][c] * b)
                        floors.append(head[i])
            if not terms:
                continue
            # The machine is idle until the earliest of these could begin, so
            # that offset can be added -- but only when it is safe with an
            # empty selection too, i.e. when cmax >= floor already holds via
            # lb_cp. Otherwise anchor at 0.
            floor = min(floors)
            if floor > lb_cp:
                floor = 0
            model.Add(cmax >= floor + sum(terms))

    model.Minimize(cmax)

    # Solution hint: a schedule some cheaper method already found. CP-SAT
    # treats it as advice, not constraint — an infeasible hint costs nothing
    # beyond being ignored.
    if m.get("hint_start") and m.get("hint_combo"):
        hs, hc = m["hint_start"], m["hint_combo"]
        hd, he = m.get("hint_dur"), m.get("hint_end")
        # Only hint an operation whose values sit inside their own domains and
        # satisfy end == start + duration. Clamping a value to fit (what this
        # did before) produces an inconsistent assignment that CP-SAT cannot
        # complete, and the whole hint is then thrown away.
        hinted = 0
        for i in range(n):
            s_i = hs[i]
            d_i = hd[i] if hd else None
            e_i = he[i] if he else None
            if d_i is None or e_i is None or e_i != s_i + d_i:
                continue
            if s_i < min_start[i] or e_i > horizon:
                continue
            if max_end[i] >= 0 and e_i > max_end[i]:
                continue
            model.AddHint(start[i], s_i)
            model.AddHint(dur_var[i], d_i)
            model.AddHint(end[i], e_i)
            for c in range(n_combos):
                if combo_lit[i][c] is not None:
                    model.AddHint(combo_lit[i][c], 1 if hc[i] == c else 0)
            hinted += 1
        # The objective variable needs hinting too. Leaving it out makes the
        # hint "incomplete" in CP-SAT's terms, and an incomplete hint cannot be
        # adopted as a solution — it is only advice for the search. That one
        # missing variable was the difference between the hint being used and
        # being thrown away.
        if hinted == n:
            model.AddHint(cmax, max(he[i] for i in targets))
        print(f"cpsat hint: {hinted}/{n} operations hinted"
              + (" + objective" if hinted == n else " (incomplete: objective not hinted)"))


    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(m["time_limit"])
    solver.parameters.num_search_workers = int(m["workers"])
    solver.parameters.random_seed = int(m.get("random_seed", 0))
    # CPSAT_LOG=1 turns on the solver's own progress log, which states whether
    # it found the supplied hint complete and feasible — the only reliable way
    # to tell a used hint from a silently-discarded one.
    if os.environ.get("CPSAT_LOG"):
        solver.parameters.log_search_progress = True
    t0 = time.time()
    status = solver.Solve(model)
    wall = time.time() - t0

    name = solver.StatusName(status)
    out = {"status": name, "wall": wall}
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        out["start"] = [solver.Value(s) for s in start]
        out["combo"] = [next(c for c in range(n_combos)
                             if combo_lit[i][c] is not None
                             and solver.Value(combo_lit[i][c]))
                        for i in range(n)]
        out["objective"] = solver.ObjectiveValue()
        out["best_bound"] = solver.BestObjectiveBound()
    json.dump(out, open(outp, "w"))
    print(f"cpsat status={name} wall={wall:.1f}s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
