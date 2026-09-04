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

    model = cp_model.CpModel()
    start, end, combo_lit, dur_var = [], [], [], []
    for i in range(n):
        lo = min_start[i]
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

    targets = [i for i in range(n) if not periodic[i]] if m["restrict_to_nonperiodic"] else list(range(n))
    if not targets:
        targets = list(range(n))
    cmax = model.NewIntVar(0, horizon, "cmax")
    model.AddMaxEquality(cmax, [end[i] for i in targets])
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
