#!/usr/bin/env python3
"""The planner's hand-off table (gap rule "table"), measured from board rounds, and what a round's plan
charged against what the board paid.

From the rounds given (frames 1..N-1 of every model, all runs; frame-first layers held for their start gate
left out), pooled over the models:
  flat start cost   mean gap of layers that carry on on the SAME worker thread, in the same model, when that
                    worker's previous entry had woken nobody (goes on every row of the profile tree)
  table             for each (sender core kind -> receiver core kind): mean gap of layers that waited last
                    for an entry on ANOTHER worker, minus the flat start cost. All four pairs must have
                    samples: a pair without one is an error, never a guess.
  sender's delay    a layer that carries on on its worker right after that worker's previous entry woke a
                    layer on another thread (the sender pays for the wake-up): mean gap minus the flat start
                    cost, by the sender's core kind (scheduler.sender_ms)
Not in the table on purpose (reported per round and model, count and us per frame):
  model switch      a layer that carries on on its worker after a layer of ANOTHER model
Per round also: hand-offs to another worker per frame, in the plan and on the board; and per hand-off type
on the board, the gap the board showed against the gap the plan held (planned start - planned ready).

usage: handoff_table.py --round r0=<raw run folder>:<runs> [--round ...] --out table.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import statistics as st
import sys
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import loop_lib as T  # noqa: E402

PLANNER_KIND = {"x100": "cpu_p", "a100": "cpu_e"}
PAIRS = [("x100", "x100"), ("x100", "a100"), ("a100", "x100"), ("a100", "a100")]


def planned_binding(E: dict) -> Dict[int, tuple]:
    """eid -> (planned ready, eid the PLAN makes it wait for last, or -1 for its start gate)."""
    out, last = {}, {}
    for eid in sorted(E):
        e = E[eid]
        c = []
        if e.gate is not None:
            c.append((e.gate, -1))
        c += [(E[d].pe, d) for d in e.deps]
        if e.tdep >= 0:
            c.append((E[e.tdep].pe, e.tdep))
        w = last.get((e.kind, e.hart))
        if w is not None:
            c.append((E[w].pe, w))
        last[(e.kind, e.hart)] = eid
        out[eid] = max(c, key=lambda x: x[0]) if c else (0.0, -1)
    return out


def mean_us(xs: List[float]) -> float:
    return st.fmean(xs) * 1e3 if xs else float("nan")


def measure(rounds: List[tuple], allow_missing: bool = False) -> dict:
    flat: List[float] = []
    pair: Dict[tuple, List[float]] = collections.defaultdict(list)
    sender: Dict[str, List[float]] = collections.defaultdict(list)       # by the sender's core kind
    sender_n: Dict[str, List[float]] = collections.defaultdict(list)     # by how many threads it had just woken
    per_round = {}
    for name, raw, reps in rounds:
        fr_count: Dict[str, int] = collections.Counter()
        acc = {n: dict(board_handoffs=0, plan_handoffs=0, sender=[], switch=[], by_type=collections.defaultdict(lambda: dict(board=[], planned=[])),
                       plan_sender=0, both_sender=0, plan_sender_gap=[])
               for n in T.NETS}
        for k in range(1, reps + 1):
            rd = T.load(raw, k, checks=False)
            E = rd.run.entries
            pb = planned_binding(E)
            woken: collections.Counter = collections.Counter()
            for p in rd.pieces:
                if p.handoff and not p.handoff.startswith("same thread"):
                    woken[p.binding_eid] += 1
            for (net, inst) in rd.run.frames:
                fr_count[net] += inst > 0
            for p in rd.pieces:
                if p.frame == 0:
                    continue
                e = E[p.eid]
                a = acc[p.net]
                pr, pw = pb[p.eid]
                if pw >= 0 and (E[pw].kind, E[pw].hart) != (e.kind, e.hart):
                    a["plan_handoffs"] += 1
                if not p.handoff:
                    continue
                b = E[p.binding_eid]
                t = a["by_type"][p.handoff]
                t["board"].append(p.gap)
                t["planned"].append(e.ps - pr)
                # does the PLAN hold this layer back on its own worker (the planner's sender delay)? planned gap of more than
                # 1 us (the flat start cost sits in the duration, not in the gap) behind an entry of the same worker
                plan_holds = pw >= 0 and (E[pw].kind, E[pw].hart) == (e.kind, e.hart) and (e.ps - pr) > 0.001
                if plan_holds:
                    a["plan_sender"] += 1
                    a["plan_sender_gap"].append(e.ps - pr)
                    a["both_sender"] += p.handoff == "same thread" and woken[p.binding_eid] > 0
                if p.handoff == "same thread":
                    if woken[p.binding_eid]:
                        a["sender"].append(p.gap)
                        sender[e.kind].append(p.gap)
                        sender_n["1" if woken[p.binding_eid] == 1 else "2 or more"].append(p.gap)
                    else:
                        flat.append(p.gap)
                elif p.handoff == "same thread, other model":
                    a["switch"].append(p.gap)
                else:
                    a["board_handoffs"] += 1
                    pair[(b.kind, e.kind)].append(p.gap)
        per_round[name] = dict(raw=raw, runs=reps, models={})
        for n in T.NETS:
            a, nf = acc[n], max(fr_count[n], 1)
            per_round[name]["models"][n] = dict(
                frames=fr_count[n], handoffs_per_frame_board=a["board_handoffs"] / nf, handoffs_per_frame_plan=a["plan_handoffs"] / nf,
                sender_delay=dict(per_frame=len(a["sender"]) / nf, mean_gap_us=mean_us(a["sender"]), us_per_frame=sum(a["sender"]) * 1e3 / nf,
                                  held_by_the_plan_per_frame=a["plan_sender"] / nf, held_by_plan_and_seen_on_board_per_frame=a["both_sender"] / nf,
                                  plan_us_per_frame=sum(a["plan_sender_gap"]) * 1e3 / nf),
                model_switch=dict(per_frame=len(a["switch"]) / nf, mean_gap_us=mean_us(a["switch"]), us_per_frame=sum(a["switch"]) * 1e3 / nf),
                by_handoff={t: dict(per_frame=len(v["board"]) / nf, board_gap_mean_us=mean_us(v["board"]),
                                    board_gap_median_us=T.quantile(v["board"], .5) * 1e3, planned_gap_mean_us=mean_us(v["planned"]))
                            for t, v in sorted(a["by_type"].items())})
    if not flat:
        raise SystemExit("no same-thread hand-off without a wake-up in these rounds: cannot measure the flat start cost")
    flat_ms = st.fmean(flat)
    missing = [f"{s} -> {r}" for s, r in PAIRS if not pair[(s, r)]]
    if missing and not allow_missing:
        raise SystemExit(f"no sample for the hand-off pair(s) {missing} in these rounds: the table is not built on a guess")
    table = {PLANNER_KIND[s]: {} for s, _ in PAIRS}
    detail = {}
    for s, r in PAIRS:
        v = pair[(s, r)]
        val = round(st.fmean(v) - flat_ms, 6) if v else None
        table[PLANNER_KIND[s]][PLANNER_KIND[r]] = val
        detail[f"{T.KIND_NAME[s]} -> {T.KIND_NAME[r]}"] = dict(n=len(v), mean_gap_us=mean_us(v), median_gap_us=(T.quantile(v, .5) * 1e3 if v else float("nan")),
                                                                table_ms=val, planner_key=[PLANNER_KIND[s], PLANNER_KIND[r]])
    # the sender's delay: a real layer that carries on on its worker right after that worker's previous entry woke a
    # layer on another thread, and that waited last for exactly that entry. Mean gap minus the flat start cost.
    sender_ms, sender_detail = {}, {}
    for kind in ("x100", "a100"):
        v = sender[kind]
        sender_ms[PLANNER_KIND[kind]] = round(st.fmean(v) - flat_ms, 6) if v else None
        sender_detail[T.KIND_NAME[kind]] = dict(n=len(v), mean_gap_us=mean_us(v), median_gap_us=(T.quantile(v, .5) * 1e3 if v else float("nan")),
                                                 delay_ms=sender_ms[PLANNER_KIND[kind]], planner_key=PLANNER_KIND[kind])
    by_woken = {k: dict(n=len(v), mean_gap_us=mean_us(v), median_gap_us=T.quantile(v, .5) * 1e3) for k, v in sorted(sender_n.items())}
    return dict(schema="k3work.t023.handoff_table/v1", rounds=[n for n, _, _ in rounds], flat_start_cost_ms=round(flat_ms, 6),
                flat_samples=len(flat), handoff_ms=table, pairs=detail, sender_ms=sender_ms, sender=sender_detail,
                sender_by_threads_woken=by_woken, per_round=per_round,
                not_in_the_table=["model switch on the same worker"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", action="append", required=True, help="name=<raw folder>:<runs>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-missing", action="store_true", help="for checking one round against a table: a pair without a sample is reported, not an error")
    a = ap.parse_args()
    rounds = []
    for r in a.round:
        name, rest = r.split("=", 1)
        folder, reps = rest.rsplit(":", 1)
        rounds.append((name, folder, int(reps)))
    doc = measure(rounds, a.allow_missing)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    print(f"flat start cost {doc['flat_start_cost_ms'] * 1e3:.2f} us ({doc['flat_samples']} samples); table (ms): {json.dumps(doc['handoff_ms'])}")
    for k, v in doc["pairs"].items():
        print(f"  {k}: n {v['n']}, mean gap {v['mean_gap_us']:.2f} us, median {v['median_gap_us']:.2f} us -> table "
              + (f"{v['table_ms'] * 1e3:.2f} us" if v["table_ms"] is not None else "no sample"))
    for k, v in doc["sender"].items():
        print(f"  sender's delay, {k} sends: n {v['n']}, mean gap {v['mean_gap_us']:.2f} us, median {v['median_gap_us']:.2f} us -> "
              + (f"{v['delay_ms'] * 1e3:.2f} us" if v["delay_ms"] is not None else "no sample"))
    print("  by threads woken: " + "; ".join(f"{k}: n {v['n']}, mean {v['mean_gap_us']:.2f}, median {v['median_gap_us']:.2f} us" for k, v in doc["sender_by_threads_woken"].items()))
    for name, r in doc["per_round"].items():
        for n, m in r["models"].items():
            print(f"  {name} {n:12s} hand-offs/frame board {m['handoffs_per_frame_board']:.1f} plan {m['handoffs_per_frame_plan']:.1f}; "
                  f"sender delay {m['sender_delay']['per_frame']:.1f}/frame = {m['sender_delay']['us_per_frame']:.0f} us "
                  f"(plan holds {m['sender_delay']['held_by_the_plan_per_frame']:.1f}/frame = {m['sender_delay']['plan_us_per_frame']:.0f} us, "
                  f"{m['sender_delay']['held_by_plan_and_seen_on_board_per_frame']:.1f} of them seen on the board); "
                  f"model switch {m['model_switch']['per_frame']:.2f}/frame = {m['model_switch']['us_per_frame']:.1f} us")
    return 0


if __name__ == "__main__":
    sys.exit(main())
