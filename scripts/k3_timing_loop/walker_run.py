#!/usr/bin/env python3
"""Read one K3 walker run: the walker's trace, its placement record, the generated dispatch table (the
dependencies, ordering edges and harts the walker obeyed), the models' IRs and the workload config.

How times line up (ModelBlaster pipeline/generate_xpurt_main.py):
  * every trace time is ns since `run_t0` (taken just before the workers start; reset after the
    warm-up pass);
  * dispatch 0 of every frame is held until run_t0 + its planned start, so the plan's t = 0 IS run_t0.
    Planned and actual times therefore share one origin with no fitting.
  * the trace's start stamp is taken BEFORE the entry locks its harts, so a recorded span can include
    waiting for another entry (a multi-core layer) to release a core. The lock is a per-hart mutex, so
    the kernels on one hart never overlap: ordering the entries that use a hart by end time gives the
    lock order, and an entry's kernel started at max(recorded start, end of the previous holder of
    each of its harts).
  * the trace's dispatch_id is a record slot; the IR id comes from k1_trace.ir_slot_map and is checked
    against the trace's own layer name (must match for every row).

Per entry: planned start/end (ps, pe), recorded start (rs: dependencies done, worker free), kernel
start (xs, after the hart locks), actual end (ae), and what it waited for last (binding): the start
gate, a data dependency, the ordering edge, or the previous entry of the same worker.

Also here: the check of a run's printed output block against a golden output (bit-exact).
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "../.."))          # the XPU-RT checkout this file is in
sys.path.insert(0, os.path.join(REPO, "xpu-rt"))
import k1_trace  # noqa: E402

NS_PER_MS = 1e6


@dataclass
class Entry:
    eid: int
    net: str
    inst: int
    slot: int
    irid: Optional[int]
    op: str
    name: str
    kind: str
    impl: str
    hart: int
    harts: Tuple[int, ...]
    ps: float               # planned start (ms, trace column, 6 decimals)
    pdur: float             # planned duration (ms)
    rs: Optional[float]     # recorded start (ms since run_t0), None for zero-cost
    ae: Optional[float]     # actual end (ms)
    deps: Tuple[int, ...] = ()
    tdep: int = -1
    gate: Optional[float] = None    # start gate (dispatch slot 0): planned start
    # derived
    xs: Optional[float] = None      # kernel start after the hart locks
    lock_holder: int = -1
    ready: Optional[float] = None   # when the last thing it waited for was done
    binding: str = ""               # gate | dep | tdep | worker
    binding_eid: int = -1
    done: Optional[float] = None    # when its completion was posted

    @property
    def real(self) -> bool:
        return self.slot >= 0 and self.rs is not None

    @property
    def pe(self) -> float:
        return self.ps + self.pdur

    @property
    def lock_wait(self) -> float:
        return (self.xs - self.rs) if self.real else 0.0

    @property
    def gap(self) -> float:
        return (self.rs - self.ready) if self.real else 0.0

    @property
    def run_extra(self) -> float:
        return (self.ae - self.xs - self.pdur) if self.real else 0.0


@dataclass
class Frame:
    net: str
    inst: int
    release: float
    deadline: float
    entries: List[int] = field(default_factory=list)
    planned_start: float = 0.0
    planned_end: float = 0.0
    actual_start: float = 0.0
    actual_end: float = 0.0

    @property
    def late(self) -> bool:
        return self.actual_end > self.deadline

    @property
    def lateness(self) -> float:
        return self.actual_end - self.deadline

    @property
    def planned_late(self) -> bool:
        return self.planned_end > self.deadline + 1e-6


@dataclass
class Run:
    run_id: str
    entries: Dict[int, Entry]
    frames: Dict[Tuple[str, int], Frame]
    periods: Dict[str, Tuple[float, float]]
    audit: Dict[str, object]
    ir_names: Dict[str, Dict[int, str]]


# ---------------------------------------------------------------- loading
def _csv_rows(path: str) -> List[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def rows_from_stdout(path: str, block: str) -> List[dict]:
    text = open(path, errors="replace").read()
    m = re.search(rf"=== MODELBLASTER_XPURT_{block}_BEGIN ===\n(.*?)=== MODELBLASTER_XPURT_{block}_END ===",
                  text, re.S)
    return list(csv.DictReader(io.StringIO(m.group(1)))) if m else []


_FIELD = re.compile(r'\.(\w+) = ("[^"]*"|[^,}]+)')


def parse_table(path: str) -> Dict[int, dict]:
    """The generated dispatch table: per entry_id, deps, time dep, harts, hart, start."""
    text = open(path).read()
    arrays = {m.group(1): tuple(int(x) for x in m.group(2).split(",") if x.strip())
              for m in re.finditer(r"static const int (sched_\d+_(?:deps|harts))\[\] = \{([^}]*)\};", text)}
    out = {}
    for line in text.splitlines():
        s = line.strip()
        if not s.startswith("{ .entry_id"):
            continue
        f = dict(_FIELD.findall(s))
        eid = int(f["entry_id"])
        deps = arrays.get(f["deps"].strip(), ()) if f["deps"].strip() != "NULL" else ()
        harts = arrays.get(f["harts"].strip(), ()) if f["harts"].strip() != "NULL" else ()
        out[eid] = dict(deps=deps, tdep=int(f["time_dep_entry_id"]), n_harts=int(f["n_harts"]),
                        harts=harts, hart=int(f["hart"]), slot=int(f["dispatch_id"]),
                        start=float(f["start_time_ms"].rstrip("f")), network=f["network"].strip('"'),
                        instance=int(f["instance"]))
    return out


def load_irs(ir_args: List[str]) -> Tuple[Dict[str, Dict[int, int]], Dict[str, Dict[int, Tuple[str, str]]]]:
    slot_maps, ops = {}, {}
    for a in ir_args:
        net, path = a.split("=", 1)
        ir = json.load(open(path))
        slot_maps[net] = k1_trace.ir_slot_map(ir)
        ops[net] = {o["dispatch_id"]: (o.get("name", ""), o.get("op", ""))
                    for o in ir.get("ops", []) if o.get("dispatch_id") is not None}
    return slot_maps, ops


def load_periods(config: str) -> Dict[str, Tuple[float, float]]:
    nets = json.load(open(config))["networks"]
    return {n: (float(v["period"]), float(v.get("window_duration", v["period"]))) for n, v in nets.items()}


def load_run(run_id: str, trace: str, placement: str, table: str, irs: List[str], config: str,
             schedule: Optional[str] = None) -> Run:
    trows = _csv_rows(trace) if trace.endswith(".csv") else rows_from_stdout(trace, "TRACE")
    prows = _csv_rows(placement) if placement.endswith(".csv") else rows_from_stdout(placement, "PLACEMENT")
    tab = parse_table(table)
    slot_maps, ir_ops = load_irs(irs)
    periods = load_periods(config)
    place = {int(r["entry_id"]): r for r in prows}
    audit: Dict[str, object] = dict(trace_rows=len(trows), placement_rows=len(prows), table_rows=len(tab),
                                    name_mismatch=0, op_mismatch=0, unmapped=0, zero_cost=0,
                                    table_mismatch=0)
    entries: Dict[int, Entry] = {}
    for r in trows:
        eid = int(r["entry_id"])
        t = tab[eid]
        p = place.get(eid, {})
        net, inst, slot = r["network"], int(r["instance"]), int(r["dispatch_id"])
        if t["slot"] != slot or t["network"] != net or t["instance"] != inst:
            audit["table_mismatch"] += 1
        irid = slot_maps.get(net, {}).get(slot) if slot >= 0 else None
        if slot >= 0:
            if irid is None:
                audit["unmapped"] += 1
            else:
                nm, op = ir_ops[net].get(irid, ("", ""))
                audit["name_mismatch"] += nm != r["name"]
                audit["op_mismatch"] += op != r["op"]
        else:
            audit["zero_cost"] += 1
        s, e = int(r["actual_start_cycles"]), int(r["actual_end_cycles"])
        real = slot >= 0 and e > 0
        harts = tuple(int(x) for x in p["harts"].split()) if p.get("harts") else (t["harts"] or (int(r["hart"]),))
        entries[eid] = Entry(
            eid=eid, net=net, inst=inst, slot=slot, irid=irid, op=r["op"], name=r["name"],
            kind=r["core_kind"], impl=p.get("impl", ""), hart=int(r["hart"]), harts=harts,
            ps=float(r["predicted_start_ms"]), pdur=float(r["predicted_duration_ms"]),
            rs=s / NS_PER_MS if real else None, ae=e / NS_PER_MS if real else None,
            deps=tuple(t["deps"]), tdep=t["tdep"], gate=float(r["predicted_start_ms"]) if slot == 0 else None)
    if schedule:
        audit.update(_check_schedule(entries, schedule))
    ir_names = {n: {i: v[0] for i, v in d.items()} for n, d in ir_ops.items()}
    run = Run(run_id, entries, {}, periods, audit, ir_names)
    _derive(run)
    _frames(run)
    return run


def _check_schedule(entries: Dict[int, Entry], schedule: str) -> dict:
    """Planned times in the trace == the schedule's, joined on the IR id."""
    d = json.load(open(schedule))["dispatches"]
    worst, missing, n = 0.0, 0, 0
    for e in entries.values():
        if e.irid is None:
            continue
        v = d.get(f"{e.net}{e.inst}_dispatch_{e.irid}")
        if v is None:
            missing += 1
            continue
        n += 1
        worst = max(worst, abs(float(v["start_time"]) - e.ps), abs(float(v["duration"]) - e.pdur))
    return dict(schedule_joined=n, schedule_missing=missing, schedule_max_diff_ms=worst)


# ---------------------------------------------------------------- derived timing
def _derive(run: Run) -> None:
    E = run.entries
    # kernel start after the hart locks: previous holder of each hart, in end order
    by_hart: Dict[int, List[Entry]] = {}
    for e in E.values():
        if e.real:
            for h in e.harts:
                by_hart.setdefault(h, []).append(e)
    lock_pred: Dict[int, Tuple[float, int]] = {}
    for h, L in by_hart.items():
        L.sort(key=lambda x: (x.ae, x.eid))
        for a, b in zip(L, L[1:]):
            if b.eid not in lock_pred or a.ae > lock_pred[b.eid][0]:
                lock_pred[b.eid] = (a.ae, a.eid)
    for e in E.values():
        if e.real:
            pe_, holder = lock_pred.get(e.eid, (-1.0, -1))
            e.xs = max(e.rs, pe_)
            e.lock_holder = holder if pe_ > e.rs else -1
    # ready time and binding constraint, in table order (all edges point backward)
    last_on_worker: Dict[Tuple[str, int], int] = {}
    for eid in sorted(E):
        e = E[eid]
        cands: List[Tuple[float, str, int]] = []
        if e.gate is not None:
            cands.append((e.gate, "gate", -1))
        for d in e.deps:
            cands.append((E[d].done, "dep", d))
        if e.tdep >= 0:
            cands.append((E[e.tdep].done, "tdep", e.tdep))
        w = last_on_worker.get((e.kind, e.hart))
        if w is not None:
            cands.append((E[w].done, "worker", w))
        last_on_worker[(e.kind, e.hart)] = eid
        cands = [c for c in cands if c[0] is not None]
        if cands:
            t, kind, b = max(cands, key=lambda c: (c[0], c[1] == "dep"))
        else:
            t, kind, b = 0.0, "start", -1
        e.ready, e.binding, e.binding_eid = t, kind, b
        e.done = e.ae if e.real else t


def _frames(run: Run) -> None:
    for e in run.entries.values():
        key = (e.net, e.inst)
        if key not in run.frames:
            period, window = run.periods[e.net]
            run.frames[key] = Frame(e.net, e.inst, e.inst * period, e.inst * period + window)
        run.frames[key].entries.append(e.eid)
    for f in run.frames.values():
        real = [run.entries[i] for i in f.entries if run.entries[i].real]
        f.planned_start = min(x.ps for x in real)
        f.planned_end = max(x.pe for x in real)
        f.actual_start = min(x.rs for x in real)
        f.actual_end = max(x.ae for x in real)


# ---------------------------------------------------------------- output check
def output_block(text, net):
    m = re.search(rf"=== MODELBLASTER_OUTPUT_BEGIN \[{re.escape(net)}\] ===\n(.*?)"
                  rf"=== MODELBLASTER_OUTPUT_END \[{re.escape(net)}\] ===", text, re.S)
    return m.group(1).strip().splitlines() if m else []


def check_output(lines, golden):
    g = golden.reshape(-1).astype(np.float64)
    if not lines:
        return False, "no output block"
    if not lines[0].startswith("OUTPUT_SAMPLED"):
        got = np.array([float(x) for x in lines])
        if got.size != g.size:
            return False, f"size {got.size} vs golden {g.size}"
        err = float(np.max(np.abs(got - g)))
        return err == 0, f"all {g.size} values, max_abs_err={err:g}"
    size = int(re.search(r"size=(\d+)", lines[0]).group(1))
    if size != g.size:
        return False, f"size {size} vs golden {g.size}"
    errs = []
    for ln in lines[1:]:
        m = re.match(r"\s*\[(\d+)\]\s+(\S+)", ln)
        if m:
            errs.append(abs(float(m.group(2)) - g[int(m.group(1))]))
    summ = dict(re.findall(r"(\w+)=(\S+)", next(l for l in lines if l.startswith("OUTPUT_SUMMARY"))))
    want = {"sum": g.sum(), "abs_sum": np.abs(g).sum(), "min": g.min(), "max": g.max()}
    s_ok = all(float(summ[k]) == float(f"{v:.6g}") for k, v in want.items())
    ok = len(errs) == 16 and max(errs) == 0 and s_ok
    return ok, f"16 samples max_abs_err={max(errs) if errs else 'n/a'}, summary {'match' if s_ok else 'DIFFER'}"
