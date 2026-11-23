"""Discrete-event model of the ModelBlaster schedule walker.

Used to evaluate scheduler feedback and walker policies offline, from real board
traces, before spending board time. It reproduces the walker's semantics:

* Table order is ingest's: a start-time-priority topological sort over data deps
  plus each ``time_dependency`` that does not close a cycle
  (ModelBlaster ``pipeline/ingest_xpurt_schedule.py``).
* One worker per hart. A worker runs its own entries strictly in table order;
  an entry waits for its data deps, its ``time_dependency`` and, for the first
  kernel dispatch of an instance, the release gate at its planned start.
* Trace ``dispatch_id`` is the *codegen* index (zero-cost IR ops removed), so
  rows are matched to schedule keys through the table position (entry_id).

Policies:

``static``     the walker as built today.
``dynamic``    ModelBlaster ``--dynamic-walker`` (Sam Bai, 1ff915f): an idle
               worker may take a *held* entry of its kind (inputs ready, owner
               busy or waiting) if ``now + safety * planned`` ends before its
               own next planned start.
``pool``       proposed: entries of non-periodic networks are not owned by any
               hart. They sit in one global ready queue; any idle worker of the
               right kind takes the oldest ready one whose *estimated* duration
               fits before its next periodic entry. Periodic entries keep their
               AOT placement (static or dynamic).

Durations come from a ``DurationModel``: empirical samples per
``(network, IR dispatch, kind)`` from board traces, so contention seen on the
board is in the samples.
"""

from __future__ import annotations

import csv
import heapq
import json
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

#: K3: CPU_P = X100 harts 0-7, CPU_E = A100 harts 8-15.
K3_HART_BASE = {"cpu_p": 0, "cpu_e": 8}


@dataclass
class Entry:
    idx: int
    key: str
    net: str
    inst: int
    ir_did: int
    kind: str            # "cpu_p" / "cpu_e"
    hart: int
    ps: float            # planned start (ms)
    pd: float            # planned duration (ms)
    zero_cost: bool
    gate: bool           # first kernel dispatch of its instance
    deps: list[int] = field(default_factory=list)
    tdep: int = -1


def _kind_hart(target: str, hart_base: dict[str, int]) -> tuple[str, int]:
    first = str(target).split("+")[0]
    kind, _, n = first.partition("#")
    kind = kind.lower()
    return kind, hart_base.get(kind, 0) + int(n or 0)


def _split_key(key: str) -> tuple[str, int, int]:
    job, _, d = key.rpartition("_dispatch_")
    i = len(job)
    while i and job[i - 1].isdigit():
        i -= 1
    return job[:i], int(job[i:]), int(d)


def load_table(schedule_path: str | Path,
               hart_base: dict[str, int] = K3_HART_BASE) -> list[Entry]:
    """Schedule JSON -> walker table, in ingest's order."""
    raw = json.loads(Path(schedule_path).read_text())["dispatches"]
    adj: dict[str, list[str]] = defaultdict(list)
    indeg = {k: 0 for k in raw}
    for k, d in raw.items():
        for dep in d.get("dependencies") or []:
            if dep in raw:
                adj[dep].append(k)
                indeg[k] += 1

    def reaches(src, dst):
        seen, stack = {src}, [src]
        while stack:
            for nxt in adj[stack.pop()]:
                if nxt == dst:
                    return True
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return False

    dropped = set()
    for k in sorted(raw):
        dep = raw[k].get("time_dependency")
        if not dep or dep not in raw:
            continue
        if dep == k or reaches(k, dep):
            dropped.add(k)
            continue
        adj[dep].append(k)
        indeg[k] += 1
    heap = [(float(raw[k].get("start_time") or 0.0), k) for k, n in indeg.items() if n == 0]
    heapq.heapify(heap)
    order = []
    while heap:
        _, k = heapq.heappop(heap)
        order.append(k)
        for nxt in adj[k]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                heapq.heappush(heap, (float(raw[nxt].get("start_time") or 0.0), nxt))
    if len(order) != len(raw):
        raise ValueError("schedule dep graph has a cycle")
    pos = {k: i for i, k in enumerate(order)}

    # Gate = first non-zero-cost dispatch of each instance (codegen index 0).
    first_kernel: dict[str, int] = {}
    for k, d in raw.items():
        if not d.get("module_name"):
            continue
        job = d.get("job_name") or k.rpartition("_dispatch_")[0]
        _, _, did = _split_key(k)
        if job not in first_kernel or did < first_kernel[job]:
            first_kernel[job] = did
    table = []
    for i, k in enumerate(order):
        d = raw[k]
        net, inst, did = _split_key(k)
        kind, hart = _kind_hart(d["hardware_target"], hart_base)
        job = d.get("job_name") or k.rpartition("_dispatch_")[0]
        td = d.get("time_dependency")
        table.append(Entry(
            idx=i, key=k, net=net, inst=inst, ir_did=did, kind=kind, hart=hart,
            ps=float(d.get("start_time") or 0.0), pd=float(d.get("duration") or 0.0),
            zero_cost=not d.get("module_name"),
            gate=bool(d.get("module_name")) and first_kernel.get(job) == did,
            deps=[pos[x] for x in (d.get("dependencies") or []) if x in pos],
            tdep=pos[td] if td in pos and k not in dropped else -1))
    return table


# --------------------------------------------------------------------- traces
def read_trace(trace_path: str | Path, table: list[Entry]) -> list[dict]:
    """Trace rows aligned to the table (row i <-> table[i]); times in ms."""
    rows = list(csv.DictReader(open(trace_path)))
    if len(rows) != len(table):
        raise ValueError(f"{trace_path}: {len(rows)} rows, table has {len(table)}")
    out = [None] * len(table)
    for r in rows:
        i = int(r["entry_id"])
        e = table[i]
        if r["network"] != e.net or int(r["instance"]) != e.inst:
            raise ValueError(f"{trace_path}: entry {i} is {r['network']}{r['instance']}, "
                             f"table says {e.key}")
        s, f = int(r["actual_start_cycles"]) / 1e6, int(r["actual_end_cycles"]) / 1e6
        out[i] = dict(s=s, f=f, dur=max(0.0, f - s), ran_on=int(r.get("ran_on_hart") or r["hart"]),
                      moved=r.get("moved") == "1", zero=e.zero_cost or (s == 0 and f == 0))
    return out


def op_samples(pairs: Iterable[tuple[list[Entry], list[dict]]]) -> dict[tuple, list[float]]:
    """Measured durations (ms) per (network, IR dispatch, kind)."""
    S: dict[tuple, list[float]] = defaultdict(list)
    for table, tr in pairs:
        for e, t in zip(table, tr):
            if not t["zero"]:
                S[(e.net, e.ir_did, e.kind)].append(t["dur"])
    return S


def gap_samples(pairs: Iterable[tuple[list[Entry], list[dict]]]) -> dict[tuple, list[float]]:
    """Board ready->start gaps (ms) per (network, kind), static-walker traces only.

    Ready = latest of: data deps, time_dependency, the previous entry on the same
    hart, and the release gate. Zero-cost entries finish when their deps do."""
    G: dict[tuple, list[float]] = defaultdict(list)
    for table, tr in pairs:
        fin = [0.0] * len(table)
        last: dict[int, int] = {}
        for i, e in enumerate(table):
            if tr[i]["zero"]:
                fin[i] = max([fin[d] for d in e.deps] + [0.0])
                continue
            fin[i] = tr[i]["f"]
            c = [fin[d] for d in e.deps]
            if e.tdep >= 0:
                c.append(fin[e.tdep])
            if e.hart in last:
                c.append(fin[last[e.hart]])
            if e.gate:
                c.append(e.ps)
            last[e.hart] = i
            G[(e.net, e.kind)].append(max(0.0, tr[i]["s"] - max(c + [0.0])))
    return G


class DurationModel:
    """Board 'truth' for the simulator: draws a measured duration per entry."""

    def __init__(self, samples: dict[tuple, list[float]], seed: int = 0,
                 gaps: Optional[dict[tuple, list[float]]] = None, scale: Optional[dict] = None):
        self.S = samples
        self.G = gaps or {}
        self.scale = scale or {}
        self.rng = random.Random(seed)

    def gap(self, e: Entry) -> float:
        xs = self.G.get((e.net, e.kind)) or self.G.get(next(iter(self.G), None), [])
        return self.rng.choice(xs) if xs else 0.002

    def __call__(self, e: Entry) -> float:
        if e.zero_cost:
            return 0.0
        xs = self.S.get((e.net, e.ir_did, e.kind))
        if xs:
            return self.rng.choice(xs) * self.scale.get(e.net, 1.0)
        return e.pd * 1.27 * self.scale.get(e.net, 1.0)       # unseen placement: the mean board/profile ratio


# ----------------------------------------------------------------- simulation
@dataclass
class SimResult:
    start: list[float]
    end: list[float]
    ran_on: list[int]
    moved: list[bool]
    n_moved: int = 0
    n_refused: int = 0


def simulate(table: list[Entry], dur: Callable[[Entry], float], *,
             policy: str = "static", safety: float = 1.5,
             aperiodic: Optional[set[str]] = None,
             estimate: Optional[Callable[[Entry], float]] = None,
             periodic_policy: str = "static",
             all_harts: Optional[dict[str, list[int]]] = None,
             pool_period: Optional[dict[str, float]] = None,
             overhead_ms: float = 0.002, scan_window: int = 600) -> SimResult:
    """Run the table under a walker policy. See module docstring."""
    n = len(table)
    D = [dur(e) for e in table]          # one draw per entry, fixed for the run
    gapf = getattr(dur, "gap", None)
    O = [0.0 if e.zero_cost else (gapf(e) if gapf else overhead_ms) for e in table]
    start = [None] * n
    end: list[Optional[float]] = [None] * n
    ran_on = [-1] * n
    moved = [False] * n
    claimed = [False] * n
    aperiodic = aperiodic or set()
    pooled = [policy == "pool" and e.net in aperiodic for e in table]
    est = estimate or (lambda e: e.pd)

    own: dict[int, list[int]] = defaultdict(list)
    kind_of: dict[int, str] = {}
    for e in table:
        if not pooled[e.idx]:
            own[e.hart].append(e.idx)
        kind_of[e.hart] = e.kind
    if policy in ("dynamic", "pool") and all_harts:
        for k, hs in all_harts.items():
            for h in hs:
                own.setdefault(h, [])
                kind_of[h] = k
    workers = sorted(own)
    ptr = {h: 0 for h in workers}
    free = {h: 0.0 for h in workers}
    pos_in_owner = {i: p for h in workers for p, i in enumerate(own[h])}
    steal_periodic = policy == "dynamic" or (policy == "pool" and periodic_policy == "dynamic")
    res = SimResult(start, end, ran_on, moved)
    lo = 0
    INF = float("inf")

    def ready_at(i):
        e = table[i]
        t = 0.0
        for d in e.deps:
            if end[d] is None:
                return None
            t = max(t, end[d])
        # Ordering edges only bind AOT-placed work: pooled entries have no slot,
        # so an edge into or out of the pool is dropped (data deps still hold).
        if e.tdep >= 0 and not pooled[i] and not pooled[e.tdep]:
            if end[e.tdep] is None:
                return None
            t = max(t, end[e.tdep])
        if e.gate:
            # Pooled work is released at its instance release (inst * period, or
            # 0 for aperiodic), not at the planned start: it may run early.
            t = max(t, (pool_period or {}).get(e.net, 0.0) * e.inst if pooled[i] else e.ps)
        return t

    waiting: set[int] = set()

    def run(h, i, t, mv):
        claimed[i] = True
        s = t + O[i]
        start[i], end[i], ran_on[i], moved[i] = s, s + D[i], h, mv
        free[h] = end[i]
        waiting.discard(h)
        # A new end time is the only thing that can make a waiting worker's
        # entry ready (besides gates, already in its wake time).
        for w in list(waiting):
            if end[i] < free[w]:
                free[w] = max(end[i], t)
                waiting.discard(w)

    def next_own(h):
        L = own[h]
        while ptr[h] < len(L) and claimed[L[ptr[h]]]:
            ptr[h] += 1
        return L[ptr[h]] if ptr[h] < len(L) else None

    def held(i, t):
        o = table[i].hart
        if free.get(o, 0.0) > t:
            return True
        nx = next_own(o)
        return nx is not None and nx != i and pos_in_owner[nx] < pos_in_owner[i] or \
            (nx == i and (ready_at(i) is None or ready_at(i) > t))

    remaining = n
    guard = 0
    while remaining:
        guard += 1
        if guard > 50 * n + 10000:
            raise RuntimeError("simulation did not converge")
        h = min(workers, key=lambda w: (free[w], w))
        t = free[h]
        while lo < n and claimed[lo]:
            lo += 1
        nx = next_own(h)
        if nx is not None:
            r = ready_at(nx)
            if r is not None and r <= t + 1e-12:
                run(h, nx, t, False); remaining -= 1; continue
        limit = table[nx].ps if nx is not None else INF
        took = False
        if policy in ("dynamic", "pool") and t < limit:
            for c in range(lo, min(n, lo + scan_window)):
                if claimed[c]:
                    continue
                e = table[c]
                if pooled[c] and e.zero_cost:
                    # A pooled view op only posts its completion: any worker does it.
                    if ready_at(c) is None or ready_at(c) > t:
                        continue
                elif e.kind != kind_of[h] or e.zero_cost:
                    continue
                elif pooled[c]:
                    r = ready_at(c)
                    if r is None or r > t:
                        continue
                    if t + est(e) > limit:
                        res.n_refused += 1
                        continue
                elif steal_periodic:
                    if e.hart == h:
                        continue
                    r = ready_at(c) if e.tdep < 0 else None
                    if r is None:   # data deps only (the walker ignores the ordering edge when moving)
                        r = max([end[d] for d in e.deps] + [0.0]) if all(end[d] is not None for d in e.deps) else None
                        if r is not None and e.gate:
                            r = max(r, e.ps)
                    if r is None or r > t or not held(c, t):
                        continue
                    if t + safety * e.pd > limit:
                        res.n_refused += 1
                        continue
                else:
                    continue
                run(h, c, t, c not in pos_in_owner or table[c].hart != h)
                res.n_moved += 1
                remaining -= 1
                took = True
                break
        if took:
            continue
        # Nothing to do now: sleep until the next thing that can change.
        cand = [x for x in (end[i] for i in range(lo, min(n, lo + scan_window)) if end[i] is not None) if x > t]
        cand += [w for w in free.values() if w > t]
        if nx is not None:
            r = ready_at(nx)
            if r is not None and r > t:
                cand.append(r)
        cand += [table[i].ps for i in range(lo, min(n, lo + scan_window))
                 if not claimed[i] and table[i].gate and table[i].ps > t]
        free[h] = min(cand) if cand else INF
        if not (nx is None and policy == "static"):
            waiting.add(h)
        if all(free[w] == INF for w in workers):
            stuck = [i for i in range(n) if not claimed[i]][:4]
            info = [(table[i].key, "pooled" if pooled[i] else f"hart{table[i].hart}",
                     ready_at(i), [table[d].key for d in table[i].deps if end[d] is None][:2],
                     table[table[i].tdep].key if table[i].tdep >= 0 and end[table[i].tdep] is None else None)
                    for i in stuck]
            raise RuntimeError(f"deadlock at t={t:.3f}: {remaining} left; first stuck {info}")
    return res


# -------------------------------------------------------------------- metrics
def instance_misses(table: list[Entry], end: list[float],
                    windows: dict[str, float], periods: dict[str, float]) -> dict[str, dict]:
    """Per network: instances, misses, worst lateness (ms). Release = inst*period."""
    last: dict[tuple, float] = {}
    for e, f in zip(table, end):
        if f is None or e.net not in windows:
            continue
        k = (e.net, e.inst)
        last[k] = max(last.get(k, 0.0), f)
    out = {}
    for net in windows:
        ks = [k for k in last if k[0] == net]
        late = [last[k] - (k[1] * periods[net] + windows[net]) for k in ks]
        out[net] = dict(n=len(ks), misses=sum(1 for x in late if x > 1e-9),
                        worst=max(late) if late else 0.0)
    return out


def spec_windows(spec_path: str | Path) -> tuple[dict, dict]:
    nets = json.loads(Path(spec_path).read_text())["networks"]
    w, p = {}, {}
    for name, info in nets.items():
        if info.get("period"):
            p[name] = float(info["period"])
            w[name] = float(info.get("window_duration") or info["period"])
    return w, p


def write_trace(table: list[Entry], res: SimResult, path: str | Path) -> None:
    """Walker-format trace (ns ticks), so board tooling reads simulated runs."""
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["entry_id", "network", "instance", "dispatch_id", "op", "name", "core_kind",
                     "hart", "predicted_start_ms", "predicted_duration_ms", "worker_kind_idx",
                     "actual_start_cycles", "actual_end_cycles", "ran_on_hart", "moved"])
        for e in table:
            s, f = res.start[e.idx], res.end[e.idx]
            z = e.zero_cost
            wr.writerow([e.idx, e.net, e.inst, -1 if z else e.ir_did, "", "", e.kind, e.hart,
                         f"{e.ps:.6f}", f"{e.pd:.6f}", 0,
                         0 if z else int(round(s * 1e6)), 0 if z else int(round(f * 1e6)),
                         res.ran_on[e.idx], int(res.moved[e.idx])])
