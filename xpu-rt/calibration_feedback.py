"""Calibrated runtime feedback: board trace -> scheduler costs + structural constraints.

Why not just measured / predicted
---------------------------------
Scaling each op by its mean measured/predicted ratio (``duration_feedback.py``)
removes about a third of the misses on the K3 and then stops improving. Board
traces show the remaining misses are structural, not per-op cost errors:

* the high-rate model (``mlp_control``, 0.625 ms period) is accurate on its own
  ops, but its first dispatch is queued on a hart behind a long op of another
  model (YOLO, DroNet); the static walker runs each hart's entries in order, so
  that op's overrun becomes the high-rate model's lateness;
* every dispatch pays a ready->start gap (median ~4 us, YOLO ~12 us cross-core
  handoff), which the cost model does not price and which sums along chains;
* one sample per instance is noisy (66x outliers on 0.1 us ops), and a mean
  under-prices the tail that actually causes misses.

What this module does with one or more board (or simulated) traces
-----------------------------------------------------------------------
1. **Robust cost calibration.** Per ``(network, IR dispatch, kind)``, samples are
   anchored to the raw profile (as in ``duration_feedback``), summarised per round
   by a quantile (p75; p90 for networks that missed), and combined across the last
   rounds by a median. Keys with few samples shrink toward their op type's factor.
   Tiny ops (raw < ``TINY_MS``) get an additive correction instead of a ratio.
2. **Dispatch gap.** The measured ready->start gap per (network, kind) is added to
   every dispatch, so the plan prices synchronisation and handoff.
3. **Blame.** Each missed instance is traced back along what it waited on (data
   dep, previous entry on the hart, release gate). Hops where a network waited for
   a *different* network's entry on the same hart are tallied as blocking.
4. **Structural feedback.**
   * Hart reservation (scheduler): a network that was blocked gets enough harts of
     the kind it was blocked on, reserved for it alone. Other networks are priced
     off those harts.
   * Blocking-bound split hints (graph partitioner): an op that blocked a network
     and is longer than that network's slack is asked to split so each piece fits
     the slack. Emitted only when the blocked network cannot get its own harts.

Document (JSON, ``schema: xpurt.calibration/v1``)::

    {"schema": "xpurt.calibration/v1", "rounds": ["r0", ...],
     "ops":  {"dronet|dispatch_12|cpu_e": {"op": "conv2d_s8", "raw_ms": 0.12,
              "round_stats": [1.31, 1.28], "n": [30, 30], "mode": "ratio",
              "value": 1.30}},
     "gap_ms": {"yolov8_nano|cpu_p": 0.012, ...},
     "reserve": {"mlp_control": ["CPU_P#0"]},
     "blame": {"mlp_control": {"blocked_by": {"yolov8_nano": 12.3}, "harts": {...}}},
     "split_hints": [...]}
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional

SCHEMA = "xpurt.calibration/v1"
TINY_MS = 0.02            # below this, correct additively (fixed overhead dominates)
Q_DEFAULT = 0.75          # per-round quantile of measured/raw
Q_MISSED = 0.90           # ... for networks that missed deadlines this round
ROUNDS_KEPT = 3           # rounds combined (median) per key
N_SHRINK = 5              # pseudo-samples of the op-type prior
CLAMP = (0.5, 4.0)
UTIL_TARGET = 0.6         # reserve enough harts to keep a network's util below this
PENALTY_MS = 100.0        # cost added on a hart reserved for another network


def empty() -> dict:
    return {"schema": SCHEMA, "rounds": [], "ops": {}, "gap_ms": {}, "reserve": {},
            "blame": {}, "split_hints": []}


def load(path: Optional[str | Path]) -> dict:
    if not path:
        return empty()
    doc = json.loads(Path(path).read_text())
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema {doc.get('schema')!r}, expected {SCHEMA}")
    return doc


def save(doc: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")


def _q(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    pos = q * (len(xs) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def key(net: str, did: int | str, kind: str) -> str:
    d = did if str(did).startswith("dispatch_") else f"dispatch_{did}"
    return f"{net}|{d}|{kind}"


# ------------------------------------------------------------------- applying
def correction(doc: dict, net: str, disp: str, kind: str, raw_ms: float) -> float:
    """Calibrated cost (ms) for one profiled value."""
    rec = doc.get("ops", {}).get(f"{net}|{disp}|{kind}")
    v = raw_ms
    if rec:
        v = raw_ms + rec["value"] if rec["mode"] == "add" else raw_ms * rec["value"]
    return v + float(doc.get("gap_ms", {}).get(f"{net}|{kind}", 0.0))


def apply(processing_times: dict[str, list], networks: Iterable[str],
          machine_combinations: list[list[str]], combo_kinds: list[str], doc: dict) -> dict:
    """Rewrite ``processing_times`` in place: calibrated costs plus reservations."""
    nets = sorted(networks, key=len, reverse=True)
    reserved_by = {m: n for n, ms in doc.get("reserve", {}).items() for m in ms}
    st = Counter()
    for name, times in processing_times.items():
        net = next((n for n in nets if name.startswith(n + "_")), None)
        if net is None:
            continue
        disp = name[len(net) + 1:]
        for i, t in enumerate(times):
            if t is None or float(t) >= 1e5 or i >= len(combo_kinds):
                continue
            if float(t) > 0:
                times[i] = correction(doc, net, disp, combo_kinds[i], float(t))
                st["values_calibrated"] += 1
            if any(reserved_by.get(m, net) != net for m in machine_combinations[i]):
                times[i] = float(times[i]) + PENALTY_MS
                st["values_priced_off_reserved"] += 1
    return dict(st)


# --------------------------------------------------------------------- update
def _runs(table, trace):
    """Per-entry facts from one trace: measured, raw-anchored ratio, gap, binding."""
    fin = [0.0] * len(table)
    last: dict[int, int] = {}
    out = []
    for i, e in enumerate(table):
        t = trace[i]
        if t["zero"]:
            fin[i] = max([fin[d] for d in e.deps] + [0.0])
            out.append(None)
            continue
        fin[i] = t["f"]
        cands = [(fin[d], d, "data") for d in e.deps]
        if e.tdep >= 0 and not t["moved"]:
            cands.append((fin[e.tdep], e.tdep, "order"))
        if e.hart in last and not t["moved"]:
            cands.append((fin[last[e.hart]], last[e.hart], "order"))
        if e.gate:
            cands.append((e.ps, None, "gate"))
        rdy, par, kind = max(cands, key=lambda c: c[0]) if cands else (0.0, None, "gate")
        if not t["moved"]:
            last[e.hart] = i
        out.append(dict(meas=t["dur"], gap=max(0.0, t["s"] - rdy), par=par, bind=kind))
    return out


def update(doc: dict, table, trace: list[dict], round_name: str, *,
           windows: dict[str, float], periods: dict[str, float],
           op_names: Optional[dict[str, str]] = None,
           hart_names: Optional[dict[int, str]] = None,
           all_harts: Optional[dict[str, list[int]]] = None,
           splittable: Iterable[str] = ("conv2d_s8", "conv2d_batchnorm2d_silu_s8",
                                        "conv2d_batchnorm2d_s8", "linear_s8")) -> dict:
    """Fold one run into a NEW document. ``table``/``trace`` from ``walker_sim``.

    ``op_names`` maps ``"net|dispatch_N"`` to the op type, for shrinkage.
    ``hart_names`` maps hart number -> schedule machine name (``CPU_P#0``).
    """
    out = json.loads(json.dumps(doc))
    out.setdefault("ops", {}); out.setdefault("gap_ms", {}); out.setdefault("reserve", {})
    out["rounds"] = list(out.get("rounds", [])) + [round_name]
    facts = _runs(table, trace)

    # ---- misses this round (per instance) and which networks missed
    end_by: dict[tuple, tuple[float, int]] = {}
    for i, e in enumerate(table):
        if facts[i] is None or e.net not in windows:
            continue
        f = trace[i]["f"]
        k = (e.net, e.inst)
        if k not in end_by or f > end_by[k][0]:
            end_by[k] = (f, i)
    missed = {k: v for k, v in end_by.items()
              if v[0] > k[1] * periods[k[0]] + windows[k[0]] + 1e-9}
    missed_nets = {k[0] for k in missed}

    # ---- 1. robust per-op factors, anchored to the raw profile
    used = lambda k: out["ops"].get(k)
    per_key: dict[str, list[float]] = defaultdict(list)
    raw_of: dict[str, float] = {}
    for i, e in enumerate(table):
        if facts[i] is None:
            continue
        k = key(e.net, e.ir_did, e.kind)
        rec = used(k)
        # raw = planned / (what this round's plan already applied)
        g = float(doc.get("gap_ms", {}).get(f"{e.net}|{e.kind}", 0.0))
        planned = max(e.pd - g, 1e-6)
        if rec and rec["mode"] == "add":
            raw = max(planned - rec["value"], 1e-6)
        elif rec:
            raw = planned / rec["value"]
        else:
            raw = planned
        raw_of[k] = raw
        per_key[k].append(facts[i]["meas"])
    by_type: dict[tuple, list[float]] = defaultdict(list)
    stats: dict[str, tuple[str, float, int]] = {}
    for k, ms in per_key.items():
        net, disp, kind = k.split("|")
        raw = raw_of[k]
        q = Q_MISSED if net in missed_nets else Q_DEFAULT
        if raw < TINY_MS:
            stats[k] = ("add", _q([m - raw for m in ms], q), len(ms))
        else:
            r = _q([m / raw for m in ms], q)
            stats[k] = ("ratio", r, len(ms))
            by_type[((op_names or {}).get(f"{net}|{disp}", "?"), kind)].append(r)
    prior = {t: statistics.median(v) for t, v in by_type.items() if v}
    for k, (mode, val, n) in stats.items():
        net, disp, kind = k.split("|")
        rec = out["ops"].get(k) or {"op": (op_names or {}).get(f"{net}|{disp}", "?"),
                                    "mode": mode, "round_stats": [], "n": []}
        if rec["mode"] != mode:          # crossed TINY_MS: restart the history
            rec = {"op": rec.get("op", "?"), "mode": mode, "round_stats": [], "n": []}
        rec["raw_ms"] = raw_of[k]
        rec["round_stats"] = (rec["round_stats"] + [val])[-ROUNDS_KEPT:]
        rec["n"] = (rec["n"] + [n])[-ROUNDS_KEPT:]
        v = statistics.median(rec["round_stats"])
        if mode == "ratio":
            tot = sum(rec["n"])
            # Shrink thin keys toward their op type; never toward an unknown type,
            # which would mix unrelated ops.
            p = prior.get((rec["op"], kind), v) if rec["op"] != "?" else v
            v = (tot * v + N_SHRINK * p) / (tot + N_SHRINK) if tot < 3 * N_SHRINK else v
            v = min(max(v, CLAMP[0]), CLAMP[1])
        else:
            v = max(v, 0.0)
        rec["value"] = v
        out["ops"][k] = rec

    # ---- 2. dispatch gap per (network, kind): p75 of ready->start
    gaps: dict[str, list[float]] = defaultdict(list)
    for i, e in enumerate(table):
        if facts[i] is not None and facts[i]["bind"] != "gate":
            gaps[f"{e.net}|{e.kind}"].append(facts[i]["gap"])
    for k, xs in gaps.items():
        out["gap_ms"][k] = round(_q(xs, 0.75), 6)

    # ---- 3. blame: walk back from each missed instance's last entry
    blk: dict[str, Counter] = defaultdict(Counter)        # victim -> blocker net -> ms waited
    blk_hart: dict[str, Counter] = defaultdict(Counter)   # victim -> hart -> ms waited
    blockers: dict[str, Counter] = defaultdict(Counter)   # victim -> blocking entry idx -> count
    for (net, inst), (f, i) in missed.items():
        seen = set()
        n = i
        while n is not None and n not in seen and facts[n] is not None:
            seen.add(n)
            par, bind = facts[n]["par"], facts[n]["bind"]
            if par is None:
                break
            if bind == "order" and table[par].net != table[n].net and facts[par] is not None:
                wait = max(0.0, trace[par]["f"] - max(
                    [trace[d]["f"] for d in table[n].deps if facts[d] is not None] + [table[n].ps]))
                blk[net][table[par].net] += wait
                blk_hart[net][table[n].hart] += wait
                blockers[net][par] += 1
            n = par
    out["blame"] = {v: {"blocked_by_ms": dict(blk[v]), "harts_ms": {str(h): w for h, w in blk_hart[v].items()},
                        "missed": sum(1 for k in missed if k[0] == v)} for v in blk}

    # ---- 4a. hart reservation for blocked networks (shortest period first)
    spans = [periods[n] * (max(e.inst for e in table if e.net == n) + 1)
             for n in periods if any(e.net == n for e in table)]
    hyper = max(spans) if spans else max([trace[i]["f"] for i in range(len(table))
                                          if facts[i] is not None] + [1.0])
    taken = {m for ms in out["reserve"].values() for m in ms}
    for net in sorted(blk, key=lambda n: periods.get(n, 1e9)):
        for kind in sorted({table[i].kind for i in range(len(table))
                            if table[i].net == net and facts[i] is not None}):
            harts_of_kind = (all_harts or {}).get(kind, [])
            work = sum(facts[i]["meas"] for i, e in enumerate(table)
                       if e.net == net and e.kind == kind and facts[i] is not None)
            need = max(1, math.ceil(work / (hyper * UTIL_TARGET)))
            mine = [m for m in out["reserve"].get(net, []) if m.startswith(kind.upper())]
            if len(mine) >= need:
                continue
            # prefer harts this network already uses, most-used first
            use = Counter(e.hart for i, e in enumerate(table)
                          if e.net == net and e.kind == kind and facts[i] is not None)
            order = [h for h, _ in use.most_common()] + [h for h in harts_of_kind if h not in use]
            for h in order:
                name = (hart_names or {}).get(h)
                if not name or name in taken:
                    continue
                if len([m for m in taken if m.startswith(kind.upper())]) >= len(harts_of_kind) - 1:
                    break                            # keep at least one shared hart per kind
                out["reserve"].setdefault(net, []).append(name)
                taken.add(name)
                mine.append(name)
                if len(mine) >= need:
                    break

    # ---- 4b. blocking-bound split hints, only where a reservation did not land
    hints = []
    for net, ents in blockers.items():
        slack = windows[net] - statistics.median(
            [f - k[1] * periods[net] for k, (f, _) in end_by.items() if k[0] == net] or [0.0])
        slack = max(slack, 0.05)
        for i, _ in ents.most_common():
            e = table[i]
            name = (hart_names or {}).get(e.hart)
            if name in set(out["reserve"].get(net, [])):
                continue                              # the victim now owns that hart
            op = (op_names or {}).get(f"{e.net}|dispatch_{e.ir_did}", "?")
            if op not in splittable or trace[i]["dur"] <= slack:
                continue
            hints.append({"network": e.net, "op": e.ir_did, "op_type": op,
                          "n_splits": max(2, math.ceil(trace[i]["dur"] / slack)),
                          "reason": f"blocked {net}: {trace[i]['dur']:.3f} ms > slack {slack:.3f} ms"})
    out["split_hints"] = hints
    out["last_round"] = {"missed": {n: sum(1 for k in missed if k[0] == n) for n in windows}}
    return out


def split_hint_contract(doc: dict) -> dict:
    """``modelblaster.split_hints/v1`` for ``apply_split_hint.py``."""
    by_net: dict[str, dict] = {}
    for h in doc.get("split_hints", []):
        rec = by_net.setdefault(h["network"], {"network": h["network"], "split_ops": []})
        if all(s["op"] != h["op"] for s in rec["split_ops"]):
            rec["split_ops"].append({"op": h["op"], "n_splits": h["n_splits"]})
    return {"contract": "modelblaster.split_hints/v1",
            "reason": "blocking bound: ops that delayed a higher-rate network",
            "networks": list(by_net.values())}
