#!/usr/bin/env python3
"""Fit timing corrections from board runs that carry the stamps.

  key        (network, layer IR id, core kind, kernel type rvv/ime, number of cores)
  run time   p75 of run / RAW profile value per round, clamped 0.5 .. 4 (raw >= 20 us), or p75 of
             (run - raw), not below 0 (raw < 20 us); several rounds: the median of the per-round
             values; fewer than 15 samples: shrunk toward the median of the same op/kind/kernel/cores
  gap        per entry: p75 of that entry's own ready -> start samples (an entry that was always
             held for its start gate: 0). --gap-from latest: only if it ran in the LATEST round.
             --gap-from seen: from the most recent round it ran in, so an entry the planner left keeps
             the gap it had (the timing loop's choice: nothing is forgotten)
  entries with no gap of their own: the MEAN gap of their (network, core kind) in the latest round
  frame 0 of every model is left out; always anchored to the raw tree.

usage: fit.py --raw-tree <raw profile tree> --round r0=<raw run folder>:<runs> [--round r1=...] --out corrections.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import loop_lib as T  # noqa: E402

SCHEMA = "k3work.t023.corrections/v1"
TINY_MS, QUANTILE, CLAMP, N_THIN, N_SHRINK = 0.020, 0.75, (0.5, 4.0), 15, 5


def key_str(key: T.Key) -> str:
    return "|".join(str(x) for x in key)


def fit(raw_tree: str, rounds: List[Tuple[str, str, int]], gap_from: str = "latest") -> dict:
    prof = T.read_profile(raw_tree)
    stats: Dict[T.Key, List[Tuple[str, float, int]]] = defaultdict(list)
    meta: Dict[T.Key, dict] = {}
    round_info = []
    layer_gap: Dict[T.Key, dict] = {}
    kind_mean: Dict[str, dict] = {}
    for idx, (name, raw_dir, reps) in enumerate(rounds):
        pieces = [p for k in range(1, reps + 1) for p in T.load(raw_dir, k, checks=False).pieces if p.frame > 0]
        round_info.append(dict(name=name, raw=raw_dir, runs=reps, layer_samples=len(pieces)))
        groups = T.group_by_key(pieces)
        for key, ps in groups.items():
            if key not in prof:
                raise KeyError(f"{name}: measured key {key} has no entry in the raw tree {raw_tree}")
            raw = prof[key]["ms"]
            meta[key] = dict(op=ps[0].op, name=ps[0].name, raw_ms=raw)
            v = T.quantile([p.run / raw for p in ps], QUANTILE) if raw >= TINY_MS else T.quantile([p.run - raw for p in ps], QUANTILE)
            stats[key].append((name, v, len(ps)))
        latest = idx == len(rounds) - 1
        if latest or gap_from == "seen":            # gaps: the latest round ("latest"), or each entry's most recent round ("seen")
            by: Dict[str, List[float]] = defaultdict(list)
            for key, ps in groups.items():
                xs = [p.gap for p in ps if not p.gated]
                layer_gap[key] = dict(value=T.quantile(xs, QUANTILE) if xs else 0.0, n=len(xs), round=name,
                                      p50=T.quantile(xs, .5) if xs else 0.0, mean=statistics.fmean(xs) if xs else 0.0)
                by[f"{key[0]}|{key[2]}"] += xs
                by[f"*|{key[2]}"] += xs
            if latest:                              # entries never run: the mean of their model and core kind in the latest round
                kind_mean = {k: dict(value=statistics.fmean(v), n=len(v)) for k, v in sorted(by.items()) if v}

    prior_pool: Dict[tuple, List[float]] = defaultdict(list)
    for key, rs in stats.items():
        if meta[key]["raw_ms"] >= TINY_MS:
            prior_pool[(meta[key]["op"],) + key[2:]].append(statistics.median(v for _, v, _ in rs))
    prior = {k: statistics.median(v) for k, v in prior_pool.items()}
    ops = {}
    for key in sorted(stats):
        rs, raw = stats[key], meta[key]["raw_ms"]
        mode = "ratio" if raw >= TINY_MS else "add"
        v = statistics.median(x for _, x, _ in rs)
        tot = sum(n for _, _, n in rs)
        unclamped, shrunk = v, False
        if mode == "ratio":
            if tot < N_THIN:
                v = (tot * v + N_SHRINK * prior.get((meta[key]["op"],) + key[2:], v)) / (tot + N_SHRINK)
                shrunk = True
            v = min(max(v, CLAMP[0]), CLAMP[1])
        else:
            v = max(v, 0.0)
        ops[key_str(key)] = dict(network=key[0], layer=key[1], core_kind=key[2], kernel_type=key[3], cores=key[4],
                                 op=meta[key]["op"], name=meta[key]["name"], raw_ms=raw, mode=mode, value=v, unclamped=unclamped,
                                 clamped=abs(v - unclamped) > 1e-12 and not shrunk, shrunk=shrunk,
                                 round_stats={r: x for r, x, _ in rs}, n={r: n for r, _, n in rs},
                                 corrected_ms_before_gap=(raw * v if mode == "ratio" else raw + v))
    return dict(schema=SCHEMA, anchor_tree=raw_tree, rounds=round_info, gap_rule="layer", gap_round=rounds[-1][0], gap_from=gap_from,
                params=dict(tiny_ms=TINY_MS, quantile=QUANTILE, clamp=list(CLAMP), n_thin=N_THIN, n_shrink=N_SHRINK,
                            measured="run = end - got_cores", frame0="left out"),
                gap_kind_mean_ms={k: v["value"] for k, v in kind_mean.items()}, gap_kind_mean_detail=kind_mean,
                gap_by_key_ms={key_str(k): v["value"] for k, v in sorted(layer_gap.items())},
                gap_by_key_detail={key_str(k): v for k, v in sorted(layer_gap.items())}, ops=ops)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-tree", required=True)
    ap.add_argument("--round", action="append", required=True, help="name=<raw folder>:<runs>; the LAST one gives the gaps")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gap-from", choices=("latest", "seen"), default="latest",
                    help="latest: an entry's gap only if it ran in the last round. seen: its gap from the most "
                         "recent round it ran in, so no measurement is forgotten (the timing loop's choice)")
    a = ap.parse_args()
    rounds = []
    for r in a.round:
        name, rest = r.split("=", 1)
        folder, reps = rest.rsplit(":", 1)
        rounds.append((name, folder, int(reps)))
    doc = fit(a.raw_tree, rounds, a.gap_from)
    with open(a.out, "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
        f.write("\n")
    ops = list(doc["ops"].values())
    print(f"wrote {a.out}: {len(ops)} measured keys, clamped {sum(o['clamped'] for o in ops)}, shrunk {sum(o['shrunk'] for o in ops)}")
    for net in T.NETS:
        r = [o["value"] for o in ops if o["network"] == net and o["mode"] == "ratio"]
        ad = [o["value"] * 1e3 for o in ops if o["network"] == net and o["mode"] == "add"]
        g = [v for k, v in doc["gap_by_key_ms"].items() if k.startswith(net + "|")]
        print(f"  {net:12s} ratios {len(r)}" + (f" ({min(r):.2f} .. {max(r):.2f}, median {statistics.median(r):.2f})" if r else "")
              + f"; additive {len(ad)}" + (f" ({min(ad):.1f} .. {max(ad):.1f} us, median {statistics.median(ad):.1f})" if ad else "")
              + f"; gaps from {doc['gap_round']}: {len(g)} entries, {sum(x >= 0.008 for x in g)} of 8 us or more, max {max(g) * 1e3:.1f} us")
    print("  mean gap per (network, core kind), for entries that did not run (us): "
          + ", ".join(f"{k} {v * 1e3:.2f}" for k, v in doc["gap_kind_mean_ms"].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
