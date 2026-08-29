#!/usr/bin/env python3
"""Translate compile_advice.json into a ModelBlaster shard hint.

The last of the five verbs to get a bridge, and the reason it was last is
worth stating: `shard_advice` has existed since `compile_advice.py` was
written, but it is gated on having the SAME model profiled at more than one
core width, and for most of this project's life only `topo_0` existed. It
fires now because dronet, ffn_block and yolov8_nano_64x96 are each profiled at
1, 2, 4 and 8 harts.

SHARD IS NOT SPLIT, and confusing them is the failure this file is shaped
around.

    split  cuts one dispatch into n dispatches. The graph grows. The
           scheduler may place the pieces on different harts, or at
           different times, or not at the same time at all.
    shard  leaves the graph alone. One dispatch is COMPILED to run its
           output channels across n cores at once -- one node, one cost,
           and that cost is a function of the width it was given.

So the 2.27x ceiling in `advice_to_split_hint.py` does not apply here and must
not be borrowed: it is the measured cost of 4-way OC SPLITTING of yolov8n
(+76% total work before any parallelism is bought). Sharding pays a different
and much smaller tax -- the pool barrier -- which the advice reports per
dispatch as `sync_overhead_us`, measured rather than assumed.

WHY THE WIDTH IS DERIVED AND NOT CHOSEN
---------------------------------------
Every candidate width was MEASURED. `shard_advice` already picked the one it
recommends and recorded, in `evidence`, the cost at 1 core, the cost at that
width, the speedup, and the parallel efficiency. This bridge's job is not to
second-guess that; it is to check the constraints the REWRITER enforces, here,
where the advice that caused a refusal is still in hand:

  * OC must be divisible by the shard count. `shard_conv_weights` skips any
    conv whose OC does not divide, silently falling back to one shard -- so an
    unchecked hint produces a build that compiles, runs, and is not sharded.
    We walk the width down to the largest measured factor that divides.
  * the op must be shardable at all (`_SHARDABLE_CONV_OPS` plus the linear
    wrappers), and it must not already be a split tile: a dispatch carrying
    `split_from` is skipped by the planner.
  * the hint must be about THIS graph. A dispatch id means nothing without
    the graph it indexes, which is why `--ir` is required and the op kind is
    checked against the advice's own record of it.

WHAT PARALLEL EFFICIENCY IS FOR
-------------------------------
`--min-efficiency` refuses a width whose measured efficiency is below a floor,
default 0.5. This is not a purity test. Efficiency below 0.5 means more than
half the added cores are being spent on the barrier rather than the work, and
on this board those cores are not free: they are the other cluster, which the
`cost_by_pred` measurement shows costs ~10% to reach across. A dispatch that
is 3.1x faster on 8 cores is worth having; the same dispatch at 1.4x is
usually worth less than leaving the cores for another network.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "xpu-rt"))

import bundle  # noqa: E402

#: Ops whose emitters have a sharded form. Mirrors ModelBlaster's
#: `_SHARDABLE_CONV_OPS` plus the linear wrappers; kept here so a refusal is
#: reported against the advice rather than discovered at build time.
SHARDABLE_OPS = {
    "conv2d_s8",
    "conv2d_batchnorm2d_s8",
    "conv2d_batchnorm2d_silu_s8",
    "conv2d_silu_s8",
    "linear_s8",
    "matmul_s8",
}


def _largest_divisor_at_most(oc: int, n: int) -> int:
    """The biggest shard count <= n that divides oc, or 1.

    Walking DOWN rather than rounding: a width that does not divide is not a
    width the planner will use, and the measured cost we are quoting is for a
    width we can actually ask for.
    """
    for k in range(min(n, oc), 1, -1):
        if oc % k == 0:
            return k
    return 1


def _oc_of(op: dict) -> int:
    sh = op.get("shape") or {}
    for key in ("OC", "N"):          # conv output channels, or linear N
        if key in sh:
            try:
                return int(sh[key])
            except (TypeError, ValueError):
                return 0
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--advice", required=True)
    ap.add_argument("--ir", required=True, help="ModelBlaster graph.json")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-efficiency", type=float, default=0.5,
                    help="refuse a width whose MEASURED parallel efficiency "
                         "is below this (default 0.5)")
    ap.add_argument("--max-shards", type=int, default=8,
                    help="cap, in harts. 8 is the whole board; 4 keeps a "
                         "dispatch inside one L2 cluster")
    a = ap.parse_args()

    advice = json.load(open(a.advice))
    ir = json.load(open(a.ir))
    ops_by_id = {op.get("dispatch_id"): op for op in ir.get("ops", [])}

    wants = [x for x in advice.get("advice", [])
             if x.get("recommendation") == "shard"
             and x.get("model") == a.model]
    if not wants:
        print(f"no shard advice for {a.model} in {a.advice}", file=sys.stderr)
        return 1

    shard_ops, notes, refused = [], [], 0
    for x in wants:
        did = x.get("dispatch_id")
        op = ops_by_id.get(did)
        if op is None:
            print(f"REFUSED dispatch {did}: not in {a.ir}. The advice is "
                  f"about a different graph than the one being rewritten.",
                  file=sys.stderr)
            refused += 1
            continue

        kind = op.get("op")
        if kind not in SHARDABLE_OPS:
            print(f"REFUSED dispatch {did}: {kind} has no sharded emitter "
                  f"(shardable: {sorted(SHARDABLE_OPS)})", file=sys.stderr)
            refused += 1
            continue

        if op.get("split_from"):
            print(f"REFUSED dispatch {did}: already a split tile. The shard "
                  f"planner skips anything carrying `split_from`, so the "
                  f"hint would build cleanly and do nothing.", file=sys.stderr)
            refused += 1
            continue

        ev = x.get("evidence") or {}
        eff = ev.get("parallel_efficiency")
        if eff is not None and eff < a.min_efficiency:
            print(f"REFUSED dispatch {did}: measured parallel efficiency "
                  f"{eff:.2f} < {a.min_efficiency:.2f} -- more than half the "
                  f"added cores go to the barrier, not the work",
                  file=sys.stderr)
            refused += 1
            continue

        asked = int((x.get("constraints") or {}).get("n_cores", 0))
        if asked <= 1:
            refused += 1
            continue
        capped = min(asked, a.max_shards)

        oc = _oc_of(op)
        if oc <= 0:
            print(f"REFUSED dispatch {did}: no OC/N in its shape; the "
                  f"planner cannot check divisibility", file=sys.stderr)
            refused += 1
            continue

        n = _largest_divisor_at_most(oc, capped)
        if n <= 1:
            print(f"REFUSED dispatch {did}: OC={oc} has no divisor in "
                  f"2..{capped}. shard_conv_weights would skip it and the "
                  f"build would be silently unsharded.", file=sys.stderr)
            refused += 1
            continue

        shard_ops.append({"op": did, "n_shards": n})
        notes.append({
            "dispatch_id": did, "op": kind, "oc": oc,
            "n_advised": asked, "n_shards": n,
            "divides": n == capped,
            "cost_1core_ms": ev.get("cost_1core_ms"),
            "cost_ncore_ms": ev.get(f"cost_{asked}core_ms"),
            "measured_speedup": ev.get("measured_speedup"),
            "parallel_efficiency": eff,
            "sync_overhead_us": ev.get("sync_overhead_us"),
        })

    if not shard_ops:
        print(f"no emittable shard for {a.model} "
              f"({refused} refused, {len(wants)} advised)", file=sys.stderr)
        return 1

    hint = bundle.shard_hint(
        {a.model: shard_ops},
        reason="; ".join(
            f"dispatch {n['dispatch_id']} ({n['op']}) OC={n['oc']} across "
            f"{n['n_shards']} cores, measured {n['measured_speedup']}x at "
            f"{n['parallel_efficiency']} efficiency" for n in notes),
        provenance={
            "from_advice": advice.get("schedule_id"),
            "min_efficiency": a.min_efficiency,
            "max_shards": a.max_shards,
            "refused": refused,
            "derivation": notes,
            "evidence": [x.get("evidence") for x in wants],
        })
    with open(a.out, "w") as f:
        json.dump(hint, f, indent=1)
    print(f"wrote {a.out}")
    for n in notes:
        flag = "" if n["divides"] else \
            f"  (WIDTH WALKED DOWN from {n['n_advised']}: OC={n['oc']})"
        print(f"  dispatch {n['dispatch_id']} {n['op']}: OC={n['oc']} in "
              f"{n['n_shards']} shards, {n['measured_speedup']}x measured"
              f"{flag}")
    if refused:
        print(f"  {refused} advised dispatch(es) refused; see above")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
