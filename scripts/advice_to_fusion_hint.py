#!/usr/bin/env python3
"""Translate compile_advice.json into a ModelBlaster fusion hint.

This is the narrow adapter the design calls for: the scheduler never edits C, it
states a recommendation with evidence, and a translator turns the subset that
ModelBlaster can act on into that project's existing contract
(`modelblaster.fusion_hints/v1`, consumed by pipeline/apply_fusion_hint.py).

Only `fuse_with_successor` / `fuse_with_predecessor` are translated. `split` and
`choose_implementation` are different mechanisms and are deliberately left to
their own paths rather than being crammed through this one.

Groups are built from the IR, not from the advice: the advisor says *this model
is launch-overhead bound*, and the legal chains are whatever the graph actually
contains. A group is emitted only where the chain is genuinely linear -- one
producer, one consumer -- which is the precondition apply_fusion_hint enforces
anyway, and it is better to fail here with a readable reason than there.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "xpu-rt"))

import bundle  # noqa: E402

CONTRACT = bundle.FUSION_CONTRACT


def linear_chains(ops, only_ops=None):
    """Maximal chains where each op has exactly one producer and one consumer."""
    by_id = {o["dispatch_id"]: o for o in ops}
    consumers = defaultdict(list)
    for o in ops:
        for d in (o.get("depends_on") or []):
            consumers[d].append(o["dispatch_id"])
    chains, used = [], set()
    for o in ops:
        i = o["dispatch_id"]
        if i in used:
            continue
        deps = o.get("depends_on") or []
        # start a chain only at an op whose producer does not itself continue one
        if len(deps) == 1 and len(consumers[deps[0]]) == 1 and deps[0] not in used:
            continue
        chain = [i]
        used.add(i)
        cur = i
        while len(consumers[cur]) == 1:
            nxt = consumers[cur][0]
            if nxt in used or len((by_id[nxt].get("depends_on") or [])) != 1:
                break
            if only_ops and by_id[nxt]["op"] not in only_ops \
               and by_id[chain[-1]]["op"] not in only_ops:
                break
            chain.append(nxt)
            used.add(nxt)
            cur = nxt
        if len(chain) > 1:
            chains.append(chain)
    return chains


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--advice", required=True)
    ap.add_argument("--ir", required=True, help="ModelBlaster graph.json")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pair-only", action="store_true",
                    help="emit adjacent pairs rather than maximal chains; a "
                         "shorter fused unit is a smaller non-preemptive "
                         "blocker, which matters for a periodic workload")
    a = ap.parse_args()

    advice = json.load(open(a.advice))
    wants_fusion = [x for x in advice["advice"]
                    if x["recommendation"].startswith("fuse_")
                    and x["model"] == a.model]
    if not wants_fusion:
        print(f"no fusion advice for {a.model}; nothing to do", file=sys.stderr)
        return 1

    graph = json.load(open(a.ir))
    chains = linear_chains(graph["ops"])
    if a.pair_only:
        pairs = []
        for c in chains:
            pairs += [c[i:i + 2] for i in range(0, len(c) - 1, 2)]
        chains = [p for p in pairs if len(p) > 1]
    if not chains:
        print("IR has no linear chain to fuse", file=sys.stderr)
        return 1

    # The schema lives in `bundle`, not here: three call sites build this
    # contract from three different analyses, and they must not drift on the
    # wire format.
    hint = bundle.fusion_hint(
        {a.model: chains},
        reason=f"overhead-bound: {len(chains)} linear chain(s) from "
               f"{advice.get('schedule_id')}",
        provenance={
            "from_advice": advice.get("schedule_id"),
            "recommendations": [x["recommendation"] for x in wants_fusion],
            "evidence": [x["evidence"] for x in wants_fusion],
        })
    json.dump(hint, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")
    print(f"  {len(chains)} fuse group(s): {chains}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
