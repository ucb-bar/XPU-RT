#!/usr/bin/env python3
"""Find a graph's independent branches in flow_c IR space.

`slice_experiment.py --tile` partitions the *ONNX* by backward closure,
which is what the converter needs.  A flow_c binding manifest indexes the
*IR* (`ops: {ranges: ...}`), and the two spaces have different op counts --
ViNT is 1931 ONNX nodes and 605 IR ops.  This does the same closure on the
IR that the slicer does on the ONNX, so a manifest's ranges are derived
rather than transcribed, and prints the dependency structure that decides
whether two tiles can occupy two lanes at once.

    python3 ir_branch_ranges.py <graph.json> <tile_out_op> [<tile_out_op> ...]

e.g.  python3 ir_branch_ranges.py ../flow_c/gen/ir/vint/int8/graph.json linear linear_1
"""
from __future__ import annotations

import json
import sys


def runs(ix: list[int]) -> list[list[int]]:
    out, lo, prev = [], ix[0], ix[0]
    for i in ix[1:]:
        if i == prev + 1:
            prev = i
            continue
        out.append([lo, prev])
        lo = prev = i
    out.append([lo, prev])
    return out


def partition(graph_json: str, tile_outs: list[str]):
    ops = json.load(open(graph_json))["ops"]
    dep = {i: set(o.get("depends_on") or []) for i, o in enumerate(ops)}
    idx = {o["name"]: i for i, o in enumerate(ops)}

    def closure(root: int) -> set[int]:
        seen, st = set(), [root]
        while st:
            i = st.pop()
            if i in seen:
                continue
            seen.add(i)
            st.extend(dep[i])
        return seen

    claimed, tiles = set(), []
    for nm in tile_outs:
        if nm not in idx:
            raise SystemExit(f"no IR op named '{nm}' in {graph_json}")
        s = closure(idx[nm]) - claimed
        tiles.append(sorted(s))
        claimed |= s
    rest = sorted(set(range(len(ops))) - claimed)
    if rest:
        tiles.append(rest)

    # a tile depends on another iff it consumes an op that tile owns
    owner = {i: k for k, s in enumerate(tiles) for i in s}
    out = []
    for k, s in enumerate(tiles):
        d = sorted({owner[p] for i in s for p in dep[i] if owner[p] != k})
        out.append({"index": k, "ranges": runs(s), "n_ops": len(s), "depends_on": d})
    indep = [[a["index"], b["index"]] for a in out for b in out
             if a["index"] < b["index"]
             and b["index"] not in a["depends_on"] and a["index"] not in b["depends_on"]]
    return {"n_ir_ops": len(ops), "tiles": out, "independent_pairs": indep}


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    res = partition(sys.argv[1], sys.argv[2:])
    print(f"{res['n_ir_ops']} IR ops")
    for t in res["tiles"]:
        print(f"  tile{t['index']}: ops {t['ranges']} ({t['n_ops']}) "
              f"depends_on={t['depends_on']}")
    print(f"  independent pairs: {res['independent_pairs'] or 'none'}")
