#!/usr/bin/env python3
"""Build a Flow C IR (graph.json) for the smolVLA vision port from the sliced ONNX.

Flow C refuses an onnx-sourced network without an IR, and rightly: the IR is
what the capability check reads to decide whether a tile's ops can run on a
backend. The repo's export_graph_json.py builds one from a DLC via
`qairt-dlc-to-json`, which is not available on this host, so this reads the
per-tile ONNX directly instead.

Ops are numbered with a global dispatch_id in tile execution order, so each
tile owns a contiguous [start, end] range and the binding can address it with
`ops: {ranges: [...]}` rather than a partition file.
"""
import argparse, json, os, sys
import onnx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binding", required=True)
    ap.add_argument("--slices", default="../smolVLA/vision_slices_v3")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    man = json.load(open(a.binding))
    ops, ranges, missing = [], {}, []
    did = 0
    for b in man["bindings"]:
        t = b["name"]
        path = None
        for sub in ("", "trampolines", "hta_convs", "conv1x1"):
            p = os.path.join(a.slices, sub, f"{t}.onnx")
            if os.path.exists(p): path = p; break
        if path is None:
            missing.append(t); continue
        m = onnx.load(path, load_external_data=False)
        start = did
        for n in m.graph.node:
            ops.append({"name": n.name or f"{t}_{did}", "op_type": n.op_type,
                        "dispatch_id": did,
                        "depends_on": [did - 1] if did else [],
                        "hardware_target": "any"})
            did += 1
        if did == start:      # a tile with no nodes still needs one op to own
            ops.append({"name": t, "op_type": "Identity", "dispatch_id": did,
                        "depends_on": [did - 1] if did else [], "hardware_target": "any"})
            did += 1
        ranges[t] = [start, did - 1]

    ir = {"name": "smolvlm_vision", "quant": "int8", "ops": ops}
    json.dump(ir, open(a.out, "w"), indent=1)
    json.dump(ranges, open(a.out + ".ranges", "w"), indent=1)
    from collections import Counter
    c = Counter(o["op_type"] for o in ops)
    print(f"  {len(ops)} ops across {len(ranges)} tiles -> {a.out}")
    print(f"  op types: {dict(c.most_common(12))}")
    if missing: print(f"  MISSING onnx for {len(missing)} tiles: {missing[:5]}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
