#!/usr/bin/env python3
"""Op census over qairt-dlc-to-json dumps.

Emits, per (network, precision): the op-type histogram and the set of tensor
data types each op type carries.  This is what tells us how many distinct
kernels a GPU int8 path would need -- the compose log tells us *that* int8 is
missing, this tells us *how much* is missing.
"""
import json, sys, os, collections

DT = {0x0008: "int8", 0x0016: "int16", 0x0032: "int32", 0x0064: "int64",
      0x0108: "uint8", 0x0116: "uint16", 0x0132: "uint32", 0x0164: "uint64",
      0x0216: "fp16", 0x0232: "fp32",
      0x0308: "sFxp8", 0x0316: "sFxp16", 0x0332: "sFxp32",
      0x0408: "uFxp8", 0x0416: "uFxp16", 0x0432: "uFxp32",
      0x0508: "bool8"}


def census(path):
    d = json.load(open(path))
    g = d["graph"]
    tensors, nodes = g["tensors"], g["nodes"]
    hist = collections.Counter()
    dtypes = collections.defaultdict(set)
    detail = {}
    for name, n in nodes.items():
        t = n["type"]
        hist[t] += 1
        for tn in list(n["input_names"]) + list(n["output_names"]):
            ti = tensors.get(tn)
            if ti is None:
                continue
            dtypes[t].add(DT.get(ti.get("data_type"), hex(ti.get("data_type", 0))))
        detail[name] = t
    return {
        "num_nodes": len(nodes),
        "op_types": dict(sorted(hist.items(), key=lambda kv: -kv[1])),
        "dtypes_per_op": {k: sorted(v) for k, v in sorted(dtypes.items())},
        "first_node": next(iter(nodes)),
    }


if __name__ == "__main__":
    out = {}
    for p in sys.argv[1:]:
        key = os.path.basename(p).replace(".qnn.json", "")
        out[key] = census(p)
    print(json.dumps(out, indent=2))
