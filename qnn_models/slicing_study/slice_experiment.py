#!/usr/bin/env python3
"""One re-runnable slicing experiment.

Given a network and a candidate cut (an ordered list of boundary tensor
names in the source ONNX), this:

  1. slices the ONNX into contiguous tiles at those boundaries
     (``onnx.utils.extract_model``), promoting every crossing edge to
     tile I/O — including skip connections that jump a tile;
  2. captures calibration activations at each tile's boundary inputs by
     running the *whole* network under onnxruntime with those tensors
     promoted to graph outputs;
  3. converts each tile to a DLC (``snpe-onnx-to-dlc``) and quantizes it
     (``qairt-quantizer``) inside the ``qnn-convert`` docker image;
  4. builds a context binary per (tile, backend) on the board with
     ``qnn-context-binary-generator``, recording compose success or the
     op the log names as the reason for failure;
  5. measures every (tile, backend) that composed with
     ``/root/qnn_runtime/profile_seg`` under the ``performance``
     governor, restoring ``schedutil`` in the same locked block;
  6. appends one JSON record per experiment to ``experiments.jsonl``.

Idempotent: every artifact is keyed by the sha256 of its input plus the
cut specification, so re-running skips work whose output already exists.
``--force`` re-does it anyway.  Every shell command is printed as it runs
and stored in the record.

Board etiquette: every board interaction is serialised behind
``flock -w 900 /tmp/qnn_board.lock`` and wrapped in ``timeout -s KILL``.

Usage
-----
    # whole network, no cut (the k=1 baseline)
    python3 slice_experiment.py --network dronet --cut ''

    # two tiles, cut after the second residual block
    python3 slice_experiment.py --network dronet \
        --cut /Add_1_output_0 --name dronet_k2_add1

    # list the cut points a network offers
    python3 slice_experiment.py --network yolov8n --list-nodes

    # per-dispatch overhead probes (synthetic 1-MAC graphs)
    python3 slice_experiment.py --overhead-probe
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
GEN = os.path.join(HERE, "gen")
LOGS = os.path.join(HERE, "logs")
JOURNAL = os.path.join(HERE, "experiments.jsonl")

# The interpreter that has onnx + onnxruntime in this checkout.
PY = os.environ.get("SLICE_PY", "/scratch2/dima/miniforge3/envs/xpurt/bin/python3")

BOARD = os.environ.get("BOARD", "root@10.44.120.201")
BOARD_DIR = os.environ.get("BOARD_DIR", "/root/slicing_study")
QAIRT_BOARD = "/root/qairt"
PROFILE_SEG = "/root/qnn_runtime/profile_seg"
LOCK = "/tmp/qnn_board.lock"
LOCK_WAIT = 900

DOCKER_IMAGE = os.environ.get("DOCKER_IMAGE", "qnn-convert")
DOCKER = os.environ.get("DOCKER", "sudo docker")

BACKEND_LIB = {"hta": "libQnnHta.so", "dsp": "libQnnDsp.so", "cpu": "libQnnCpu.so"}

# --------------------------------------------------------------------------
# Network registry.  Paths are repo-relative.
# --------------------------------------------------------------------------

CAL = "qnn_models/boards/qrb5165_v66/calibration_data"

NETWORKS: dict[str, dict[str, Any]] = {
    "dronet": {
        "onnx": "qnn_models/dronet_full_hta.onnx",
        # BN folded offline, FC head re-expressed as 1x1 conv, trailing
        # Reshape dropped -- the graph flow_c's dronet_full binding names.
        "inputs": {"input": {"shape": [1, 3, 112, 112], "dtype": "float32",
                              "samples": f"{CAL}/calibration_data_dronet/input_*.raw"}},
        "iters": 40,
        "lowering": ["fold_batchnorm_into_conv", "fuse_activation_into_conv",
                      "conv_head_for_fc", "drop_trailing_views"],
    },
    "yolov8n": {
        "onnx": "qnn_models/yolov8n.onnx",
        "inputs": {"images": {"shape": [1, 3, 640, 640], "dtype": "float32",
                               "samples": f"{CAL}/calibration_data_yolov8n/input_*.raw"}},
        "iters": 20,
        "lowering": [],
    },
    # Same network after the Split -> 2x Conv1x1 channel-selector rewrite
    # (qnn_models/optimizations.md #14).  HTA has no Split op, so this is a
    # precondition for *any* yolov8n tile reaching HTA -- exactly the role
    # the offline BN fold plays for dronet.
    "yolov8n_nosplit": {
        "onnx": "qnn_models/yolov8n_nosplit.onnx",
        "inputs": {"images": {"shape": [1, 3, 640, 640], "dtype": "float32",
                               "samples": f"{CAL}/calibration_data_yolov8n/input_*.raw"}},
        "iters": 20,
        "lowering": ["split_to_conv1x1"],
    },
    "fused_full": {
        "onnx": "qnn_models/flow_c/gen/onnx/fused_full.onnx",
        "inputs": {
            "front_grey": {"shape": [1, 1, 60, 90], "dtype": "float32",
                            "samples": "qnn_models/flow_c/gen/convert/cal/front_grey_*.raw"},
            "tof_cross": {"shape": [1, 4, 8, 8], "dtype": "float32",
                           "samples": "qnn_models/flow_c/gen/convert/cal/tof_cross_*.raw"},
            "lowdim": {"shape": [1, 21], "dtype": "float32",
                        "samples": "qnn_models/flow_c/gen/convert/cal/lowdim_*.raw"},
        },
        "iters": 40,
        "lowering": [],
    },
    "vint": {
        "onnx": "qnn_models/flow_c/gen/onnx/vint.onnx",
        "inputs": {
            "obs_img": {"shape": [1, 18, 64, 85], "dtype": "float32",
                         "samples": "qnn_models/flow_c/gen/convert/cal_vint/obs_img_*.raw"},
            "goal_img": {"shape": [1, 3, 64, 85], "dtype": "float32",
                          "samples": "qnn_models/flow_c/gen/convert/cal_vint/goal_img_*.raw"},
        },
        "iters": 5,
        "lowering": [],
    },
    "mlp_control": {
        "onnx": "qnn_models/flow_c/gen/onnx/mlp_control.onnx",
        "inputs": {"obs": {"shape": [1, 16], "dtype": "float32",
                            "samples": "SYNTHETIC:normal:0"}},
        "iters": 40,
        "lowering": [],
    },
}

# --------------------------------------------------------------------------
# shell plumbing
# --------------------------------------------------------------------------


class Runner:
    """Runs commands, echoing each one and keeping the transcript."""

    def __init__(self) -> None:
        self.commands: list[str] = []

    def sh(self, cmd: str, check: bool = True, quiet: bool = False,
           timeout: int | None = None) -> subprocess.CompletedProcess:
        self.commands.append(cmd)
        if not quiet:
            print(f"  $ {cmd}", flush=True)
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                            timeout=timeout)
        if check and p.returncode != 0:
            print(p.stdout[-4000:], file=sys.stderr)
            print(p.stderr[-4000:], file=sys.stderr)
            raise SystemExit(f"command failed ({p.returncode}): {cmd}")
        return p

    def board(self, script: str, timeout: int = 1200,
              check: bool = False) -> subprocess.CompletedProcess:
        """Run a shell snippet on the board behind the shared lock."""
        inner = f"flock -w {LOCK_WAIT} {LOCK} -c {shlex.quote(script)}"
        return self.sh(f"ssh -o ConnectTimeout=15 {BOARD} {shlex.quote(inner)}",
                        check=check, timeout=timeout)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(p: str) -> str:
    return os.path.relpath(os.path.abspath(p), REPO)


def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


# --------------------------------------------------------------------------
# ONNX side -- these run under PY (the env that has onnx/onnxruntime), as a
# subprocess, so that this driver stays importable from stock python3.
# --------------------------------------------------------------------------

SLICE_HELPER = r'''
import json, os, re, sys
import onnx
from onnx import shape_inference
import onnx.utils

req = json.load(open(sys.argv[1]))
src, out_dir, cuts = req["src"], req["out_dir"], req["cuts"]
os.makedirs(out_dir, exist_ok=True)

model = onnx.load(src)
try:
    model = shape_inference.infer_shapes(model, strict_mode=False)
except Exception:
    pass
g = model.graph
nodes = list(g.node)
init = {i.name for i in g.initializer}
graph_in = [i.name for i in g.input if i.name not in init]
graph_out = [o.name for o in g.output]
producer = {}
for i, n in enumerate(nodes):
    for o in n.output:
        if o:
            producer[o] = i

vi = {v.name: v for v in list(g.value_info) + list(g.output) + list(g.input)}
def dims(t):
    v = vi.get(t)
    if v is None:
        return None
    d = v.type.tensor_type.shape.dim
    return [(x.dim_value if x.dim_value else (x.dim_param or 0)) for x in d]

# ---- static subgraphs belong to every tile, not to one ------------------
# A node whose value does not depend on any graph input is compile-time
# data (Constant, and anything folded from Constants -- ViNT's attention
# masks are Unsqueeze/Range/ConstantOfShape chains over constants).  If
# such a node were assigned to one tile its output would cross the cut as
# a *tensor*, which the converter rejects
# (KeyError on the promoted Constant) and the quantizer cannot calibrate.
# So they are shared: every tile that needs one gets its own copy.
static = set()
for i, n in enumerate(nodes):
    dyn_inputs = [t for t in n.input if t and t not in init]
    if all(producer.get(t) in static for t in dyn_inputs if t in producer) \
       and not any(t in graph_in for t in dyn_inputs):
        static.add(i)
dyn = [i for i in range(len(nodes)) if i not in static]
dyn_pos = {i: k for k, i in enumerate(dyn)}

# ---- subgraph mode: tiles named by BOTH their inputs and their outputs --
# The only way to name a tile that starts in the middle of the graph.  ViNT's
# obs encoder is an EfficientNet run on a batch of 6 stacked frames; the tile
# that holds one frame starts at the Concat that stacks them, which no
# backward closure from an output can express.  `batch1` then rewrites the
# leading dimension of every tile boundary from B to 1 and re-infers, turning
# the batch-6 graph into the batch-1 graph that gets dispatched six times.
subgraphs = req.get("subgraphs") or []
if subgraphs:
    tiles = []
    for k, sg in enumerate(subgraphs):
        ins, outs = list(sg["inputs"]), list(sg["outputs"])
        path = os.path.join(out_dir, f"tile{k}.onnx")
        onnx.utils.extract_model(src, path, ins, outs)
        sub = onnx.load(path)
        renames = {}
        for gi in sub.graph.input:
            if gi.name in graph_in:
                continue
            nn = "t_" + re.sub(r"[^A-Za-z0-9_]", "_", gi.name).strip("_")
            if nn != gi.name:
                renames[gi.name] = nn
        if renames:
            for gi in sub.graph.input:
                if gi.name in renames:
                    gi.name = renames[gi.name]
            for n in sub.graph.node:
                for j, t in enumerate(n.input):
                    if t in renames:
                        n.input[j] = renames[t]
            for v in sub.graph.value_info:
                if v.name in renames:
                    v.name = renames[v.name]
        if req.get("batch1"):
            _b = int(req.get("batch") or 1)
            for io in list(sub.graph.input) + list(sub.graph.output):
                d = io.type.tensor_type.shape.dim
                if len(d):
                    d[0].ClearField("dim_param")
                    d[0].dim_value = _b
            del sub.graph.value_info[:]
            try:
                sub = shape_inference.infer_shapes(sub, strict_mode=False, data_prop=True)
            except Exception as e:
                print(f"  batch1 shape inference: {e}")
        onnx.save(sub, path)
        sub = onnx.load(path)
        names = {n.name for n in sub.graph.node}
        seg = sorted(i for i, n in enumerate(nodes) if n.name in names and i not in static)
        rr = [[seg[0], seg[-1]]] if seg else [[0, 0]]
        if seg:
            rr, lo, prev = [], seg[0], seg[0]
            for i in seg[1:]:
                if all((j == i) or (j in static) for j in range(prev + 1, i + 1)):
                    prev = i
                    continue
                rr.append([lo, prev]); lo = prev = i
            rr.append([lo, prev])
        vinfo = {v.name: v for v in list(sub.graph.input) + list(sub.graph.output)}
        def sdims(t):
            v = vinfo.get(t)
            if v is None:
                return dims(t)
            return [(x.dim_value if x.dim_value else (x.dim_param or 0))
                    for x in v.type.tensor_type.shape.dim]
        tiles.append({
            "index": k, "op_range": [rr[0][0], rr[-1][1]], "ranges": rr,
            "n_src_nodes": len(seg), "n_tile_nodes": len(sub.graph.node),
            "inputs": [{"src_name": t, "dlc_name": renames.get(t, t),
                         "shape": sdims(renames.get(t, t)),
                         "is_net_input": t in graph_in} for t in ins],
            "outputs": [{"name": t, "shape": sdims(t), "is_net_output": t in graph_out}
                        for t in outs],
            "onnx": path,
            "op_types": sorted({nodes[i].op_type for i in seg}),
            "depends_on": [],
        })
    json.dump({"tiles": tiles, "n_src_nodes": len(nodes),
                "n_static_nodes": len(static), "n_dynamic_nodes": len(dyn),
                "mode": "subgraph", "batch1": bool(req.get("batch1")),
                "independent_pairs": [],
                "graph_inputs": graph_in, "graph_outputs": graph_out},
              open(req["result"], "w"), indent=2)
    print(f"sliced {len(tiles)} tile(s) from {src} in SUBGRAPH mode"
          + (" with batch1 rewrite" if req.get("batch1") else ""))
    for t in tiles:
        print(f"  tile{t['index']}: ops {t['ranges']} ({t['n_src_nodes']} src nodes -> "
              f"{t['n_tile_nodes']} tile nodes) "
              f"in={[(i['dlc_name'], i['shape']) for i in t['inputs']]} "
              f"out={[(o['name'], o['shape']) for o in t['outputs']]}")
    raise SystemExit(0)

# ---- branch mode: a tile is a SET of ops, named by what it produces -----
# A contiguous span cannot express two independent branches: whichever comes
# second in topological order ends up consuming the first one's output and
# the two serialise inside one graph.  In branch mode each tile is given its
# output tensors and claims the backward closure of them over the dynamic
# nodes, minus whatever earlier tiles already claimed.  Independent branches
# then fall out as tiles whose input sets touch no other tile's outputs.
tile_outputs = req.get("tile_outputs") or []
if tile_outputs:
    consumers = {}
    for i, n in enumerate(nodes):
        for t in n.input:
            if t:
                consumers.setdefault(t, []).append(i)

    def closure(outs):
        seen, stack = set(), []
        for t in outs:
            if t not in producer:
                raise SystemExit(f"tile output '{t}' is not produced by any node in {src}")
            stack.append(producer[t])
        while stack:
            i = stack.pop()
            if i in seen or i in static:
                continue
            seen.add(i)
            for t in nodes[i].input:
                if t and t in producer:
                    stack.append(producer[t])
        return seen

    claimed, node_sets = set(), []
    for outs in tile_outputs:
        s = closure(outs) - claimed
        if not s:
            raise SystemExit(f"tile {outs} claims no ops (all already claimed)")
        node_sets.append(sorted(s))
        claimed |= s
    rest = sorted(set(dyn) - claimed)
    if rest:
        node_sets.append(rest)

    def boundary_set(seg_list):
        seg = set(seg_list)
        others = set(dyn) - seg
        produced_dyn, consumed = set(), set()
        for i in seg:
            for t in nodes[i].input:
                if t:
                    consumed.add(t)
            for t in nodes[i].output:
                if t:
                    produced_dyn.add(t)
        produced_static = {t for i in static for t in nodes[i].output if t}
        outside = {t for i in others for t in nodes[i].input if t}
        ins = sorted(t for t in consumed
                     if t not in produced_dyn and t not in produced_static and t not in init)
        outs = sorted(t for t in produced_dyn if t in graph_out or t in outside)
        return ins, outs

    def _runs(ix, filler=frozenset()):
        """Contiguous runs over sorted node indices. `filler` indices do not
        break a run: static (constant) nodes belong to every tile, so a gap
        made only of them is not a real discontinuity -- without this ViNT's
        decoder declares 200 one-node ranges instead of one span."""
        out, lo, prev = [], ix[0], ix[0]
        for i in ix[1:]:
            if all((j == i) or (j in filler) for j in range(prev + 1, i + 1)):
                prev = i
                continue
            out.append([lo, prev]); lo = prev = i
        out.append([lo, prev])
        return out

    def runs(ix):
        return _runs(ix, filler=static)

    tiles = []
    for k, seg in enumerate(node_sets):
        ins, outs = boundary_set(seg)
        if not ins or not outs:
            raise SystemExit(f"tile {k} has empty boundary in={ins} out={outs}")
        path = os.path.join(out_dir, f"tile{k}.onnx")
        onnx.utils.extract_model(src, path, ins, outs)
        sub = onnx.load(path)
        renames = {}
        for gi in sub.graph.input:
            if gi.name in graph_in:
                continue
            nn = "t_" + re.sub(r"[^A-Za-z0-9_]", "_", gi.name).strip("_")
            if nn != gi.name:
                renames[gi.name] = nn
        if renames:
            for gi in sub.graph.input:
                if gi.name in renames:
                    gi.name = renames[gi.name]
            for n in sub.graph.node:
                for j, t in enumerate(n.input):
                    if t in renames:
                        n.input[j] = renames[t]
            for v in sub.graph.value_info:
                if v.name in renames:
                    v.name = renames[v.name]
            onnx.save(sub, path)
            sub = onnx.load(path)
        rr = runs(seg)
        rr_exact = _runs(seg)
        tiles.append({
            "index": k, "op_range": [rr[0][0], rr[-1][1]], "ranges": rr,
            "ranges_exact_dynamic_only": rr_exact,
            "n_src_nodes": len(seg), "n_tile_nodes": len(sub.graph.node),
            "inputs": [{"src_name": t, "dlc_name": renames.get(t, t), "shape": dims(t),
                         "is_net_input": t in graph_in} for t in ins],
            "outputs": [{"name": t, "shape": dims(t), "is_net_output": t in graph_out}
                        for t in outs],
            "onnx": path,
            "op_types": sorted({nodes[i].op_type for i in seg}),
        })
    prod_of = {o["name"]: t["index"] for t in tiles for o in t["outputs"]}
    for t in tiles:
        t["depends_on"] = sorted({prod_of[i["src_name"]] for i in t["inputs"]
                                  if i["src_name"] in prod_of
                                  and prod_of[i["src_name"]] != t["index"]})
    parallel = [[a["index"], b["index"]] for a in tiles for b in tiles
                if a["index"] < b["index"]
                and b["index"] not in a["depends_on"] and a["index"] not in b["depends_on"]]
    json.dump({"tiles": tiles, "n_src_nodes": len(nodes),
                "n_static_nodes": len(static), "n_dynamic_nodes": len(dyn),
                "mode": "branch", "independent_pairs": parallel,
                "graph_inputs": graph_in, "graph_outputs": graph_out},
              open(req["result"], "w"), indent=2)
    print(f"sliced {len(tiles)} tile(s) from {src} in BRANCH mode ({len(nodes)} nodes)")
    for t in tiles:
        print(f"  tile{t['index']}: ops {t['ranges']} ({t['n_src_nodes']} src nodes -> "
              f"{t['n_tile_nodes']} tile nodes) depends_on={t['depends_on']} "
              f"in={[i['dlc_name'] for i in t['inputs']]} "
              f"out={[o['name'] for o in t['outputs']]}")
    print(f"  independent tile pairs (may run concurrently): {parallel or 'none'}")
    raise SystemExit(0)

# ---- resolve the cut into contiguous node-index ranges ------------------
idx = []
for c in cuts:
    if c not in producer:
        raise SystemExit(f"cut tensor '{c}' is not produced by any node in {src}")
    if producer[c] in static:
        raise SystemExit(f"cut tensor '{c}' is produced by a constant subgraph; "
                          f"it carries no activation and cannot be a tile boundary")
    idx.append(producer[c])
if idx != sorted(idx) or len(set(idx)) != len(idx):
    raise SystemExit(f"cut tensors must be in strictly increasing topological order; got {idx}")
bounds = [-1] + [dyn_pos[i] for i in idx] + [len(dyn) - 1]
chunks = [(bounds[k] + 1, bounds[k + 1]) for k in range(len(bounds) - 1)]
chunks = [c for c in chunks if c[0] <= c[1]]
ranges = [(dyn[a], dyn[b]) for a, b in chunks]

def boundary(a, b):
    """a,b index into `dyn`; static nodes are replicated into the tile."""
    seg = set(dyn[a:b + 1])
    others = set(dyn) - seg
    produced_dyn, consumed = set(), set()
    for i in seg:
        for t in nodes[i].input:
            if t:
                consumed.add(t)
        for t in nodes[i].output:
            if t:
                produced_dyn.add(t)
    produced_static = set()
    for i in static:
        for t in nodes[i].output:
            if t:
                produced_static.add(t)
    outside_consumers = set()
    for i in others:
        for t in nodes[i].input:
            if t:
                outside_consumers.add(t)
    ins = sorted(t for t in consumed
                 if t not in produced_dyn and t not in produced_static
                 and t not in init)
    outs = sorted(t for t in produced_dyn
                  if t in graph_out or t in outside_consumers)
    return ins, outs

tiles = []
for k, ((a, b), (lo, hi)) in enumerate(zip(chunks, ranges)):
    ins, outs = boundary(a, b)
    if not ins or not outs:
        raise SystemExit(f"tile {k} ops[{lo},{hi}] has empty boundary in={ins} out={outs}")
    path = os.path.join(out_dir, f"tile{k}.onnx")
    onnx.utils.extract_model(src, path, ins, outs)
    sub = onnx.load(path)
    # The converter's -d matching breaks on names containing '/', so
    # rename every tile input that is not already a whole-network input.
    renames = {}
    for gi in sub.graph.input:
        if gi.name in graph_in:
            continue
        new = "t_" + re.sub(r"[^A-Za-z0-9_]", "_", gi.name).strip("_")
        if new != gi.name:
            renames[gi.name] = new
    if renames:
        for gi in sub.graph.input:
            if gi.name in renames:
                gi.name = renames[gi.name]
        for n in sub.graph.node:
            for j, t in enumerate(n.input):
                if t in renames:
                    n.input[j] = renames[t]
        for v in sub.graph.value_info:
            if v.name in renames:
                v.name = renames[v.name]
        onnx.save(sub, path)
        sub = onnx.load(path)
    tiles.append({
        "index": k,
        "op_range": [lo, hi],
        "n_src_nodes": b - a + 1,
        "src_node_span": [lo, hi],
        "n_tile_nodes": len(sub.graph.node),
        "inputs": [{"src_name": t, "dlc_name": renames.get(t, t), "shape": dims(t),
                     "is_net_input": t in graph_in} for t in ins],
        "outputs": [{"name": t, "shape": dims(t), "is_net_output": t in graph_out} for t in outs],
        "onnx": path,
        "op_types": sorted({nodes[i].op_type for i in dyn[a:b + 1]}),
    })

json.dump({"tiles": tiles, "n_src_nodes": len(nodes),
            "n_static_nodes": len(static), "n_dynamic_nodes": len(dyn),
            "graph_inputs": graph_in, "graph_outputs": graph_out},
          open(req["result"], "w"), indent=2)
print(f"sliced {len(tiles)} tile(s) from {src} ({len(nodes)} nodes)")
for t in tiles:
    print(f"  tile{t['index']}: src ops [{t['op_range'][0]},{t['op_range'][1]}] "
          f"({t['n_src_nodes']} src nodes -> {t['n_tile_nodes']} tile nodes) "
          f"in={[i['dlc_name'] for i in t['inputs']]} "
          f"out={[o['name'] for o in t['outputs']]}")
'''

CALIB_HELPER = r'''
import json, os, re, sys, glob
import numpy as np
import onnx
import onnxruntime as ort

req = json.load(open(sys.argv[1]))
src = req["src"]
want = req["capture"]              # internal tensors needing calibration
net_inputs = req["net_inputs"]     # name -> {shape, dtype, samples}
out_dir = req["out_dir"]
n_samples = req["n_samples"]
os.makedirs(out_dir, exist_ok=True)

def safe(t):
    return re.sub(r"[^A-Za-z0-9_]", "_", t)

# ---- assemble the sample feed ------------------------------------------
feeds = []
for name, spec in net_inputs.items():
    pat = spec["samples"]
    shape = spec["shape"]
    dt = np.dtype(spec["dtype"])
    if pat.startswith("SYNTHETIC:"):
        _, kind, seed = pat.split(":")
        rng = np.random.default_rng(int(seed))
        arrs = [rng.standard_normal(shape).astype(dt) for _ in range(n_samples)]
        # persist them so the experiment is reproducible from its record
        sdir = os.path.join(out_dir, "_synthetic", name)
        os.makedirs(sdir, exist_ok=True)
        paths = []
        for i, a in enumerate(arrs):
            p = os.path.join(sdir, f"sample_{i:04d}.raw")
            a.tofile(p)
            paths.append(p)
        feeds.append((name, arrs, paths))
    else:
        files = sorted(glob.glob(os.path.join(req["repo"], pat)))[:n_samples]
        if not files:
            raise SystemExit(f"no calibration samples matched {pat}")
        arrs = [np.fromfile(f, dtype=dt).reshape(shape) for f in files]
        feeds.append((name, arrs, files))

n = min(len(a) for _, a, _ in feeds)
net_sample_paths = {name: paths[:n] for name, _, paths in feeds}

captured = {}
shapes = {}
if want:
    model = onnx.load(src)
    have = ({v.name for v in model.graph.value_info}
            | {o.name for o in model.graph.output}
            | {i.name for i in model.graph.input}
            | {o for nd in model.graph.node for o in nd.output})
    missing = [t for t in want if t not in have]
    if missing:
        raise SystemExit(f"boundary tensors absent from {src}: {missing[:5]}")
    ext = onnx.ModelProto()
    ext.CopyFrom(model)
    existing = {o.name for o in ext.graph.output}
    for t in want:
        if t in existing:
            continue
        v = next((v for v in ext.graph.value_info if v.name == t), None)
        if v is None:
            v = onnx.helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None)
        ext.graph.output.append(v)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(ext.SerializeToString(), so,
                                 providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]
    for t in want:
        d = os.path.join(out_dir, "tensors", safe(t))
        os.makedirs(d, exist_ok=True)
        captured[t] = []
    for i in range(n):
        feed = {name: arrs[i] for name, arrs, _ in feeds}
        res = sess.run(want, feed)
        for t, a in zip(want, res):
            arr = np.asarray(a, dtype=np.float32)
            bs = int(req.get("batch_split") or 0)
            if bs and arr.ndim >= 2 and arr.shape[0] >= bs:
                # a batch-N tile needs calibration in batch-N chunks, not one
                # sample per forward pass
                for b in range(0, arr.shape[0] - bs + 1, bs):
                    p = os.path.join(out_dir, "tensors", safe(t),
                                      f"sample_{i:04d}_{b:02d}.raw")
                    arr[b:b + bs].tofile(p)
                    captured[t].append(p)
                shapes.setdefault(t, [bs] + list(arr.shape[1:]))
            else:
                p = os.path.join(out_dir, "tensors", safe(t), f"sample_{i:04d}.raw")
                arr.tofile(p)
                captured[t].append(p)
                shapes.setdefault(t, list(arr.shape))

json.dump({"n": n, "net_sample_paths": net_sample_paths,
            "captured": captured, "shapes": shapes},
          open(req["result"], "w"), indent=2)
print(f"calibration: {n} samples, {len(want)} boundary tensor(s) captured")
'''



FREEZE_HELPER = r'''
import json, sys
import onnx
from onnx import shape_inference

req = json.load(open(sys.argv[1]))
done = []
for spec in req["tiles"]:
    m = onnx.load(spec["onnx"])
    for gi in m.graph.input:
        shp = spec["input_shapes"].get(gi.name)
        if not shp:
            continue
        d = gi.type.tensor_type.shape.dim
        while len(d) < len(shp):
            d.add()
        for k, v in enumerate(shp):
            d[k].ClearField("dim_param")
            d[k].dim_value = int(v)
    # Re-infer with the boundary now concrete, then pin whatever the
    # graph outputs turned out to be: the converter refuses a model that
    # still carries a symbolic dim anywhere on its I/O.
    del m.graph.value_info[:]
    try:
        m = shape_inference.infer_shapes(m, strict_mode=False, data_prop=True)
    except Exception as e:
        print(f"  shape inference on {spec['onnx']}: {e}")
    vi = {v.name: v for v in m.graph.value_info}
    for go in m.graph.output:
        src = vi.get(go.name)
        if src is None:
            continue
        go.type.CopyFrom(src.type)
    onnx.save(m, spec["onnx"])
    sym = []
    for io in list(m.graph.input) + list(m.graph.output):
        for dd in io.type.tensor_type.shape.dim:
            if dd.dim_param:
                sym.append(f"{io.name}:{dd.dim_param}")
    done.append({"onnx": spec["onnx"], "unresolved": sym})
    print(f"  froze {spec['onnx']} -- unresolved dims: {sym or 'none'}")
json.dump({"tiles": done}, open(req["result"], "w"), indent=2)
'''


def run_helper(runner: Runner, helper_src: str, req: dict, tag: str) -> dict:
    os.makedirs(LOGS, exist_ok=True)
    hp = os.path.join(LOGS, f"_helper_{tag}.py")
    rp = os.path.join(LOGS, f"_req_{tag}.json")
    resp = os.path.join(LOGS, f"_res_{tag}.json")
    with open(hp, "w") as f:
        f.write(helper_src)
    req = dict(req, result=resp, repo=REPO)
    with open(rp, "w") as f:
        json.dump(req, f, indent=2)
    p = runner.sh(f"{PY} {shlex.quote(hp)} {shlex.quote(rp)}")
    for ln in p.stdout.splitlines():
        print(f"    {ln}")
    with open(resp) as f:
        return json.load(f)


# --------------------------------------------------------------------------
# conversion
# --------------------------------------------------------------------------


def docker_run(runner: Runner, inner: str, timeout: int = 3600) -> subprocess.CompletedProcess:
    cmd = (f"{DOCKER} run --rm -v {REPO}:/work {DOCKER_IMAGE} bash -c "
           f"{shlex.quote(inner)}")
    return runner.sh(cmd, check=False, timeout=timeout)


def convert_tile(runner: Runner, tile: dict, out_dlc: str, force: bool) -> dict:
    """ONNX -> DLC (fp32).  Returns {'ok':bool,'log':str}."""
    log = out_dlc + ".convert.log"
    if os.path.exists(out_dlc) and not force:
        print(f"    [skip] {rel(out_dlc)} exists")
        return {"ok": True, "log": log, "skipped": True}
    dflags = ""
    for i in tile["inputs"]:
        shp = i.get("shape")
        if shp and all(isinstance(x, int) and x > 0 for x in shp):
            dflags += f" -d {shlex.quote(i['dlc_name'])} {','.join(str(x) for x in shp)}"
    inner = (f"python3.10 /qnn/bin/x86_64-linux-clang/snpe-onnx-to-dlc "
             f"--input_network /work/{rel(tile['onnx'])} "
             f"--output_path /work/{rel(out_dlc)}{dflags} 2>&1")
    p = docker_run(runner, inner)
    with open(log, "w") as f:
        f.write(p.stdout + p.stderr)
    ok = os.path.exists(out_dlc)
    if not ok:
        print(f"    convert FAILED, see {rel(log)}")
        print("    " + (p.stdout + p.stderr)[-800:].replace("\n", "\n    "))
    return {"ok": ok, "log": log, "cmd": inner}


def quantize_tile(runner: Runner, dlc: str, out_q: str, input_list: str,
                  force: bool) -> dict:
    log = out_q + ".quant.log"
    if os.path.exists(out_q) and not force:
        print(f"    [skip] {rel(out_q)} exists")
        return {"ok": True, "log": log, "skipped": True}
    inner = (f"/qnn/bin/x86_64-linux-clang/qairt-quantizer "
             f"--input_dlc /work/{rel(dlc)} --output_dlc /work/{rel(out_q)} "
             f"--input_list /work/{rel(input_list)} 2>&1")
    p = docker_run(runner, inner)
    with open(log, "w") as f:
        f.write(p.stdout + p.stderr)
    ok = os.path.exists(out_q)
    if not ok:
        print(f"    quantize FAILED, see {rel(log)}")
        print("    " + (p.stdout + p.stderr)[-800:].replace("\n", "\n    "))
    return {"ok": ok, "log": log, "cmd": inner}


def fix_perms(runner: Runner, d: str) -> None:
    """docker writes converter output as root; hand it back so scp can read it."""
    runner.sh(f"sudo chown -R {os.getuid()}:{os.getgid()} {shlex.quote(d)} "
              f"2>/dev/null || true", check=False, quiet=True)


# --------------------------------------------------------------------------
# board: compose + measure
# --------------------------------------------------------------------------

COMPOSE_ERR_RE = re.compile(
    r"(unsupported op \w+|Op \S+ not supported|validation failed for \S+|"
    r"Param\[\d+\] has incorrect \w+ \S*|Input\[\d+\] has incorrect \S+ \S*|"
    r"Output\[\d+\] has incorrect \S+ \S*|\S+ not supported|"
    r"could not create graph|failed to (?:validate|finalize) \S+)", re.I)


_PREFIX_RE = re.compile(r"^\s*\d+(?:\.\d+)?ms\s*\[\s*\w+\s*\]\s*")


def parse_compose_failure(log: str) -> str:
    """Pull the op the compose log names out of its error lines."""
    lines = [_PREFIX_RE.sub("", l).strip()
             for l in log.splitlines() if "ERROR" in l or "error" in l]
    for l in lines:
        m = COMPOSE_ERR_RE.search(l)
        if m:
            return m.group(0).strip()
    for l in lines[:6]:
        if l:
            return l[:240]
    return "unknown (see log)"


def board_compose_and_measure(runner: Runner, jobs: list[dict], iters_default: int,
                              governor: str = "performance") -> dict:
    """jobs: [{'base':..,'dlc_local':..,'backend':..,'iters':..}].

    One locked ssh round trip: push nothing (caller scp'd already), set the
    governor, compose each job, measure what composed, restore schedutil.
    """
    script_lines = [
        "set +e",
        f"mkdir -p {BOARD_DIR}/ctx {BOARD_DIR}/dlc {BOARD_DIR}/logs",
        f"QNN={QAIRT_BOARD}",
        "export LD_LIBRARY_PATH=$QNN/lib/target",
        'export ADSP_LIBRARY_PATH="$QNN/lib/hexagon-v66;/dsp/cdsp;/dsp"',
        "GOV_OLD=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)",
        f"for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo {governor} > $c; done",
        "echo \"GOVERNOR_SET=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)\"",
    ]
    for j in jobs:
        base, be = j["base"], j["backend"]
        lib = BACKEND_LIB[be]
        ctx = f"ctx_{base}__{be}"
        dlc = f"{BOARD_DIR}/dlc/{os.path.basename(j['dlc_board'])}"
        it = j.get("iters", iters_default)
        script_lines += [
            f'echo "===JOB {base} {be}"',
            f'if [ ! -f {BOARD_DIR}/ctx/{ctx}.bin ]; then',
            f'  timeout -s KILL 900 $QNN/bin/target/qnn-context-binary-generator '
            f'--backend $QNN/lib/target/{lib} --model $QNN/lib/target/libQnnModelDlc.so '
            f'--dlc_path {dlc} --binary_file {ctx} --output_dir {BOARD_DIR}/ctx '
            f'> {BOARD_DIR}/logs/{ctx}.log 2>&1',
            f'  echo "COMPOSE_RC=$?"',
            "else",
            '  echo "COMPOSE_RC=0"; echo "COMPOSE_CACHED=1"',
            "fi",
            f'if [ -f {BOARD_DIR}/ctx/{ctx}.bin ]; then',
            f'  echo "COMPOSE_OK size=$(stat -c%s {BOARD_DIR}/ctx/{ctx}.bin)"',
            f'  echo "CTX_SHA256=$(sha256sum {BOARD_DIR}/ctx/{ctx}.bin | cut -d\" \" -f1)"',
            f'  timeout -s KILL 300 {PROFILE_SEG} {BOARD_DIR}/ctx/{ctx}.bin {lib} {it} 2>/dev/null',
            f'  echo "MEASURE_RC=$?"',
            "else",
            '  echo "COMPOSE_FAIL"',
            f'  grep -E "ERROR|error" {BOARD_DIR}/logs/{ctx}.log 2>/dev/null | head -6',
            "fi",
        ]
    script_lines += [
        "for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo $GOV_OLD > $c; done",
        'echo "GOVERNOR_RESTORED=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"',
    ]
    script = "\n".join(script_lines)
    sp = os.path.join(LOGS, "_board_script.sh")
    with open(sp, "w") as f:
        f.write(script)
    runner.sh(f"scp -q {shlex.quote(sp)} {BOARD}:{BOARD_DIR}/_run.sh", check=True)
    inner = f"timeout -s KILL 3000 sh {BOARD_DIR}/_run.sh"
    p = runner.board(inner, timeout=3600)
    out = p.stdout + p.stderr
    with open(os.path.join(LOGS, "_board_out.txt"), "w") as f:
        f.write(out)
    return parse_board_output(out, jobs)


def parse_board_output(out: str, jobs: list[dict]) -> dict:
    res: dict[str, Any] = {"governor": None, "governor_restored": None, "jobs": {}}
    cur = None
    buf: list[str] = []

    def flush():
        if cur is None:
            return
        text = "\n".join(buf)
        entry: dict[str, Any] = {"raw": text}
        if "COMPOSE_OK" in text:
            entry["compose"] = "ok"
            m = re.search(r"COMPOSE_OK size=(\d+)", text)
            if m:
                entry["ctx_bytes"] = int(m.group(1))
            m = re.search(r"CTX_SHA256=([0-9a-f]{64})", text)
            if m:
                entry["ctx_sha256"] = m.group(1)
            entry["cached"] = "COMPOSE_CACHED=1" in text
            mj = re.search(r'\{"dlc".*?\}', text)
            if mj:
                try:
                    entry["stats"] = json.loads(mj.group(0))
                except json.JSONDecodeError:
                    entry["stats"] = None
            if entry.get("stats") is None:
                entry["measure"] = "failed"
        else:
            entry["compose"] = "fail"
            entry["reason"] = parse_compose_failure(text)
        res["jobs"][cur] = entry

    for line in out.splitlines():
        if line.startswith("GOVERNOR_SET="):
            res["governor"] = line.split("=", 1)[1].strip()
            continue
        if line.startswith("GOVERNOR_RESTORED="):
            res["governor_restored"] = line.split("=", 1)[1].strip()
            continue
        if line.startswith("===JOB "):
            flush()
            _, base, be = line.split()
            cur = f"{base}::{be}"
            buf = []
            continue
        buf.append(line)
    flush()
    return res


# --------------------------------------------------------------------------
# the experiment
# --------------------------------------------------------------------------


def cut_id(src_sha: str, cuts: list[str],
           tile_outputs: list[list[str]] | None = None) -> str:
    h = hashlib.sha256()
    h.update(src_sha.encode())
    for c in cuts:
        h.update(b"\0")
        h.update(c.encode())
    for group in (tile_outputs or []):
        h.update(b"\1")
        for t in group:
            h.update(b"\0")
            h.update(t.encode())
    return h.hexdigest()[:12]


def experiment(net: str, cuts: list[str], name: str | None, backends: list[str],
               iters: int | None, n_samples: int, force: bool,
               precisions: list[str], measure: bool,
               tile_outputs: list[list[str]] | None = None,
               subgraphs: list[dict] | None = None, batch1: int = 0) -> dict:
    spec = NETWORKS[net]
    src = os.path.join(REPO, spec["onnx"])
    if not os.path.exists(src):
        raise SystemExit(f"source ONNX missing: {src}")
    runner = Runner()
    src_sha = sha256(src)
    tile_outputs = tile_outputs or []
    subgraphs = subgraphs or []
    key_extra = [sg["inputs"] + [">"] + sg["outputs"] for sg in subgraphs]
    if batch1:
        key_extra.append([f"batch{int(batch1)}"])
    cid = cut_id(src_sha, cuts, tile_outputs + key_extra)
    n_declared = len(subgraphs) or len(tile_outputs) or (len(cuts) + 1)
    label = name or (f"{net}_k{n_declared}_{cid}")
    work = os.path.join(GEN, net, f"{label}__{cid}")
    os.makedirs(work, exist_ok=True)
    mode = ("subgraph" if subgraphs else "branch" if tile_outputs else "contiguous")
    print(f"\n=== {label}  ({net}, {mode} mode, ~{n_declared} tile(s), cut_id {cid})")
    print(f"    src  {rel(src)}  sha256:{src_sha[:16]}")

    # 1. slice
    sl = run_helper(runner, SLICE_HELPER,
                     {"src": src, "out_dir": work, "cuts": cuts,
                      "tile_outputs": tile_outputs,
                 "subgraphs": subgraphs,
                 "batch": int(batch1), "subgraphs": subgraphs,
                      "batch1": bool(batch1), "batch": int(batch1)},
                     tag=f"slice_{label}")
    tiles = sl["tiles"]

    # 2. calibration
    want = sorted({i["src_name"] for t in tiles for i in t["inputs"]
                   if not i["is_net_input"]})
    cal = run_helper(runner, CALIB_HELPER,
                      {"src": src, "capture": want, "net_inputs": spec["inputs"],
                       "out_dir": os.path.join(work, "calib"),
                       "batch_split": int(batch1),
                       "n_samples": n_samples},
                      tag=f"calib_{label}")

    # A dynamic dim in the source ONNX (ViNT's per-frame batch, say) leaves
    # the tile's declared boundary shape non-concrete; the shape onnxruntime
    # actually produced during calibration capture is the ground truth the
    # converter needs for -d.
    obs_shapes = dict(cal.get("shapes", {}))
    for name, spec_in in spec["inputs"].items():
        obs_shapes.setdefault(name, spec_in["shape"])
    for t in tiles:
        for tin in t["inputs"]:
            shp = tin.get("shape")
            if shp and all(isinstance(x, int) and x > 0 for x in shp):
                continue
            got = obs_shapes.get(tin["src_name"])
            if got:
                tin["shape"] = got
                tin["shape_from"] = "observed during calibration capture"

    # Pin the now-known boundary shapes into the tile ONNX itself: -d alone
    # does not survive a graph whose interior still carries symbolic dims
    # (ViNT's decoder), and the converter refuses it.
    if any(i.get("shape_from") for t in tiles for i in t["inputs"]):
        run_helper(runner, FREEZE_HELPER,
                    {"tiles": [{"onnx": t["onnx"],
                                 "input_shapes": {i["dlc_name"]: i["shape"]
                                                   for i in t["inputs"] if i.get("shape")}}
                                for t in tiles]},
                    tag=f"freeze_{label}")

    # 3/4. per-tile convert + quantize
    for t in tiles:
        t["onnx_sha256"] = sha256(t["onnx"])
        base = f"{label}_t{t['index']}"
        t["base"] = base
        dlc = os.path.join(work, base + ".dlc")
        qdlc = os.path.join(work, base + "_q.dlc")
        print(f"  -- tile{t['index']} {base}")
        c = convert_tile(runner, t, dlc, force)
        t["convert"] = {k: v for k, v in c.items() if k != "log"}
        t["convert_log"] = rel(c["log"])
        if not c["ok"]:
            t["status"] = "convert_failed"
            continue
        fix_perms(runner, work)
        t["dlc"] = dlc
        t["dlc_sha256"] = sha256(dlc)
        # input_list, in tile-input order, name:=path form
        rows = []
        for i in range(cal["n"]):
            parts = []
            for tin in t["inputs"]:
                if tin["is_net_input"]:
                    p = cal["net_sample_paths"][tin["src_name"]][i]
                else:
                    p = cal["captured"][tin["src_name"]][i]
                parts.append(f"{tin['dlc_name']}:=/work/{rel(p)}")
            rows.append(" ".join(parts))
        il = os.path.join(work, base + "_input_list.txt")
        with open(il, "w") as f:
            f.write("\n".join(rows) + "\n")
        t["input_list"] = rel(il)
        q = quantize_tile(runner, dlc, qdlc, il, force)
        t["quantize"] = {k: v for k, v in q.items() if k != "log"}
        t["quantize_log"] = rel(q["log"])
        if q["ok"]:
            fix_perms(runner, work)
            t["qdlc"] = qdlc
            t["qdlc_sha256"] = sha256(qdlc)
            t["status"] = "built"
        else:
            t["status"] = "quantize_failed"

    record: dict[str, Any] = {
        "timestamp": now(),
        "label": label,
        "network": net,
        "cut_id": cid,
        "source_onnx": rel(src),
        "source_onnx_sha256": src_sha,
        "cut": {"boundary_tensors": cuts,
                 "tile_outputs": tile_outputs,
                 "subgraphs": subgraphs,
                 "batch": int(batch1),
                 "mode": sl.get("mode", "contiguous"),
                 "n_tiles": len(tiles),
                 "op_ranges": [t["op_range"] for t in tiles],
                 "op_range_sets": [t.get("ranges", [t["op_range"]]) for t in tiles],
                 "independent_pairs": sl.get("independent_pairs", []),
                 "static_nodes_shared": sl.get("n_static_nodes", 0),
                 "src_node_count": sl["n_src_nodes"]},
        "calibration": {"n_samples": cal["n"],
                         "boundary_tensors_captured": want,
                         "net_inputs": {k: v["samples"] for k, v in spec["inputs"].items()}},
        "tiles": [{k: v for k, v in t.items()
                   if k in ("index", "base", "op_range", "ranges", "depends_on",
                             "n_src_nodes", "n_tile_nodes",
                             "op_types", "inputs", "outputs", "status",
                             "onnx_sha256", "dlc_sha256", "qdlc_sha256",
                             "convert_log", "quantize_log", "input_list")}
                  for t in tiles],
        "artifacts_dir": rel(work),
        "backends_requested": backends,
        "precisions": precisions,
    }

    if not measure:
        record["measurements"] = {"skipped": True}
        record["commands"] = runner.commands
        return record

    # 5. board: stage, compose, measure
    jobs = []
    push = []
    for t in tiles:
        if t.get("status") != "built":
            continue
        for prec in precisions:
            local = t["qdlc"] if prec == "int8" else t["dlc"]
            bn = os.path.basename(local)
            push.append(local)
            for be in backends:
                if prec == "fp32" and be != "cpu":
                    continue  # DSP/HTA need the quantized graph
                jobs.append({"base": f"{t['base']}_{prec}", "backend": be,
                              "dlc_board": bn, "tile": t["index"], "precision": prec,
                              "iters": iters or spec.get("iters", 40)})
    if not jobs:
        record["measurements"] = {"error": "no tile built"}
        record["commands"] = runner.commands
        return record

    runner.sh(f"ssh -o ConnectTimeout=15 {BOARD} "
              f"{shlex.quote(f'mkdir -p {BOARD_DIR}/dlc {BOARD_DIR}/ctx {BOARD_DIR}/logs')}")
    for p in sorted(set(push)):
        runner.sh(f"scp -q {shlex.quote(p)} {BOARD}:{BOARD_DIR}/dlc/")
    board = board_compose_and_measure(runner, jobs, iters or spec.get("iters", 40))

    cells: dict[str, Any] = {}
    for j in jobs:
        key = f"{j['base']}::{j['backend']}"
        e = board["jobs"].get(key, {"compose": "missing"})
        cells[key] = {
            "tile": j["tile"], "backend": j["backend"], "precision": j["precision"],
            "iters": j["iters"], "compose": e.get("compose"),
            "reason": e.get("reason"), "ctx_bytes": e.get("ctx_bytes"),
            "ctx_sha256": e.get("ctx_sha256"), "cached": e.get("cached"),
            "stats": e.get("stats"),
        }
    record["measurements"] = {
        "governor": board.get("governor"),
        "governor_restored": board.get("governor_restored"),
        "cpu_affinity": "unmasked (no taskset) -- the QNN CPU op package's "
                          "thread pool is not confined by the lane mask",
        "harness": "/root/qnn_runtime/profile_seg (qnn_models/runtime/profile_segments.cpp)",
        "cells": cells,
    }
    record["commands"] = runner.commands
    return record


# --------------------------------------------------------------------------
# per-dispatch overhead probes
# --------------------------------------------------------------------------

PROBE_HELPER = r'''
import json, os, sys
import numpy as np
import onnx
from onnx import helper, TensorProto

req = json.load(open(sys.argv[1]))
out_dir = req["out_dir"]
os.makedirs(out_dir, exist_ok=True)
made = []
for h, w in req["sizes"]:
    name = f"probe_{h}x{w}"
    wt = helper.make_tensor("W", TensorProto.FLOAT, [1, 1, 1, 1], [1.0])
    bt = helper.make_tensor("B", TensorProto.FLOAT, [1], [0.0])
    node = helper.make_node("Conv", ["x", "W", "B"], ["y"], name="probe_conv",
                             kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0])
    g = helper.make_graph([node], name,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, h, w])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, h, w])],
        [wt, bt])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    p = os.path.join(out_dir, name + ".onnx")
    onnx.save(m, p)
    cdir = os.path.join(out_dir, name + "_cal")
    os.makedirs(cdir, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(req["n_samples"]):
        a = rng.standard_normal((1, 1, h, w)).astype(np.float32)
        f = os.path.join(cdir, f"s{i}.raw")
        a.tofile(f)
        rows.append(f"x:=/work/{os.path.relpath(f, req['repo'])}")
    il = os.path.join(out_dir, name + "_input_list.txt")
    open(il, "w").write("\n".join(rows) + "\n")
    made.append({"name": name, "h": h, "w": w, "bytes_int8": h * w,
                  "onnx": p, "input_list": il})
json.dump({"probes": made}, open(req["result"], "w"), indent=2)
print(f"built {len(made)} probe graph(s)")
'''


def overhead_probe(sizes: list[tuple[int, int]], iters: int, force: bool) -> dict:
    runner = Runner()
    work = os.path.join(GEN, "_overhead")
    os.makedirs(work, exist_ok=True)
    res = run_helper(runner, PROBE_HELPER,
                      {"out_dir": work, "sizes": [list(s) for s in sizes],
                       "n_samples": 8}, tag="probe")
    probes = res["probes"]
    jobs, push = [], []
    for p in probes:
        dlc = os.path.join(work, p["name"] + ".dlc")
        qdlc = os.path.join(work, p["name"] + "_q.dlc")
        tile = {"onnx": p["onnx"],
                 "inputs": [{"dlc_name": "x", "shape": [1, 1, p["h"], p["w"]]}]}
        c = convert_tile(runner, tile, dlc, force)
        if not c["ok"]:
            p["status"] = "convert_failed"
            continue
        fix_perms(runner, work)
        q = quantize_tile(runner, dlc, qdlc, p["input_list"], force)
        fix_perms(runner, work)
        if not q["ok"]:
            p["status"] = "quantize_failed"
            continue
        p["status"] = "built"
        p["dlc_sha256"] = sha256(dlc)
        p["qdlc_sha256"] = sha256(qdlc)
        push += [dlc, qdlc]
        for be in ("hta", "dsp", "cpu"):
            jobs.append({"base": p["name"] + "_int8", "backend": be,
                          "dlc_board": os.path.basename(qdlc), "iters": iters,
                          "tile": 0, "precision": "int8"})
        jobs.append({"base": p["name"] + "_fp32", "backend": "cpu",
                      "dlc_board": os.path.basename(dlc), "iters": iters,
                      "tile": 0, "precision": "fp32"})
    runner.sh(f"ssh -o ConnectTimeout=15 {BOARD} "
              f"{shlex.quote(f'mkdir -p {BOARD_DIR}/dlc {BOARD_DIR}/ctx {BOARD_DIR}/logs')}")
    for p in sorted(set(push)):
        runner.sh(f"scp -q {shlex.quote(p)} {BOARD}:{BOARD_DIR}/dlc/")
    board = board_compose_and_measure(runner, jobs, iters)
    cells = {}
    for j in jobs:
        key = f"{j['base']}::{j['backend']}"
        e = board["jobs"].get(key, {"compose": "missing"})
        cells[key] = {"backend": j["backend"], "precision": j["precision"],
                       "iters": j["iters"], "compose": e.get("compose"),
                       "reason": e.get("reason"), "stats": e.get("stats")}
    return {
        "timestamp": now(),
        "label": "overhead_probe",
        "network": "_synthetic",
        "cut_id": "probe",
        "source_onnx": rel(work),
        "source_onnx_sha256": "",
        "cut": {"boundary_tensors": [], "n_tiles": 1, "op_ranges": [[0, 0]],
                 "src_node_count": 1},
        "probes": probes,
        "measurements": {"governor": board.get("governor"),
                          "governor_restored": board.get("governor_restored"),
                          "cpu_affinity": "unmasked",
                          "harness": "/root/qnn_runtime/profile_seg",
                          "cells": cells},
        "commands": runner.commands,
        "note": ("Single 1x1x1x1 Conv (one MAC per pixel, one channel) on an "
                  "HxW input. The intercept over tensor size is the per-dispatch "
                  "overhead; the slope is the per-byte boundary cost."),
    }


# --------------------------------------------------------------------------


def append_record(rec: dict) -> None:
    with open(JOURNAL, "a") as f:
        f.write(json.dumps(rec, sort_keys=False) + "\n")
    print(f"    -> appended {rec['label']} to {rel(JOURNAL)}")


def list_nodes(net: str) -> None:
    spec = NETWORKS[net]
    src = os.path.join(REPO, spec["onnx"])
    runner = Runner()
    helper = r'''
import sys, onnx
from onnx import shape_inference
m = onnx.load(sys.argv[1])
try: m = shape_inference.infer_shapes(m, strict_mode=False)
except Exception: pass
g = m.graph
vi = {v.name: v for v in list(g.value_info)+list(g.output)+list(g.input)}
def shp(t):
    v = vi.get(t)
    if v is None: return "?"
    return "x".join(str(d.dim_value or d.dim_param or "?") for d in v.type.tensor_type.shape.dim)
init = {i.name for i in g.initializer}
prod = {o: i for i, n in enumerate(g.node) for o in n.output}
for i, n in enumerate(g.node):
    ins = ",".join(str(prod.get(x, "IN")) for x in n.input if x and x not in init)
    for o in n.output:
        print(f"{i:5d} {n.op_type:16s} <-{ins:18s} {o}  [{shp(o)}]")
'''
    hp = os.path.join(LOGS, "_list_nodes.py")
    os.makedirs(LOGS, exist_ok=True)
    with open(hp, "w") as f:
        f.write(helper)
    p = runner.sh(f"{PY} {shlex.quote(hp)} {shlex.quote(src)}", quiet=True)
    print(p.stdout)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--network", choices=sorted(NETWORKS))
    ap.add_argument("--cut", nargs="*", default=[],
                    help="contiguous mode: ordered boundary tensor names; empty = whole network")
    ap.add_argument("--tile", action="append", default=[], metavar="OUT[,OUT...]",
                    help="branch mode (repeatable): the tensors THIS tile must produce. "
                          "The tile claims the backward closure of them minus what earlier "
                          "--tile entries claimed; leftovers become a final tile. Tiles are "
                          "sets of ops, so independent branches stay independent.")
    ap.add_argument("--subgraph", action="append", default=[], metavar="IN[,IN]:OUT[,OUT]",
                    help="subgraph mode (repeatable): one tile named by BOTH its input and "
                          "its output tensors. The only way to name a tile that starts in the "
                          "middle of the graph.")
    ap.add_argument("--batch1", type=int, default=0, metavar="N",
                    help="with --subgraph: rewrite the leading dim of every tile boundary to N "
                          "and chunk captured calibration along it, turning a batched tile into "
                          "the batch-N tile the runtime would dispatch B/N times.")
    ap.add_argument("--name", default=None, help="label for this slice set")
    ap.add_argument("--backends", default="hta,dsp,cpu")
    ap.add_argument("--precisions", default="int8,fp32")
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--samples", type=int, default=8, help="calibration samples")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-measure", action="store_true",
                    help="build artifacts only; do not touch the board")
    ap.add_argument("--list-nodes", action="store_true")
    ap.add_argument("--overhead-probe", action="store_true")
    ap.add_argument("--probe-sizes", default="1x1,64x64,256x256,512x512,1024x1024")
    args = ap.parse_args()

    os.makedirs(GEN, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)

    if args.overhead_probe:
        sizes = [tuple(int(x) for x in s.split("x")) for s in args.probe_sizes.split(",")]
        rec = overhead_probe(sizes, args.iters or 200, args.force)
        append_record(rec)
        return
    if not args.network:
        ap.error("--network is required (or use --overhead-probe)")
    if args.list_nodes:
        list_nodes(args.network)
        return
    cuts = [c for c in args.cut if c.strip()]
    tile_outputs = [[t.strip() for t in grp.split(",") if t.strip()]
                    for grp in args.tile]
    subgraphs = []
    for spec in args.subgraph:
        ins, _, outs = spec.partition(":")
        subgraphs.append({"inputs": [t.strip() for t in ins.split(",") if t.strip()],
                           "outputs": [t.strip() for t in outs.split(",") if t.strip()]})
    if sum(bool(x) for x in (cuts, tile_outputs, subgraphs)) > 1:
        ap.error("--cut, --tile and --subgraph are different modes; pass one")
    if args.batch1 and not subgraphs:
        ap.error("--batch1 only applies to --subgraph mode")
    rec = experiment(args.network, cuts, args.name,
                      [b for b in args.backends.split(",") if b],
                      args.iters, args.samples, args.force,
                      [p for p in args.precisions.split(",") if p],
                      measure=not args.no_measure, tile_outputs=tile_outputs,
                      subgraphs=subgraphs, batch1=args.batch1)
    append_record(rec)
    m = rec.get("measurements", {})
    if "cells" in m:
        print("\n  measured:")
        for k, c in m["cells"].items():
            if c["compose"] == "ok" and c.get("stats"):
                s = c["stats"]
                print(f"    {k:56s} mean {s['mean_us']:10.2f} us  "
                      f"median {s['median_us']:10.2f}  min {s['min_us']:10.2f}")
            else:
                print(f"    {k:56s} {c['compose']}: {c.get('reason')}")


if __name__ == "__main__":
    main()
