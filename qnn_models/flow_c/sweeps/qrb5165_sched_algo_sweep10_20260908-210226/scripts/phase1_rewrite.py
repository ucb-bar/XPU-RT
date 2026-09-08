#!/usr/bin/env python3
"""Phase 1R — remove each measured backend blocker with a graph rewrite, build
the rewritten variant, and let the cost model take whichever MEASURES better.

Phase 1 built every network whole and recorded three HTA blockers with the
verbatim validator string. `qnn_models/PARTITIONING_GUIDE.md` §5 says a rewrite
is the first move and a cut is the fallback -- on dronet, full HTA after the BN
rewrite measured 4.14 ms against 7.23 ms for the best cut and 31.89 ms full DSP
-- so each blocker is removed here rather than routed around.

    base network        blocker                       chain
    dronet_s[a-g]       QnnHtaunsupported op          onnxsim, bn_to_mul_add,
                        Batchnorm                     flatten_gemm_to_conv
    yolov8_nano_s[cefh] QnnHtaunsupported op          onnxsim,
                        StridedSlice                  channel_slice_to_conv1x1
    mlp_control_sf      unsupported elementwise       onnxsim, elu_to_relu
                        neuson op 0                   (NUMERICS-CHANGING probe)

Every variant is built exactly as the base was -- fresh ONNX, fresh DLC, fresh
calibration, fresh qairt-quantizer pass (PARTITIONING_GUIDE §8 rule 5: never
splice pre-quantized constants into a rewritten graph) -- and composed on all
five (backend, precision) pairs, so the before/after comparison is like for
like.

A variant is NOT adopted because it composes somewhere new. `phase1_adopt.py`
takes the per-lane minimum over {base, variant} on measured evidence, and
records which variant each cell came from and whether that variant is
numerics-preserving.

    python3 phase1_rewrite.py [--only dronet_sc] [--force]
"""
from __future__ import annotations

import argparse, importlib.util, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import onnx_rewrites                                              # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "pb", os.path.join(HERE, "phase1_build.py"))
pb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pb)

STATE = os.path.join(SWEEP, "results", "phase1r_state.json")
OUT = os.path.join(SWEEP, "results", "phase1r_variants.json")

#: base network -> (variant suffix, rewrite chain, why)
PLAN = {}
for k in "abcdefg":
    PLAN[f"dronet_s{k}"] = (
        "hta", ["onnxsim", "bn_to_mul_add", "flatten_gemm_to_conv"],
        "HTA rejects Batchnorm; dronet's BNs follow the residual Add so there "
        "is nothing to fold them into, and the Flatten before the two heads is "
        "what the converter turns into the Transpose HTA also rejects.")
for k in ("sc", "se", "sf", "sh"):
    PLAN[f"yolov8_nano_{k}"] = (
        "hta", ["onnxsim", "channel_slice_to_conv1x1"],
        "HTA rejects StridedSlice, which is how the converter lowers the C2f "
        "channel chunk that modelblaster exports as a Slice pair. The 1x1 "
        "selector convolution computes the same tensor with an op HTA has.")
#: ONE numerics-changing probe, on the largest rung, to answer "would HTA ever
#: be worth it for the control loop?" by measurement instead of by argument.
PLAN["mlp_control_sf"] = (
    "htaprobe", ["onnxsim", "elu_to_relu"],
    "NUMERICS-CHANGING probe. Elu has no frozen-constant linear equivalent, so "
    "substituting the activation is the only way mlp_control reaches HTA at "
    "all. Run on the largest rung only: if HTA loses there it loses on every "
    "smaller one, since HTA's dispatch floor is fixed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-board", action="store_true")
    a = ap.parse_args()
    want = set(a.only.split(",")) if a.only else None
    st = json.load(open(STATE)) if os.path.exists(STATE) else {}
    base_rows = {r["id"]: r for r in
                 json.load(open(os.path.join(SWEEP, "results",
                                             "phase1_compose.json")))}
    todo = [(b, v) for b, v in PLAN.items() if not want or b in want]
    print(f"{len(todo)} variant(s)")
    for base, (sfx, chain, why) in todo:
        vid = f"{base}_{sfx}"
        rec = st.setdefault(vid, {})
        if rec.get("stage") == "composed" and not a.force:
            ok = sorted(k for k, v in (rec.get("compose") or {}).items()
                        if v.get("status") == "ok")
            print(f"  {vid:<26} already built (ok={ok}) — skipping")
            continue
        t0 = time.time()
        rec.update(id=vid, base=base, family=base_rows[base]["family"],
                   rewrite_chain=chain, rationale=why,
                   zoo=base_rows[base]["zoo"])
        src = os.path.join(SWEEP, base_rows[base]["onnx"])
        dst = os.path.join(pb.ONNX_DIR, f"{vid}.onnx")
        print(f"  {vid:<26} rewrite…", end="", flush=True)
        try:
            report, label = onnx_rewrites.apply_chain(src, dst, chain)
        except Exception as e:
            rec["stage"] = "rewrite_failed"
            rec["error"] = f"{type(e).__name__}: {e}"
            json.dump(st, open(STATE, "w"), indent=1)
            print(f" FAILED {e}")
            continue
        rec["rewrite"] = report
        rec["numerics"] = label
        rec["onnx"] = os.path.relpath(dst, SWEEP)
        rec["onnx_bytes"] = os.path.getsize(dst)
        rec["inputs"] = base_rows[base]["inputs"]
        # The rewritten graph is no longer modelblaster's PyTorch IR, so its IR
        # comes through the graph_json door -- the same door the shipped
        # yolov8n binding uses, and for the same reason.
        irdir = os.path.join(pb.IR_DIR, vid, "int8")
        os.makedirs(irdir, exist_ok=True)
        p = pb.run([sys.executable, os.path.join(HERE, "ir_from_onnx.py"),
                       "--onnx", dst, "--out", os.path.join(irdir, "graph.json"),
                       "--name", vid], f"ir_{vid}.log", timeout=1800)
        if p.returncode != 0:
            rec["stage"] = "ir_failed"
            rec["error"] = (p.stderr or "")[-300:]
            json.dump(st, open(STATE, "w"), indent=1)
            print(" IR FAILED")
            continue
        d = json.load(open(os.path.join(irdir, "graph.json")))
        rec["graph_json"] = os.path.relpath(os.path.join(irdir, "graph.json"), SWEEP)
        rec["ir_ops"] = len(d["ops"])
        from collections import Counter
        rec["ir_op_kinds"] = dict(Counter(o["op"] for o in d["ops"]))
        rec["ir_source"] = ("scripts/ir_from_onnx.py (graph_json door: the "
                            "rewritten graph is no longer the PyTorch IR)")
        print(" dlc…", end="", flush=True)
        m = {"id": vid}
        if not pb.onnx_to_dlc(m, rec):
            rec["stage"] = "dlc_failed"
            json.dump(st, open(STATE, "w"), indent=1)
            print(" FAILED")
            continue
        print(" quant…", end="", flush=True)
        pb.calibrate_and_quantize(m, rec)
        rec["stage"] = "converted"
        json.dump(st, open(STATE, "w"), indent=1)
        if a.skip_board:
            print("  (board skipped)")
            continue
        print(" compose…", end="", flush=True)
        pb.compose_on_board(m, rec)
        rec["stage"] = "composed"
        rec["build_s"] = round(time.time() - t0, 1)
        json.dump(st, open(STATE, "w"), indent=1)
        cm = rec.get("compose") or {}
        good = sorted(k for k, v in cm.items() if v.get("status") == "ok")
        bad = sorted(k for k, v in cm.items() if v.get("status") != "ok")
        print(f" ok={good} fail={bad}  {rec['ir_ops']} ops  "
              f"[{rec['numerics']}]  {rec['build_s']}s")
    rows = [st[f"{b}_{PLAN[b][0]}"] for b in PLAN if f"{b}_{PLAN[b][0]}" in st]
    json.dump(rows, open(OUT, "w"), indent=1)
    nh = sum(1 for r in rows if ((r.get("compose") or {}).get("hta") or {})
             .get("status") == "ok")
    print(f"\n  {len(rows)} variants; {nh} now compose on HTA -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
