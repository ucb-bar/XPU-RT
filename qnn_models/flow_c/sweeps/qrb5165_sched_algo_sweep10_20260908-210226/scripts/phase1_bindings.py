#!/usr/bin/env python3
"""Phase 1b — write a Flow C binding manifest per network from what actually
composed, and record what did not.

The manifest is generated from the board's own compose verdicts, never from an
assumption about what a backend supports. That is the same discipline
`gen_smolvla_binding.py` uses (it builds its manifest from the ctx inventory on
the board) and it is the reason a manifest here can't declare a lane the board
would reject at bringup.

One tile per network (`ops: "all"`): Phase 1 built each network as a single
graph. See mk_workloads_qrb5165.py's docstring for why there is no split arm.

`cpu` is declared at fp32, not int8, even where the int8 context composed:

  * QnnCpu's int8 conv is a reference kernel (12.2 ms for fused_split's vision
    branch against 0.014 ms for its 8x8 depth branch -- flow_c/README.md), so
    the int8 CPU cell describes a kernel nobody would deploy; and
  * for mlp_control the int8 CPU path is NUMERICALLY DEAD -- three different
    inputs give byte-identical outputs, verified with qnn-net-run and recorded
    in measurements/qrb5165_v66.json. Since mlp_control rungs are 4 of the 16
    networks here, taking int8 on the CPU lane would put a wrong number in the
    cost model for a quarter of the set.

The int8 CPU compose result is still recorded (as backend `cpu@int8`) and its
cell is still measured, so the choice is auditable rather than asserted.

    python3 phase1_bindings.py [--write]
"""
from __future__ import annotations

import argparse, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.abspath(os.path.join(HERE, ".."))
COMPOSE = os.path.join(SWEEP, "results", "phase1_compose.json")
BDIR = os.path.join(SWEEP, "bindings")

#: registry kind -> the backend key in the compose record whose context the
#: manifest declares for that kind.
LANE_SOURCE = {"hta": "hta", "dsp": "dsp", "cpu": "cpu", "gpu": "gpu"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--compose", default=COMPOSE)
    ap.add_argument("--candidates", action="store_true",
                    help="also emit a manifest per rewritten variant, so every "
                         "candidate lane gets a measured cell before anything "
                         "is adopted")
    a = ap.parse_args()
    rows = json.load(open(a.compose))
    if a.candidates:
        vp = os.path.join(SWEEP, "results", "phase1r_variants.json")
        if os.path.exists(vp):
            rows = list(rows) + json.load(open(vp))
    if isinstance(rows, dict):
        # phase1_state.json is a dict keyed by model id and is written after
        # EVERY model, so it can be read while a build is still in flight;
        # phase1_compose.json is the flat record written at the end.
        rows = [v for v in rows.values() if v.get("stage") == "composed"]
    os.makedirs(BDIR, exist_ok=True)
    n, fails = 0, []
    print(f"  {'network':<18}{'ir ops':>7}{'macs':>12}  declared lanes")
    print("  " + "-" * 78)
    for r in rows:
        net = r["id"]
        cm = r.get("compose") or {}
        backends = {}
        for kind, src in LANE_SOURCE.items():
            v = cm.get(src) or {}
            if v.get("status") == "ok":
                backends[kind] = {"ctx": v["ctx"], "graph": r["dlc_graph_name"],
                                  "precision": v["precision"],
                                  "ctx_bytes": v["bytes"]}
            else:
                fails.append(dict(cell=f"{net}/{net}_full", backend=kind,
                                  precision=v.get("precision"),
                                  reason=v.get("reason") or v.get("status")
                                  or "not attempted"))
        alt = cm.get("cpu@int8") or {}
        if alt.get("status") == "ok":
            if a.candidates:
                # A candidate manifest declares int8-on-CPU as its own lane so
                # phase1_measure gives it a cell; phase1_adopt then picks the
                # CPU precision per tile on the measured numbers. The study this
                # follows found int8 beats fp32 by 4x on ViNT's encoders and
                # loses by 5x on dronet -- precision is not a per-network
                # constant.
                backends["cpu@int8"] = {
                    "ctx": alt["ctx"], "graph": r["dlc_graph_name"],
                    "precision": "int8", "ctx_bytes": alt["bytes"]}
        else:
            fails.append(dict(cell=f"{net}/{net}_full", backend="cpu@int8",
                              precision="int8",
                              reason=alt.get("reason") or alt.get("status")
                              or "not attempted"))
        doc = {
            "network": net,
            "ir": {"source": f"graph_json:{os.path.join(SWEEP, r['graph_json'])}",
                   "quant": "int8"},
            "_comment": (
                f"One tile, the whole network ({r['ir_ops']} IR ops, "
                f"{r.get('macs')} MACs). @generated by phase1_bindings.py from "
                f"the board's own compose verdicts (results/phase1_compose.json) "
                f"— a lane appears here only if qnn-context-binary-generator "
                f"returned rc=0 for it on this board. Model is "
                f"modelblaster.models.{net} from the zoo at {r['zoo']}, exported "
                f"to ONNX at opset 17 and converted with snpe-onnx-to-dlc; the "
                f"int8 lanes come from the qairt-quantizer output over "
                f"{(r.get('calib') or {}).get('n')} seeded synthetic "
                f"calibration samples."),
            "_scaled_variant_note": (
                "dronet and mlp_control non-default rungs keep SEEDED RANDOM "
                "INIT — the trained checkpoint only fits the default geometry — "
                "so they are valid latency and scheduling targets and must "
                "never be quoted as an accuracy result. yolov8_nano rungs are "
                "the exception: channel counts are input-size independent, so "
                "get_model() loads the COCO checkpoint at every rung. fastdepth "
                "has no committed checkpoint at all and is always "
                "seeded-random."),
            "bindings": [{
                "id": 0,
                "name": f"{net}_full",
                "ops": "all",
                "source_onnx": r["onnx"],
                "backends": backends,
            }],
        }
        if not backends:
            doc["_no_lane_note"] = ("Nothing composed. This network cannot be "
                                    "scheduled on this board at all.")
        print(f"  {net:<18}{r['ir_ops']:>7}{str(r.get('macs')):>12}  "
              f"{sorted(backends) or 'NONE'}")
        if a.write:
            with open(os.path.join(BDIR, f"{net}.json"), "w") as f:
                json.dump(doc, f, indent=1)
            n += 1
    if a.write:
        with open(os.path.join(SWEEP, "results", "compose_failures.json"), "w") as f:
            json.dump(fails, f, indent=1)
    print(f"\n  {n} manifest(s) -> {BDIR}" if a.write else
          "\n  (dry run — pass --write)")
    print(f"  {len(fails)} (tile, backend) compose failures recorded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
