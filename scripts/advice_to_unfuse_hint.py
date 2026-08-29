#!/usr/bin/env python3
"""Translate `unfuse` advice into a ModelBlaster unfuse hint.

The last of the four bridges. `unfuse_advice` and `apply_unfuse_hint` both
existed; nothing connected them, so the verb could be emitted and applied but
never by the loop itself.

WHAT MAKES THIS BRIDGE DIFFERENT FROM THE OTHERS
------------------------------------------------
Fusion and split advice are about TIME: something is too fine or too coarse for
the schedule. Unfuse advice is about a KERNEL LOOKUP, and the trigger is a
failure this project actually shipped. Curated kernels are matched by exact op
name, so a fused op like `conv2d_batchnorm2d_silu_s8` matches no
per-constituent kernel and silently falls back to the scalar reference inside a
build labelled `rvv_x60`. Measured, before the curated fused kernel existed:

    yolov8_nano  rvv_x60   57 of 90 dispatches on reference, 99.8% of the
                           4974.8 ms total -- 0.81x against pure scalar

So the bridge does not re-derive the decision. `unfuse_advice` already gates on
the two measured facts -- the fused op ran `reference/*`, and every constituent
has a curated kernel -- and that gate is narrow on purpose: unfusing a WORKING
fused kernel is a large loss (`conv2d_batchnorm2d_silu_s8` is 97% of yolov8n's
runtime and applies BN and SiLU inside the conv's register tile). This bridge's
job is to refuse to widen it.

WHAT IT REFUSES
---------------
  * an op the IR does not record as fused (`sub_ops` shorter than 2) -- the
    rewriter would refuse it, and the reason belongs next to the advice
  * a graph the advice did not come from (`advice_join.verify_graph_identity`)
  * advice whose evidence does not name a reference fallback: unfusing on any
    other basis is the 0.81x result, deliberately reproduced
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "xpu-rt"))

from advice_join import verify_graph_identity  # noqa: E402

CONTRACT = "modelblaster.unfuse_hints/v1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--advice", required=True)
    ap.add_argument("--ir", required=True, help="ModelBlaster graph.json")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-identity-check", action="store_true")
    a = ap.parse_args()

    advice = json.load(open(a.advice))
    model_items = [x for x in advice.get("advice", [])
                   if x.get("model") == a.model]
    graph = json.load(open(a.ir))
    by_id = {o["dispatch_id"]: o for o in graph["ops"]
             if o.get("dispatch_id") is not None}

    n_checked, disagreements = verify_graph_identity(model_items, by_id)
    if disagreements:
        print(f"REFUSED: the advice did not come from {a.ir}. "
              f"{len(disagreements)} of {n_checked} checkable dispatch(es) "
              f"disagree:", file=sys.stderr)
        for d in disagreements[:5]:
            print(f"  {d}", file=sys.stderr)
        return 2
    if n_checked == 0 and not a.skip_identity_check:
        print(f"REFUSED: nothing in the advice confirms it came from {a.ir}.",
              file=sys.stderr)
        return 2

    wants = [x for x in model_items if x.get("recommendation") == "unfuse"]
    if not wants:
        print(f"no unfuse advice for {a.model}; nothing to do", file=sys.stderr)
        return 1

    unfuse_ops, notes, refused = [], [], 0
    for x in wants:
        did = x.get("dispatch_id")
        ev = x.get("evidence") or {}
        if did not in by_id:
            print(f"REFUSED dispatch {did}: not in {a.ir}", file=sys.stderr)
            refused += 1
            continue
        op = by_id[did]
        subs = op.get("sub_ops") or []
        if len(subs) < 2:
            print(f"REFUSED dispatch {did}: op {op['op']!r} records "
                  f"{len(subs)} sub_ops, so there is no fusion to undo",
                  file=sys.stderr)
            refused += 1
            continue
        fused_impl = str(ev.get("fused_impl", ""))
        trigger = str(ev.get("trigger", "") or "reference_fallback")
        if fused_impl.split("/")[0] != "reference" and trigger != "runtime_share_probe":
            # Two justifications are accepted, and nothing else:
            #
            #   reference_fallback   the fused op matched no curated kernel and
            #                        silently ran the scalar reference. Measured
            #                        before the curated fused kernel existed: 57
            #                        of 90 dispatches, 99.8% of runtime, 0.81x.
            #   runtime_share_probe  the fused KIND dominates the model and all
            #                        constituents have curated kernels, so
            #                        whether the fused kernel wins is a question
            #                        for the board. Measured on yolov8_nano's
            #                        SHIPPING build: unfusing is -19.1% and was
            #                        accepted at term 5.
            #
            # This refusal used to have no second branch, and that is why the
            # 19% was invisible: the belief that a curated fused kernel beats
            # its constituents was encoded in the producer AND here, so fixing
            # only the producer left the bridge still refusing.
            print(f"REFUSED dispatch {did}: evidence says the fused op ran "
                  f"{fused_impl!r} and carries no runtime-share probe. "
                  f"Unfusing on any other basis loses the epilogue fusion and "
                  f"doubles the dispatch count.", file=sys.stderr)
            refused += 1
            continue
        unfuse_ops.append({"op": did})
        notes.append({"dispatch_id": did, "op": op["op"],
                      "n_constituents": len(subs),
                      "constituents": [s.get("op") for s in subs],
                      "fused_impl": fused_impl,
                      "trigger": trigger,
                      "constituent_impls": ev.get("constituent_impls"),
                      "service_time_us": ev.get("service_time_us")})

    if not unfuse_ops:
        print(f"no emittable unfuse for {a.model} "
              f"({refused} refused, {len(wants)} advised)", file=sys.stderr)
        return 1

    hint = {
        "contract": CONTRACT,
        "reason": "; ".join(
            f"dispatch {n['dispatch_id']} ({n['op']}) ran {n['fused_impl']} "
            f"while all {n['n_constituents']} constituents have curated kernels"
            for n in notes),
        "networks": [{"network": a.model, "unfuse_ops": unfuse_ops}],
        "_provenance": {
            "from_advice": advice.get("schedule_id"),
            "identity_checked_dispatches": n_checked,
            "refused": refused,
            "derivation": notes,
            "invariant": ("a rewrite may not change modelled work unless "
                          "kernels exist for the result -- run "
                          "check_kernel_coverage on the rewritten graph before "
                          "building"),
        },
    }
    with open(a.out, "w") as f:
        json.dump(hint, f, indent=1)
    print(f"wrote {a.out}")
    for n in notes:
        print(f"  unfuse dispatch {n['dispatch_id']} {n['op']} -> "
              f"{n['n_constituents']} ops ({', '.join(n['constituents'])}); "
              f"ran {n['fused_impl']}")
    if refused:
        print(f"  {refused} advised dispatch(es) refused; see above")
    return 0


if __name__ == "__main__":
    sys.exit(main())
