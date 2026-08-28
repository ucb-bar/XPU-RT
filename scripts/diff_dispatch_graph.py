#!/usr/bin/env python3
"""Prove a hint rewrite actually changed dispatch granularity. The gate.

Why this exists, precisely
--------------------------
`gen/vmfb/mlp/spacemit_x60/RVV_fused/` contains *the same five dispatches with
the same names* as the baseline it was supposed to have fused. Nothing was
fused; the build was labelled `RVV_fused`, profiled, and the numbers were
recorded as a fusion result. Nobody looked at the graph. So this runs BEFORE any
profiling and refuses to pass a rewrite that did not change anything:

    exit 0  the graph changed -- go and profile it
    exit 3  the graph is identical -- a negative result, report it as one
    exit 4  the rewriter's `id_remap` disagrees with the signatures

Joins on the OP SIGNATURE, never on `dispatch_id`
-------------------------------------------------
`apply_fusion_hint` and `apply_split_hint` both reassign ids contiguously, so
fusing ops 0+1 shifts every later op down even though it was untouched, and
splitting shifts them up. A `dispatch_id` join across a rewrite therefore
compares different ops and reports differences that are pure renumbering.

For a `results.csv` this delegates to `dispatch_lineage.op_signature`, which is
the repo's one definition of "the part of a module name a renumbering cannot
change" -- it strips *every* occurrence of the index, because the IREE-era
profiler embeds it twice. A second parser here would be a second answer to the
same question.

An IR `graph.json` has no `module_name` yet -- it is assigned later, by
`pipeline/profile_writer._module_name`, as
`<model>$dispatch_<id>_<backend>_<op>_<shapetag>`. So for that input the
signature is rebuilt from the op's own `op` + `shape` fields, which is exactly
what will land in the module name. The two forms are not interchangeable (the
profile signature carries a backend tag and the IR one does not), which is why
`--before` and `--after` must be the same kind of file: comparing an IR against
a profile would report every op as both added and removed.

Signatures are compared as MULTISETS. Three identical maxpools in yolov8_nano
share one signature, and collapsing them to a set would hide two of them.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "xpu-rt"))

from dispatch_lineage import (  # noqa: E402
    _unwrap_singleton, check_id_remap, op_signature,
)


def _shape_tag(shape) -> str:
    """Mirror `profile_writer._shape_concise` closely enough to join on it."""
    if isinstance(shape, dict):
        parts = [f"{k}{v}" for k, v in shape.items()
                 if not isinstance(v, (list, dict))]
        return "x".join(parts) if parts else "noshape"
    if isinstance(shape, str) and shape:
        return "x".join(kv.replace("=", "") for kv in shape.split(";") if kv)
    return "noshape"


def signatures_from_ir(path: str) -> list[str]:
    """Op signatures from a ModelBlaster `graph.json` / `graph.fused.json`.

    A fused op's signature includes its members, so `linear_s8_elu_s8` over
    (M1xK16xN256, n256) cannot collide with the unfused pair it replaced.
    """
    g = json.load(open(path))
    out = []
    for op in g.get("ops", []):
        if op.get("dispatch_id") is None:
            continue          # view op: no dispatch, no cost, no signature
        sig = f"{op['op']}_{_shape_tag(op.get('shape'))}"
        subs = op.get("sub_ops") or []
        if subs:
            sig += "[" + "+".join(
                f"{s['op']}_{_shape_tag(s.get('shape'))}" for s in subs) + "]"
        out.append(sig)
    return out


def signatures_from_profile(path: str) -> list[str]:
    """Op signatures from a `results.csv` `module_name` column."""
    import csv
    out = []
    for row in csv.DictReader(open(path, newline="")):
        sig = op_signature(row.get("module_name", "") or "")
        if sig:
            out.append(sig)
    return out


def _load(path: str) -> list[str]:
    return (signatures_from_profile(path) if path.endswith(".csv")
            else signatures_from_ir(path))


def _ir_signature(op) -> str:
    """One op's signature, the same string `signatures_from_ir` builds."""
    sig = f"{op['op']}_{_shape_tag(op.get('shape'))}"
    subs = op.get("sub_ops") or []
    if subs:
        sig += "[" + "+".join(
            f"{s['op']}_{_shape_tag(s.get('shape'))}" for s in subs) + "]"
    return sig


def _by_dispatch_id(path: str) -> dict:
    """{dispatch_id: op} for an IR graph.json. Empty for a results.csv."""
    if path.endswith(".csv"):
        return {}
    g = json.load(open(path))
    return {op["dispatch_id"]: op for op in g.get("ops", [])
            if op.get("dispatch_id") is not None}


def audit_id_remap(before_path: str, after_path: str) -> list[str]:
    """Check the rewriter's `id_remap` against the signatures, if it made one.

    `id_remap` is the rewriter saying which op became which. The signatures are
    what the artifact says. Checking one against the other is the point of
    keeping both -- and until now nothing did: `check_id_remap` had no caller
    outside its own tests, while this script already holds exactly the two
    graphs it needs.

    Returns `(problems, n_checked)`. `n_checked` is what separates "the remap
    agrees" from "there was nothing to check" -- and the second was the silent
    reality on every split graph before `_unwrap_singleton`.
    """
    if after_path.endswith(".csv"):
        return [], 0
    after_doc = json.load(open(after_path))
    remap = after_doc.get("id_remap")
    if not remap:
        return [], 0
    before, after = _by_dispatch_id(before_path), _by_dispatch_id(after_path)
    if not before or not after:
        return [], 0
    n = sum(1 for v in remap.values()
            if not isinstance(_unwrap_singleton(v), (list, tuple)))
    return check_id_remap(before, after, remap,
                          signature_of=_ir_signature), n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True,
                    help="baseline graph.json (or results.csv)")
    ap.add_argument("--after", required=True,
                    help="rewritten graph.fused.json / graph.split.json")
    ap.add_argument("--json", default=None, help="also write the diff here")
    a = ap.parse_args()

    before, after = _load(a.before), _load(a.after)
    cb, ca = collections.Counter(before), collections.Counter(after)
    added = sorted((ca - cb).elements())
    removed = sorted((cb - ca).elements())
    changed = bool(added or removed) or len(before) != len(after)

    print(f"before  {len(before):>4} dispatches, {len(cb)} distinct signatures"
          f"   ({a.before})")
    print(f"after   {len(after):>4} dispatches, {len(ca)} distinct signatures"
          f"   ({a.after})")
    print(f"op count delta: {len(after) - len(before):+d}")
    for s in removed:
        print(f"  - {s}")
    for s in added:
        print(f"  + {s}")

    verdict = ("GRANULARITY CHANGED" if changed else
               "GRANULARITY UNCHANGED -- negative result, do not profile this "
               "as a rewrite")
    print(verdict)

    # The rewriter's own claim about which op became which, checked against
    # the signatures. A remap that disagrees means the rewrite moved an op
    # somewhere its module name says it did not go, and every downstream join
    # that trusts the remap -- advice, lineage, the schedule ingest -- would
    # carry that error silently.
    remap_problems, remap_checked = audit_id_remap(a.before, a.after)
    if remap_checked:
        # Say how many were checked. A silent pass is indistinguishable from a
        # check that ran on nothing, which is exactly what this check used to
        # do on a split graph.
        print(f"id_remap: {remap_checked} one-to-one entr"
              f"{'y' if remap_checked == 1 else 'ies'} checked against the "
              f"signatures, {len(remap_problems)} disagreement(s)")
    if remap_problems:
        print(f"id_remap DISAGREES with the signatures "
              f"({len(remap_problems)} entr{'y' if len(remap_problems) == 1 else 'ies'}):")
        for line in remap_problems:
            print(f"  {line}")
        print("Refusing: the rewrite's own bookkeeping does not match its "
              "output. Fix the rewriter before profiling this graph.")
        return 4

    if a.json:
        json.dump({
            "before": a.before, "after": a.after,
            "n_before": len(before), "n_after": len(after),
            "op_count_delta": len(after) - len(before),
            "signatures_removed": removed,
            "signatures_added": added,
            "granularity_changed": changed,
            "verdict": verdict,
            "id_remap_problems": remap_problems,
        }, open(a.json, "w"), indent=1)
        print(f"wrote {a.json}")
    return 0 if changed else 3


if __name__ == "__main__":
    sys.exit(main())
