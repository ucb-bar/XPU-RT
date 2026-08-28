#!/usr/bin/env python3
"""Turn `choose_implementation` advice into a codegen kernel-selection change.

The third bridge, and the one whose consumer had to be found rather than
written. `choose_implementation` has been emitted since `implementation_advice`
existed; its only consumer was `scripts/apply_compile_advice.py`, which rewrote
`.vmfb` module paths inside a schedule -- an IREE mechanism that does not exist
on this path. So the verb was emitted and ignored.

WHAT THE LIVE EQUIVALENT IS
---------------------------
ModelBlaster picks kernels at codegen: `generate_kernels` starts every op on
the portable reference implementation and SWAPS IN a curated one where the
target has it (`kernel_picks.json` records which). A build is one target, so
"use implementation X for dispatch 7" has no representation. What does have one
is the swap decision itself, per op kind:

    curated kernel is SLOWER than the reference  ->  stop swapping it in

which is `generate_kernels --keep-reference-ops <op,...>`.

Measured, yolov8_nano on the board:

    dispatch 40, 41   maxpool2d_s8  N1xC128xIH5xIW5xKH5xKW5xSH1xSW1xPH2xPW2
                      curated[rvv]/direct  502us / 456us
                      scalar build         394us / 359us   (-21.5%, -21.2%)

THE GRANULARITY GAP, AND WHY IT IS SURFACED RATHER THAN AVERAGED
----------------------------------------------------------------
The advice is per DISPATCH; the pin is per OP KIND. When two dispatches of the
same kind disagree -- one wants the curated kernel, one does not -- there is no
pin that satisfies both, and picking the majority would silently make one of
them slower. That is refused and printed, because the honest resolutions are
outside this script's reach (a shape-specialised kernel, or accepting the loss
on one dispatch) and both are decisions.

WHAT THIS DOES NOT PROMISE
--------------------------
The proposed timing came from a DIFFERENT BUILD. `scalar` in the profile tree
is the whole model compiled with `-march=rv64gc`; keeping the reference kernel
inside an `rvv_x60` build compiles the same C with the vector flags, where the
compiler may auto-vectorise it differently. The mechanism transfers, the number
does not. Every emitted pin is a rung to be measured, not a saving to be
booked.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Optional

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "xpu-rt"))

from advice_join import verify_graph_identity  # noqa: E402

CONTRACT = "modelblaster.kernel_choice/v1"

#: Implementation names that mean "the portable reference", i.e. the thing
#: `generate_kernels` starts from and a pin keeps. Anything else names a
#: curated/vector build, which is what the swap already produces by default.
_REFERENCE_IMPLS = {"scalar", "reference"}


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

    wants = [x for x in model_items
             if x.get("recommendation") == "choose_implementation"]
    if not wants:
        print(f"no choose_implementation advice for {a.model}; nothing to do",
              file=sys.stderr)
        return 1

    # Group per op KIND, which is the granularity the pin has.
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped: list[str] = []
    for x in wants:
        did = x.get("dispatch_id")
        ev = x.get("evidence") or {}
        if did not in by_id:
            skipped.append(f"dispatch {did}: not in {a.ir}")
            continue
        proposed = str(ev.get("proposed_impl", ""))
        if proposed not in _REFERENCE_IMPLS:
            # "prefer the vector build here" is what the swap already does.
            skipped.append(
                f"dispatch {did} ({by_id[did]['op']}): proposes "
                f"{proposed!r}, which is what the curated swap already "
                f"selects -- no pin expresses it")
            continue
        by_kind[by_id[did]["op"]].append(x)

    # WHICH SIBLINGS ACTUALLY OBJECT. The pin is per op KIND, so every
    # dispatch of a pinned kind is affected. But "did not get a
    # choose_implementation item" is NOT the same as "the curated kernel won
    # here": `implementation_advice` has two floors, one relative and one
    # absolute, and a dispatch can miss them while still measuring faster on
    # the reference. It records that in its `unchanged` item, as
    # `best_alternative` plus the gain -- so the direction is in the advice
    # and does not need the profiles re-read.
    #
    # Measured, and the reason this distinction exists: yolov8_nano's three
    # maxpool2d_s8 dispatches are the same kind AND the same shape, and all
    # three measure faster on scalar (21.5%, 21.2%, 10.4%). Only two cleared
    # the 0.05 ms absolute floor; dispatch 42 missed it by 0.0033 ms. Treating
    # its silence as an objection refuses a pin that all three agree with.
    by_did = {x["dispatch_id"]: x for x in model_items
              if isinstance(x.get("dispatch_id"), int)}

    def objects_to_pin(did: int) -> Optional[str]:
        """Why dispatch `did` would be hurt by pinning its kind, or None."""
        item = by_did.get(did)
        if item is None:
            return f"{did} has no advice item at all, so its direction is unknown"
        ev = item.get("evidence") or {}
        if item.get("recommendation") == "choose_implementation":
            return None                      # already asking for the pin
        alt = str(ev.get("best_alternative") or "")
        if alt in _REFERENCE_IMPLS:
            return None                      # agrees, just under a floor
        if not alt:
            return (f"{did} measured no alternative faster than "
                    f"{ev.get('baseline_impl')}, so the curated kernel wins "
                    f"there")
        return f"{did} prefers {alt!r}, not the reference"

    pins, conflicts = [], []
    for kind, items in sorted(by_kind.items()):
        same_kind = sorted(d for d, o in by_id.items() if o["op"] == kind)
        advised = {d["dispatch_id"] for d in items}
        objections = [r for d in same_kind if d not in advised
                      for r in [objects_to_pin(d)] if r]
        if objections:
            conflicts.append(
                f"{kind}: advice pins dispatch(es) {sorted(advised)} to the "
                f"reference, but the pin is per op KIND and dispatch "
                + "; ".join(objections)
                + ". No pin satisfies both.")
            continue
        concurring = sorted(set(same_kind) - advised)
        gains = [float((d.get('evidence') or {}).get('gain_fraction', 0))
                 for d in items]
        pins.append({
            "op": kind,
            "dispatches": sorted(d["dispatch_id"] for d in items),
            "baseline_impl": (items[0].get("evidence") or {}).get("baseline_impl"),
            "baseline_kernel": (items[0].get("evidence") or {}).get("baseline_kernel"),
            "proposed_impl": (items[0].get("evidence") or {}).get("proposed_impl"),
            "advised_gain_fraction": round(sum(gains) / len(gains), 4),
            "advised_gain_is_from_another_build": True,
            # Same kind, same direction, but under one of the advisor's
            # floors. Named so the pin's true blast radius is on the record.
            "concurring_below_threshold": concurring,
        })

    for s in skipped:
        print(f"skipped {s}", file=sys.stderr)
    for c in conflicts:
        print(f"REFUSED {c}", file=sys.stderr)
    if not pins:
        print(f"no emittable kernel pin for {a.model}", file=sys.stderr)
        return 1

    out = {
        "contract": CONTRACT,
        "network": a.model,
        "keep_reference_ops": [p["op"] for p in pins],
        "generate_kernels_flag":
            "--keep-reference-ops " + ",".join(p["op"] for p in pins),
        "_provenance": {
            "from_advice": advice.get("schedule_id"),
            "identity_checked_dispatches": n_checked,
            "pins": pins,
            "refused": conflicts,
            "skipped": skipped,
            "caveat": ("the advised gain was measured in the proposed impl's "
                       "own BUILD; keeping the reference kernel inside the "
                       "baseline build compiles the same C with different "
                       "flags, so the result must be re-profiled"),
        },
    }
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {a.out}")
    for p in pins:
        print(f"  keep reference for {p['op']} (dispatches {p['dispatches']}, "
              f"{p['baseline_kernel']} is {p['advised_gain_fraction']*100:.1f}% "
              f"slower than {p['proposed_impl']} -- IN THAT BUILD; re-profile)")
    print(f"  {out['generate_kernels_flag']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
