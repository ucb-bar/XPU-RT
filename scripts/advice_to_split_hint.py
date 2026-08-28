#!/usr/bin/env python3
"""Translate compile_advice.json into a ModelBlaster split hint.

The dual of `advice_to_fusion_hint.py`, and the piece that was missing: the
scheduler has emitted `split` advice since `blocking_advice` existed, and there
was no way to turn it into anything ModelBlaster consumes. The one split that
reached the board (DroNet dispatch 0, OC 32 -> 2x16, +13.7%, rejected) was
hand-written.

WHY `n_splits` IS DERIVED HERE AND NOT CHOSEN
---------------------------------------------
Split advice carries what fusion advice does not: a real `dispatch_id` and a
`constraints.max_target_piece_us` -- the periodic slot the dispatch has to fit
into. So the split factor is a CONSEQUENCE of the measurement,

    n_needed = ceil(service_time_us / max_target_piece_us)

rather than a number someone picked. `ModelBlaster/scripts/decision_loop.py`
hard-codes `n_splits=2`, which is right only by coincidence.

`n_needed` is then rounded UP to a divisor of the tilable dimension, because
`apply_split_hint` requires the axis to divide cleanly and would otherwise
refuse the hint. Emitting a hint the rewriter refuses is the failure mode this
bridge exists to avoid, so every constraint the rewriter enforces is checked
HERE, where the reason can still be printed against the advice that caused it.

WHY THERE IS A CAP
------------------
Splitting is not free. B4 measured 4-way OC sharding of yolov8n costing **+76%
total work** before it buys any parallelism, so its ceiling is 2.27x rather
than 4x. Past that the extra pieces mostly add work. `--max-splits` defaults
to 4 for that reason; a run that wants more must say so, and the hint records
what was asked for alongside what was emitted.

WHAT IT REFUSES, LOUDLY
-----------------------
  * an op kind the rewriter cannot split -- the advice is real, the mechanism
    is missing, and a hint would be rejected downstream with less context
  * a dispatch_id whose op kind or shape disagrees with the advice's own
    `module_name` -- ids renumber across rewrites, so an id that resolves to a
    DIFFERENT op is the silent-corruption case, not a missing-op case
  * a tilable dimension with no usable divisor
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from typing import Any, Optional, Sequence

CONTRACT = "modelblaster.split_hints/v1"

#: Op kinds `ModelBlaster/pipeline/apply_split_hint.py::_SPLITTABLE` accepts,
#: and the axis each one tiles. Mirrored rather than imported so this script
#: stays runnable without ModelBlaster on the path;
#: `xpu-rt/tests/test_advice_to_split_hint.py` asserts the two agree, so drift
#: fails a test instead of producing refused hints.
SPLITTABLE_AXIS = {
    "linear_s8": "N",
    "conv2d_s8": "OC",
    "conv2d_batchnorm2d_s8": "OC",
    "conv2d_batchnorm2d_silu_s8": "OC",
}


def op_shape(op: dict[str, Any]) -> dict[str, Any]:
    """A conv's shape, whether the op is plain or fused.

    A fused op carries no shape of its own; the geometry is on the conv
    sub-op. Reading `op["shape"]` for one is how 57 of yolov8n's dispatches
    ended up profiled as `noshape`.
    """
    if op.get("shape"):
        return op["shape"]
    for sub in (op.get("sub_ops") or ()):
        if str(sub.get("op", "")).startswith("conv2d") and sub.get("shape"):
            return sub["shape"]
    return {}


def divisors_up_to(n: int, cap: int) -> list[int]:
    return [d for d in range(2, min(n, cap) + 1) if n % d == 0]


def choose_n_splits(dim: int, n_needed: int, cap: int) -> tuple[Optional[int], bool]:
    """`(n_splits, reaches_target)` -- the smallest legal factor that meets the
    target, or the largest legal one below it.

    A split that does not fully close the gap is still worth emitting: it moves
    the piece toward the slot, and the loop measures the result either way. It
    is reported as not reaching the target rather than presented as if it did.
    """
    legal = divisors_up_to(dim, cap)
    if not legal:
        return None, False
    for d in legal:
        if d >= n_needed:
            return d, True
    return legal[-1], False


#: `<model>$dispatch_<id>_<impl>_<op>_<signature>`, as written by
#: `pipeline/profile_writer.py`.
_MODULE_RE = re.compile(r"^(?P<model>[^$]+)\$dispatch_(?P<did>\d+)_.*")

#: What `pipeline/profile_writer.py` writes in place of a shape signature when
#: it cannot read one off the op.
_NOSHAPE = "noshape"


def module_name_agrees(module_name: str, did: int, op_kind: str,
                       shape: dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    """`(refusal, warning)` for whether the advice's `module_name` describes the
    IR op we resolved. Both None means fully confirmed.

    dispatch_ids renumber on every rewrite, so an id that still resolves is not
    evidence that it resolves to the SAME op. The module name carries the kind
    and the full shape signature, so it can usually settle it.

    `noshape` IS NOT A DISAGREEMENT. `profile_writer` writes the literal string
    `noshape` where it could not read a shape, and it could not read one for
    any fused op until `_conv_shape_of` existed -- which is every fused
    dispatch in every profile on disk today, i.e. exactly the ops this bridge
    is for. Treating a missing signature as a mismatch would refuse all of
    them. It is downgraded to a warning that names what was actually
    confirmed (id and kind) and what was not, rather than being silently
    skipped or silently accepted.
    """
    if not module_name:
        return None, "advice carries no module_name; identity is the id alone"
    m = _MODULE_RE.match(module_name)
    if m and int(m.group("did")) != did:
        return (f"module_name says dispatch {m.group('did')}, advice says "
                f"{did}"), None
    if op_kind not in module_name:
        return f"module_name does not mention op kind {op_kind!r}", None
    if _NOSHAPE in module_name:
        return None, (f"profiled as {_NOSHAPE}, so only the id and the op kind "
                      f"confirm identity, not the geometry")
    for key in ("OC", "N"):
        want = shape.get(key)
        if want is not None and f"{key}{want}" not in module_name:
            return (f"module_name does not carry {key}={want} from the IR; "
                    f"the id may point at a different op after a rewrite"), None
    return None, None


def shape_tokens(shape: dict[str, Any]) -> set[str]:
    """`{"N": 1, "IC": 3}` -> `{"N1", "IC3"}` -- `profile_writer._shape_concise`'s
    tokens, compared as a SET so key order cannot cause a false mismatch.

    A LIST value is joined with `|`, matching `generate_skeleton`'s shape-string
    emitter. Rendering `[128, 64]` as Python does instead makes every concat
    dispatch look like a mismatch: 16 of 33 checkable dispatches, all of them
    false.
    """
    out = set()
    for k, v in (shape or {}).items():
        if isinstance(v, (list, tuple)):
            v = "|".join(str(x) for x in v)
        out.add(f"{k}{v}")
    return out


def verify_graph_identity(items: Sequence[dict[str, Any]],
                          by_id: dict[int, dict[str, Any]]
                          ) -> tuple[int, list[str]]:
    """`(n_checked, disagreements)` for whether the advice and the IR describe
    the same graph.

    WHY THIS IS NOT THE PER-OP CHECK. The ops worth splitting are fused convs,
    and every fused dispatch in every profile on disk is `noshape` -- so the
    per-op signature check cannot fire for precisely the ops that matter, and
    the op KIND alone is worthless here because yolov8n's topology is identical
    at every input size. Only the geometry differs.

    Measured: joining a 320x320 profile (226.86 ms, 90 dispatches) against the
    64x96 IR (46.4 ms) passed every per-op check -- same ids, same kinds, all
    `noshape` -- and would have derived split factors from service times 25x
    too large. `add_s8_n25600` in the same advice file is 6144 elements in the
    64x96 graph, and says so immediately.

    So identity is established from the WHOLE advice set: the elementwise and
    concat dispatches carry real signatures even when the convs do not. One
    disagreement anywhere means the profile did not come from this IR, and
    nothing derived from it is safe to emit.
    """
    checked, bad = 0, []
    for x in items:
        did = x.get("dispatch_id")
        name = (x.get("evidence") or {}).get("op", "")
        if not isinstance(did, int) or not name or _NOSHAPE in name:
            continue
        op = by_id.get(did)
        if not op or not op.get("shape"):
            continue
        want = shape_tokens(op["shape"])
        # The signature follows the op kind, which itself contains underscores,
        # so it is located by that separator rather than by splitting on `_`.
        parts = name.split(f"_{op['op']}_", 1)
        if len(parts) != 2:
            continue
        signature = parts[1]
        missing = want - set(signature.split("x"))
        checked += 1
        if missing:
            bad.append(f"dispatch {did} ({op['op']}): IR has "
                       f"{sorted(missing)}, profile says {signature}")
    return checked, bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--advice", required=True)
    ap.add_argument("--ir", required=True, help="ModelBlaster graph.json")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-splits", type=int, default=4,
                    help="cap on the split factor (default 4; 4-way OC "
                         "sharding was measured at +76%% total work, so the "
                         "return past this is small)")
    ap.add_argument("--skip-identity-check", action="store_true",
                    help="proceed even when no dispatch in the advice carries "
                         "a shape signature to confirm it came from this IR")
    ap.add_argument("--allow-below-target", action="store_true",
                    help="emit a split that does not fully close the gap to "
                         "max_target_piece_us instead of skipping it")
    a = ap.parse_args()

    advice = json.load(open(a.advice))
    wants = [x for x in advice.get("advice", [])
             if x.get("recommendation") == "split" and x.get("model") == a.model]
    if not wants:
        print(f"no split advice for {a.model}; nothing to do", file=sys.stderr)
        return 1

    graph = json.load(open(a.ir))
    by_id = {o["dispatch_id"]: o for o in graph["ops"]
             if o.get("dispatch_id") is not None}

    model_items = [x for x in advice.get("advice", [])
                   if x.get("model") == a.model]
    n_checked, disagreements = verify_graph_identity(model_items, by_id)
    if disagreements:
        print(f"REFUSED: the advice did not come from {a.ir}. "
              f"{len(disagreements)} of {n_checked} checkable dispatch(es) "
              f"disagree:", file=sys.stderr)
        for d in disagreements[:5]:
            print(f"  {d}", file=sys.stderr)
        if len(disagreements) > 5:
            print(f"  ... and {len(disagreements) - 5} more", file=sys.stderr)
        print("Splitting on service times measured from a different graph "
              "would derive the factor from the wrong numbers.", file=sys.stderr)
        return 2
    if n_checked == 0 and not a.skip_identity_check:
        print(f"REFUSED: no dispatch in the advice carries a shape signature, "
              f"so nothing confirms it came from {a.ir}. Op kinds alone do not "
              f"-- a model's topology is identical at every input size. Pass "
              f"--skip-identity-check to proceed anyway.", file=sys.stderr)
        return 2
    print(f"graph identity: {n_checked} dispatch signature(s) agree with "
          f"{a.ir}", file=sys.stderr)

    split_ops: list[dict[str, Any]] = []
    notes: list[dict[str, Any]] = []
    refused = 0
    for x in wants:
        did = x.get("dispatch_id")
        ev = x.get("evidence") or {}
        con = x.get("constraints") or {}
        if did not in by_id:
            print(f"REFUSED dispatch {did}: not in {a.ir}", file=sys.stderr)
            refused += 1
            continue
        op = by_id[did]
        kind = op["op"]
        axis = SPLITTABLE_AXIS.get(kind)
        if axis is None:
            print(f"REFUSED dispatch {did}: op kind {kind!r} is not "
                  f"split-capable (apply_split_hint accepts "
                  f"{sorted(SPLITTABLE_AXIS)}). The advice stands; the "
                  f"mechanism is missing.", file=sys.stderr)
            refused += 1
            continue
        shape = op_shape(op)
        disagreement, caveat = module_name_agrees(
            ev.get("op", ""), did, kind, shape)
        if disagreement:
            print(f"REFUSED dispatch {did}: {disagreement}", file=sys.stderr)
            refused += 1
            continue
        if caveat:
            print(f"note dispatch {did}: {caveat}", file=sys.stderr)
        dim = int(shape.get(axis, 0))
        if dim < 2:
            print(f"REFUSED dispatch {did}: {axis}={dim} leaves nothing to "
                  f"tile", file=sys.stderr)
            refused += 1
            continue
        svc_us = float(ev.get("service_time_us") or 0.0)
        target_us = float(con.get("max_target_piece_us") or 0.0)
        if svc_us <= 0 or target_us <= 0:
            print(f"REFUSED dispatch {did}: advice carries no measured "
                  f"service_time_us / max_target_piece_us to derive a split "
                  f"factor from", file=sys.stderr)
            refused += 1
            continue
        n_needed = max(2, math.ceil(svc_us / target_us))
        n, reaches = choose_n_splits(dim, n_needed, a.max_splits)
        if n is None:
            print(f"REFUSED dispatch {did}: {axis}={dim} has no divisor in "
                  f"[2, {a.max_splits}]", file=sys.stderr)
            refused += 1
            continue
        if not reaches and not a.allow_below_target:
            print(f"SKIPPED dispatch {did}: needs {n_needed} pieces to reach "
                  f"{target_us:.0f}us, but {axis}={dim} capped at "
                  f"--max-splits={a.max_splits} allows only {n}. Pass "
                  f"--allow-below-target to emit it anyway.", file=sys.stderr)
            continue
        split_ops.append({"op": did, "n_splits": n})
        notes.append({
            "dispatch_id": did, "op": kind, "axis": axis, f"{axis}": dim,
            "service_time_us": svc_us, "max_target_piece_us": target_us,
            "n_needed": n_needed, "n_splits": n,
            "reaches_target": reaches,
            "piece_us_estimate": round(svc_us / n, 2),
            "identity_caveat": caveat,
        })

    if not split_ops:
        print(f"no emittable split for {a.model} "
              f"({refused} refused, {len(wants)} advised)", file=sys.stderr)
        return 1

    hint = {
        "contract": CONTRACT,
        "reason": "; ".join(
            f"dispatch {n['dispatch_id']} ({n['op']}) {n['service_time_us']:.0f}us "
            f"vs {n['max_target_piece_us']:.0f}us slot -> {n['axis']} "
            f"{n[n['axis']]} in {n['n_splits']}" for n in notes),
        "networks": [{"network": a.model, "split_ops": split_ops}],
        "_provenance": {
            "from_advice": advice.get("schedule_id"),
            "max_splits": a.max_splits,
            "refused": refused,
            "derivation": notes,
            "evidence": [x.get("evidence") for x in wants],
        },
    }
    with open(a.out, "w") as f:
        json.dump(hint, f, indent=1)
    print(f"wrote {a.out}")
    for n in notes:
        flag = "" if n["reaches_target"] else "  (BELOW TARGET)"
        print(f"  dispatch {n['dispatch_id']} {n['op']}: {n['axis']}="
              f"{n[n['axis']]} in {n['n_splits']} -> ~{n['piece_us_estimate']}us "
              f"per piece vs {n['max_target_piece_us']:.0f}us slot{flag}")
    if refused:
        print(f"  {refused} advised dispatch(es) refused; see above")
    return 0


if __name__ == "__main__":
    sys.exit(main())
