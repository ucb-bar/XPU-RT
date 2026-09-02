"""Stage 2b — declared lowerings: what the converter did to the IR.

A binding claims that a set of IR ops is executed by a particular compiled
graph.  That claim is only true modulo the transformations the ONNX/DLC
build applied on the way — and on this board those are not all automatic.
dronet is the sharp case: its HTA graph only composes because the ONNX was
*rewritten* offline (BatchNorm folded into the conv weights, the FC head
replaced by a 1x1 conv, the trailing Reshape dropped — steps 6/8/11 of
optimization_flow.md).  A capability check run against the raw PyTorch IR
would reject that binding, and a check that silently assumed the folds
would accept bindings that do not actually compile.

So the manifest states which lowerings its artifact carries, this module
applies exactly those, and the registry check runs on the result.  A
binding that forgets to declare a lowering fails loudly, with the op that
blocked it named.
"""

from __future__ import annotations

from typing import Callable

_CONVISH = {"conv2d", "conv2d_s8", "linear", "linear_s8"}
_ACTS = {"relu", "relu_s8", "relu6", "relu6_s8", "hardswish", "hardswish_s8"}
_BN = {"batchnorm2d", "batchnorm2d_s8"}
_VIEWISH = {"reshape", "transpose", "view", "flatten"}


def _consumers(ops: list[dict]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {o["dispatch_id"]: [] for o in ops}
    for o in ops:
        for d in o.get("depends_on", []):
            if d in out:
                out[d].append(o)
    return out


def _producers(ops: list[dict]) -> dict[int, list[dict]]:
    by_id = {o["dispatch_id"]: o for o in ops}
    return {o["dispatch_id"]: [by_id[d] for d in o.get("depends_on", []) if d in by_id]
            for o in ops}


def fold_batchnorm_into_conv(ops: list[dict]) -> list[dict]:
    """An inference-time BatchNorm is an affine map, so the converter can
    absorb it into the conv on either side of it — into the producer's
    weights when it follows a conv, into the consumer's when it precedes
    one (dronet's residual blocks are add->BN->conv, so both directions
    occur). Requires the ONNX to present them foldably: dronet needed the
    offline bnfree rewrite before HTA would compose."""
    prods = _producers(ops)
    cons = _consumers(ops)
    drop = {o["dispatch_id"] for o in ops
            if o["op"] in _BN
            and (any(p["op"] in _CONVISH for p in prods[o["dispatch_id"]])
                 or any(c["op"] in _CONVISH for c in cons[o["dispatch_id"]]))}
    return _drop_and_rewire(ops, drop)


def fuse_activation_into_conv(ops: list[dict]) -> list[dict]:
    """QNN carries relu/relu6/hardswish as an activation attribute on the op
    that produces the tensor — conv, fc, or elementwise add (modelblaster
    calls the last one add_relu_fuse)."""
    fusable = _CONVISH | {"add", "add_s8"}
    prods = _producers(ops)
    drop = {o["dispatch_id"] for o in ops
            if o["op"] in _ACTS and any(p["op"] in fusable for p in prods[o["dispatch_id"]])}
    return _drop_and_rewire(ops, drop)


def conv_head_for_fc(ops: list[dict]) -> list[dict]:
    """optimization_flow.md #8 — the FC head is re-expressed as a 1x1 conv
    so the tile stays inside the HTA op set."""
    out = []
    for o in ops:
        o = dict(o)
        if o["op"] in ("linear", "linear_s8"):
            o["op"] = "conv2d_s8" if o["op"].endswith("_s8") else "conv2d"
            o["lowered_from"] = "linear"
        out.append(o)
    return out


def drop_trailing_views(ops: list[dict]) -> list[dict]:
    """optimization_flow.md #11 — a trailing Reshape/Transpose is a host-side
    concern; the graph ends at the last compute op."""
    drop = set()
    for o in reversed(ops):
        if o["op"] in _VIEWISH:
            drop.add(o["dispatch_id"])
        else:
            break
    return _drop_and_rewire(ops, drop)


def _drop_and_rewire(ops: list[dict], drop: set[int]) -> list[dict]:
    if not drop:
        return ops
    by_id = {o["dispatch_id"]: o for o in ops}
    def resolve(d: int, seen: set[int]) -> list[int]:
        if d not in drop:
            return [d]
        if d in seen:
            return []
        seen.add(d)
        out: list[int] = []
        for p in by_id.get(d, {}).get("depends_on", []):
            out += resolve(p, seen)
        return out
    kept = []
    for o in ops:
        if o["dispatch_id"] in drop:
            continue
        o = dict(o)
        deps: list[int] = []
        for d in o.get("depends_on", []):
            deps += resolve(d, set())
        o["depends_on"] = sorted(set(deps))
        kept.append(o)
    return kept


TRANSFORMS: dict[str, Callable[[list[dict]], list[dict]]] = {
    "fold_batchnorm_into_conv": fold_batchnorm_into_conv,
    "fuse_activation_into_conv": fuse_activation_into_conv,
    "conv_head_for_fc": conv_head_for_fc,
    "drop_trailing_views": drop_trailing_views,
}


def apply(ops: list[dict], names: list[str]) -> tuple[list[dict], list[str]]:
    """Apply the declared lowerings in order; returns (ops, log)."""
    log = []
    for name in names:
        fn = TRANSFORMS.get(name)
        if fn is None:
            raise ValueError(f"unknown lowering {name!r}; known: {sorted(TRANSFORMS)}")
        before = len(ops)
        ops = fn(ops)
        log.append(f"{name}: {before} -> {len(ops)} ops")
    return ops, log
