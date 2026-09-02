"""Joining measured advice to an IR, safely.

The scheduler's advice is keyed on `dispatch_id`, and dispatch_ids renumber on
every rewrite -- so an id that still resolves is not evidence that it resolves
to the SAME op. Everything here exists to establish that a `compile_advice.json`
and a `graph.json` describe the same graph BEFORE anything is derived from the
pairing.

Shared by `scripts/advice_to_split_hint.py` and
`scripts/advice_to_kernel_choice.py`, which is why it is a module rather than a
copy in each: they must agree about what counts as the same graph, or one of
them will act on a join the other would have refused.
"""

from __future__ import annotations

import re
from typing import Any, Optional, Sequence

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


