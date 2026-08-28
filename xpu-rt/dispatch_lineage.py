"""Per-dispatch identity that survives a fuse/split IR rewrite.

THE HAZARD
----------
A closed-loop iteration compares a profile taken **before** a granularity
rewrite against one taken **after** it: did fusing those seven MLP ops into four
actually pay? The obvious join key is `dispatch_id`, and it is wrong.

`realize-hint` (ModelBlaster's `pipeline/apply_fusion_hint.py` /
`apply_split_hint.py`) *reassigns dispatch_ids contiguously* over the rewritten
graph -- that is in its contract, not an accident. Fusing ops 2 and 3 of a
21-dispatch DroNet therefore renumbers 4..20 down to 3..19, and splitting one op
into two tiles renumbers everything after it up by one. Join on `dispatch_id`
across that boundary and every downstream row is compared against a different
op: a 22 ms convolution lines up with an 0.06 ms batchnorm and the round reports
a 99% "speedup" it did not get. Nothing raises, the shapes match, the numbers
are plausible, and the conclusion is fiction.

`module_name` is the field that carries the op *signature* -- backend tag, op
name and shape -- so it is what identity has to be built from. Two subtleties
make that less trivial than `key=module_name`:

* **The index is inside the module name, sometimes twice.** ModelBlaster writes
  `dronet$dispatch_4_rvv_x60_conv2d_s8_N1xIC32x...`; the IREE-era profiler
  writes `dronet$async_dispatch_4_embedded_elf_riscv_64_dronet$async_dispatch_4_conv_...`
  -- the same index embedded twice. Stripping only the first occurrence leaves a
  key that still changes when the graph is renumbered, which is the original bug
  wearing a different hat. `op_signature` removes *every* occurrence.

* **Signatures are not unique.** DroNet's dispatches 18 and 19 are both
  `linear_s8_M1xK2048xN1`; the two heads of the network genuinely run the same
  kernel on the same shape. A plain dict keyed on the signature silently drops
  one of them. `lineage_keys` disambiguates by order of appearance, which is
  stable exactly because the rewrite renumbers *in topological order* -- the
  k-th `linear_s8` stays the k-th `linear_s8`.

  That stability fails in one case, and it is not detectable from a single side:
  if the rewrite *consumes* one member of a repeated-signature family (fuses
  dispatch 18 into its predecessor), the survivor's ordinal shifts and pairing
  by ordinal would match the wrong op. `join` therefore reports any family whose
  multiplicity changed as `ambiguous` rather than matching it. Refusing to
  answer is the only honest option there; guessing reproduces the defect this
  module exists to prevent.

**The backend tag stays in the signature on purpose.** `..._rvv_x60_conv2d_s8_...`
and `..._scalar_conv2d_s8_...` are the same op compiled two ways, and joining
them as one dispatch would report a kernel change as a granularity change. A
cross-implementation comparison is `compile_advice.implementation_advice`'s job,
which works within one graph and needs no lineage.

WHY NOT `id_remap`
------------------
`apply_fusion_hint` emits an `id_remap` field mapping every pre-rewrite id to
its post-rewrite id, and using it is correct when you have it. You do not always
have it: a profile measured on the board carries dispatch ids and module names
and nothing else, and a rewrite realised by hand, by an older rewriter, or by a
different tool carries no remap at all. `id_remap` is the fast path when
present; this module is what makes the join possible when it is not, and it is
what can *check* a remap rather than trusting it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Tuple

#: `$dispatch_12` / `$async_dispatch_12` -- the renumbered part of a module name.
_DISPATCH_INDEX = re.compile(r"\$(?:async_)?dispatch_\d+")

#: What every occurrence of the index is replaced with. A fixed token rather
#: than "" so the signature keeps the structure of the name it came from and two
#: different naming conventions cannot collide by accident.
_INDEX_PLACEHOLDER = "$dispatch"


def op_signature(module_name: str) -> str:
    """The part of a module name that a renumbering cannot change.

    >>> op_signature("dronet$dispatch_4_rvv_x60_conv2d_s8_N1xIC32")
    'dronet$dispatch_rvv_x60_conv2d_s8_N1xIC32'

    Every occurrence of the dispatch index is removed, not just the first: the
    IREE profiler's module names embed it twice.
    """
    if not module_name:
        return ""
    return _DISPATCH_INDEX.sub(_INDEX_PLACEHOLDER, str(module_name))


def _id_order(dispatch_id: Any) -> Tuple[int, float, str]:
    """Sort ints numerically, anything else lexically, ints first.

    Profiles are keyed by int in every producer in this repo, but a schedule can
    carry `"*"` (`compile_advice.overhead_advice`), and sorting a mixed dict
    must not raise.
    """
    if isinstance(dispatch_id, bool):
        return (1, 0.0, str(dispatch_id))
    if isinstance(dispatch_id, (int, float)):
        return (0, float(dispatch_id), "")
    return (1, 0.0, str(dispatch_id))


def lineage_keys(profile: Mapping[Any, Mapping[str, Any]]) -> Dict[Any, str]:
    """`{dispatch_id: "<signature>#<k>"}` for one profile or graph.

    `k` counts repeated signatures in dispatch_id order, so two dispatches
    running the identical kernel on the identical shape stay distinguishable.
    Ordering by dispatch_id is ordering topologically: every producer in this
    repo numbers dispatches in topological order, and every rewrite that
    renumbers preserves it.
    """
    seen: Dict[str, int] = {}
    out: Dict[Any, str] = {}
    for did in sorted(profile, key=_id_order):
        sig = op_signature((profile[did] or {}).get("module_name", ""))
        k = seen.get(sig, 0)
        seen[sig] = k + 1
        out[did] = f"{sig}#{k}"
    return out


@dataclass
class LineageJoin:
    """The result of joining two profiles across a rewrite.

    `matched` is the only part a caller may compare costs over. The other three
    exist so that "this op vanished", "this op is new" and "this op cannot be
    identified" are distinct outcomes instead of all being absence.
    """

    #: lineage key -> (dispatch_id before, dispatch_id after)
    matched: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)
    #: lineage key -> dispatch_id, present only before (consumed by the rewrite)
    only_before: Dict[str, Any] = field(default_factory=dict)
    #: lineage key -> dispatch_id, present only after (created by the rewrite)
    only_after: Dict[str, Any] = field(default_factory=dict)
    #: signature -> (ids before, ids after) for families whose multiplicity
    #: changed, so ordinals are not trustworthy. Never matched, never silently
    #: dropped -- a caller that ignores this field is choosing to.
    ambiguous: Dict[str, Tuple[List[Any], List[Any]]] = field(default_factory=dict)

    @property
    def is_unambiguous(self) -> bool:
        return not self.ambiguous


def join(before: Mapping[Any, Mapping[str, Any]],
         after: Mapping[Any, Mapping[str, Any]]) -> LineageJoin:
    """Pair the dispatches of `before` and `after` by op signature.

    Both arguments are `{dispatch_id: record}` with a `module_name` in each
    record -- the shape `compile_advice.load_profiles*` and
    `profile_loader.load_profiled_times` both return.
    """
    keys_before = lineage_keys(before)
    keys_after = lineage_keys(after)

    def _by_signature(keys: Mapping[Any, str]) -> Dict[str, List[Any]]:
        fam: Dict[str, List[Any]] = {}
        for did, key in sorted(keys.items(), key=lambda kv: _id_order(kv[0])):
            fam.setdefault(key.rsplit("#", 1)[0], []).append(did)
        return fam

    fam_before = _by_signature(keys_before)
    fam_after = _by_signature(keys_after)

    out = LineageJoin()
    for sig in sorted(set(fam_before) | set(fam_after)):
        ids_b = fam_before.get(sig, [])
        ids_a = fam_after.get(sig, [])
        if ids_b and ids_a and len(ids_b) != len(ids_a):
            # The ordinal within this family moved. Which survivor is which is
            # not recoverable from the names alone, so say so.
            out.ambiguous[sig] = (list(ids_b), list(ids_a))
            continue
        if not ids_a:
            for k, did in enumerate(ids_b):
                out.only_before[f"{sig}#{k}"] = did
            continue
        if not ids_b:
            for k, did in enumerate(ids_a):
                out.only_after[f"{sig}#{k}"] = did
            continue
        for k, (b, a) in enumerate(zip(ids_b, ids_a)):
            out.matched[f"{sig}#{k}"] = (b, a)
    return out


def check_id_remap(before: Mapping[Any, Mapping[str, Any]],
                   after: Mapping[Any, Mapping[str, Any]],
                   id_remap: Mapping[Any, Any]) -> List[str]:
    """Disagreements between a rewriter's `id_remap` and the signatures.

    Returns one human-readable line per pre-rewrite dispatch whose remapped
    target carries a different op signature. Empty means the remap and the
    module names tell the same story.

    This is the reason to keep both mechanisms: `id_remap` is a claim made by
    the tool that did the rewrite, and a claim about which op became which is
    exactly the kind that is worth checking against the artifact rather than
    believing. Many-to-one entries (a fuse group) are skipped -- the fused op's
    signature is by construction not any member's.
    """
    targets: Dict[Any, List[Any]] = {}
    for src, dst in id_remap.items():
        for d in (dst if isinstance(dst, (list, tuple)) else [dst]):
            targets.setdefault(d, []).append(src)

    # Coerce before sorting, not after: a remap that has been through JSON has
    # string keys, and sorting those lexically would report the same findings in
    # a different order than the int-keyed remap it came from -- enough to make
    # two runs of the same check look like they disagreed.
    entries = [(_coerce_id(src, before), dst) for src, dst in id_remap.items()]

    problems: List[str] = []
    for src, dst in sorted(entries, key=lambda kv: _id_order(kv[0])):
        if isinstance(dst, (list, tuple)):
            continue                      # one-to-many: a split, new ops
        if len(targets.get(dst, [])) > 1:
            continue                      # many-to-one: a fuse, new op
        if src not in before or dst not in after:
            continue
        sb = op_signature(before[src].get("module_name", ""))
        sa = op_signature(after[dst].get("module_name", ""))
        if sb and sa and sb != sa:
            problems.append(
                f"id_remap says {src} -> {dst}, but the signatures differ: "
                f"{sb!r} became {sa!r}")
    return problems


def _coerce_id(src: Any, profile: Mapping[Any, Any]) -> Any:
    """JSON object keys are strings; profiles are keyed by int."""
    if src in profile:
        return src
    try:
        return int(src)
    except (TypeError, ValueError):
        return src
