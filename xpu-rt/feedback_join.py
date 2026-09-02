"""Join a measured run's runtime feedback onto the static compile advice.

THE GAP THIS CLOSES, and it was mine. `xpurt_feedback.json` had a producer on
both paths -- `run_xpurt_schedule.py --emit-feedback` and
`streaming_feedback.py` -- and no reader anywhere. A producer with no consumer
is exactly the shape of the problem the shard chain was written to fix,
pointing the other way.

WHY THE CONSUMER IS THE ADVICE PRODUCER, and not ModelBlaster directly. The
obvious move is a `feedback_to_hints.py` beside the five `advice_to_*_hint.py`
bridges. It cannot be written honestly:

    prefer_finer              -> a split needs `n = ceil(service / slot)`,
                                 and the feedback carries no slot budget
    consider_fuse_with_pred   -> a fusion needs the GROUP of dispatches,
                                 which needs the graph
    pin_target=<combination>  -> a machine combination, not a kernel impl

Every one of those needs the graph and the periodic budget, and the only thing
holding both is `emit_compile_advice`. A bridge that guessed them would be
inventing the numbers the loop exists to measure.

WHAT THIS DOES INSTEAD. The two channels reach similar conclusions from
different evidence:

    compile advice     profiles + solved schedule       PREDICTED
    runtime feedback   a run that actually happened     OBSERVED

So the run is used to CORROBORATE or CONTRADICT the static advice, never to
manufacture new advice. An item the run agrees with gains confidence; one the
run contradicts is demoted to `unchanged` with the reason recorded. Nothing is
invented, and the arithmetic that decides anything still comes from the
measured profiles.

INERT BY DEFAULT. With no feedback file, `join()` returns its input unchanged
and every existing invocation is byte-identical.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

#: Runtime hint -> the static recommendations it AGREES with.
#:
#: `prefer_finer` says the dispatch ran slower than the scheduler predicted or
#: sat behind an idle gap, which is the same conclusion `blocking_advice`
#: reaches from the periodic budget and `shard_advice` from the per-width
#: profiles. `prefer_coarser` says the opposite, and agrees with fusing.
CORROBORATES: Dict[str, set] = {
    "prefer_finer": {"split", "shard"},
    "prefer_coarser": {"fuse"},
    "consider_fuse_with_pred": {"fuse"},
    "consider_split_backend": {"choose_implementation"},
}

#: Runtime hint -> the static recommendations it CONTRADICTS.
#:
#: Deliberately not the complement of the above. `prefer_coarser` contradicts
#: splitting -- the run had slack where the model predicted pressure -- but
#: `prefer_finer` does NOT contradict fusing: a dispatch can be both too slow
#: and worth fusing with a neighbour, and treating them as opposites would
#: suppress correct advice.
CONTRADICTS: Dict[str, set] = {
    "prefer_coarser": {"split", "shard"},
}

_CONFIDENCE_UP = {"low": "medium", "medium": "high", "high": "high"}


def load(path: str | Path) -> Optional[Dict[str, Any]]:
    """Read `xpurt_feedback.json`, or None when it is absent or unreadable.

    Absent is the normal case and not an error: the feedback channel is
    additive, and a first pass has no measured run to join against.
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        doc = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc.get("dispatches"), dict) else None


def hints_by_dispatch(doc: Dict[str, Any], model: str,
                      known: Optional[set] = None) -> Dict[Any, List[str]]:
    """`{dispatch_id: [hint, ...]}` for one model, across all its instances.

    Feedback keys are `<network><instance>_dispatch_<id>` -- per INSTANCE,
    because a run has many. Advice is per DISPATCH, because a rewrite applies
    to the graph rather than to one instance of it. So the instances are
    unioned: a dispatch that earned `prefer_finer` in any instance carries it.

    Union rather than majority on purpose. These hints already survived
    `streaming_feedback`'s own rate thresholds, so a hint reaching here means
    the condition held often enough to report; requiring it in most instances
    would be filtering twice with the second filter undocumented.
    """
    import job_names

    out: Dict[Any, List[str]] = {}
    for key, rec in (doc.get("dispatches") or {}).items():
        job, _, did_s = key.rpartition("_dispatch_")
        if not job or not did_s:
            continue
        if job_names.model_of(job, known) != model:
            continue
        try:
            did = int(did_s)
        except ValueError:
            continue
        for h in rec.get("hints") or []:
            out.setdefault(did, [])
            if h not in out[did]:
                out[did].append(h)
    return out


def join(advice: List[Any], doc: Optional[Dict[str, Any]], model: str,
         known: Optional[set] = None) -> tuple[List[Any], Dict[str, int]]:
    """Corroborate or contradict `advice` with a measured run.

    Returns `(advice, counts)`. The list is the same objects, mutated in
    place: confidence raised where the run agrees, recommendation demoted to
    `unchanged` where it disagrees, and `evidence.extra` carrying what the run
    said either way — so a reader can always see WHY an item's confidence is
    what it is, and can undo the judgement by ignoring the field.
    """
    counts = {"corroborated": 0, "contradicted": 0,
              "not_applicable": 0, "silent": 0}
    if not doc:
        return advice, counts

    by_did = hints_by_dispatch(doc, model, known)
    for item in advice:
        hints = by_did.get(item.dispatch_id)
        if not hints:
            # The run reported nothing about this dispatch at all.
            counts["silent"] += 1
            continue

        rec = item.recommendation
        agree = [h for h in hints if rec in CORROBORATES.get(h.split("=")[0], ())]
        against = [h for h in hints if rec in CONTRADICTS.get(h.split("=")[0], ())]

        extra = item.evidence.extra if item.evidence.extra is not None else {}
        extra = dict(extra)
        extra["runtime_hints"] = list(hints)
        extra["runtime_run_id"] = doc.get("run_id")

        if against:
            extra["demoted_by_measurement"] = against
            item.rationale = (
                f"{item.rationale} -- DEMOTED: the measured run reported "
                f"{', '.join(against)} for this dispatch, which contradicts "
                f"{rec}").strip(" -")
            item.recommendation = "unchanged"
            item.priority = max(item.priority, 5)
            counts["contradicted"] += 1
        elif agree:
            extra["corroborated_by_measurement"] = agree
            item.confidence = _CONFIDENCE_UP.get(item.confidence,
                                                 item.confidence)
            item.rationale = (
                f"{item.rationale} -- corroborated: the measured run also "
                f"reported {', '.join(agree)}").strip(" -")
            counts["corroborated"] += 1
        else:
            # The run DID report on this dispatch, but none of its hints
            # bears on this particular recommendation -- `prefer_finer`
            # against an `unchanged`, say. Counted separately from silence,
            # because "the run said nothing" and "the run said something
            # unrelated" are different facts and only one of them means the
            # measurement missed the dispatch.
            counts["not_applicable"] += 1
        item.evidence.extra = extra

    return advice, counts
