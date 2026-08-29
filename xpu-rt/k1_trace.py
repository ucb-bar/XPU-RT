"""Reading a measured K1 trace, whichever producer wrote it.

TWO PRODUCERS, ONE MEANING. merlin's runner and ModelBlaster's
`harness_xpurt` both emit a measured per-dispatch trace, and they disagree on
spelling, not on meaning:

    merlin          dispatch_key  start_us  run_us  queue_delay_us  job_name
    ModelBlaster    (none)        actual_start_cycles / actual_end_cycles,
                                  network + instance, predicted_start_ms

Every consumer that reads only one of them is a tool that works on the retired
path and not the live one. `scripts/join_k1_trace.py` was exactly that -- the
tool that answers "is this a slow kernel or a long queue", which is the
question a deadline miss turns on, and it could not read a single trace this
project has taken since. Normalising once, here, is what stops the next
renderer from being the fourth reader of exactly one producer.

CYCLES ARE rdtime TICKS AT 24 MHz. Not the 1.6 GHz core clock and not 1 MHz:
the device-tree `timebase-frequency` is 24000000. `rdcycle` SIGILLs from
userspace on this board, so `rdtime` is what the harness reads.
"""

from __future__ import annotations

import csv
from typing import Any, Dict, List

#: rdtime on the K1. Every cycles->time conversion in this project uses it.
K1_RDTIME_HZ = 24_000_000.0

#: Ops that emit no kernel call, so `generate_skeleton` allocates them no
#: profile record. They are why the trace's `dispatch_id` is not the IR's --
#: see `ir_slot_map`. Kept in step with `generate_skeleton._zero_cost_ops`.
ZERO_COST_OPS = {"view", "chunk2_c1", "chunk2_c1_f16", "chunk2_c1_s8"}


def ir_slot_map(ir: Dict[str, Any]) -> Dict[int, int]:
    """`{trace slot index: IR dispatch_id}` for one model's graph.

    THE MISALIGNMENT THIS EXISTS TO CLOSE. `generate_skeleton` sizes the
    harness's profile record array by the ops that actually emit a kernel call
    -- `view` and the `chunk2_c1` family emit none -- and the harness stamps
    each record with its SLOT in that array. The trace column is called
    `dispatch_id`, but it is that slot, and it drifts from the IR's
    `dispatch_id` by the number of zero-cost ops seen so far.

    The schedule, the profile CSV and the advice all use the IR's numbering.
    So for any model containing a zero-cost op, joining a schedule to a trace
    on `dispatch_id` silently compares different ops from the first such op
    onward.

    Measured on yolov8_nano, which has 8 of them (ids 3, 11, 22, 33, 48, 56,
    64, 72): 95 of its 98 dispatches join to the wrong op. It reads as a
    prediction error rather than as a join error --

        yolov8_nano0_dispatch_81  predicted 17.465 ms, "measured" 0.577 ms

    -- because trace slot 81 is `detect.cv3_1_2` (a 0.577 ms conv2d_s8) while
    IR dispatch 81 is `detect.cv3_0_1.conv` (a 17.5 ms fused conv). Both
    numbers are real; they are just not of the same op.

    dronet and mlp_control have no zero-cost ops, which is why every earlier
    per-dispatch validation on this path was clean.
    """
    slot = 0
    out: Dict[int, int] = {}
    for op in ir.get("ops", []):
        did = op.get("dispatch_id")
        if did is None or op.get("op") in ZERO_COST_OPS:
            continue
        out[slot] = did
        slot += 1
    return out


def op_kind_matches(module_name: str, op: str) -> bool:
    """Does a schedule entry's `module_name` name the op kind the trace ran?

    `module_name` is `<model>$dispatch_<id>_<impl>_<op>[_<shape signature>]`,
    and the shape suffix is OPTIONAL: `profile_writer` appends `noshape` when
    it cannot read a shape, but a profile written before that behaviour existed
    ends at the op kind. Both forms are on disk right now --

        mlp_control$dispatch_0_rvv_x60_linear_s8_M1xK16xN256
        yolov8_nano$dispatch_0_rvv_x60_conv2d_batchnorm2d_silu_s8

    -- so testing only for `_<op>_` reports every dispatch of the second form
    as a mismatch. That is a false alarm in a check whose whole job is to
    refuse a bad join, which makes it worse than no check: it would train a
    reader to pass the override flag.

    Matching on a trailing field as well as an embedded one keeps it exact:
    `_conv2d_s8` does not match `..._conv2d_batchnorm2d_silu_s8`.
    """
    if not module_name or not op:
        return True                      # nothing to check against
    return f"_{op}_" in module_name or module_name.endswith(f"_{op}")


def is_modelblaster(rows: List[Dict[str, Any]]) -> bool:
    return bool(rows) and "actual_start_cycles" in rows[0]


def normalise(rows: List[Dict[str, Any]],
              slot_maps: Dict[str, Dict[int, int]] | None = None,
              fill_queue_delay: bool = True,
              ) -> List[Dict[str, Any]]:
    """Map a ModelBlaster trace onto merlin's column names; pass others through.

    The run is stamped from the FIRST TICK OBSERVED rather than from 0, so the
    axis starts at the run's own t0 -- `rdtime` is a free-running counter and
    its absolute value is boot time, not run time.

    `dispatch_key` is synthesised as `<network><instance>_dispatch_<id>`, which
    is the same string the scheduler emits, so a join against a schedule keys on
    identity rather than on array position.

    `queue_delay_us` is derived as actual start minus PREDICTED start, which is
    NOT the quantity merlin measures (submit-to-start inside the runtime): it
    is scheduler slack, not runtime queueing. It answers the related question
    "did this wait?", so renderers get it, and `schedule_slip_us` always
    carries it under its own honest name.

    `fill_queue_delay=False` suppresses BOTH the derived value and the zero
    default, leaving the column absent. `trace_metrics.summarise_trace` then
    reports `queue_us: None` and omits `queue_share_pct` -- which is right,
    because calling scheduler slack a "queue share" would put a number on a
    thing this producer never measured. `schedule_slip_us` is still there for
    anyone who wants the slack itself.
    """
    if not is_modelblaster(rows):
        return rows
    t0 = min(int(r["actual_start_cycles"]) for r in rows)
    out = []
    for r in rows:
        s, e = int(r["actual_start_cycles"]), int(r["actual_end_cycles"])
        d = dict(r)
        start_us = (s - t0) / K1_RDTIME_HZ * 1e6
        d["start_us"] = start_us
        d["end_us"] = (e - t0) / K1_RDTIME_HZ * 1e6
        d["run_us"] = max(e - s, 0) / K1_RDTIME_HZ * 1e6
        d["job_name"] = f'{r.get("network", "")}{r.get("instance", "")}'
        # `dispatch_id` in the file is a record SLOT. With the model's IR
        # available it is translated to the IR id everything else uses; without
        # one it is passed through and `trace_slot` records that no translation
        # happened, so a consumer can say so rather than assume.
        slot = int(r.get("dispatch_id", -1))
        d["trace_slot"] = slot
        smap = (slot_maps or {}).get(r.get("network", ""))
        if smap is not None and slot in smap:
            d["dispatch_id"] = smap[slot]
            d["dispatch_id_is_ir"] = True
        else:
            d["dispatch_id_is_ir"] = smap is None and slot_maps is not None
        d["dispatch_key"] = f'{d["job_name"]}_dispatch_{d["dispatch_id"]}'
        if r.get("predicted_duration_ms") not in (None, ""):
            d["planned_duration_us"] = float(r["predicted_duration_ms"]) * 1000.0
        if r.get("predicted_start_ms") not in (None, ""):
            planned = float(r["predicted_start_ms"]) * 1000.0
            d["planned_start_us"] = planned
            d["schedule_slip_us"] = start_us - planned
            if fill_queue_delay:
                d["queue_delay_us"] = max(start_us - planned, 0.0)
        # `fill_queue_delay=False` leaves it ABSENT when the producer measured
        # none, so `trace_metrics.summarise_trace` reports `queue_us: None`
        # rather than a 0 indistinguishable from a run that genuinely never
        # queued. Renderers that index the column unconditionally want the
        # fill; the scorer does not.
        if fill_queue_delay:
            d.setdefault("queue_delay_us", 0.0)
        # merlin's `target` is CPU_P/CPU_E; the equivalent identity here is the
        # worker the walker actually ran it on.
        d.setdefault("target", f'{r.get("core_kind", "")}#{r.get("hart", "")}')
        out.append(d)
    return out


def read(path: str, slot_maps: Dict[str, Dict[int, int]] | None = None,
         fill_queue_delay: bool = True) -> List[Dict[str, Any]]:
    with open(path, newline="") as f:
        return normalise(list(csv.DictReader(f)), slot_maps, fill_queue_delay)


def slot_maps_from_irs(ir_paths: List[str]) -> Dict[str, Dict[int, int]]:
    """`{network: slot map}` from a list of `graph.json` paths."""
    import json
    out = {}
    for path in ir_paths:
        ir = json.load(open(path))
        name = ir.get("name")
        if name:
            out[name] = ir_slot_map(ir)
    return out
