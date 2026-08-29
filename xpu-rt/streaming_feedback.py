"""Host-side daemon that turns live board telemetry into XPU-RT feedback.

Tails the JSON-Lines stream ModelBlaster's `harness_xpurt` writes when built
with `-DMODELBLASTER_XPURT_STREAM`, keeps a rolling window of recent
instances, derives per-dispatch hints, and writes `xpurt_feedback.json`.

WHY STREAM AT ALL, when the walker already dumps a trace. The trace block is
printed when the run ENDS. That is enough to explain a run afterwards and
useless for responding to one: a schedule that starts missing deadlines in
instance 3 of 60 keeps missing them for the remaining 57, and nothing can act
until it is over. Streaming is the same numbers, available while there is
still something to do about them.

This used to target merlin -- it tailed a format merlin's runner was going to
emit and posted the result through an `ingest_xpurt_feedback` MCP tool. Neither
end ever existed: no merlin runner emitted that format, the MCP tool was not in
the merlin checkout, and nothing imported this module. It is pointed at the
real producer now, and writes a file instead of calling a tool, which also
means it does not need an MCP server to be reachable.

Telemetry line (one JSON object per DISPATCH END, from the walker):

  {"entry_id": int, "network": str, "instance": int, "dispatch_id": int,
   "impl": str, "hart": int,
   "predicted_start_ms": float, "predicted_duration_ms": float,
   "start_ticks": int, "end_ticks": int}

TICKS ARE rdtime AT 24 MHz, not core cycles: `rdcycle` SIGILLs from userspace
on this board, so the harness reads `rdtime`, whose device-tree
`timebase-frequency` is 24000000.

WHAT THE BOARD DOES NOT SAY, and this matters for reading the output. The
walker does not know its own deadlines -- periods and `window_duration` live
in the workload spec, not in the binary -- so it emits no `deadline_miss`, and
there is no skip mechanism, so no `skip_fired`. The signals derivable from the
stream alone are therefore about the COST MODEL: measured duration against the
duration the scheduler predicted. Pass `--windows-from <spec.json>` to get real
deadline misses as well; without it the miss rate is reported as unknown rather
than as zero, because a structural zero that looks like a measurement is how
the `yolov8_nano_64x96` deadline bug survived as long as it did.

Example:

  python xpu-rt/streaming_feedback.py \\
    --telemetry-stream /tmp/xpurt.jsonl \\
    --out schedules/xpurt_feedback.json \\
    --epoch-window 32 --post-every-n-epochs 8 --follow
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

# Hint-derivation thresholds. Same vocabulary as xpu-rt/feedback.py
# but operating on streamed run_us / planned_duration_us / miss / skip
# stats rather than solver state.
_RUNS_OVER_PLANNED_FOR_FINER = 1.30   # mean run / mean planned > this → finer
_RUNS_UNDER_PLANNED_FOR_COARSER = 0.60  # < this and no misses → coarser
_DEADLINE_MISS_RATE_FOR_FINER = 0.10   # > 10% of epochs missed → finer / fuse
_SKIP_RATE_FOR_FINER = 0.05            # > 5% of epochs skipped → finer
_PIN_TARGET_STICKINESS = 0.95          # if 95% of epochs ran on same target → pin


#: Board `rdtime` is a fixed 24.000 MHz (device-tree timebase-frequency).
#: Not the 1.6 GHz core clock, and not 1 MHz.
TICKS_PER_US = 24.0


def normalise_event(line: dict[str, Any],
                    windows_ms: Optional[dict[str, float]] = None
                    ) -> Optional[dict[str, Any]]:
    """One walker telemetry line -> the event the hint derivation consumes.

    Returns None for a line that is not a dispatch-end record, so a stream
    carrying other chatter does not have to be pre-filtered.

    The key is `<network><instance>_dispatch_<id>` -- the same spelling the
    schedule uses for a job, so a hint can be joined back to the dispatch that
    produced it without a second naming convention.

    `deadline_miss` is None unless `windows_ms` supplies the network's
    deadline. None is not False: the derivation counts only explicit True, so
    an unknown miss cannot be silently read as "no misses".
    """
    try:
        net = str(line["network"])
        inst = int(line["instance"])
        did = int(line["dispatch_id"])
        t0 = float(line["start_ticks"])
        t1 = float(line["end_ticks"])
    except (KeyError, TypeError, ValueError):
        return None

    run_us = max(0.0, (t1 - t0) / TICKS_PER_US)
    planned_us = float(line.get("predicted_duration_ms", 0.0)) * 1000.0

    miss = None
    if windows_ms and net in windows_ms:
        # End of this dispatch against the instance's deadline: instance k of
        # a network with deadline D must be done by (k+1)*D from run start.
        deadline_us = float(windows_ms[net]) * 1000.0 * (inst + 1)
        miss = (t1 / TICKS_PER_US) > deadline_us

    return {
        "epoch": inst,
        "dispatch_id": f"{net}{inst}_dispatch_{did}",
        "target": str(line.get("impl") or "?"),
        "run_us": run_us,
        "planned_duration_us": planned_us,
        "deadline_miss": miss,
        "skip_fired": None,     # the walker has no skip mechanism
    }


def _derive_streaming_hints(window: list[dict[str, Any]],
                            run_id: str) -> dict[str, Any]:
    """Aggregate the windowed telemetry into an xpurt_feedback payload."""
    by_id: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for ev in window:
        d = ev.get("dispatch_id")
        if isinstance(d, str):
            by_id[d].append(ev)

    dispatches: dict[str, dict[str, Any]] = {}
    n_total_epochs = len({ev.get("epoch") for ev in window
                          if ev.get("epoch") is not None})
    skip_triggered: list[str] = []

    for d_id, evs in by_id.items():
        if not evs:
            continue
        run_us = [float(e.get("run_us", 0)) for e in evs
                  if isinstance(e.get("run_us"), (int, float))]
        planned_us = [float(e.get("planned_duration_us", 0)) for e in evs
                      if isinstance(e.get("planned_duration_us"), (int, float))]
        misses = sum(1 for e in evs if e.get("deadline_miss") is True)
        skips = sum(1 for e in evs if e.get("skip_fired") is True)
        targets = collections.Counter(e.get("target") or "?" for e in evs)
        target_now, target_n = targets.most_common(1)[0]
        stickiness = target_n / max(1, len(evs))

        mean_run = sum(run_us) / max(1, len(run_us)) if run_us else 0.0
        mean_planned = (sum(planned_us) / max(1, len(planned_us))
                        if planned_us else 0.0)
        ratio = (mean_run / mean_planned) if mean_planned > 0 else 1.0
        miss_rate = misses / max(1, len(evs))
        skip_rate = skips / max(1, len(evs))

        hints: list[str] = []
        rationale: list[str] = []

        if skip_rate > _SKIP_RATE_FOR_FINER:
            hints.append("prefer_finer")
            rationale.append(
                f"skip rate {skip_rate:.0%} over {len(evs)} epochs")
            skip_triggered.append(d_id)

        if miss_rate > _DEADLINE_MISS_RATE_FOR_FINER:
            if "prefer_finer" not in hints:
                hints.append("prefer_finer")
            rationale.append(
                f"deadline miss rate {miss_rate:.0%}")
            # If misses persist, the cross-cluster cost is likely the
            # dominant cause — surface a fuse hint too.
            hints.append("consider_fuse_with_pred")

        if ratio >= _RUNS_OVER_PLANNED_FOR_FINER:
            if "prefer_finer" not in hints:
                hints.append("prefer_finer")
            rationale.append(
                f"observed/planned = {ratio:.2f} (slower than expected)")
        elif (ratio <= _RUNS_UNDER_PLANNED_FOR_COARSER
              and miss_rate == 0 and skip_rate == 0):
            hints.append("prefer_coarser")
            rationale.append(
                f"observed/planned = {ratio:.2f} (lots of slack)")

        if stickiness >= _PIN_TARGET_STICKINESS and len(evs) >= 4:
            hints.append(f"pin_target={target_now}")
            rationale.append(
                f"ran on {target_now} for {target_n}/{len(evs)} epochs")

        # Drop duplicates while preserving order.
        seen: set[str] = set()
        uniq: list[str] = []
        for h in hints:
            if h not in seen:
                seen.add(h)
                uniq.append(h)
        if not uniq:
            continue

        dispatches[d_id] = {
            "current_target": target_now,
            "idle_fraction": None,  # not derivable from streaming alone
            "transfer_cost_ratio": round(ratio, 4),
            "deadline_slack_us": None,
            "hints": uniq,
            "rationale": "; ".join(rationale),
        }

    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "source_schedule": "streaming_feedback",
        "model_signals": {
            "stream_epochs_in_window": n_total_epochs,
            "stream_events_in_window": len(window),
            "skip_triggered": skip_triggered,
        },
        "dispatches": dispatches,
    }
    return payload


def write_payload(payload: dict[str, Any], out_path: Path) -> dict[str, Any]:
    """Write `xpurt_feedback.json`, MERGING with what is already there.

    The merge is a set-union on hints per dispatch, keyed on `run_id`, which
    is what makes it safe to call repeatedly during a long run: each window
    contributes what it saw, and a dispatch that earned `prefer_finer` in
    instance 4 does not lose it because instance 40 was quiet. A DIFFERENT
    run_id starts fresh -- otherwise a new campaign would inherit the
    conclusions of the last one.
    """
    existing: dict[str, Any] = {}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
        except (OSError, ValueError):
            existing = {}

    merged = payload
    if existing.get("run_id") == payload.get("run_id"):
        dispatches = dict(existing.get("dispatches") or {})
        for did, rec in payload.get("dispatches", {}).items():
            prev = dispatches.get(did)
            if not prev:
                dispatches[did] = rec
                continue
            hints = list(prev.get("hints") or [])
            for h in rec.get("hints") or []:
                if h not in hints:
                    hints.append(h)
            merged_rec = dict(rec)
            merged_rec["hints"] = hints
            dispatches[did] = merged_rec
        merged = dict(payload)
        merged["dispatches"] = dispatches

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, indent=1, sort_keys=True))
    counts: dict[str, int] = {}
    for rec in merged.get("dispatches", {}).values():
        for h in rec.get("hints") or []:
            counts[h.split("=")[0]] = counts.get(h.split("=")[0], 0) + 1
    return {"n_dispatches_with_hints": len(merged.get("dispatches", {})),
            "merged_with_existing": merged is not payload,
            "hint_counts": counts,
            "path": str(out_path)}


def _stream_lines(path: Path, follow: bool, poll_s: float):
    """Generator yielding parsed JSON objects from a JSON-Lines file.

    When follow=True, behaves like `tail -f`: blocks waiting for new
    bytes, with rotation detection (re-opens the file if it shrinks).
    """
    last_inode: Optional[int] = None
    f = None
    try:
        while True:
            if f is None:
                if not path.exists():
                    if not follow:
                        return
                    time.sleep(poll_s)
                    continue
                f = path.open("r", buffering=1)
                try:
                    last_inode = os.fstat(f.fileno()).st_ino
                except OSError:
                    last_inode = None
            line = f.readline()
            if not line:
                if not follow:
                    return
                # Detect rotation: stat the path; if inode differs from
                # what we hold, reopen.
                try:
                    st = path.stat()
                    if last_inode is not None and st.st_ino != last_inode:
                        f.close()
                        f = None
                        continue
                except OSError:
                    pass
                time.sleep(poll_s)
                continue
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Bad line; skip without bringing down the daemon.
                continue
    finally:
        if f is not None:
            f.close()


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--telemetry-stream", required=True, type=Path,
                   help="JSON-Lines file written by harness_xpurt built with "
                        "-DMODELBLASTER_XPURT_STREAM (or `ssh ... | tee` "
                        "redirected to a local file).")
    p.add_argument("--out", required=True, type=Path,
                   help="xpurt_feedback.json to write. Re-written on every "
                        "post, merging hints for the same run_id.")
    p.add_argument("--windows-from", type=Path, default=None,
                   help="workload spec, so a real DEADLINE MISS can be "
                        "computed. Without it the miss rate is unknown "
                        "rather than zero -- the board does not know its own "
                        "deadlines, and a structural zero that reads as a "
                        "measurement is exactly how the digit-suffixed "
                        "network bug survived.")
    p.add_argument("--run-id", required=True,
                   help="Run identifier for this streaming session. "
                        "Reuse across consecutive POSTs to accumulate "
                        "hints (MCP merge semantics).")
    p.add_argument("--epoch-window", type=int, default=32,
                   help="How many epochs to keep in the rolling window.")
    p.add_argument("--post-every-n-epochs", type=int, default=8,
                   help="Post incremental feedback every N completed "
                        "epochs (counted as distinct epoch ids seen).")
    p.add_argument("--follow", action="store_true",
                   help="tail -f the stream (default exits at EOF).")
    p.add_argument("--poll-s", type=float, default=0.25,
                   help="Sleep between EOF polls when --follow.")
    args = p.parse_args(argv)

    windows_ms: Optional[dict[str, float]] = None
    if args.windows_from:
        spec = json.loads(args.windows_from.read_text())
        windows_ms = {}
        for net in spec.get("networks", []):
            nm = net.get("name")
            w = net.get("window_duration", net.get("period"))
            if nm and w:
                windows_ms[str(nm)] = float(w)
        print(f"[stream-fb] deadlines from {args.windows_from}: "
              f"{windows_ms}", flush=True)
    else:
        print("[stream-fb] no --windows-from: deadline misses are UNKNOWN, "
              "not zero; hints come from measured-vs-predicted duration only",
              flush=True)

    window: collections.deque = collections.deque(maxlen=args.epoch_window
                                                   * 256)
    last_post_epoch: Optional[int] = None
    epochs_since_last_post = 0

    def post_now(reason: str) -> None:
        if not window:
            return
        # Drop events outside the rolling epoch window before deriving.
        epochs_seen = sorted({e.get("epoch") for e in window
                              if e.get("epoch") is not None})
        if not epochs_seen:
            return
        cutoff = epochs_seen[-args.epoch_window] if (
            len(epochs_seen) > args.epoch_window) else epochs_seen[0]
        recent = [e for e in window if (e.get("epoch") is not None
                                        and e["epoch"] >= cutoff)]
        payload = _derive_streaming_hints(recent, args.run_id)
        if not payload["dispatches"]:
            return
        try:
            result = write_payload(payload, args.out)
        except OSError as exc:
            print(f"[stream-fb] write failed ({reason}): {exc}",
                  file=sys.stderr)
            return
        print(f"[stream-fb] wrote ({reason}): "
              f"n_dispatches={result['n_dispatches_with_hints']}, "
              f"merged={result['merged_with_existing']}, "
              f"hint_counts={result['hint_counts']}", flush=True)

    seen_epochs: set[int] = set()
    try:
        for raw in _stream_lines(args.telemetry_stream, args.follow,
                                 args.poll_s):
            ev = normalise_event(raw, windows_ms)
            if ev is None:
                continue          # not a dispatch-end record
            window.append(ev)
            ep = ev.get("epoch")
            if isinstance(ep, int) and ep not in seen_epochs:
                seen_epochs.add(ep)
                if last_post_epoch is None:
                    last_post_epoch = ep
                epochs_since_last_post += 1
                if epochs_since_last_post >= args.post_every_n_epochs:
                    post_now(f"every-{args.post_every_n_epochs}-epochs")
                    epochs_since_last_post = 0
    except KeyboardInterrupt:
        pass

    # Final post on shutdown so partial windows don't go to waste.
    post_now("final")
    return 0


if __name__ == "__main__":
    sys.exit(main())
