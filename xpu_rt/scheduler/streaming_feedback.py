"""Host-side daemon that converts on-board telemetry into XPU-RT feedback.

Tails the JSON-Lines stream emitted by `samples/common/xpu-rt/
scheduler_runner.cc`'s telemetry sink, maintains a rolling window of
recent epochs, derives per-dispatch hints, and ingests them into the
Merlin output dir via the `ingest_xpurt_feedback` MCP tool. The MCP tool's
merge semantics (set-union on the same `run_id`) make repeated calls
during a long run safe — hints accumulate rather than overwrite.

Two ingestion paths are supported:
  1. Direct (default): import targetgen_mcp.tools and call
     `dispatch_tool('ingest_xpurt_feedback', {...})`. Requires the merlin
     repo on PYTHONPATH (or invocation from a merlin-dev env).
  2. CLI: subprocess to `<merlin>/tools/targetgen_cmd.py` if available.
     Fallback so this script can be deployed independently of the merlin
     conda env.

Telemetry stream format (one JSON object per line, produced by the C++
TelemetrySink::EmitDispatchEnd):
  {"epoch": int, "dispatch_id": str, "target": str,
   "planned_start_us": int, "start_us": int, "end_us": int,
   "run_us": int, "planned_duration_us": int,
   "deadline_miss": bool, "skip_fired": bool}

Example:
  python xpu-rt/streaming_feedback.py \\
    --telemetry-stream /tmp/telemetry.jsonl \\
    --merlin-dir /scratch2/agustin/merlin/eval/qrb5165/dronet \\
    --run-id qrb5165_dronet_2026-04-28 \\
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


def _post_payload_direct(merlin_dir: Path, payload: dict[str, Any]) -> dict:
    """Path 1: import targetgen_mcp.tools and call dispatch_tool."""
    try:
        from targetgen_mcp.tools import dispatch_tool  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "direct ingest path requires targetgen_mcp on PYTHONPATH; "
            f"import failed: {e}")
    return dispatch_tool("ingest_xpurt_feedback", {
        "merlin_dir": str(merlin_dir),
        "payload": payload,
    })


def _post_payload_subprocess(merlin_dir: Path, payload: dict[str, Any],
                             python_bin: str) -> dict:
    """Path 2: spawn a subprocess that imports targetgen_mcp.tools.

    Used when running outside the merlin-dev env — we shell into it.
    """
    import subprocess
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(payload, f)
        payload_path = f.name
    try:
        cmd = [
            python_bin, "-c",
            "import json, sys, os\n"
            "sys.path.insert(0, os.environ.get('TARGETGEN_TOOLS_DIR', ''))\n"
            "from targetgen_mcp.tools import dispatch_tool\n"
            "merlin_dir = sys.argv[1]\n"
            "with open(sys.argv[2]) as f: payload = json.load(f)\n"
            "result = dispatch_tool('ingest_xpurt_feedback', "
            "{'merlin_dir': merlin_dir, 'payload': payload})\n"
            "sys.stdout.write(json.dumps(result))\n",
            str(merlin_dir),
            payload_path,
        ]
        cp = subprocess.run(cmd, capture_output=True, text=True, check=True,
                            env={**os.environ})
        return json.loads(cp.stdout) if cp.stdout else {}
    finally:
        os.unlink(payload_path)


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
                   help="JSON-Lines file written by scheduler_runner.cc's "
                        "TelemetrySink (or `ssh tail -f` redirected to a "
                        "local file).")
    p.add_argument("--merlin-dir", required=True, type=Path,
                   help="Merlin output dir whose breakdowns/feedback.json "
                        "the MCP ingest tool will write.")
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
    p.add_argument("--ingest-mode", choices=["direct", "subprocess"],
                   default="direct")
    p.add_argument("--python-bin", default=sys.executable,
                   help="Python binary used by --ingest-mode=subprocess "
                        "(typically the merlin-dev env's python).")
    args = p.parse_args(argv)

    if not args.merlin_dir.is_dir():
        print(f"merlin-dir not found: {args.merlin_dir}", file=sys.stderr)
        return 1

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
            if args.ingest_mode == "direct":
                result = _post_payload_direct(args.merlin_dir, payload)
            else:
                result = _post_payload_subprocess(args.merlin_dir, payload,
                                                   args.python_bin)
        except Exception as exc:
            print(f"[stream-fb] POST failed ({reason}): {exc}",
                  file=sys.stderr)
            return
        n = result.get("n_dispatches_with_hints", "?")
        merged = result.get("merged_with_existing", False)
        print(f"[stream-fb] POST ({reason}): "
              f"n_dispatches={n}, merged={merged}, "
              f"hint_counts={result.get('hint_counts', {})}", flush=True)

    seen_epochs: set[int] = set()
    try:
        for ev in _stream_lines(args.telemetry_stream, args.follow,
                                args.poll_s):
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
