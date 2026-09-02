#!/usr/bin/env python3
"""What a workload spec says, and the fields that are load-bearing.

    .venv/bin/python examples/workloads/anatomy_of_a_spec.py

A spec in `data/toplevel/*.json` is the only place several facts live, and
four of its fields have caused wrong answers when misunderstood. This walks a
real one and says what each does — with the failure mode, not just the
meaning, because the meaning is guessable and the failure mode is not.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import REPO, head, note, step            # noqa: E402

SPEC = REPO / "data" / "toplevel" / "networks_k1_mb.json"

FIELDS = [
    ("window_duration",
     "THE DEADLINE, and it is not the period.",
     """
`trace_metrics.summarise_trace` uses `D = windows_ms.get(m, T)` -- the spec's
window_duration IS the deadline, defaulting to the period only when omitted.
Scoring without passing it means scoring against the wrong deadline, and the
score still comes out looking fine.

This is why compare_candidates.py takes --windows-from and streaming_feedback
takes it too."""),

    ("period",
     "how often an instance is released. Instance k is due at k*T + D.",
     """
The instance index comes from the JOB NAME -- `dronet3` is instance 3. Which
is fine until a network name ends in a digit: `yolov8_nano_64x960` split as
`yolov8_nano_64x` + 960, the deadline became ~48 s, and the detector reported
ZERO deadline misses forever. A structural zero that reads exactly like a
pass.

`xpu-rt/job_names.py` owns that split now and needs the KNOWN NETWORK NAMES to
do it. Four copies of the splitter existed before it, each written after the
previous one broke."""),

    ("dispatch_deps_path",
     "the dispatch graph: which dispatches exist and what depends on what.",
     """
Points at ModelBlaster's emitted `*_dispatch_graph.json`. Rewriting a graph
(split, fuse, unfuse) means pointing this at the REWRITTEN one -- and then the
candidate and the baseline are no longer scheduling the same amount of work
unless the instance counts match. compare_candidates.py checks exactly that,
because a refinement loop once grew mlp_control from 32 instances to 91 and
the resulting file sat on disk under the baseline's name."""),

    ("gen_root",
     "which profile tree the costs come from.",
     """
`gen_mb` is ModelBlaster's measured tree; `gen` is the retired IREE one.
Mixing them compares timings from two different runtimes. `gen_mb/profile` is
a SYMLINK to `gen/profile_mb`, which is worth knowing because `find` does not
follow it and will report an empty tree."""),

    ("topo_tag_override",
     "whether a solve may pick a core WIDTH per dispatch.",
     """
`false` is load-bearing for shard-mode solves: it lets the solver read the
per-width profiles (topo_0 / topo_0_1 / topo_0_1_2_3 / ...) and choose,
instead of being told one width for the whole model. Per-dispatch scaling
varies 4.8x WITHIN one model, so one width per model is wrong for nearly
every dispatch in it."""),

    ("enable_impls",
     "whether a solve may pick an IMPLEMENTATION per dispatch.",
     """
Under `scheduler`. With it on, every core-group combination is emitted once
per legal implementation, and each dispatch records the winner as `impl`.
That is how one core runs a MAC-unit GEMM and then a vector one --
`hardware_target` names WHERE, `impl` names WITH WHAT.

The binary honours it: ingest_xpurt_schedule reads `impl` and the walker
selects its dispatch table on it. Before that it selected on core_kind, so a
heterogeneous schedule produced a binary that quietly ran one implementation
everywhere and reported the runtime it got."""),
]


def main() -> int:
    head("Anatomy of a workload spec")
    if not SPEC.exists():
        print(f"SKIP: no spec at {SPEC}")
        return 0

    spec = json.loads(SPEC.read_text())
    step(1, f"{SPEC.relative_to(REPO)}")
    if spec.get("_comment"):
        note(spec["_comment"][:400])

    # `networks` is a MAPPING of name -> spec, not a list. Worth noting
    # because several sibling formats in this repo use a list and the two
    # look identical until you index one.
    nets = spec.get("networks", {})
    items = sorted(nets.items()) if isinstance(nets, dict) else \
        [(n.get("name", "?"), n) for n in nets]
    step(2, f"{len(items)} network(s)")
    for name, net in items:
        bits = [f"{name}"]
        for k in ("period", "window_duration", "instances", "count"):
            if k in net:
                bits.append(f"{k}={net[k]}")
        print(f"    {'  '.join(bits)}")
        if "period" in net and "window_duration" in net:
            if net["window_duration"] != net["period"]:
                print(f"        deadline {net['window_duration']} ms is NOT "
                      f"the period {net['period']} ms -- scoring against the "
                      f"period would be scoring the wrong thing")

    step(3, "the fields that are load-bearing")
    flat = json.dumps(spec)
    for name, headline, why in FIELDS:
        present = f'"{name}"' in flat
        mark = "present" if present else "not in this spec"
        print(f"\n    --- {name}  ({mark})")
        print(f"        {headline}")
        note(why)

    print()
    note("""
IF YOU CHANGE ONE THING, change it in the spec rather than on the command
line. The spec is what gets committed beside a result; a flag is not, and a
figure whose spec does not reproduce it is a figure nobody can check.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
