#!/usr/bin/env python3
"""The K1 board flow: build, deploy, profile, trace, Gantt.

    K1_HOST=root@<board> .venv/bin/python examples/k1_board/board_flow.py

Prints the flow and checks every precondition it can check from here. It does
NOT run the board steps for you — a board is shared, and an example that
silently starts a ten-minute sweep on someone else's hardware is a bad
example.

Without `K1_HOST` it is still worth running: the preconditions are where the
traps are, and all of them can be checked dry.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import MB, REPO, head, need_board, note, step      # noqa: E402


def check_toolchain() -> str | None:
    cross = os.environ.get("CROSS", "")
    if cross and shutil.which(cross + "gcc"):
        try:
            v = subprocess.run([cross + "gcc", "-dumpversion"],
                               capture_output=True, text=True).stdout.strip()
        except OSError:
            v = "?"
        return f"CROSS is set, gcc {v}"
    p = REPO / "scripts" / "setup_spacemit_toolchain.sh"
    if p.exists():
        r = subprocess.run(["bash", str(p), "--path"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            return f"available (not exported): {r.stdout.strip()}"
    return None


def main() -> int:
    head("The K1 board flow")

    step(0, "THE TOOLCHAIN, and it is not a preference")
    tc = check_toolchain()
    print(f"    {tc or 'NOT FOUND'}")
    note("""
    eval "$(scripts/setup_spacemit_toolchain.sh)"

GCC 13.2 -- what CROSS defaults to via chipyard's riscv-tools -- REORDERS the
RVV vsetvl intrinsics so a widening instruction runs under the narrow vtype:

    vsetvli e32,m4      <- sets SEW=32
    vsetvli e8,m1       <- clobbers it to SEW=8
    vle8.v / vle8.v
    vsext.vf4           <- ILLEGAL: widening 8->32 needs SEW=32

The board binary SIGILLs with no stdout at all. The setup script refuses
anything below 14.

GCC 14.3 is wrong in the OPPOSITE DIRECTION and it is worse: it substitutes a
wrong AVL on a CHAINED vsetvl, which computes a wrong answer instead of
crashing. Two committed kernels shipped like that (lstm_s8 err=20,
avgpool2d_s8 err=68). The only safe form is to pass the ELEMENT COUNT to every
width, and `ModelBlaster/scripts/check_rvv_avl.py` is what enforces it.""")

    step(1, "BUILD + PROFILE — one model, one backend, one core width")
    note("""
    PROFILE_OUT_ROOT=$PWD/gen_mb/profile \\
      bash ModelBlaster/scripts/run_model_k1.sh dronet int8 rvv_x60 0

Correctness is not a separate step: run_model_k1.sh golden-compares in-binary
on every run. A profile from a run whose output did not match is not written.""")

    step(2, "MULTI-CORE — MB_CORES derives everything from one place")
    note("""
    MB_CORES=0,1,2,3 ITERS=7 \\
      bash ModelBlaster/scripts/run_model_k1.sh dronet int8 rvv_x60 0

MB_CORES sets the worker-pool width, the affinity mask, the binary suffix AND
the profile's topo_ tag together, so a profile cannot claim a core count it
did not run on. They used to be set separately, and a run tagged topo_0_1_2_3
that actually ran on one core is indistinguishable afterwards from a real one.

Watch out for `int("0_1_2_3") == 123` in Python -- digit separators. That is
how a `topo_123` directory appeared.""")

    step(3, "THE BOARD'S TWO SURPRISES")
    note("""
LOAD AVERAGE HAS A FLOOR OF EXACTLY 2.00, from two D-state kernel threads
(vq0, vq1) that never leave uninterruptible sleep. Any idleness check reading
loadavg concludes the board is busy, forever. Per-CPU /proc/stat is the only
valid busy signal here.

IME (`smt.vmadot`) EXISTS ONLY ON CLUSTER 0. Harts 0-3 execute it; 4-7 SIGILL.
So `{"cpu_p": 8}` -- the runbook's own recommended config -- maps CPU_P#4..7
onto cluster 1 and dies. Use {"cpu_p": 4, "cpu_e": 4} for anything with IME,
and let scripts/check_schedule_feasibility.py refuse the schedule before it is
ever deployed:

    python scripts/check_schedule_feasibility.py <schedule.json>""")

    step(4, "SCHEDULE, then RUN")
    note("""
    python scripts/run_xpurt_schedule.py --networks-json data/toplevel/<spec>.json
    bash ModelBlaster/scripts/run_xpurt_k1.sh <schedule.json>

Add MB_XPURT_STREAM=1 and pipe through `tee` for live per-dispatch telemetry
that xpu-rt/streaming_feedback.py can act on while the run is still going --
the trace block only prints at exit.""")

    step(5, "TRACE -> GANTT")
    note("""
    python scripts/join_k1_trace.py   <trace.csv> <schedule.json>
    python scripts/plot_k1_trace_gantt.py --composite A=t1.csv:s1.json ...

The join answers "is this a slow kernel or a long queue", which is the
question a deadline miss turns on. Ticks are rdtime at 24 MHz -- not the
1.6 GHz core clock, and rdcycle SIGILLs from userspace here.""")

    step(6, "reachability")
    if need_board():
        host = os.environ["K1_HOST"]
        r = subprocess.run(["ssh", "-o", "BatchMode=yes",
                            "-o", "ConnectTimeout=5", host, "true"],
                           capture_output=True)
        print(f"    K1_HOST={host}: "
              f"{'reachable' if r.returncode == 0 else 'NOT reachable'}")
    else:
        print("    K1_HOST not set — nothing was run against hardware.")
        print("    Every number in this repo's board docs came from a real run;")
        print("    none of them came from an example that guessed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
