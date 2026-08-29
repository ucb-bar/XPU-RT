#!/usr/bin/env python3
"""One full revolution of the compiler↔scheduler loop, on real measured data.

    .venv/bin/python examples/feedback_loop/one_revolution.py

    profile -> schedule -> advice -> hint -> rewrite -> (board) -> verdict

Everything up to the board runs here, from profiles already measured on the
K1. The two steps that need hardware say so and stop rather than substituting
a number.

WHAT MAKES IT A LOOP rather than a pipeline is the last arrow. For a long time
this project went profile → schedule → advice → hint → rewrite → reprofile →
**stop**, and every rung was adjudicated by eye on a service-time percentage.
That percentage is term 9 of the 9 in `xpu-rt/candidate_objective.py`, the one
its own docstring calls "never the deciding term". Closing the loop means
scheduling the rewritten graph and asking `accept()`.

Its two worked examples are exactly the cases a percentage gets backwards:

  a split making a kernel 5% slower in total cycles but letting DroNet meet
  30 Hz instead of missing 20% of deadlines is a WIN;
  a fusion making a model 10% faster in isolation but creating an 8 ms
  non-preemptible dispatch that breaks a 100 Hz MLP is a LOSS.

Neither is visible without scheduling the rewritten graph.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import (MB, REPO, gantt, head, need_board, note,   # noqa: E402
                     out_dir, step)

import compile_advice                                            # noqa: E402

SPEC = REPO / "data" / "toplevel" / "networks_k1_mb.json"
MODEL, BASENAME = "ffn_block", "ffn_block.int8"


def main() -> int:
    head("One revolution of the loop")
    d = out_dir("feedback_loop")

    # --------------------------------------------------------------- 1 ----
    step(1, "PROFILE — already measured on the board, per core width")
    by_cores = compile_advice.load_profiles_by_cores_csv(
        "gen_mb", "spacemit_x60", MODEL, BASENAME, "rvv_x60")
    if not by_cores:
        print(f"    SKIP: no profiles for {MODEL} under gen_mb/profile")
        print("    See examples/k1_board/ for how they are produced.")
        return 0
    print(f"    core widths measured: {sorted(by_cores)}")
    print(f"    dispatches: { {k: len(v) for k, v in sorted(by_cores.items())} }")
    note("""
FOUR WIDTHS, not one. That is what lets `shard_advice` fire at all: it is a
claim about how a dispatch's cost changes when given more cores, which cannot
be inferred from a single-core profile. Sharding from op SIZE alone gets the
MLP exactly backwards.""")

    # --------------------------------------------------------------- 2 ----
    step(2, "ADVICE — what the measurement says to do")
    advice = compile_advice.shard_advice(MODEL, by_cores, free_slot_ms=5.0)
    if not advice:
        print("    no shard advice: every dispatch fits its slot, or none "
              "shards profitably. That is an outcome, not a gap.")
        return 0
    for a in advice:
        ev = a.evidence.extra if hasattr(a.evidence, "extra") else {}
        print(f"    dispatch {a.dispatch_id}: {a.recommendation} "
              f"x{a.constraints.get('n_cores')}  "
              f"({ev.get('measured_speedup')}x measured, "
              f"{ev.get('parallel_efficiency')} efficient)")
    advice_path = d / "compile_advice.json"
    compile_advice.write_advice(str(advice_path), advice, schedule_id="example")
    print(f"    -> {advice_path.relative_to(REPO)}")

    # --------------------------------------------------------------- 3 ----
    step(3, "HINT — the same claim in a form the compiler accepts")
    ir_path = MB / "build" / "k1" / MODEL / "int8" / "graph.json"
    if not ir_path.exists():
        print(f"    SKIP: no IR at {ir_path}")
        return 0
    hint_path = d / "shard_hint.json"
    p = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "advice_to_shard_hint.py"),
         "--advice", str(advice_path), "--ir", str(ir_path),
         "--model", MODEL, "--out", str(hint_path)],
        cwd=REPO, capture_output=True, text=True)
    print("    " + "\n    ".join(p.stdout.strip().split("\n")))
    if p.returncode != 0:
        print("    " + "\n    ".join(p.stderr.strip().split("\n")[-4:]))
        return 0
    note("""
The bridge is where a refusal belongs. It re-checks every constraint the
REWRITER enforces -- OC divisibility, op shardability, not-already-a-split --
here, where the advice that caused the refusal is still in hand. A hint the
rewriter rejects gives you an error about a graph, three steps removed from
the measurement that asked for it.""")

    # --------------------------------------------------------------- 4 ----
    step(4, "REWRITE")
    p = subprocess.run(
        [sys.executable, str(MB / "pipeline" / "apply_shard_hint.py"),
         "--ir", str(ir_path), "--hint", str(hint_path),
         "--network", MODEL, "--out", str(d / "graph.rewritten.json")],
        cwd=MB, capture_output=True, text=True)
    print("    " + "\n    ".join((p.stdout or p.stderr).strip().split("\n")))
    if p.returncode != 0:
        return 0

    before = json.loads(ir_path.read_text())
    after = json.loads((d / "graph.rewritten.json").read_text())
    print(f"    dispatch count {len(before['ops'])} -> {len(after['ops'])}")
    note("""
Unchanged, because this is a shard. A split here would have grown the graph
and renumbered every dispatch after the cut, and the profile join would need
the applier's `id_remap` to survive it.""")

    # --------------------------------------------------------------- 5 ----
    step(5, "VERIFY — bit-exact, before any timing is quoted")
    note("""
`ModelBlaster/scripts/verify_ir_rewrite_host.py` compiles baseline and
rewritten and compares outputs. NOTHING downstream of here is meaningful
without max_abs_err=0: a rewrite that changes the answer has not made anything
faster, it has made something else.""")

    # --------------------------------------------------------------- 6 ----
    step(6, "BOARD — reprofile the rewritten graph")
    if not need_board():
        print("    SKIP: no K1_HOST set. This step needs the board, and")
        print("    substituting a modelled cost here is exactly the")
        print("    bookkeeping fiction the loop exists to prevent:")
        print("    a rewrite may not reduce modelled work unless a kernel")
        print("    performs it.")
    else:
        print("    (run ModelBlaster/scripts/run_model_k1.sh with "
              "PROFILE_OUT_ROOT set)")

    # --------------------------------------------------------------- 7 ----
    step(7, "VERDICT — nine terms, not one percentage")
    note("""
    python scripts/compare_candidates.py \\
        --baseline-schedule  schedules/<baseline>.json \\
        --candidate-schedule schedules/<candidate>.json \\
        --windows-from       data/toplevel/<spec>.json

It checks two things BEFORE comparing any term, because each produces a
verdict that looks well-formed and means nothing:

  * `pdb_hash` differing, which proves the two solves read DIFFERENT measured
    costs. Without it the verdict is about scheduler noise.
  * per-model INSTANCE COUNTS matching, which proves they scheduled the same
    amount of work. Not hypothetical: a refinement loop once grew mlp_control
    from 32 instances to 91, the file landed under the baseline's name, every
    term still computed, and the figure reported the opposite verdict.

A tie is a REJECTION. accept() needs the candidate strictly better on some
term before any term it is worse on.""")

    for sched in sorted((REPO / "schedules").glob(f"scheduled_{SPEC.stem}_*.json"))[:1]:
        png = d / "predicted.png"
        if gantt(sched, png, title=f"{sched.stem}"):
            note(f"\nGantt per iteration is a deliverable, not a nicety: "
                 f"scripts/plot_loop_iterations.py renders the series plus a "
                 f"composite, so a rung's verdict can be read against the "
                 f"schedule that produced it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
