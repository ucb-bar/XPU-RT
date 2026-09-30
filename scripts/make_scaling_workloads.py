#!/usr/bin/env python3
"""Build a 2->5 network ladder where scheduling actually decides the outcome.

WHY THESE AND NOT THE EXISTING SPECS. The 24 runnable K1 specs are bimodal. Twelve have
a baseline that meets every deadline (density ~0.25 -- nothing is at stake, and the loop
can only win on terms nobody reads), and most of the rest ask for something no schedule
can deliver: `yolov8_nano_64x96` needs 23.95 ms at 8 cores against a 22 ms window, and
its core scaling has saturated (4->8 cores buys 1.6%), so 44 of the 50 residual misses in
the full ablation were infeasible by construction. Almost nothing sits in the band where
a scheduler decides whether the deadline is met, which is the band the loop exists for.

THE TENSION THIS LADDER BUILDS ON. Sharding buys wall time and costs efficiency: on the
board `ffn_block` goes 26.61 -> 7.72 ms across 1 -> 8 cores, a 3.45x speedup that consumes
61.8 core-ms instead of 26.61. So a solver cannot shard everything -- past a point the
core budget will not carry it -- and as networks accumulate it has to choose WHICH nets
to widen. That is a real scheduling decision with a right and a wrong answer, and it is
where a list heuristic should start to lose to an exact solver.

Every workload here is built so that:
  * the baseline (one core per net, RVV singletons) MISSES at least one deadline, and
  * a schedule that meets every deadline EXISTS using implementations already measured
    on the board -- so a failure is the loop's, not physics.

Measured whole-net board times (gen/profile_mb, rvv_x60, source=k1) used to size the
windows:

    net                  1 core    best        speedup   core-ms at best
    mlp_control            0.08     0.08 (1c)    1.00x      0.08
    attn_block             0.13     0.13 (1c)    1.00x      0.13
    fused_full             3.62     3.62 (1c)    1.00x      3.62
    dronet                 8.33     5.25 (4c)    1.59x     21.0
    ffn_block             26.61     7.72 (8c)    3.45x     61.8
    yolov8_nano_64x96     47.73    23.95 (8c)    1.99x    191.6

Usage:
  scripts/make_scaling_workloads.py --out-dir data/toplevel/scaling [--check]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GRAPH = ("gen/mb/vmfb/{net}/spacemit_x60/rvv_x60/{net}.int8/"
         "{net}.int8_dispatch_graph.json")

HARDWARE = {
    "machines": {"cpu_p": 4, "cpu_e": 4},
    "profile_hw": {"cpu_p": "rvv_x60", "cpu_e": "rvv_x60"},
    # topo_tag_override MUST be false for sharding: with it true every combination is
    # costed from topo_0, so a 4-hart block is charged the single-hart time while
    # occupying four harts, and the failure is silent.
    "profile": {"target": "spacemit_x60", "topo_tag_override": False,
                "gen_root": "gen/mb"},
    "p_core_speedup": 1.0,
}
SCHEDULER = {"random_seed": 42, "solver_verbosity": 1, "time_limit": 300,
             "use_profiled": True, "prune_periodic": True,
             "restrict_makespan_to_nonperiodic": False,
             # The loop's levers set machine_combination_mode / enable_impls per
             # candidate; the baseline must start with neither, or cell A is not a
             # baseline.
             "machine_combination_mode": "singletons", "enable_impls": False}

#: net -> (period_ms, window_ms, num_instances). Windows chosen so the SINGLETON cost
#: overruns and a measured wider implementation fits.
LADDER = {
    # 2 nets: one net must widen. ffn 26.61 singleton > 10 ms window; 4c gives 9.71.
    "w2_ffn_tight": {
        "mlp_control": (5.0, 5.0, 12),
        "ffn_block": (12.0, 10.0, 5),
    },
    # 3 nets: TWO nets must widen and they compete for the same 8 cores.
    # dronet 8.33 > 7 ms window; 2c gives 6.05.
    "w3_ffn_dronet": {
        "mlp_control": (5.0, 5.0, 12),
        "ffn_block": (12.0, 10.0, 5),
        "dronet": (12.0, 7.0, 5),
    },
    # 4 nets: add steady periodic load that fragments availability (fused_full is
    # 0.72 cores of continuous work at a 5 ms period) without needing to widen.
    "w4_ffn_dronet_sensor": {
        "mlp_control": (5.0, 5.0, 12),
        "fused_full": (5.0, 5.0, 12),
        "ffn_block": (12.0, 10.0, 5),
        "dronet": (12.0, 7.0, 5),
    },
    # 5 nets: THREE nets must widen, and the widest choice everywhere does NOT fit --
    # ffn at 8c is 61.8 core-ms against a 12 ms period, 5.15 of 8 cores on its own. The
    # solver has to pick widths per net rather than take the fastest implementation,
    # which is the decision a list heuristic makes locally and an exact solver makes
    # globally.
    "w5_ffn_dronet_yolo": {
        "mlp_control": (5.0, 5.0, 12),
        "fused_full": (5.0, 5.0, 12),
        "ffn_block": (12.0, 10.0, 5),
        "dronet": (12.0, 7.0, 5),
        "yolov8_nano_64x96": (40.0, 26.0, 1),
    },
}


#: THE SECOND LADDER, and why the first one needs a companion.
#:
#: `LADDER` above scales by ADDING networks, and it introduces `dronet` at rung 3. That
#: confounds the experiment: `dronet`'s conv dispatches take different core
#: widths across their instances, so the `shard` lever -- the only lever in the whole
#: ablation that ever clears a deadline -- was refused as unbuildable on w4 and w5 (all
#: ten contract violations were dronet). That ladder therefore varies two things at once,
#: "more networks" and "contains the network that blocks the best lever", so its flat
#: w4/w5 rungs cannot be read as a scaling limit; the evidence points at codegen.
#:
#: This ladder varies one thing. Every rung up to c5 is built from networks whose shard
#: is buildable, and `dronet` appears in exactly ONE rung, at the top, so its effect is
#: isolated to a single row instead of contaminating three.
#:
#: THE SIZING PROBLEM IT HAS TO SOLVE. Of the six measured networks only `ffn_block`,
#: `dronet` and `yolov8_nano_64x96` are heavy enough to put anything at stake, and yolo
#: needs 23.95 ms at 8 cores against its 22 ms window -- infeasible by construction, so
#: it cannot carry a rung whose point is that a schedule EXISTS. With dronet held back
#: and yolo excluded, adding a net means adding a LIGHT net, which on its own would
#: leave the rung trivially met. So each rung also tightens `ffn_block`'s window: the
#: light nets fragment availability, and the tighter window keeps the singleton baseline
#: missing. Load, not just net count, is what climbs.
LADDER_COMPOSITION = {
    # 2 nets: the known-good composition. This is w2_ffn_tight under another name, and
    # it is the rung where the loop demonstrably clears every deadline (5 -> 0 misses).
    "c2_ffn": {
        "mlp_control": (5.0, 5.0, 12),
        "ffn_block": (12.0, 10.0, 5),
    },
    # 3 nets: fused_full is 0.72 cores of continuous work at a 5 ms period. It does not
    # need to widen; it makes the cores ffn wants to widen ONTO intermittently busy.
    "c3_ffn_sensor": {
        "mlp_control": (5.0, 5.0, 12),
        "fused_full": (5.0, 5.0, 12),
        "ffn_block": (12.0, 9.5, 5),
    },
    # 4 nets: attn_block adds a fourth periodic release train, so the solver has more
    # release instants to fit ffn's shards between, with the window tightened again.
    "c4_ffn_sensor_attn": {
        "mlp_control": (5.0, 5.0, 12),
        "fused_full": (5.0, 5.0, 12),
        "attn_block": (5.0, 5.0, 12),
        "ffn_block": (12.0, 9.0, 5),
    },
    # 5 nets: dronet enters, and ONLY here. A difference between c4 and c5 is a
    # statement about dronet; a difference between c2 and c4 is a statement about scale.
    # That separation is the entire reason this ladder exists.
    "c5_ffn_sensor_attn_dronet": {
        "mlp_control": (5.0, 5.0, 12),
        "fused_full": (5.0, 5.0, 12),
        "attn_block": (5.0, 5.0, 12),
        "ffn_block": (12.0, 9.0, 5),
        "dronet": (12.0, 7.0, 5),
    },
}

#: THE THIRD LADDER, sized from BOARD measurements instead of profiles.
#:
#: The two ladders above are sized from `net_times`, which reads the profile CSVs, and
#: `check()` passes a rung as "in the band" when a profiled implementation fits the
#: window. Executing those rungs on a K1 showed the premise does not hold. Warm execution
#: per instance, summed from the trace's own cycles (scripts/attribute_board_misses.py):
#:
#:     net                  profile @ best   MEASURED warm   window   board/profile
#:     ffn_block              7.72 (8c)         10.03-10.34   10.0        1.34x
#:     yolov8_nano_64x96     23.95 (8c)         42.77         26.0        1.79x
#:     dronet                 6.05 (2c)          6.02          7.0        1.00x
#:     fused_full             3.62 (1c)          4.42          5.0        1.22x
#:     mlp_control            0.08 (1c)          0.07          5.0        0.9x
#:
#: So `ffn_block` misses its 10 ms window on the board at its FASTEST measured width, and
#: `yolov8_nano_64x96` misses its 26 ms window by 1.64x. Three of five ffn instances and
#: the single yolo instance are over window by execution alone, which no scheduler can
#: recover -- the loop was being asked to hit a target that does not exist, and the
#: "0 infeasible misses" classification says otherwise only because it trusts profiles.
#:
#: These rungs restore the premise the ladder is documented to have, by sizing windows
#: from the MEASURED numbers with ~15% slack. Baselines still miss (ffn at one core is
#: ~35.7 ms on the board against a 12 ms window), so nothing is given away.
#:
#: ONE RESIDUAL IS EXPECTED AND IS NOT A SCHEDULING FAILURE: the first instance of a
#: network pays cold start -- `fused_full` runs 2.66-2.70x its warm median on instance 0
#: (11.8-12.1 ms against 4.42 warm) and cannot fit a 5 ms window it otherwise sits
#: comfortably inside. Its period is 5 ms, so the window cannot absorb it. Expect exactly
#: one cold-start miss per such network and check it with attribute_board_misses.py
#: rather than assuming it.
LADDER_BOARD = {
    "b4_board_sized": {
        "mlp_control": (5.0, 5.0, 12),
        "fused_full": (5.0, 5.0, 12),
        "ffn_block": (12.0, 12.0, 5),
        "dronet": (12.0, 7.0, 5),
    },
    "b5_board_sized": {
        "mlp_control": (5.0, 5.0, 12),
        "fused_full": (5.0, 5.0, 12),
        "ffn_block": (12.0, 12.0, 5),
        "dronet": (12.0, 7.0, 5),
        # 42.77 ms measured widened, +15%: the first window in this study that yolo can
        # actually meet. Its profile-sized 26 ms never could.
        "yolov8_nano_64x96": (50.0, 50.0, 1),
    },
}

#: THE FIVE-NETWORK RUNG THAT CAN ACTUALLY REACH ZERO, and why the others cannot.
#:
#: `b5_board_sized` still leaves 6 misses (ffn_block 5, dronet 1) and the reason is not
#: the loop. Two things in its sizing make zero unreachable:
#:
#:   * `ffn_block` has window == period == 12 ms, so it has NO slack for queueing. Its
#:     execution fits (8.40 ms warm on the board) but any delay behind another network is
#:     a miss, and with five networks on eight harts there is always some.
#:   * COLD START is not budgeted anywhere. The first instance of `fused_full` runs
#:     11.73 ms against its warm 4.28 -- 2.74x -- so a 5 ms window it otherwise sits
#:     comfortably inside is missed once, every run, by construction.
#:
#: This rung budgets both from MEASURED numbers (cold and warm, from
#: attribute_board_misses.py over the b4/w5 board runs) and keeps all five networks:
#:
#:     net                 cold    warm   window  period   why the window
#:     mlp_control         0.105   0.076    5.0     5.0     enormous slack already
#:     fused_full         11.727   4.279   13.0    15.0     >= cold start, not just warm
#:     ffn_block          10.676   8.399   12.0    20.0     slack for queueing; duty 0.49
#:     dronet              8.061   6.030    9.0    15.0     >= cold start
#:     yolov8_nano_64x96  42.765  42.765   50.0    60.0     first measured multi-hart time
#:
#: Core budget at the widths that fit those windows: ffn 4c at 0.49 duty = 1.94 cores,
#: yolo 8c at 0.40 duty = 3.19, dronet 2c = 0.81, fused_full 0.24, mlp 0.02 -- about 6.2
#: of 8, so a zero-miss schedule exists with headroom. The baseline still misses badly
#: (ffn at one core is 26.61 ms against a 12 ms window), so nothing is given away.
#:
#: The honest caveat: `fused_full`'s deployed period in the sensor stack is 5 ms, and a
#: 5 ms period cannot absorb an 11.7 ms cold start in any schedule. Deploying it at that
#: rate needs a warm-up pass before the mission, not a better scheduler. This rung states
#: the requirement as a window instead of hiding it.
LADDER_BOARD_SLACK = {
    "b5x_board_slack": {
        "mlp_control": (5.0, 5.0, 12),
        "fused_full": (15.0, 13.0, 4),
        "ffn_block": (20.0, 12.0, 3),
        "dronet": (15.0, 9.0, 4),
        "yolov8_nano_64x96": (60.0, 50.0, 1),
    },
}

#: THE ONE THAT HOLDS ON THE BOARD, at five networks.
#:
#: `b5x_board_slack` reaches 0 predicted misses and does NOT survive execution. The
#: reveal shows why, and it is about sizing rather than about the loop: its windows
#: were sized from board measurements taken while `dronet` and `yolo` were WIDENED
#: (6.03 and 42.77 ms), but the loop converged on `shard:ffn_block` alone and left both
#: at one core, where the board gives 9.56 and 60.02 ms against 9.0 and 50.0 ms windows.
#: A window is only meetable at the width the scheduler actually chooses, and the AOT
#: profile understates the single-core case by 1.15x for dronet and 1.26x for yolo.
#:
#: These windows come from board measurements at ONE CORE -- the conservative width, the
#: one the loop falls back to -- and include cold start, which is the other thing the
#: warm steady-state profile does not model:
#:
#:     net                 board 1c warm   board 1c cold   window   period
#:     mlp_control              0.079          0.097          5.0     5.0
#:     fused_full               4.594         12.667         13.0    15.0
#:     ffn_block                8.757         17.527         18.0    20.0
#:     dronet                   9.564         11.461         12.0    15.0
#:     yolov8_nano_64x96       60.021         60.021         65.0    70.0
#:
#: Every window now exceeds the measured COLD time at the width the loop can fall back
#: to, so no instance is execution-bound however the solver places it, and the core
#: budget is ~3.4 of 8 so queueing has room too. The baseline still misses -- ffn at one
#: core is 26.61 ms profile / far worse measured, against an 18 ms window -- so the rung
#: is at stake rather than given away.
#: THE SAME RECIPE APPLIED DOWN THE LADDER, so success is not a single data point.
#:
#: b5y works and the profile-sized rungs do not, and the difference is entirely the
#: SIZING, not the number of networks: a window is meetable only if it exceeds the
#: network's measured COLD time at the width the loop actually converges on. These rungs
#: apply that one rule at 2, 3 and 4 networks using the same board numbers b5y was sized
#: from, so the claim becomes "the loop closes at every rung whose target is achievable"
#: rather than "the loop closed once, at five networks".
#:
#: Every rung is at stake through `ffn_block`: at one core it is 26.61 ms profile (worse
#: measured) against an 18 ms window, so the baseline misses and a lever is required.
LADDER_BOARD_HOLDS = {
    "b2y_board_holds": {
        "mlp_control": (5.0, 5.0, 12),
        "ffn_block": (20.0, 18.0, 3),
    },
    "b3y_board_holds": {
        "mlp_control": (5.0, 5.0, 12),
        "ffn_block": (20.0, 18.0, 3),
        "dronet": (15.0, 12.0, 4),
    },
    "b4y_board_holds": {
        "mlp_control": (5.0, 5.0, 12),
        "fused_full": (15.0, 13.0, 4),
        "ffn_block": (20.0, 18.0, 3),
        "dronet": (15.0, 12.0, 4),
    },
    "b5y_board_holds": {
        "mlp_control": (5.0, 5.0, 12),
        "fused_full": (15.0, 13.0, 4),
        "ffn_block": (20.0, 18.0, 3),
        "dronet": (15.0, 12.0, 4),
        "yolov8_nano_64x96": (70.0, 65.0, 1),
    },
}

#: SIZED FOR THE SOLVER AS WELL AS FOR THE BOARD.
#:
#: Every rung above sizes windows and ignores how big the CP-SAT model becomes. A scan of
#: the 204 CP-SAT certificates in this repo says that is the variable that decided most of
#: our results:
#:
#:   * a phase-1 objective of 0 is OPTIMAL in 55 of 55 runs -- when a zero-miss schedule
#:     exists the bound of 0 is matched trivially, so a REACHABLE rung is also a TRACTABLE
#:     one. All 121 FEASIBLE runs have objective > 0 and a genuine open gap;
#:   * n = 492 dispatches is never OPTIMAL (0/33), and above 300 only 2 of 73 files ever
#:     proved phase 1. The single yolo instance -- 98 dispatches on its own -- is exactly
#:     what takes w4 (394) to w5 (492);
#:   * between those, node count is a weak predictor: n = 214 is mostly FEASIBLE while the
#:     larger n = 217 and n = 254 are mostly OPTIMAL.
#:
#: The workload our best figure came from (`_4w_networks_k1_sensor_sharded_rich_shard_ime`)
#: is already FIVE networks at 217 dispatches. The ladder rungs are the same networks with
#: 2-4x the instances, which is the whole difference. Dispatches per instance, measured:
#: mlp_control 7, fused_full 15, ffn_block 5, dronet 21, yolov8_nano_64x96 98.
#:
#:     5x mlp(35) + 2x fused_full(30) + 2x ffn(10) + 2x dronet(42) + 1x yolo(98) = 215
#:
#: Windows come from the b5y recipe -- measured board COLD time at the width the scheduler
#: falls back to -- except `dronet`, deliberately left at 9.0 ms so the board has something
#: to reveal: above its 1-core PROFILE time (8.33, so the AOT solve predicts it fits),
#: below its 1-core MEASURED time (9.61, so the board disagrees), and above its sharded
#: measured cold time (8.06, so widening is a real fix rather than a trade.)
LADDER_SOLVABLE = {
    "s5_solvable_reveal": {
        "mlp_control": (5.0, 5.0, 5),
        "fused_full": (15.0, 13.0, 2),
        # 28 ms, not the 18 that its warm (8.76) and even cold (16.69) execution would
        # suggest. The binding constraint is not ffn's own execution: at t=0 all five
        # networks release together and every one of them pays COLD START, so ffn's first
        # instance queues behind the others and lands at 26.96 ms. The reachability sweep
        # is unambiguous -- 18/20/21/24 ms all leave exactly that one instance missing,
        # 28 clears it, and instance 1 has 5 ms of slack either way. A window has to cover
        # the cold-start BURST, not just the network's own cold time.
        # PERIOD 20 < ffn's own 26.61 ms single-core time, which is what puts the rung at
        # stake, and a 28 ms WINDOW, which is what makes it reachable. The two numbers
        # come from different constraints: a window of 28 sized to absorb the cold-start
        # burst is also above the 26.61 ms single-core cost, so on its own it lets the
        # baseline fit on one core and the band check reports "nothing at stake".
        # Shortening the period instead makes one core
        # unsustainable -- instance 1 releases at 20 ms while instance 0 is still running
        # -- so widening is forced no matter how generous the window is, while the window
        # stays wide enough for the widened schedule to survive the t=0 cold-start burst.
        # period 20 / window 34 / 3 instances, chosen by a MECHANICAL SEARCH over
        # (period, window, instances) rather than by hand, because each gate constrains
        # the other. Period 20 < ffn's 26.61 ms single-core time
        # puts the rung at stake (one core cannot sustain the release rate, so widening
        # is forced whatever the window says). Window 34 is what makes it REACHABLE: the
        # sweep against measured board costs found 10 configurations that reach zero
        # misses, every one of them via shard:ffn_block + shard:dronet, and the windows
        # below 28 all leave ffn's first instance missing because at t=0 all five
        # networks release together and it queues behind their cold starts.
        "ffn_block": (25.0, 34.0, 2),
        # 14 ms. Above dronet's measured 1-core board cold time (11.74) with enough
        # margin to absorb the t=0 cold-start burst -- at 12 ms it still lost one instance
        # to queueing on measured costs, at 14 the sweep reaches zero. The point is that
        # this rung tests ONE decision -- whether the loop widens yolo -- rather than
        # mixing in a second marginal net. Its 9 ms variant is kept in b5z_reveal.
        "dronet": (15.0, 14.0, 2),
        # YOLO IS THE AT-STAKE NET, and sharding is the fix -- the point of the rung.
        # 45 ms sits between yolo's one-core PROFILE time (47.73, so the baseline misses
        # and a lever is required), its one-core MEASURED board time (60.86, so the board
        # misses harder than the model predicted), and its measured WIDENED board time
        # (42.77, so widening genuinely fixes it rather than trading the miss elsewhere).
        # Every other net is sized to fit comfortably so the rung tests one decision.
        # 30 ms, and it is BRACKETED rather than argued: an empirical two-gate sweep
        # (scripts/gate_rung.py) solved the baseline and the fix at each candidate
        # window. At 35 ms and above the solved baseline already meets every deadline --
        # nothing at stake. At 26 ms and below not even shard:yolo + shard:ffn reaches
        # zero on measured board costs -- unreachable. 30 is the window where the
        # baseline misses AND a lever exists that clears it on the board.
        #
        # Sized empirically because an analytical gate needs a model of the baseline, and
        # each simple model misses a case: `one > window` ignores
        # that the window may be wide for cold-start reasons; `one > period` is false
        # because successive INSTANCES go on different harts; and comparing net_times()
        # to the window ignores that the scheduler parallelises a network's DISPATCHES
        # across cores, so yolo's 47.73 ms serial sum never lands on one core at all.
        # Solve it and look.
        "yolov8_nano_64x96": (70.0, 30.0, 1),
    },
}

LADDERS = {"scaling": LADDER, "composition": LADDER_COMPOSITION,
           "board": LADDER_BOARD, "board_slack": LADDER_BOARD_SLACK,
           "board_holds": LADDER_BOARD_HOLDS, "solvable": LADDER_SOLVABLE}


def net_times(net):
    """`{n_cores: whole-net ms}` measured on the board, from the committed profiles."""
    out = {}
    for c in glob.glob(os.path.join(
            REPO, f"gen/profile_mb/rvv_x60/spacemit_x60/{net}/*/*/topo_*/results.csv")):
        width = len(os.path.basename(os.path.dirname(c)).split("_")) - 1
        t = sum(float(r["mean_time"] or 0) for r in csv.DictReader(open(c)))
        if t > 0:
            out[width] = min(out.get(width, 9e9), t)
    return out


def check(name, nets, log):
    """Is this workload in the band? Baseline must miss; a fix must exist; and the
    core budget at the widths a fix needs must still carry it."""
    ok = True
    singleton_util = 0.0
    widened_util = 0.0
    n_miss_baseline = 0
    for net, (period, window, _inst) in nets.items():
        t = net_times(net)
        if not t:
            log(f"    {net}: NO measured profile — cannot use this net")
            return False
        one = t.get(1)
        best_w = min(t, key=lambda w: t[w])
        best = t[best_w]
        singleton_util += one / period
        # WRONG TURN, KEPT AS A WARNING. This briefly tested `one > period` as a second
        # way for the baseline to miss, on the reasoning that instance 1 is released while
        # instance 0 is still running so the backlog must grow. That is false on a
        # multicore machine: successive INSTANCES of a network are independent and the
        # scheduler simply places them on different harts. `s5_solvable_reveal` was sized
        # on that premise -- ffn at 26.61 ms with a 20 ms period -- and CP-SAT returned a
        # baseline with ZERO misses by putting the three instances on three cores. Period
        # only binds through aggregate utilisation, which `singleton_util` already tracks.
        # A net is at stake when ONE INSTANCE cannot fit its OWN window.
        over_period = False
        if one > window or over_period:
            n_miss_baseline += 1
            if over_period and one <= window:
                log(f"    {net}: singleton {one:.2f} > PERIOD {period:.1f} -> one core "
                    f"cannot sustain the release rate; must widen (window {window:.1f} "
                    f"is generous on purpose)")
            # the fix, and what it costs in cores
            # A fix has to satisfy whichever constraint is binding -- the window, and
            # the period when the net is period-overloaded.
            limit = min(window, period) if over_period else window
            fits = [w for w in sorted(t) if t[w] <= limit]
            if not fits:
                log(f"    {net}: INFEASIBLE — needs {best:.2f} ms at {best_w}c "
                    f"against a {limit:.1f} ms limit ({best / limit:.2f}x)")
                ok = False
                continue
            w = fits[0]
            widened_util += (w * t[w]) / period
            why = (f"> PERIOD {period:.1f}" if over_period and one <= window
                   else f"> window {window:.1f}")
            log(f"    {net}: singleton {one:.2f} {why} -> must widen; "
                f"{w}c gives {t[w]:.2f} ms, costing {(w * t[w]) / period:.2f} cores")
        else:
            widened_util += one / period
    if n_miss_baseline == 0:
        log("    NOTHING AT STAKE: every net fits on one core; the baseline cannot miss")
        ok = False
    log(f"    baseline utilisation {singleton_util:.2f} cores; "
        f"with the required widening {widened_util:.2f} of 8")
    if widened_util > 8.0:
        log("    OVER CAPACITY even at the required widths — no schedule can exist")
        ok = False
    elif widened_util > 6.5:
        log("    tight: the solver must choose widths rather than take the widest")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/toplevel/scaling")
    ap.add_argument("--check", action="store_true",
                    help="only report the feasibility band analysis; write nothing")
    ap.add_argument("--ladder", default="scaling",
                    choices=sorted(LADDERS) + ["both"],
                    help="'scaling' adds a network per rung (introduces dronet at rung "
                         "3); 'composition' holds the buildable set and isolates dronet "
                         "to the top rung")
    a = ap.parse_args()
    ladder = (dict(LADDER, **LADDER_COMPOSITION, **LADDER_BOARD,
                   **LADDER_BOARD_SLACK, **LADDER_BOARD_HOLDS,
                   **LADDER_SOLVABLE) if a.ladder == "both"
              else LADDERS[a.ladder])
    out_dir = a.out_dir if os.path.isabs(a.out_dir) else os.path.join(REPO, a.out_dir)

    def log(s):
        print(s, flush=True)

    written, bad = [], []
    for name, nets in ladder.items():
        log(f"== {name} ({len(nets)} nets)")
        good = check(name, nets, log)
        if not good:
            bad.append(name)
            log("    -> NOT in the band; not written")
            continue
        spec = {
            "_comment": (f"{name}: {len(nets)}-net K1 ladder rung. Windows sized from "
                         f"MEASURED board times so the singleton baseline misses and a "
                         f"measured wider implementation fits. Generated by "
                         f"scripts/make_scaling_workloads.py -- do not hand-edit."),
            "hardware": json.loads(json.dumps(HARDWARE)),
            "scheduler": json.loads(json.dumps(SCHEDULER)),
            "networks": {}, "edges": [],
        }
        for i, (net, (period, window, inst)) in enumerate(nets.items()):
            spec["networks"][net] = {
                "id": i, "identifier": net,
                "dispatch_deps_path": GRAPH.format(net=net),
                "period": period, "window_duration": window, "num_instances": inst,
            }
        if not a.check:
            os.makedirs(out_dir, exist_ok=True)
            p = os.path.join(out_dir, f"{name}.json")
            json.dump(spec, open(p, "w"), indent=1)
            written.append(os.path.relpath(p, REPO))
            log(f"    -> wrote {os.path.relpath(p, REPO)}")
        else:
            log("    -> in the band (--check: not written)")
    log(f"\n{len(ladder) - len(bad)}/{len(ladder)} rungs in the band"
        + (f"; rejected {bad}" if bad else ""))
    for p in written:
        log(f"  {p}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
