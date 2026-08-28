#!/usr/bin/env bash
# B3 -- THE GRANULARITY RUNG. The board step, ready to run.
#
# A runbook, not a driver: it executes nothing by itself. Read section 0, then
# run section 1, then section 2.
#
# All host-side work is DONE and does not have to be reconstructed. Same shape as
# artifacts/k1_run/board_next_commands.sh section 2, with the gate before the
# board step -- but every "must pass first" step in that generic sequence has
# already been run here, and its output is committed beside this file.
set -euo pipefail

XPURT_ROOT=/scratch2/agustin/XPU-RT
MB_ROOT="$XPURT_ROOT/ModelBlaster"
ROUND="$XPURT_ROOT/artifacts/k1_run/round_B3_dronet_split"
PY=python3            # must import `modelblaster` from THIS checkout; the
                      # runners assert that and refuse otherwise.
CROSS=/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-linux-gnu-


# =============================================================================
# 0.  WHAT IS ALREADY ESTABLISHED -- do not redo any of this
# =============================================================================
#
# THE CHOICE, FROM MEASUREMENT (not intuition).
#   split_candidate_ranking.json ranks every split-capable dispatch of the k1
#   3-model workload -- apply_split_hint._SPLITTABLE is {linear_s8, conv2d_s8}
#   only -- by cost relative to its OWN model's free slot:
#
#     model        disp  op          ms       % of model   % of free slot
#     dronet         0   conv2d_s8   2.1651      22.1           9.06   <- chosen
#     dronet        14   conv2d_s8   1.6349      16.7           6.84
#     dronet         9   conv2d_s8   0.9349       9.6           3.91
#     yolov8_nano   82   conv2d_s8   2.4395       1.1           2.51
#
#   yolov8_nano's d82 is marginally larger in absolute terms and much smaller
#   against its 97.3 ms slot. dronet d0 also has depends_on == [], so its tiles
#   are the only pair of dronet dispatches that can start with nothing
#   serialising them -- a split of a mid-graph op buys parallelism its
#   predecessor immediately spends.
#
#   yolov8_nano's HEAVY ops (17.5, 14.8, 9.8 ms) are conv2d_batchnorm2d_silu_s8
#   and are NOT splittable today. That is a real blocker, not a misconfiguration.
#
# THE GATE -- scripts/diff_dispatch_graph.py, exit 0.
#   granularity_diff_x2.json:  21 -> 22 dispatches, op count delta +1
#       - conv2d_s8_N1xIC3xIH112xIW112xOC32x...   (removed)
#       + conv2d_s8_N1xIC3xIH112xIW112xOC16x...   (added, x2)
#   granularity_diff_x4.json:  21 -> 24 dispatches, +3, OC32 -> 4x OC8
#   Both verdicts: GRANULARITY CHANGED. Compare with the precedent this gate
#   exists for -- gen/vmfb/mlp/spacemit_x60/RVV_fused/ holds the SAME five
#   dispatches with the SAME names as its baseline.
#
# CORRECTNESS -- bit-exact, and MEASURED, before any board time.
#   An OC split partitions the set of output channels; it does not reorder any
#   accumulation, because each output element's sum over (IC, KH, KW) is
#   unchanged. Requantization is elementwise integer. So max_abs_err MUST be
#   exactly 0 -- numeric drift is NOT inherent here and any nonzero value is a
#   bug, not tolerance. (A K-dim/reduction split would be the case where that
#   argument has to be made rather than asserted; this rewrite does not do one.)
#
#   Measured on the BUILD HOST with the new `generate_skeleton --platform host`
#   (clock_gettime instead of rdtime, so model.c assembles on x86):
#     tile_tensor_equality.json
#       x2, x4: dronet conv_modules_0, all 100352 int8 elements, max_abs_diff 0,
#               and the tiles are NOT copies of each other
#       golden verify (MODELBLASTER_VERIFY): max_abs_err=0 for base, x2, x4
#       negative control: tile 1's weight/bias offsets stripped from the emitted
#               C -- the historical `tile_offset_N` defect, reintroduced by hand
#               -- gives max_abs_diff 180 over 49631 elements and golden
#               max_abs_err=16. The check has teeth.
#
#   ihwoc_tile_weight_check.json -- A REAL BUG FOUND AND FIXED HERE.
#       The rvv backends pack conv weights IHWOC (IC, KH, KW, OC), OC innermost
#       (pipeline/generate_skeleton._backend_pack_weight), so an OC slice is
#       STRIDED. The codegen's `weight + t*tile_oc*IC*KH*KW` is an OIHW formula:
#       correct on scalar, wrong on rvv_x60 for EVERY tile including tile 0 --
#       and additionally the kernel was handed OC=tile_oc, striding the packed
#       array by 16 where it strides by 32. Silent: right IR, passing gate,
#       clean build, wrong numbers. Each tile now gets its own re-packed array
#       (split_conv_tile_weights); the parent is then dead and is not emitted,
#       so it costs no bytes. Verified against weights.npz and pinned by
#       pipeline/tests/test_split_codegen_offsets.py.
#
#   THE LIMIT OF THE HOST CHECK, stated plainly. Only the scalar backend
#   assembles on x86 -- the rvv kernels are intrinsics -- so the host run proves
#   the REWRITE (tiling, output aliasing, dependency rewiring, OIHW pointer
#   math) and NOT the vector kernels. The rvv_x60 tile weights are proven
#   separately and statically: each emitted array equals pack(oihw_slice)
#   element for element under the kernel's own index formula
#   `weight[((ic*KH + kh)*KW + kw)*OC + oc]` -- 0 of 864 entries mismatch --
#   which is a complete argument about the DATA but is not a run. The first
#   execution of the rvv_x60 tiles is section 1, and its max_abs_err is the
#   thing to read.
#
#   To reproduce the host verification (one command, no board):
#     cd $MB_ROOT && python3 scripts/verify_ir_rewrite_host.py \
#       --baseline-ir  $ROUND/graph.baseline.json \
#       --rewritten-ir $ROUND/graph.split_x2.json \
#       --weights build/k1/dronet/int8/weights.npz \
#       --io      build/k1/dronet/int8/io.npz \
#       --tensor  conv_modules_0 --json $ROUND/host_verify_x2.json
#   -> host_verify_x2.json / host_verify_x4.json, both bit_exact: true.
#
# BUILD READINESS -- the board binary already builds and passes both pre-deploy
#   gates on this host:
#     build/b3_cross/dronet_int8_rvv_x60_split_x2_harness
#     scripts/check_kernel_coverage.py  OK (every weighted op has an rvv_x60 kernel)
#     scripts/check_rvv_vtype.py        OK (no vtype-unsafe instruction)
#   So section 1 cannot fail at build time.
#
# WHAT B3 IS FOR, stated honestly. The standing advice set emits ZERO split and
#   ZERO shard recommendations, because the largest single dispatch of any model
#   is at most 18% of its own free slot. B3 is therefore NOT expected to improve
#   deadlines, and a null or negative result IS the deliverable: the rung exists
#   to measure what a granularity change COSTS on this hardware and to prove the
#   loop can execute one end to end. Record whatever comes out.


# =============================================================================
# 1.  BOARD: profile the split dronet.  ONE COMMAND.
# =============================================================================
#
# MB_IR is new and it is the reason this is one command. run_model_k1.sh's step
# 1/5 used to re-extract the model unconditionally, so the obvious recipe --
# copy the rewrite over build/k1/<model>/int8/graph.json, then run the script --
# profiled the BASELINE and filed the results under the rewrite's name. Exactly
# the RVV_fused failure, reached through a runbook step. MB_IR skips the
# extraction and reuses the baseline weights.npz + io.npz, which is also what
# makes max_abs_err a statement about the rewrite rather than about a fresh
# calibration.
#
# Back the baseline profile up first: gen_mb/profile is a symlink to
# gen/profile_mb, so the re-profile OVERWRITES the dronet results.csv that the
# B0/B1/round-5 numbers were solved from.
DRONET_PROF="$XPURT_ROOT/gen/profile_mb/rvv_x60/spacemit_x60/dronet/dronet.int8/dronet_spacemit_x60_rvv_x60_dronet.int8/topo_0"
cp -r "$DRONET_PROF" "$ROUND/baseline_profile_topo_0"
cp "$XPURT_ROOT/gen_mb/vmfb/dronet/spacemit_x60/rvv_x60/dronet.int8/dronet.int8_dispatch_graph.json" \
   "$ROUND/baseline_dispatch_graph.json"

cd "$MB_ROOT"
MB_IR="$ROUND/graph.split_x2.json" \
PROFILE_OUT_ROOT="$XPURT_ROOT/gen_mb/profile" \
  bash scripts/run_model_k1.sh dronet int8 rvv_x60 0
# CORRECTNESS GATE ON THE STDOUT. It must say max_abs_err=0. Anything else and
# the round stops: a rewrite that changes the answer is ineligible
# (candidate_objective GATE_CORRECTNESS) and its timings mean nothing. The host
# run already says 0, so a nonzero here is an rvv_x60-specific fault -- look at
# the tile weight arrays in build/k1/dronet/int8/generated/weights.c first.
#
# -> 22 rows under
#    gen/profile_mb/rvv_x60/spacemit_x60/dronet/dronet.int8/.../topo_0/results.csv
#
# Then the cost question B3 exists to answer, joined on module_name and NEVER on
# dispatch_id (the split renumbers 1..20 -> 2..21):
cd "$XPURT_ROOT"
"$PY" - <<'PYEOF'
import csv, os, sys
sys.path.insert(0, "xpu-rt")
from dispatch_lineage import op_signature
ROUND = "artifacts/k1_run/round_B3_dronet_split"
NEW = ("gen/profile_mb/rvv_x60/spacemit_x60/dronet/dronet.int8/"
       "dronet_spacemit_x60_rvv_x60_dronet.int8/topo_0/results.csv")
OLD = f"{ROUND}/baseline_profile_topo_0/results.csv"
def rows(p):
    return [(op_signature(r["module_name"]), float(r["mean_time_ns"]) / 1e6)
            for r in csv.DictReader(open(p, newline=""))]
if not os.path.exists(NEW):
    sys.exit("no post-split profile yet -- run section 1 first")
old, new = rows(OLD), rows(NEW)
tiles = [(s, ms) for s, ms in new if "OC16" in s]
parent = [ms for s, ms in old if "OC32xOH56" in s]
print(f"baseline dispatches {len(old)}  ->  split {len(new)}")
print(f"parent conv2d_s8 OC32 : {parent[0]:.4f} ms")
for s, ms in tiles:
    print(f"  tile OC16           : {ms:.4f} ms")
tsum = sum(ms for _, ms in tiles)
print(f"sum of tiles          : {tsum:.4f} ms   "
      f"({tsum / parent[0]:.3f}x the parent -- >1 is the granularity COST, "
      f"and the split only pays if the scheduler recovers more than that in "
      f"wall clock)")
print(f"total service         : {sum(ms for _, ms in old):.3f} -> "
      f"{sum(ms for _, ms in new):.3f} ms")
# Everything NOT the split op must be unchanged; a shifted number there means
# the run measured something else.
import collections
ob, nb = collections.Counter(), collections.Counter()
for s, ms in old:
    if "OC32xOH56" not in s: ob[s] += 1
for s, ms in new:
    if "OC16" not in s: nb[s] += 1
print("untouched signature multiset identical:", ob == nb)
PYEOF


# =============================================================================
# 2.  RE-EMIT THE SCHEDULER'S GRAPH, RE-SOLVE, RUN, SCORE
# =============================================================================
#
# The solver reads the dispatch graph and the results.csv as a PAIR. Re-emit the
# graph or `load_profiled_processing_times` (strict=True) raises on dispatch 21
# having no cost -- which is the right failure, but do not spend a board slot
# discovering it.
cd "$MB_ROOT"
PYTHONPATH="$MB_ROOT/src:$MB_ROOT" "$PY" -m modelblaster.pipeline.emit_dispatch_graph \
  --ir "$ROUND/graph.split_x2.json" --out-root "$XPURT_ROOT/gen_mb/vmfb" \
  --target spacemit_x60 --hw rvv_x60
# -> 22 entries; dispatch_0 and dispatch_1 both have dependencies [], and
#    dispatch_2 (the maxpool) depends on BOTH. That pair of empty dependency
#    lists is the whole point of the rung: it is the only concurrency a
#    granularity change can hand the scheduler here.

# 2f. Re-solve. `pdb_hash` in the output MUST change -- if it did not, the solve
#     read the old costs and the round is void.
cd "$XPURT_ROOT"
"$PY" scripts/run_xpurt_schedule.py \
  --networks-json data/toplevel/networks_k1_mb_3model_4hz.json \
  --solver greedy --profiled
# Check the tiles actually landed on different cores; if the solver serialised
# them on one hart there is no parallelism to measure and the rung is answered
# already (report that -- it is a result about the scheduler, not a failure).
"$PY" - <<'PYEOF'
import json
s = json.load(open("schedules/scheduled_networks_k1_mb_3model_4hz_greedy_profiled.json"))
ent = [e for e in s.get("schedule", s.get("entries", []))
       if str(e.get("network", e.get("model", ""))).startswith("dronet")
       and int(e.get("dispatch_id", -1)) in (0, 1)]
for e in sorted(ent, key=lambda e: (e.get("network"), e["dispatch_id"]))[:8]:
    print(e.get("network"), e["dispatch_id"], e.get("machine") or e.get("core"),
          e.get("start"), e.get("end"))
PYEOF

# 2g. Run the multi-model schedule on the board.
cd "$MB_ROOT"
CORE_KINDS=rvv bash scripts/run_xpurt_k1.sh \
  --schedule "$XPURT_ROOT/schedules/scheduled_networks_k1_mb_3model_4hz_greedy_profiled.json" \
  --models mlp_control,dronet,yolov8_nano --backends rvv_x60
#   NOTE: run_xpurt_k1.sh reuses build/k1_xpurt/<model>/int8/graph.json when it
#   exists, so stage the split IR there and remove the stale generated dir:
#     cp $ROUND/graph.split_x2.json $MB_ROOT/build/k1_xpurt/dronet/int8/graph.json
#     rm -rf $MB_ROOT/build/k1_xpurt/dronet/int8/rvv_x60

# 2h. Score it. Lexicographic, deadline-first. Incumbent:
#     results/k1_ladder_mb/trace_3model_4hz.csv -- 0/123 misses, makespan 908.40 ms.
cd "$XPURT_ROOT"
"$PY" - <<'PYEOF'
import json, sys
sys.path.insert(0, "xpu-rt")
import trace_metrics as tm
sched = "schedules/scheduled_networks_k1_mb_3model_4hz_greedy_profiled.json"
periods = json.load(open(sched))["metadata"]["periodic_networks"]
for label, trace in (("B0/B1 incumbent", "results/k1_ladder_mb/trace_3model_4hz.csv"),
                     ("B3 split_x2",     "<paste the new *_trace.csv here>")):
    try:
        s = tm.summarise_trace(tm.read_trace(trace), periods)
    except OSError:
        print(f"{label}: {trace} not readable yet"); continue
    print(label, s["instance_deadline_misses"], "misses;",
          "makespan_ms", round(s["makespan_us"] / 1000, 2),
          {m: (d["instance_deadline_misses"], d["achieved_frequency_hz"],
               d["response_p99_ms"]) for m, d in s["per_model"].items()})
PYEOF


# =============================================================================
# 3.  RESTORE, and record the round
# =============================================================================
#
# Put the baseline back so B0/B1/round-5 stay reproducible:
#   cp -f $ROUND/baseline_profile_topo_0/results.csv "$DRONET_PROF/results.csv"
#   cp -f $ROUND/baseline_dispatch_graph.json \
#         $XPURT_ROOT/gen_mb/vmfb/dronet/spacemit_x60/rvv_x60/dronet.int8/dronet.int8_dispatch_graph.json
#
# Then append one row to artifacts/k1_run/lineage.jsonl. Transcription only --
# every number copied from an artifact named in `_provenance`, nulls where no
# artifact exists. `changed_dispatches` comes from granularity_diff_x2.json,
# `correctness` from the board stdout's max_abs_err (host-side: 0), and
# `execution_delta` stays null unless section 2g actually ran.
#
# If the 4-way variant is wanted as the second point on the cost-vs-tiles curve,
# repeat section 1 with MB_IR=$ROUND/graph.split_x4.json. It is gated
# (granularity_diff_x4.json) and host-verified (tile_tensor_equality.json) too.
