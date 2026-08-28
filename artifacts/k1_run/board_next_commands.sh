#!/usr/bin/env bash
# The board step for round 5, ready to run. HOST-SIDE WORK IS DONE; nothing here
# has to be reconstructed. Read the three sections in order and run the one you
# want -- this is a runbook, not a driver, so it does not execute anything by
# itself.
#
# Everything below was derived on 2026-08-28 from:
#   artifacts/k1_run/compile_advice_mb_3model_4hz.json   the regenerated advice
#   artifacts/k1_run/lineage.jsonl                       every round so far
#   artifacts/k1_run/round1_mlp_control/                 the rewrite + its gate
#   docs/k1_modelblaster_xpurt_closed_loop.md            sections 3-6
set -euo pipefail

XPURT_ROOT=/scratch2/agustin/XPU-RT
MB_ROOT="$XPURT_ROOT/ModelBlaster"
PY="$XPURT_ROOT/.venv/bin/python"


# =============================================================================
# 0.  READ THIS FIRST -- what NOT to spend a board slot on
# =============================================================================
#
# * DO NOT re-profile the mlp_control linear_s8+elu_s8 fusion.
#   It has already gone all the way through, on hardware, and it was REJECTED:
#   2892 ticks fused vs 2122 unfused (+36%), correctness PASS. Recorded in
#   lineage.jsonl round 4 with the two causes (elu's per-element expf does not
#   vectorise; the Codex fused kernel emits no vwmacc at all). The host-side
#   rewrite was reproduced in this round only to prove the gate works --
#   artifacts/k1_run/round1_mlp_control/granularity_diff.json, 7 ops -> 4 --
#   not to ask for another measurement.
#
# * The regenerated advice contains NO granularity recommendation. Against the
#   corrected rvv_x60 profiles, the largest single dispatch of any model is at
#   most 18% of that model's own free slot:
#
#       model         period   free slot    total   util%   max disp   max/slot
#       dronet          33.3      23.909    9.789    29.4      2.165      0.091
#       mlp_control     10.0       9.917    0.083     0.8      0.029      0.003
#       yolov8_nano    250.0      97.348  226.865    90.7     17.465      0.179
#
#   so `split` correctly finds nothing, and `shard` cannot be judged at all
#   because the ModelBlaster profile tree holds only topo_0. Section 1 is the
#   measurement that changes that, and it is the highest-value board work
#   available.


# =============================================================================
# 1.  HIGHEST VALUE: multi-core profiles, so `shard` becomes decidable
# =============================================================================
#
# WHY. With 0/123 deadline misses at 4 Hz, every objective term above the last
# is already tied (xpu-rt/candidate_objective.py). The only terms with headroom
# are heavy-model max latency and heavy-model throughput, and yolov8_nano is the
# heavy model: 226.87 ms of work, a 152.65 ms critical path, 90.7% of its period.
# The one lever that shortens a critical path without touching semantics is
# sharding, and `shard_advice` refuses to emit without measured cost-vs-cores.
# That measurement does not exist for any ModelBlaster build.
#
# NOTE the topo_tag/machine_combination_mode pairing trap (runbook section 4):
# 4-hart profiles must be used with `machine_combination_mode: "shard"` and
# `topo_tag_override: false`, never with `"singletons"`.

cd "$MB_ROOT"
for m in yolov8_nano dronet; do
  PROFILE_OUT_ROOT="$XPURT_ROOT/gen_mb/profile" \
    bash scripts/run_model_k1.sh "$m" int8 rvv_x60 0,1,2,3
done
# -> gen_mb/profile/rvv_x60/spacemit_x60/<m>/<m>.int8/<spec>/topo_0_1_2_3/results.csv
# Check the golden verify in the stdout (max_abs_err=0) BEFORE using the numbers.

# Then re-advise. `load_profiles_by_cores_csv` picks the new topo tag up with no
# further change, and `shard_advice` will emit -- or refuse and say why, which is
# equally a result and is recorded either way.
cd "$XPURT_ROOT"
"$PY" scripts/emit_compile_advice.py \
  --gen-root gen_mb --profile-format csv --target spacemit_x60 \
  --impls rvv_x60,scalar --baseline-impl rvv_x60 \
  --models mlp_control:mlp_control.int8,dronet:dronet.int8,yolov8_nano:yolov8_nano.int8 \
  --schedule schedules/scheduled_networks_k1_mb_3model_4hz_greedy_profiled.json \
  --trace results/k1_ladder_mb/trace_3model_4hz.csv \
  --out artifacts/k1_run/compile_advice_mb_3model_4hz_multicore.json


# =============================================================================
# 2.  THE GENERIC REPROFILE + RE-SOLVE SEQUENCE for a rewritten model
# =============================================================================
#
# Use this for any granularity rewrite. Set MODEL and HINT; everything else is
# fixed. Steps 2a-2c are host-side and must ALL pass before step 2d touches the
# board -- 2c is the gate that the RVV_fused precedent skipped.

MODEL=mlp_control                                     # substitute
HINT="$XPURT_ROOT/artifacts/k1_run/round1_mlp_control/fusion_hint.json"
ROUND="$XPURT_ROOT/artifacts/k1_run/round_${MODEL}"
mkdir -p "$ROUND"

# 2a. advice -> hint (fuse_with_successor / fuse_with_predecessor only).
"$PY" "$XPURT_ROOT/scripts/advice_to_fusion_hint.py" \
  --advice "$XPURT_ROOT/artifacts/k1_run/compile_advice_mb_3model_4hz.json" \
  --ir "$MB_ROOT/build/k1/$MODEL/int8/graph.json" \
  --model "$MODEL" --pair-only --out "$HINT"
#   For the dual, splitting one op across cores, use apply_split_hint instead;
#   _SPLITTABLE is {linear_s8, conv2d_s8} only, so yolov8_nano's heavy
#   conv2d_batchnorm2d_silu_s8 ops CANNOT be split today. That is a real
#   blocker, not a configuration mistake.

# 2b. hint -> rewritten IR. Never edit graph.json in place.
cd "$MB_ROOT"
PYTHONPATH="$MB_ROOT" "$PY" -m pipeline.apply_fusion_hint \
  --hint "$HINT" --model "$MODEL" \
  --ir  "$MB_ROOT/build/k1/$MODEL/int8/graph.json" \
  --out "$ROUND/graph.fused.json"

# 2c. THE GATE. Exits 3 if the graph did not change; stop there and report the
#     negative result rather than profiling a rewrite that is not one.
#     Exits 4 if the rewriter's own `id_remap` disagrees with the op
#     signatures -- that is a broken rewriter, not a negative result, and
#     every downstream join that trusts the remap would carry the error.
cd "$XPURT_ROOT"
"$PY" scripts/diff_dispatch_graph.py \
  --before "$MB_ROOT/build/k1/$MODEL/int8/graph.json" \
  --after  "$ROUND/graph.fused.json" \
  --json   "$ROUND/granularity_diff.json"

# 2d. Board: build, run, verify, profile the rewritten IR. run_model_k1.sh reads
#     build/k1/<model>/int8/graph.json, so stage the rewrite there first and keep
#     the original.
cp "$MB_ROOT/build/k1/$MODEL/int8/graph.json" "$ROUND/graph.baseline.json"
cp "$ROUND/graph.fused.json" "$MB_ROOT/build/k1/$MODEL/int8/graph.json"
rm -rf "$MB_ROOT/build/k1/$MODEL/int8/generated"   # stage 1 reuses stale output
cd "$MB_ROOT"
PROFILE_OUT_ROOT="$XPURT_ROOT/gen_mb/profile" \
  bash scripts/run_model_k1.sh "$MODEL" int8 rvv_x60 0
cp "$ROUND/graph.baseline.json" "$MB_ROOT/build/k1/$MODEL/int8/graph.json"
#   Correctness gate: the stdout must carry max_abs_err=0. A rewrite that
#   changes the answer is ineligible (candidate_objective GATE_CORRECTNESS) and
#   its timings mean nothing.

# 2e. Re-emit the dispatch graph the scheduler consumes (7 nodes -> 4).
cd "$MB_ROOT"
PYTHONPATH="$MB_ROOT" "$PY" -m pipeline.emit_dispatch_graph \
  --ir "$ROUND/graph.fused.json" --out-root "$XPURT_ROOT/gen_mb/vmfb" \
  --target spacemit_x60 --hw rvv_x60

# 2f. Re-solve against the NEW profile. This is the step that must not be faked:
#     the scheduler's active cost source is the results.csv under gen_mb/profile,
#     and `pdb_hash` in the output must CHANGE. If it did not, the solve read the
#     old costs and the whole round is void.
cd "$XPURT_ROOT"
"$PY" scripts/run_xpurt_schedule.py \
  --networks-json data/toplevel/networks_k1_mb_3model_4hz.json \
  --solver greedy --profiled

# 2g. Run the multi-model schedule on the board and measure.
cd "$MB_ROOT"
CORE_KINDS=rvv bash scripts/run_xpurt_k1.sh \
  --schedule "$XPURT_ROOT/schedules/scheduled_networks_k1_mb_3model_4hz_greedy_profiled.json" \
  --models mlp_control,dronet,yolov8_nano --backends rvv_x60

# 2h. Score it. Lexicographic, deadline-first; standalone kernel cycles LAST.
#     Compare against the incumbent trace results/k1_ladder_mb/trace_3model_4hz.csv
#     (0/123 misses, makespan 908.40 ms). Join on module_name, never dispatch_id
#     (xpu-rt/dispatch_lineage.py).
"$PY" - <<'PYEOF'
import json, sys
sys.path.insert(0, "xpu-rt")
import trace_metrics as tm
sched = "schedules/scheduled_networks_k1_mb_3model_4hz_greedy_profiled.json"
periods = json.load(open(sched))["metadata"]["periodic_networks"]
for label, trace in (("baseline", "results/k1_ladder_mb/trace_3model_4hz.csv"),
                     ("candidate", "<paste the new *_trace.csv here>")):
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
# 3.  CHEAP AND OPTIONAL: the 4 surviving choose_implementation items
# =============================================================================
#
# The regenerated advice's only actionable items are 3x maxpool2d_s8 in
# yolov8_nano (0.502/0.456/0.448 ms on rvv_x60 vs 0.394/0.359/0.401 scalar) and
# a 1-element sigmoid_s8 in dronet (1.42 us vs 1.21 us). Total at stake: 0.30 ms
# of 226.87 ms, 0.13%.
#
# THE EVIDENCE IS WEAKER THAN THE THRESHOLD. Those three maxpools are the same
# op at the same shape, so they are three replicates of one measurement: the
# within-shape spread is 12.1% (rvv) and 11.6% (scalar) on a single sample each,
# against a 17.9% gap at the mean (per dispatch: 21.5%, 21.2%, 10.4%) and a
# `min_gain` gate of 5%. The 10.4% item (dispatch 42) is not
# distinguishable from that spread at all.
#
# So the honest prerequisite is repetition, not application: add MB_REPS to
# ModelBlaster/harness_linux/src/main.c and take the median (plan Phase 5). Until
# then `results.csv` carries `stat_basis: single_sample_mean` and every item in
# the document is confidence "medium" for exactly this reason.
