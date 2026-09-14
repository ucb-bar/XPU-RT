#!/usr/bin/env bash
# Regenerate every deliverable clip from the recorded episodes + the MEASURED
# board traces. Episode choices are fixed here so the clips are reproducible
# even though the harness that recorded them is not.
set -euo pipefail
source /scratch2/dima/miniforge3/etc/profile.d/conda.sh
conda activate octo_sim
R=/scratch2/dima/misc_sw/octo_work/sim_eval/roselite
V=$R/videos
T=/scratch2/dima/misc_sw/XPU-RT/qnn_models/octo/repro_runs
cd "$R"; mkdir -p "$V"

L3="3-way CPU+DSP+HTA int8, median wall of 20 iterations (ungated_20260905-131037.log)"
L1="CPU-only int8 monolith, median wall of 20 iterations (mono_20260905-130719.log)"
LF="fp32 Octo path on the QRB5165 (OCTO_INT8_QRB5165.md section 1)"

# ---------------------------------------------------------------- Task 1
python make_case_video.py \
  --video runs_video/base_0ms_ens-stock/ep02_success_True.mp4 \
  --out "$V/case_0ms_baseline_ensembler-on_SUCCESS.mp4" \
  --latency-ms 0 --schedule serial --latency-note "no latency: the stock free-running configuration" \
  --arm-sr 52.8 --arm-n 72 --episode-success True \
  --source "runs_video/base_0ms_ens-stock ep02  (that recording run: 7/12)" \
  --headline "The reference. Zero latency, stock action ensembler on: a fresh inference every 200 ms and every action acts on the observation that produced it." \
  --note "Representative SUCCESS. This arm is 52.8% — about half its episodes fail."

python make_case_video.py \
  --video runs_video/lat283_pipelined/ep05_success_True.mp4 \
  --out "$V/case_283.4ms_pipelined_SUCCESS.mp4" \
  --latency-ms 283.4 --schedule pipelined --latency-note "$L3" \
  --arm-sr 56.9 --arm-n 72 --episode-success True --n-avg 2 \
  --source "runs_video/lat283_pipelined ep05  (that recording run: 8/12)" \
  --headline "THE HEADLINE. 283.4 ms of MEASURED board latency costs nothing — 56.9% vs the 52.8% baseline, p = 0.74, not significant — IF a fresh inference can be issued every 200 ms. That is a MODELLED control strategy, not a schedule the board has run." \
  --note "Representative SUCCESS. PIPELINED is MODELLED, not measured hardware."

python make_case_video.py \
  --video runs_video/lat283_serial/ep00_success_False.mp4 \
  --out "$V/case_283.4ms_serial_FAILURE.mp4" \
  --latency-ms 283.4 --schedule serial --latency-note "$L3" \
  --arm-sr 15.8 --arm-n 120 --episode-success False \
  --source "runs_video/lat283_serial ep00  (that recording run: 3/12)" \
  --headline "The contrast, and the deployable one. The same 283.4 ms, but one inference in flight: 15.8% vs the pipelined arm's 56.9% (p = 6e-09). It still grasps the eggplant; it cannot place it." \
  --note "Representative FAILURE — 84% of this arm's episodes fail. It grasps but never reaches the basket."

python make_case_video.py \
  --video runs_video/lat555_serial/ep04_success_False.mp4 \
  --out "$V/case_555ms_serial_FAILURE.mp4" \
  --latency-ms 555 --schedule serial --latency-note "$LF" \
  --arm-sr 9.7 --arm-n 72 --episode-success False \
  --source "runs_video/lat555_serial ep04  (that recording run: 2/12)" \
  --headline "555 ms serial: the grasp itself starts to go. Grasp rate falls from 61.7% at 283.4 ms to 47.2% here, and success to 9.7%." \
  --note "Representative FAILURE — never achieves a grasp (53% of this arm's episodes do not)."

python make_case_video.py \
  --video runs_video/lat555_pipelined/ep04_success_False.mp4 \
  --out "$V/case_555ms_pipelined_FAILURE.mp4" \
  --latency-ms 555 --schedule pipelined --latency-note "$LF" \
  --arm-sr 20.8 --arm-n 72 --episode-success False --n-avg 1 \
  --source "runs_video/lat555_pipelined ep04  (that recording run: 4/8)" \
  --headline "At 555 ms even the optimistic pipelined strategy breaks: only ONE prediction still targets the current step, so the averaging that rescued 283.4 ms is gone. 20.8% vs 52.8% baseline, p = 1e-04." \
  --note "Representative FAILURE. PIPELINED is MODELLED, not measured hardware."

python make_case_video.py \
  --video runs_video/lat684_serial/ep03_success_False.mp4 \
  --out "$V/case_684.8ms_serial_FAILURE.mp4" \
  --latency-ms 684.8 --schedule serial --latency-note "$L1" \
  --arm-sr 3.3 --arm-n 120 --episode-success False \
  --source "runs_video/lat684_serial ep03  (that recording run: 1/12)" \
  --headline "The CPU-only int8 monolith: 684.8 ms, one observation every 4 control steps. 3.3% success, and only 26.7% of episodes manage a grasp at all." \
  --note "Representative FAILURE — flails near the eggplant and never grasps it."

python make_case_video.py \
  --video runs_video/lat684_pipelined/ep00_success_False.mp4 \
  --out "$V/case_684.8ms_pipelined_DEAD_never-moves.mp4" \
  --latency-ms 684.8 --schedule pipelined --latency-note "$L1" \
  --arm-sr 0.0 --arm-n 8 --episode-success False --n-avg 0 \
  --source "runs_video/lat684_pipelined ep00  (that recording run: 0/6, all identical)" \
  --headline "STRUCTURALLY DEAD. At D = 4 the 4-entry action chunk is already stale when it lands, so ZERO predictions target the current step, the timestep-aligned average is empty, and the robot holds forever. The stillness is the result." \
  --note "Every applied action is the zero-delta HOLD: sum |action| = 0.000 over all 120 steps, in all 6 recorded episodes."

# ---------------------------------------------------------------- Task 2
python animate_octo_gantt.py \
  --trace "$T/ungated_20260905-112925.log" \
  --video runs_video/lat283_serial/ep00_success_False.mp4 \
  --out "$V/gantt_283.4ms_serial_3way-CPU-DSP-HTA.mp4" \
  --latency-ms 283.4 --config-label "3-way CPU+DSP+HTA int8" \
  --latency-note "(283.4 ms = MEASURED median over 20 iterations)" \
  --arm-sr 15.8 --arm-n 120 --episode-success False \
  --episode-label "runs_video/lat283_serial ep00" --window-ms 1600

python animate_octo_gantt.py \
  --trace "$T/mono_20260905-130719.log" \
  --video runs_video/lat684_serial/ep03_success_False.mp4 \
  --out "$V/gantt_684.8ms_serial_CPU-only-monolith.mp4" \
  --latency-ms 684.8 --config-label "CPU-only int8 monolith" \
  --latency-note "(684.8 ms = MEASURED median over 20 iterations)" \
  --arm-sr 3.3 --arm-n 120 --episode-success False \
  --episode-label "runs_video/lat684_serial ep03" --window-ms 2600 --slow-steps 12
echo "### VIDEOS DONE"

# ---------------------------------------------------------------- Task 3
# The two MEASURED PIPELINED replays. Unlike Task 2 these draw a genuinely
# OVERLAPPING execution: multi-instance walks actually run on the QRB5165,
# coloured by Octo instance. Every lane/overlap/latency/cadence number on the
# frame is recomputed from the trace being drawn -- nothing below is hardcoded
# into the figure; the values in --latency-ms are cross-checked against the
# trace and warn if they disagree by >12%.
#
# --control-trace is the SERIAL walk from the SAME board session. Its overlap
# is recomputed too and printed beside the pipelined one, so "overlapping" is a
# measured contrast against a measured baseline (0.0%) rather than a claim.
#
# Both use the same rollout on purpose: at the 110 ms cadence per-inference
# latency rises 260 -> 385 ms, which is still D = 2, so the CONTROL side is
# identical to the 200 ms arm. The extra throughput buys the controller
# nothing, and the clip says so instead of implying a better arm.

python animate_octo_gantt.py --schedule pipelined \
  --trace "$T/pipe200_gated_20260905-164357.log" \
  --control-trace "$T/ungated_20260905-164735.log" \
  --video runs_video/lat283_pipelined/ep05_success_True.mp4 \
  --out "$V/gantt_pipelined_200ms_MEASURED.mp4" \
  --latency-ms 260.5 --issue-period-ms 200 --config-label "3-way CPU+DSP+HTA int8" \
  --latency-note "(pipe200_gated_20260905-164357.log)" \
  --throughput-note " wall 1159.2 ms / 5 inf = 231.8 ms, 4.31 inf/s (median of 6)" \
  --headline "THE HEADLINE — the board sustains one completed inference per 200 ms control tick, the cadence the 56.9% pipelined arm assumes. MEASURED, overlapping, on hardware." \
  --arm-sr 56.9 --arm-n 72 --episode-success True \
  --episode-label "runs_video/lat283_pipelined ep05" --window-ms 1600

python animate_octo_gantt.py --schedule pipelined \
  --trace "$T/pipe110x10_20260905-165246.log" \
  --control-trace "$T/ungated_20260905-164735.log" \
  --video runs_video/lat283_pipelined/ep05_success_True.mp4 \
  --out "$V/gantt_pipelined_110ms_MEASURED.mp4" \
  --latency-ms 385.1 --issue-period-ms 110 --config-label "3-way CPU+DSP+HTA int8" \
  --latency-note "(pipe110x10_20260905-165246.log)" \
  --throughput-note " wall 1176.6 ms / 10 inf = 117.7 ms, 8.50 inf/s (median of 20)" \
  --headline "THE THROUGHPUT CEILING — 8.50 inf/s, all three lanes busy at once. But the env still steps at 200 ms, so half these inferences re-read a frame already read, and per-inference latency rises 260 → 385 ms. D is still 2: the extra throughput buys the controller nothing." \
  --arm-sr 56.9 --arm-n 72 --episode-success True \
  --episode-label "runs_video/lat283_pipelined ep05" \
  --arm-note " same arm as the 200 ms walk — D unchanged at 2" --window-ms 1600
echo "### PIPELINED VIDEOS DONE"
