#!/usr/bin/env bash
# Fine-grain replay clips: the zero-order hold on stale data drawn per 40 ms
# tick instead of quantised into 200 ms granules.
set -euo pipefail
source /scratch2/dima/miniforge3/etc/profile.d/conda.sh
conda activate octo_sim
cd /scratch2/dima/misc_sw/octo_work/sim_eval/roselite/finegrain
mkdir -p videos
A () { python animate_fine_replay.py "$@"; }

pick () {  # dir -> first episode index with the requested outcome (True/False)
  local d=$1 want=$2
  for f in "$d"/ep*_success_${want}.mp4; do
    [ -e "$f" ] || continue
    basename "$f" | sed 's/^ep0*\([0-9]*\)_.*/\1/'; return
  done
  echo ""
}

# ---- 1. zero-latency control: the validation arm -------------------------
E=$(pick runs_video/vid_ctrl_lat0 True)
A --run runs_video/vid_ctrl_lat0 --ep "${E:-0}" \
  --out videos/fine_0ms_control_VALIDATED.mp4 \
  --latency-ms 0 --cadence-ms 200 --arm-sr 52.8 --arm-n 72 \
  --config-label "zero-latency control (validation arm)" \
  --headline "THE VALIDATION. Fine 40 ms stepping with the deltas rescaled by 1/5 reproduces the 5 Hz baseline EXACTLY: 38/72 = 52.8%, the same 38/72 the stock harness gets. Every latency number below rests on this." \
  --note "Green = a fresh policy result landed on this tick. Red = zero-order hold on an already-applied action. At zero latency the policy still only runs every 200 ms, so 4 of every 5 ticks are held: that is the 5 Hz baseline's OWN staleness, mean 80 ms / max 160 ms."

# ---- 2. 117.7 ms pipelined: faster than the nominal 5 Hz -----------------
E=$(pick runs_video/vid_pipe110 True)
A --run runs_video/vid_pipe110 --ep "${E:-0}" \
  --out videos/fine_117.7ms_pipelined-110.mp4 \
  --latency-ms 117.7 --cadence-ms 117.6 --arm-sr 59.7 --arm-n 72 \
  --config-label "pipelined 110 ms cadence, 4 instances in flight" \
  --headline "FASTER THAN THE BASELINE. 8.50 inf/s MEASURED on the board means a fresh result every 117.6 ms - faster than Octo's own 200 ms nominal rate. 59.7% vs the 52.8% baseline: the staleness costs less than the extra rate buys." \
  --note "Only 2-3 ticks of hold between results. MEASURED latency and cadence; the mapping onto a 40 ms tick is MODELLED."

# ---- 3. 283.4 ms serial: the headline reversal ---------------------------
E=$(pick runs_video/vid_serial283 False); E=${E:-$(pick runs_video/vid_serial283 True)}
A --run runs_video/vid_serial283 --ep "${E:-0}" \
  --out videos/fine_283.4ms_serial.mp4 \
  --latency-ms 283.4 --cadence-ms 283.4 --arm-sr 23.6 --arm-n 72 \
  --config-label "3-way CPU+DSP+HTA int8, serial" \
  --headline "THE REVERSAL. The coarse 200 ms model called 283.4 ms indistinguishable from baseline (56.9% vs 52.8%, p=0.74). At a 40 ms tick the same MEASURED latency costs 29 points: 23.6% vs 52.8%, p=0.0005. The coarse model could not see this because D=ceil(283.4/200)=2 hid 7-8 ticks of hold." \
  --note "7-8 ticks of zero-order hold between results; the arm runs on an observation 405 ms old on average, 560 ms at worst."

# ---- 4. 684.8 ms CPU-only monolith: structural collapse ------------------
E=$(pick runs_video/vid_cpu685 False)
A --run runs_video/vid_cpu685 --ep "${E:-0}" \
  --out videos/fine_684.8ms_cpu-monolith.mp4 \
  --latency-ms 684.8 --cadence-ms 684.8 --arm-sr 2.8 --arm-n 72 \
  --config-label "CPU-only int8 monolith" \
  --headline "COLLAPSE, AND THE COARSE MODEL AGREED. 684.8 ms = 17-18 ticks of hold; the arm spends 94% of its ticks on stale data and acts on an observation 1.0 s old on average. 2.8% here vs 3.3% coarse - the one place the two models agree." \
  --note "94.3% of ticks are held. Mean observation age at actuation 1008 ms, max 1360 ms."
echo "### FINE VIDEOS DONE"
ls -la videos/fine_*.mp4
