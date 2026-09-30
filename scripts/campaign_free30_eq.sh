#!/usr/bin/env bash
# The same 30 Hz census, flown with ONE gain for both arms instead of one per command rate.
#
# scripts/campaign_free30.sh gives each arm the gain its own rate calls for (0.5 / eff_hz). That keeps
# the per-step impulse right for each arm, but it also hands the two arms two different controllers, so
# the comparison moves two things at once. This census moves one: both arms fly moment_scale 0.00500,
# and only the command rate differs.
#
# 0.00500 is the gain the scheduled arm already flew, so its 48 flights carry over unchanged and only
# the baseline is flown here. It is also the baseline's best gain of the four measured in
# campaign_rosgain (0.00550 -> 9/24 flights reach the first gate, 0.01667 -> 3/24), so the single gain
# is the one that favours the baseline rather than the scheduled arm.
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
T="$WT/results/codesign_feedback/ctrl_traces"
OUT="${OUT:-$WT/results/codesign_feedback/campaign_free30_eq}"
ARMLIST="${ARMLIST:-ros_cp3n4_30:$T/ros_cp3n430.csv:30.1:33.3:0.00500}" \
  SPEEDS="${SPEEDS:-1.4 1.2 1.0 1.6}" SEEDS="${SEEDS:-12}" MAX_SIMS="${MAX_SIMS:-3}" OUT="$OUT" \
  bash "$WT/scripts/campaign_rate30_gain.sh"

# The scheduled arm's rows are the rate-correct census's own, unchanged: same trace, same latency, same
# hold, same seeds, same gain. They are copied rather than re-flown, and only rows at this gain qualify.
.venv/bin/python - "$OUT/campaign.csv" "$WT/results/codesign_feedback/campaign_free30/campaign.csv" <<'PY'
import csv, sys
dst, src = sys.argv[1], sys.argv[2]
have = list(csv.DictReader(open(dst))); cols = have[0].keys() if have else None
add = [r for r in csv.DictReader(open(src))
       if r["ctrl_trace"].endswith("xpu_p30free.csv") and abs(float(r["moment_scale"]) - 0.005) < 1e-9]
seen = {(r["seed"], r["cruise_speed"], r["ctrl_trace"]) for r in have}
add = [r for r in add if (r["seed"], r["cruise_speed"], r["ctrl_trace"]) not in seen]
with open(dst, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(cols))
    for r in add:
        w.writerow({k: r.get(k, "") for k in cols})
print(f"carried over {len(add)} scheduled-arm rows at gain 0.00500")
PY
echo "CAMPAIGN_FREE30_EQ_DONE $(date +%H:%M:%S)"
