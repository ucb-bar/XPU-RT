#!/usr/bin/env bash
# When the XPU-RT sweep driver has finished course A at density 0.30, borrow its slot for the
# perception-latency replay cells (both arms), then hand the slot back to the sweep.
set -u; cd "$(dirname "$0")/.."
R=results/codesign_feedback; T=$R/ctrl_traces; E=$R/campaign_env
have_all() { python3 - $E/env_sweep.csv <<'PY'
import csv, sys, collections
c = collections.Counter()
for r in csv.DictReader(open(sys.argv[1])):
    if r["ctrl_trace"] == "xpu_a_cpsat_hard.csv" and r.get("course","a") == "a" and abs(float(r["prop_density"]) - 0.30) < 1e-9: c[float(r["cruise_speed"])] += 1
sys.exit(0 if all(c[s] >= 12 for s in (1.0, 1.2, 1.4, 1.6, 1.8)) else 1)
PY
}
until have_all; do sleep 120; done
D1=$(pgrep -f "env_sweep.sh 1 " | head -n 1); echo "=== $(date +%H:%M:%S) course A d0.30 complete for XPU-RT; pausing driver 1 ($D1)"
[ -n "$D1" ] && { for tpid in $(pgrep -P $D1); do pkill -9 -P $tpid; kill -9 $tpid; done; kill $D1; }
sleep 5
# the same-scene display pair first, at the speed the tall-scene cells single out (1.0 m/s: XPU-RT 6/12, the baseline 1/12 with nine flights crashing after one or two gates)
CRUISE=${DISPLAY_CRUISE:-1.0} bash scripts/display_same_env.sh > $R/campaign_v2/display_same_c1.0.log 2>&1
ARMS="xpu_cpsat:$T/xpu_a_cpsat_hard.csv:56.8 ros_vanilla4:$T/ros_vanilla445.csv:242" SPEEDS="1.8 1.4 1.0 1.6 1.2" bash scripts/campaign_percep.sh
ARMS="ros_vanilla4_q1:$T/ros_vanilla4_q145.csv:42.0 xpu_greedy:$T/xpu_a_greedy.csv:748" SPEEDS="1.8 1.4 1.0" bash scripts/campaign_percep.sh
echo "=== $(date +%H:%M:%S) resuming driver 1"
exec bash scripts/env_sweep.sh 1 "xpu_cpsat:$T/xpu_a_cpsat_hard.csv"
