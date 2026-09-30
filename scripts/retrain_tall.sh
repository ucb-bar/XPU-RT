#!/usr/bin/env bash
# Guidance net re-trained for the 2.4 m people (third GPU slot, sequential): expert demonstrations
# in the crowded aisle and on the prop-free course with the tall people, clean-demo filter, CNN
# training as for the shipped net, then the new weights flown against both replayed cadences.
. "$(dirname "$0")/env.sh"
set -u; cd "$(dirname "$0")/.."
WT=$PWD; R=$WT/results/codesign_feedback; T=$R/ctrl_traces
TO="$TRAIN_OUT"; D=$TO/fused_bc_warehouse_tall; mkdir -p $D/tmp; export TMPDIR=$D/tmp
PY="$ISAAC_PY"
say(){ echo "=== $(date +%H:%M:%S) $*"; }
until [ -f $D/pilot_tall_d030.pt ]; do sleep 30; done
say "pilot: $(grep -c 'gates=4/4' $D/pilot_tall_d030.log)/$(grep -c '\[ep' $D/pilot_tall_d030.log) complete"
cd "$SIM_TREE"
collect(){ local out=$1; shift; [ -f $out ] && { say "have $out"; return; }
  say "collect $(basename $out)"; $PY sims/training/collect_fused_warehouse.py --headless --max_steps 1500 --planned_expert --obstacle_level 8 --base_speed 1.4 --noise_std 0.06 --turn_slow "$@" --out $out > ${out%.pt}.log 2>&1
  grep -E "\[done\]|Traceback" ${out%.pt}.log | tail -n 2; }
collect $D/crowded_tall_d030.pt --episodes 60 --seed 600 --prop_density 0.30
collect $D/gate_tall_d000.pt   --episodes 30 --seed 700 --prop_density 0.0
say "clean filter (gates >= 3)"
$PY - <<PY
import torch
d = torch.load("$D/crowded_tall_d030.pt", map_location="cpu", weights_only=False)
keep = [e for e in d["episodes"] if int(e.get("gates_reached", 0)) >= 3]
print(f"crowded: {len(keep)}/{len(d['episodes'])} episodes reach >= 3 gates")
d["episodes"] = keep; d["meta"]["n_episodes"] = len(keep); d["meta"]["filter"] = "gates>=3"; d["meta"]["n_frames"] = sum(len(e["label"]) for e in keep)
torch.save(d, "$D/crowded_tall_d030_clean.pt")
PY
say "train"
$PY sims/training/train_fused.py --data $D/crowded_tall_d030_clean.pt $D/gate_tall_d000.pt --vision_encoder cnn --epochs 60 --out_dir $TO/fused_bc_warehouse_v20_tall_cnn > $D/train_v20.log 2>&1
tail -n 2 $D/train_v20.log
BEST=$(ls -t $TO/fused_bc_warehouse_v20_tall_cnn/*/best.pt | head -n 1); cp $BEST $WT/sims/models/warehouse/nav_fused_v20_tall_cnn.pt; cp $BEST sims/models/warehouse/nav_fused_v20_tall_cnn.pt
say "new weights $BEST -> sims/models/warehouse/nav_fused_v20_tall_cnn.pt"
cd $WT
W=sims/models/warehouse/nav_fused_v20_tall_cnn.pt OUT=$R/campaign_tallnet SPEEDS="1.8 1.4 1.0 1.6 1.2" \
  ARMS="xpu_cpsat:$T/xpu_a_cpsat_hard.csv:0 ros_vanilla4:$T/ros_vanilla445.csv:0" bash scripts/campaign_percep.sh
say "RETRAIN_TALL_DONE"
bash scripts/queue_extra.sh > results/codesign_feedback/queue_extra.log 2>&1
bash scripts/campaign_v2_cal_now.sh >> $R/campaign_v2/cal_now.log 2>&1
until grep -q GRID_FINAL_DONE $R/gain_controlled/resume_final.log 2>/dev/null; do sleep 300; done
SPEEDS="1.8 1.4 1.0" DENS="0.30" COURSES="a" bash scripts/env_sweep.sh 3 "ros_vanilla4x2:$T/ros_vanilla4x245.csv ros_vanilla4_q1:$T/ros_vanilla4_q145.csv"
