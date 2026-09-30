#!/usr/bin/env bash
set -u; cd "$(dirname "$0")/.."
while ! grep -q XPU_AGAIN_DONE results/codesign_feedback/campaign_v2/display_v3_xpu_again.log 2>/dev/null; do sleep 120; done
bash scripts/campaign_v2_cal_now.sh
