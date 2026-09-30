#!/bin/bash
# Side-by-side video of a display pair: the XPU-RT flight left, the ROS 2 flight right, at the recorder's frame rate.
# The baseline clip ends at its collision; its last frame is held (tpad, stop_mode=clone) until the XPU-RT clip ends,
# so both panels run for the completed flight's duration. A half-size copy is written alongside.
#
#   scripts/compose_pair_video.sh <xpu.mp4> <ros.mp4> <out.mp4> ["left banner"] ["right banner"]
#   e.g. scripts/compose_pair_video.sh results/codesign_feedback/campaign_v2/display_same/xpu_s1005.mp4 \
#          results/codesign_feedback/campaign_v2/display_same/ros_s1005.mp4 \
#          results/codesign_feedback/refined/warehouse_showdown_final_pair_s1005.mp4 \
#          "XPU-RT · CP-SAT (96 Hz control) — clears the course" "ROS 2 vanilla (39 Hz control) — clears G1, hits a crate"
set -euo pipefail
. "$(dirname "$0")/env.sh"
X=$1; R=$2; OUT=$3; LB=${4:-XPU-RT}; RB=${5:-ROS 2}
dur() { ffprobe -v error -show_entries format=duration -of csv=p=0 "$1"; }
DX=$(dur "$X"); DR=$(dur "$R"); HOLD=$(python3 -c "print(max(0.0, $DX - $DR))")
FONT="fontsize=44:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=14:x=24:y=24"
ffmpeg -v error -y -i "$X" -i "$R" -filter_complex \
  "[0:v]drawtext=text='$LB':$FONT[l];[1:v]tpad=stop_mode=clone:stop_duration=$HOLD,drawtext=text='$RB':$FONT[r];[l][r]hstack=inputs=2[v]" \
  -map "[v]" -c:v libx264 -pix_fmt yuv420p -crf 18 "$OUT"
ffmpeg -v error -y -i "$OUT" -vf scale=iw/2:-2 -c:v libx264 -pix_fmt yuv420p -crf 20 "${OUT%.mp4}_half.mp4"
echo "wrote $OUT and ${OUT%.mp4}_half.mp4 ($(ffprobe -v error -show_entries stream=width,height,nb_frames -of csv=p=0 "$OUT"))"
