#!/usr/bin/env bash
# Snapshot every board-local source the measurements depend on into the repo, and verify it later.
#
# WHY. The ROS 2 baselines are defined by a node source that lived only on the K1
# (/root/ros_mb/ros_mb_chain_traced.cpp and friends): the arms are flags into that one program, so
# without it in the repo no ROS rung can be rebuilt and a reimaged board would take every baseline
# with it. The per-network kernels are NOT snapshotted -- those are generated (ModelBlaster builds,
# see docs/Baselines/ros_baseline_reproduction.md) and their sha256 is recorded per run in kernel_sha256.txt.
#
#   scripts/board_source_snapshot.sh            pull the board's sources into board/k1_ros_mb/
#   scripts/board_source_snapshot.sh --verify   fail if the board differs from the repo copy
set -u
cd "$(dirname "$0")/.."
HOST="${MODELBLASTER_K1_HOST:-k1}"; DEST=board/k1_ros_mb; MODE="${1:-pull}"
FILES="ros_mb_chain_traced.cpp ros_mb_chain.cpp ros_ctrl_starve.cpp cpu_sampler.c cpu_hog.c rdtime_now.c
       pool/modelblaster_pool.c pool/modelblaster_pool.h pool/mb_posix_compat.h"
mkdir -p "$DEST/pool"
if [ "$MODE" = "--verify" ]; then
  rc=0
  for f in $FILES; do
    want=$(sha256sum "$DEST/$f" 2>/dev/null | cut -d' ' -f1)
    got=$(ssh "$HOST" "sha256sum /root/ros_mb/$f 2>/dev/null" | cut -d' ' -f1)
    if [ -z "$got" ]; then echo "MISSING ON BOARD  $f"; rc=1
    elif [ "$want" != "$got" ]; then echo "DIFFERS  $f  (repo ${want:0:12} board ${got:0:12})"; rc=1
    else echo "ok       $f"; fi
  done
  exit $rc
fi
for f in $FILES; do scp -q "$HOST:/root/ros_mb/$f" "$DEST/$f" 2>/dev/null || echo "  (absent on board: $f)"; done
( cd "$DEST" && sha256sum $(echo $FILES) > MANIFEST.sha256 2>/dev/null )
echo "snapshot in $DEST:"; sed 's/^/  /' "$DEST/MANIFEST.sha256"
