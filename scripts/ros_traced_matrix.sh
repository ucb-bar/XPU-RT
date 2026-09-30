#!/usr/bin/env bash
# Run the traced ROS 2 baseline on the K1 across camera rates, for one named arm, with the
# per-core sampler around every run, and pull everything back.
#
#   scripts/ros_traced_matrix.sh <arm> [replicate]
#
# Arms are real deployments, named by what the launcher does to the process -- the program
# itself never pins anything, and its manifest records the mask it actually ran under:
#
#   ship     one process, default executor, serial kernels, unpinned        (ROS 2 as it ships)
#   multi    one process, MultiThreadedExecutor, serial kernels, unpinned
#   spin     one process, default executor, YOLO on a 4-hart pool P0-3, executor thread on P0
#   smte     as spin, MultiThreadedExecutor
#   p3       three processes: perception (pool P0-3), nav on E4, control on E5
#   p8       three processes: perception (pool P0-3), nav on E4-5, control on E6-7
#   yproc    two processes: perception alone (pool P0-3); nav + control together on E4
#   rspin / rp3 / rmulti  the heavier stack: + ffn_block @10 Hz + dronet @30 Hz as two more nodes, in the
#            spin / p3 (extras in a fourth process on E#2-3) / multi layouts
#   cship / cspin / cp3   the same three layouts with control CHAINED to the goal topic (no
#            control timer): control runs once per perception result, the classic ROS pipeline
#   nproc    two processes: nav alone on E4; perception (pool P0-3) + control together
#   vanilla4x2c0  vanilla4x2 with the whole deployment confined to cluster 0 (both perception
#            pools on harts 0-3, every process tasksetted there) -- the placement an IME build
#            has no choice about, and the RVV control for it
#
# BINSUF picks WHICH KERNELS the arm runs: BINSUF=_ime uses ros_mb_chain_traced_pool_ime, the
# same traced program linked against the K1-IME YOLO. Pass SUFFIX as well so the tags differ.
#
# The last four are the starvation matrix: which callbacks share a thread with the control
# timer. In p3/p8 none do; in yproc only nav; in nproc only perception; in spin/ship both.
# The pool arms need the binary built with MB_WITH_POOL (scripts/board_campaign.sh step 2).
set -u
ARM="${1:?arm}"; REP="${2:-1}"
RATES="${RATES:-5 8 10 12 15 20 25 30}"
SECS="${SECS:-20}"
CTRL_HZ="${CTRL_HZ:-100}"; QOS="${QOS:-10}"; HOGS="${HOGS:-0}"; SUFFIX="${SUFFIX:-}"   # sensitivity knobs; the tag carries them
# Which KERNELS the arm runs. An arm names a LAYOUT (which process, which hart, which
# executor); the generated kernels linked into the node are a separate axis, and BINSUF
# selects a differently-linked copy of the same traced program -- e.g. BINSUF=_ime runs
# ros_mb_chain_traced_pool_ime, the same C linked against the K1-IME YOLO instead of the
# RVV one. It is NOT recorded in the tag by itself, so pass SUFFIX too when both kernel
# sets are run at the same rate, or the second run overwrites the first.
BINSUF="${BINSUF:-}"
HOST="${MODELBLASTER_K1_HOST:-k1}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO/results/codesign_feedback/ros_traced"
SHA="$(git -C "$REPO" rev-parse --short HEAD) ir=$(sed -n 's/^sha256=//p' "$REPO/ModelBlaster/build/k1_xpurt/yolov8_nano_64x96/int8/.staged_from" 2>/dev/null | head -c 12)"
mkdir -p "$DEST"

case "$ARM" in
  ship)  BIN=ros_mb_chain_traced;      PRE="";                 EXTRA="--executor single" ;;
  cship) BIN=ros_mb_chain_traced;      PRE="";                 EXTRA="--executor single --ctrl-mode chained" ;;
  cspin) BIN=ros_mb_chain_traced_pool; PRE="taskset -c 0-3";   EXTRA="--executor single --ctrl-mode chained --yolo-pool 4 --pool-harts 0,1,2,3 --pin-main 0" ;;
  multi) BIN=ros_mb_chain_traced;      PRE="";                 EXTRA="--executor multi" ;;
  spin)  BIN=ros_mb_chain_traced_pool; PRE="taskset -c 0-3";   EXTRA="--executor single --yolo-pool 4 --pool-harts 0,1,2,3 --pin-main 0" ;;
  rspin) BIN=ros_mb_chain_traced_rich; PRE="taskset -c 0-3";   EXTRA="--executor single --yolo-pool 4 --pool-harts 0,1,2,3 --pin-main 0 --extra ffn_block:10,dronet:30" ;;
  rmulti) BIN=ros_mb_chain_traced_rich; PRE="";                EXTRA="--executor multi --yolo-pool 4 --pool-harts 0,1,2,3 --extra ffn_block:10,dronet:30" ;;
  x2spin) BIN=ros_mb_chain_traced_pool; PRE="taskset -c 0-3";  EXTRA="--executor single --cameras 2 --nodes camera,perception,camera2,perception2,nav,control --yolo-pool 4 --pool-harts 0,1,2,3 --pin-main 0" ;;
  x2p) BIN=ros_mb_chain_traced_pool ;;
  smte)  BIN=ros_mb_chain_traced_pool; PRE="taskset -c 0-3";   EXTRA="--executor multi --yolo-pool 4 --pool-harts 0,1,2,3" ;;
  p3|p8|cp3|yproc|nproc) BIN=ros_mb_chain_traced_pool ;;
  # navigation sharded four ways over the efficiency cluster, control sharing those harts:
  # nav and control are consecutive links of one chain, so they are not resident at the same
  # time for a frame. Needs the node linked against the 4-way nav build (docs/K1/nav_sharding.md).
  cp3n4) BIN=ros_mb_chain_traced_pool_nav4 ;;
  part8) BIN=ros_mb_chain_traced_pool ;;   # the machine partitioned between the three networks: YOLO on a 4-hart pool, nav on a 2-hart pool, control on a 2-hart pool
  rp3) BIN=ros_mb_chain_traced_rich ;;
  rp8) BIN=ros_mb_chain_traced_rich ;;   # the p8 layout with the heavier stack: perception 0-3, nav 4-5, control 6-7, extras unpinned
  x2rp3) BIN=ros_mb_chain_traced_rich ;;      # two cameras AND the heavier stack: x2p plus the extras process, unpinned
  vanilla) BIN=ros_mb_chain_traced ;;         # ROS 2 as written out of the box: one node per stage, one process each, unpinned, serial kernels, control in the goal callback
  rvanilla) BIN=ros_mb_chain_traced_rich ;;   # the same plus ffn_block and dronet, each its own unpinned process
  vanilla4) BIN=ros_mb_chain_traced_pool ;;
  vanilla8|vanilla8tm) BIN=ros_mb_chain_traced_pool ;;
  vanilla4f|vanilla8f) BIN=ros_mb_chain_traced_pool ;;   # the pool unpinned: its workers are ordinary threads the operating system places and may migrate   # the whole machine for perception: an 8-wide YOLO pool across every hart
   # as vanilla, the perception node using the model's 4-hart build (the pool pins its own workers; ROS itself unpinned)
  rvanilla4) BIN=ros_mb_chain_traced_rich ;;  # vanilla4 plus ffn_block and dronet as their own unpinned processes: every core carries a node
  rvanilla8|rvanilla8tm) BIN=ros_mb_chain_traced_rich ;;   # the heavier stack with the pool asked for every hart (0-7) and nothing pinned: the whole machine, out of the box
  rvanilla4x2|rvanilla4x2tm) BIN=ros_mb_chain_traced_rich ;;   # the heavier stack pipelined by hand across both clusters: the ceiling of what ROS 2 can be given here
  vanilla4x2|vanilla4x2tm) BIN=ros_mb_chain_traced_pool ;; # pipelining by hand: the camera alternates frames between two perception processes (4-hart pools on 0-3 and 4-7); nav, control unpinned
  # vanilla4x2 with the WHOLE deployment confined to cluster 0: both perception pools on
  # harts 0-3 and every process tasksetted there. That is the placement an IME build has no
  # choice about -- smt.vmadot executes on cluster 0 and raises SIGILL on cluster 1
  # (ModelBlaster/artifacts/ime_isa_probe/FINDINGS.md) -- so it is also the arm the RVV
  # kernels have to be run in to separate the accelerator from the half machine it costs.
  vanilla4x2c0) BIN=ros_mb_chain_traced_pool ;;
  vanilla4tm) BIN=ros_mb_chain_traced_pool ;; # as vanilla4, control on its own 100 Hz timer in its own unpinned process (the held goal), the other natural default
  vanilla4t) BIN=ros_mb_chain_traced_pool; PRE=""; EXTRA="--executor single --yolo-pool 4 --pool-harts 0,1,2,3" ;;   # one process, default executor, 4-hart YOLO, control on its 100 Hz timer; nothing pinned by hand
  x2rspin) BIN=ros_mb_chain_traced_rich; PRE="taskset -c 0-3"; EXTRA="--executor single --cameras 2 --nodes camera,perception,camera2,perception2,nav,control,extra --extra ffn_block:10,dronet:30 --yolo-pool 4 --pool-harts 0,1,2,3 --pin-main 0" ;;   # as deployed, under the combined load
  x2rmulti) BIN=ros_mb_chain_traced_rich; PRE="";               EXTRA="--executor multi --cameras 2 --nodes camera,perception,camera2,perception2,nav,control,extra --extra ffn_block:10,dronet:30 --yolo-pool 4 --pool-harts 0,1,2,3" ;;   # the multi-threaded executor, under the combined load
  *) echo "unknown arm $ARM"; exit 2 ;;
esac

BIN="${BIN}${BINSUF}"

ros_remote() {  # runs on the board with the ROS env sourced
  ssh "$HOST" "source /opt/ros/jazzy_prebuilt/setup.bash; cd /root/ros_mb; $*"
}

for HZ in $RATES; do
  TAG="${HZ}_${ARM}${SUFFIX}_r${REP}"; KN="--ctrl-hz $CTRL_HZ --qos-depth $QOS"
  echo "=== $(date +%H:%M:%S) $TAG ==="
  ros_remote "rm -rf out/$TAG; mkdir -p out/$TAG; (nohup ./cpu_sampler 100 out/$TAG/cpu.csv >/dev/null 2>&1 & echo \$! > out/$TAG/cpu.pid); for h in \$(seq 1 $HOGS); do (nohup ./cpu_hog >/dev/null 2>&1 & echo \$! >> out/$TAG/hog.pid); done; sleep 0.5"
  if [[ "$ARM" == p3 || "$ARM" == p8 || "$ARM" == cp3 || "$ARM" == cp3n4 || "$ARM" == part8 || "$ARM" == rp8 ]]; then
    if [[ "$ARM" == p8 || "$ARM" == part8 || "$ARM" == rp8 ]]; then NAVC="4,5"; CTLC="6,7";
    elif [[ "$ARM" == cp3n4 ]]; then NAVC="4,5,6,7"; CTLC="4,5,6,7";
    else NAVC="4"; CTLC="5"; fi
    # rp8 adds ffn_block and dronet as their own unpinned processes, as the other r* arms do
    RXTRA=""; [[ "$ARM" == rp8 ]] && RXTRA="./$BIN --nodes none --extra ffn_block:10 --rate $HZ --seconds $SECS $KN --t0 \$T0 --tag $TAG --out-dir out/$TAG/extra --kernels-sha '$SHA' > out/$TAG/extra.log 2>&1 &
      ./$BIN --nodes none --extra dronet:30 --rate $HZ --seconds $SECS $KN --t0 \$T0 --tag $TAG --out-dir out/$TAG/extra2 --kernels-sha '$SHA' > out/$TAG/extra2.log 2>&1 &"
    CM=""; [[ "$ARM" == cp3 || "$ARM" == cp3n4 ]] && CM="--ctrl-mode chained"
    # part8 also builds a pool for each of the two small networks, so every hart carries one network
    NAVP=""; CTLP=""; [[ "$ARM" == part8 ]] && { NAVP="--nav-pool 2 --nav-harts 4,5"; CTLP="--ctrl-pool 2 --ctrl-harts 6,7"; }
    [[ "$ARM" == cp3n4 ]] && NAVP="--nav-pool 4 --nav-harts 4,5,6,7"
    ros_remote "T0=\$(./rdtime_now);
      taskset -c $CTLC ./$BIN --nodes control $CM $CTLP --rate $HZ --seconds $SECS $KN --t0 \$T0 --tag $TAG --out-dir out/$TAG/control --kernels-sha '$SHA' > out/$TAG/control.log 2>&1 &
      taskset -c $NAVC ./$BIN --nodes nav $NAVP --rate $HZ --seconds $SECS $KN --t0 \$T0 --tag $TAG --out-dir out/$TAG/nav --kernels-sha '$SHA' > out/$TAG/nav.log 2>&1 &
      $RXTRA
      sleep 1;
      taskset -c 0-3 ./$BIN --nodes camera,perception --rate $HZ --seconds $SECS $KN --t0 \$T0 --tag $TAG --out-dir out/$TAG/percep --yolo-pool 4 --pool-harts 0,1,2,3 --pin-main 0 --kernels-sha '$SHA' > out/$TAG/percep.log 2>&1;
      wait; cat out/$TAG/*.log | grep ROS_TRACED"
  elif [[ "$ARM" == x2p ]]; then
    ros_remote "T0=\$(./rdtime_now);
      ./$BIN --nodes control --cameras 2 --rate $HZ --seconds $SECS $KN --t0 \$T0 --tag $TAG --out-dir out/$TAG/control --kernels-sha '$SHA' > out/$TAG/control.log 2>&1 &
      ./$BIN --nodes nav --cameras 2 --rate $HZ --seconds $SECS $KN --t0 \$T0 --tag $TAG --out-dir out/$TAG/nav --kernels-sha '$SHA' > out/$TAG/nav.log 2>&1 &
      taskset -c 4-7 ./$BIN --nodes camera2,perception2 --cameras 2 --rate $HZ --seconds $SECS $KN --t0 \$T0 --tag $TAG --out-dir out/$TAG/percep2 --yolo-pool 4 --pool-harts 4,5,6,7 --pin-main 4 --kernels-sha '$SHA' > out/$TAG/percep2.log 2>&1 &
      sleep 1;
      taskset -c 0-3 ./$BIN --nodes camera,perception --cameras 2 --rate $HZ --seconds $SECS $KN --t0 \$T0 --tag $TAG --out-dir out/$TAG/percep --yolo-pool 4 --pool-harts 0,1,2,3 --pin-main 0 --kernels-sha '$SHA' > out/$TAG/percep.log 2>&1;
      wait; cat out/$TAG/*.log | grep ROS_TRACED"
  elif [[ "$ARM" == vanilla4x2c0 ]]; then
    ros_remote "T0=\$(./rdtime_now);
      taskset -c 0-3 ./$BIN --nodes control --ctrl-mode chained --cameras 2 --rate $HZ --seconds $SECS $KN --t0 \$T0 --tag $TAG --out-dir out/$TAG/control --kernels-sha '$SHA' > out/$TAG/control.log 2>&1 &
      taskset -c 0-3 ./$BIN --nodes nav --cameras 2 --rate $HZ --seconds $SECS $KN --t0 \$T0 --tag $TAG --out-dir out/$TAG/nav --kernels-sha '$SHA' > out/$TAG/nav.log 2>&1 &
      taskset -c 0-3 ./$BIN --nodes perception --yolo-pool 4 --pool-harts 0,1,2,3 --rate $HZ --seconds $SECS $KN --t0 \$T0 --tag $TAG --out-dir out/$TAG/percep --kernels-sha '$SHA' > out/$TAG/percep.log 2>&1 &
      taskset -c 0-3 ./$BIN --nodes perception2 --cameras 2 --yolo-pool 4 --pool-harts 0,1,2,3 --rate $HZ --seconds $SECS $KN --t0 \$T0 --tag $TAG --out-dir out/$TAG/percep2 --kernels-sha '$SHA' > out/$TAG/percep2.log 2>&1 &
      sleep 1;
      taskset -c 0-3 ./$BIN --nodes camera --alternate --rate $HZ --seconds $SECS $KN --t0 \$T0 --tag $TAG --out-dir out/$TAG/camera --kernels-sha '$SHA' > out/$TAG/camera.log 2>&1;
      wait; cat out/$TAG/*.log | grep ROS_TRACED"
  elif [[ "$ARM" == vanilla4x2 || "$ARM" == vanilla4x2tm || "$ARM" == rvanilla4x2 || "$ARM" == rvanilla4x2tm ]]; then
    X2CM="--ctrl-mode chained"; [[ "$ARM" == vanilla4x2tm || "$ARM" == rvanilla4x2tm ]] && X2CM=""   # tm: control on its own 100 Hz timer, as vanilla4tm
    # NAVPOOL gives the navigation node a worker pool of its own, the way the perception nodes have
    # one. It needs a binary linked against a nav built with MB_SHARD_FACTOR > 1 (BINSUF selects it):
    # the shard is an output-channel split re-packed at codegen time, so a pool alone does nothing.
    X2NP=""; [ -n "${NAVPOOL:-}" ] && X2NP="--nav-pool $NAVPOOL${NAVHARTS:+ --nav-harts $NAVHARTS}"
    # the r* variants add ffn_block and dronet as their own unpinned processes, as rvanilla4 does
    X2XTRA=""; [[ "$ARM" == rvanilla4x2 || "$ARM" == rvanilla4x2tm ]] && X2XTRA="./$BIN --nodes none --extra ffn_block:10 --rate $HZ --seconds $SECS $KN --t0 \$T0 --tag $TAG --out-dir out/$TAG/extra --kernels-sha '$SHA' > out/$TAG/extra.log 2>&1 &
      ./$BIN --nodes none --extra dronet:30 --rate $HZ --seconds $SECS $KN --t0 \$T0 --tag $TAG --out-dir out/$TAG/extra2 --kernels-sha '$SHA' > out/$TAG/extra2.log 2>&1 &"
    ros_remote "T0=\$(./rdtime_now);
      ./$BIN --nodes control $X2CM --cameras 2 --rate $HZ --seconds $SECS $KN --t0 \$T0 --tag $TAG --out-dir out/$TAG/control --kernels-sha '$SHA' > out/$TAG/control.log 2>&1 &
      ./$BIN --nodes nav $X2NP --cameras 2 --rate $HZ --seconds $SECS $KN --t0 \$T0 --tag $TAG --out-dir out/$TAG/nav --kernels-sha '$SHA' > out/$TAG/nav.log 2>&1 &
      ./$BIN --nodes perception --yolo-pool 4 --pool-harts 0,1,2,3 --rate $HZ --seconds $SECS $KN --t0 \$T0 --tag $TAG --out-dir out/$TAG/percep --kernels-sha '$SHA' > out/$TAG/percep.log 2>&1 &
      ./$BIN --nodes perception2 --cameras 2 --yolo-pool 4 --pool-harts 4,5,6,7 --rate $HZ --seconds $SECS $KN --t0 \$T0 --tag $TAG --out-dir out/$TAG/percep2 --kernels-sha '$SHA' > out/$TAG/percep2.log 2>&1 &
      $X2XTRA
      sleep 1;
      ./$BIN --nodes camera --alternate --rate $HZ --seconds $SECS $KN --t0 \$T0 --tag $TAG --out-dir out/$TAG/camera --kernels-sha '$SHA' > out/$TAG/camera.log 2>&1;
      wait; cat out/$TAG/*.log | grep ROS_TRACED"
  elif [[ "$ARM" == vanilla || "$ARM" == rvanilla || "$ARM" == vanilla4 || "$ARM" == rvanilla4 || "$ARM" == vanilla4tm || "$ARM" == vanilla8 || "$ARM" == vanilla8tm || "$ARM" == vanilla4f || "$ARM" == vanilla8f || "$ARM" == rvanilla8 || "$ARM" == rvanilla8tm ]]; then
    XTRA=""; POOL=""; [[ "$ARM" == vanilla4 || "$ARM" == rvanilla4 || "$ARM" == vanilla4tm ]] && POOL="--yolo-pool 4 --pool-harts 0,1,2,3"
    [[ "$ARM" == vanilla8 || "$ARM" == vanilla8tm ]] && POOL="--yolo-pool 8 --pool-harts 0,1,2,3,4,5,6,7"
    [[ "$ARM" == rvanilla8 || "$ARM" == rvanilla8tm ]] && POOL="--yolo-pool 8 --pool-harts 0,1,2,3,4,5,6,7"
    [[ "$ARM" == vanilla4f ]] && POOL="--yolo-pool 4"; [[ "$ARM" == vanilla8f ]] && POOL="--yolo-pool 8"
    CMODE="--ctrl-mode chained"; [[ "$ARM" == vanilla4tm || "$ARM" == vanilla8tm || "$ARM" == rvanilla8tm ]] && CMODE=""
    [[ "$ARM" == rvanilla || "$ARM" == rvanilla4 || "$ARM" == rvanilla8 || "$ARM" == rvanilla8tm ]] && XTRA="./$BIN --nodes none --extra ffn_block:10 --rate $HZ --seconds $SECS $KN --t0 \$T0 --tag $TAG --out-dir out/$TAG/extra --kernels-sha '$SHA' > out/$TAG/extra.log 2>&1 &
      ./$BIN --nodes none --extra dronet:30 --rate $HZ --seconds $SECS $KN --t0 \$T0 --tag $TAG --out-dir out/$TAG/extra2 --kernels-sha '$SHA' > out/$TAG/extra2.log 2>&1 &"
    ros_remote "T0=\$(./rdtime_now);
      ./$BIN --nodes control $CMODE --rate $HZ --seconds $SECS $KN --t0 \$T0 --tag $TAG --out-dir out/$TAG/control --kernels-sha '$SHA' > out/$TAG/control.log 2>&1 &
      ./$BIN --nodes nav --rate $HZ --seconds $SECS $KN --t0 \$T0 --tag $TAG --out-dir out/$TAG/nav --kernels-sha '$SHA' > out/$TAG/nav.log 2>&1 &
      ./$BIN --nodes perception $POOL --rate $HZ --seconds $SECS $KN --t0 \$T0 --tag $TAG --out-dir out/$TAG/percep --kernels-sha '$SHA' > out/$TAG/percep.log 2>&1 &
      $XTRA
      sleep 1;
      ./$BIN --nodes camera --rate $HZ --seconds $SECS $KN --t0 \$T0 --tag $TAG --out-dir out/$TAG/camera --kernels-sha '$SHA' > out/$TAG/camera.log 2>&1;
      wait; cat out/$TAG/*.log | grep ROS_TRACED"
  elif [[ "$ARM" == x2rp3 ]]; then
    ros_remote "T0=\$(./rdtime_now);
      ./$BIN --nodes control --cameras 2 --rate $HZ --seconds $SECS $KN --t0 \$T0 --tag $TAG --out-dir out/$TAG/control --kernels-sha '$SHA' > out/$TAG/control.log 2>&1 &
      ./$BIN --nodes nav --cameras 2 --rate $HZ --seconds $SECS $KN --t0 \$T0 --tag $TAG --out-dir out/$TAG/nav --kernels-sha '$SHA' > out/$TAG/nav.log 2>&1 &
      ./$BIN --nodes none --extra ffn_block:10,dronet:30 --cameras 2 --rate $HZ --seconds $SECS $KN --t0 \$T0 --tag $TAG --out-dir out/$TAG/extra --kernels-sha '$SHA' > out/$TAG/extra.log 2>&1 &
      taskset -c 4-7 ./$BIN --nodes camera2,perception2 --cameras 2 --rate $HZ --seconds $SECS $KN --t0 \$T0 --tag $TAG --out-dir out/$TAG/percep2 --yolo-pool 4 --pool-harts 4,5,6,7 --pin-main 4 --kernels-sha '$SHA' > out/$TAG/percep2.log 2>&1 &
      sleep 1;
      taskset -c 0-3 ./$BIN --nodes camera,perception --cameras 2 --rate $HZ --seconds $SECS $KN --t0 \$T0 --tag $TAG --out-dir out/$TAG/percep --yolo-pool 4 --pool-harts 0,1,2,3 --pin-main 0 --kernels-sha '$SHA' > out/$TAG/percep.log 2>&1;
      wait; cat out/$TAG/*.log | grep ROS_TRACED"
  elif [[ "$ARM" == rp3 ]]; then
    ros_remote "T0=\$(./rdtime_now);
      taskset -c 5 ./$BIN --nodes control --rate $HZ --seconds $SECS $KN --t0 \$T0 --tag $TAG --out-dir out/$TAG/control --kernels-sha '$SHA' > out/$TAG/control.log 2>&1 &
      taskset -c 4 ./$BIN --nodes nav --rate $HZ --seconds $SECS $KN --t0 \$T0 --tag $TAG --out-dir out/$TAG/nav --kernels-sha '$SHA' > out/$TAG/nav.log 2>&1 &
      taskset -c 6,7 ./$BIN --nodes none --extra ffn_block:10,dronet:30 --rate $HZ --seconds $SECS $KN --t0 \$T0 --tag $TAG --out-dir out/$TAG/extra --kernels-sha '$SHA' > out/$TAG/extra.log 2>&1 &
      sleep 1;
      taskset -c 0-3 ./$BIN --nodes camera,perception --rate $HZ --seconds $SECS $KN --t0 \$T0 --tag $TAG --out-dir out/$TAG/percep --yolo-pool 4 --pool-harts 0,1,2,3 --pin-main 0 --kernels-sha '$SHA' > out/$TAG/percep.log 2>&1;
      wait; cat out/$TAG/*.log | grep ROS_TRACED"
  elif [[ "$ARM" == yproc ]]; then
    ros_remote "T0=\$(./rdtime_now);
      taskset -c 4 ./$BIN --nodes nav,control --rate $HZ --seconds $SECS $KN --t0 \$T0 --tag $TAG --out-dir out/$TAG/control --kernels-sha '$SHA' > out/$TAG/control.log 2>&1 &
      sleep 1;
      taskset -c 0-3 ./$BIN --nodes camera,perception --rate $HZ --seconds $SECS $KN --t0 \$T0 --tag $TAG --out-dir out/$TAG/percep --yolo-pool 4 --pool-harts 0,1,2,3 --pin-main 0 --kernels-sha '$SHA' > out/$TAG/percep.log 2>&1;
      wait; cat out/$TAG/*.log | grep ROS_TRACED"
  elif [[ "$ARM" == nproc ]]; then
    ros_remote "T0=\$(./rdtime_now);
      taskset -c 4 ./$BIN --nodes nav --rate $HZ --seconds $SECS $KN --t0 \$T0 --tag $TAG --out-dir out/$TAG/nav --kernels-sha '$SHA' > out/$TAG/nav.log 2>&1 &
      sleep 1;
      taskset -c 0-3 ./$BIN --nodes camera,perception,control --rate $HZ --seconds $SECS $KN --t0 \$T0 --tag $TAG --out-dir out/$TAG/control --yolo-pool 4 --pool-harts 0,1,2,3 --pin-main 0 --kernels-sha '$SHA' > out/$TAG/control.log 2>&1;
      wait; cat out/$TAG/*.log | grep ROS_TRACED"
  else
    ros_remote "$PRE ./$BIN --rate $HZ --seconds $SECS $KN $EXTRA --tag $TAG --out-dir out/$TAG --kernels-sha '$SHA' 2>&1 | grep ROS_TRACED"
  fi
  ros_remote "kill \$(cat out/$TAG/cpu.pid); [ -f out/$TAG/hog.pid ] && kill \$(cat out/$TAG/hog.pid); sleep 0.3; sha256sum yolo/kernels.o nav/kernels.o ctrl/kernels.o yolo4/kernels.o yolo4_ime256/kernels.o 2>/dev/null > out/$TAG/kernel_sha256.txt; sha256sum \$(readlink -f ./$BIN) >> out/$TAG/kernel_sha256.txt; sync"
  # Pull as one archive, in chunks, checked against the board's own digest.
  #
  # A plain `scp -r` of a run is not safe here: on this link a sustained copy is closed by the
  # remote host after a few hundred KB, and a per-frame trace is megabytes. A completeness
  # test of "is one trace.csv non-empty?" passes on such a partial copy, and a copy preceded by
  # `rm -rf` on the destination would let a truncated pull destroy the previous run. So:
  # archive on the board, fetch in chunks the link survives, verify the digest, and only then
  # replace the destination -- a failed pull leaves the previous run untouched.
  for attempt in 1 2 3; do
    STAGE="$(mktemp -d)"; ok=0
    ros_remote "cd /root/ros_mb/out && rm -f $TAG.tgz && tar czf $TAG.tgz $TAG && split -b 65536 -d -a 4 $TAG.tgz $TAG.part. && md5sum $TAG.tgz | cut -d' ' -f1 > $TAG.md5 && ls $TAG.part.* | wc -l" > "$STAGE/nparts" 2>/dev/null
    WANT="$(ros_remote "cat /root/ros_mb/out/$TAG.md5" 2>/dev/null | tr -d '\r')"
    N="$(tr -dc '0-9' < "$STAGE/nparts" | tail -c 6)"
    if [ -n "$N" ] && [ "$N" -gt 0 ] 2>/dev/null; then
      i=0; while [ "$i" -lt "$N" ]; do
        PN="$(printf '%04d' "$i")"
        ssh "$HOST" "cat /root/ros_mb/out/$TAG.part.$PN" > "$STAGE/part.$PN" 2>/dev/null || break
        i=$((i+1))
      done
      cat "$STAGE"/part.* > "$STAGE/$TAG.tgz" 2>/dev/null
      GOT="$(md5sum "$STAGE/$TAG.tgz" 2>/dev/null | cut -d' ' -f1)"
      if [ -n "$WANT" ] && [ "$GOT" = "$WANT" ]; then
        tar xzf "$STAGE/$TAG.tgz" -C "$STAGE" && [ -d "$STAGE/$TAG" ] && ok=1
      fi
    fi
    ros_remote "cd /root/ros_mb/out && rm -f $TAG.part.* $TAG.tgz $TAG.md5" >/dev/null 2>&1
    if [ "$ok" = 1 ]; then
      rm -rf "$DEST/$TAG"; mv "$STAGE/$TAG" "$DEST/$TAG"; rm -rf "$STAGE"; break
    fi
    rm -rf "$STAGE"; echo "  pull of $TAG incomplete (digest mismatch), retrying"; sleep 2
  done
done
echo "MATRIX_DONE $ARM r$REP $(date +%H:%M:%S)"
