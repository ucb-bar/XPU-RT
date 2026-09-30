#!/usr/bin/env bash
# Execute a solved coupled-chain schedule on the K1 with the per-core sampler running, and keep
# everything the run produced: the per-dispatch trace, the per-hart accounting block, the
# sampler's per-core busy series, and a manifest that says which solver and policy produced it.
#
#   scripts/run_xpurt_long.sh <schedule.json> <label> [replicates]
#
# Primary runs are SCHED_OTHER, the policy the ROS arms run under; a FIFO run is a labelled
# variant (MODELBLASTER_K1_RT_PRIORITY=80), never the only row.
set -u
SCHED="${1:?schedule json}"; LABEL="${2:?label}"; REPS="${3:-3}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
SCHED="$(readlink -f "$SCHED")"      # the board runner resolves paths from its own tree
HOST="${MODELBLASTER_K1_HOST:-k1}"
DEST="results/codesign_feedback/xpurt_long"; mkdir -p "$DEST"
# the networks are whatever the schedule names (the heavier stack adds ffn_block and dronet)
MODELS=$(python3 -c "
import json,sys; sys.path.insert(0,'xpu-rt'); from job_names import split_job_name
known={'yolov8_nano_64x96','fused_full','mlp_control','ffn_block','dronet'}
s=json.load(open(sys.argv[1]))['dispatches']; nets=[]
for v in s.values():
    n,_=split_job_name(v['job_name'],known)
    if n not in nets: nets.append(n)
print(','.join(nets))" "$1")
eval "$(bash scripts/setup_spacemit_toolchain.sh 2>/dev/null)"
STAGED=""; for m in ${MODELS//,/ }; do STAGED="$STAGED --staged-ir $m:$PWD/ModelBlaster/build/k1_xpurt/$m/int8"; done
SNAME="$(basename "$SCHED" .json)"
POLICY="${MODELBLASTER_K1_RT_PRIORITY:+fifo${MODELBLASTER_K1_RT_PRIORITY}}"; POLICY="${POLICY:-other}"
HOGS="${HOGS:-0}"; [ "$HOGS" != 0 ] && POLICY="${POLICY}_hog${HOGS}"      # background CPU hogs, unpinned, started before the run

for k in $(seq 1 "$REPS"); do
  TAG="${LABEL}_${POLICY}_run${k}"
  echo "=== $(date +%H:%M:%S) $TAG ==="
  ssh "$HOST" "pkill -x cpu_sampler; mkdir -p /root/mb_k1/xpurt; cd /root/ros_mb; (nohup ./cpu_sampler 100 /root/mb_k1/xpurt/cpu_$TAG.csv >/dev/null 2>&1 & echo \$! > /root/mb_k1/xpurt/cpu.pid); rm -f /root/mb_k1/xpurt/hog.pid; for h in \$(seq 1 $HOGS); do (nohup ./cpu_hog >/dev/null 2>&1 & echo \$! >> /root/mb_k1/xpurt/hog.pid); done; sleep 0.5"
  WALL0=$(ssh "$HOST" "date +%s%3N")
  GEN="$REPO/ModelBlaster/tmp/${LABEL}_board/_gen/$SNAME"        # --out-root is taken relative to ModelBlaster/
  rm -f "$GEN/${SNAME}_trace.csv" "$GEN/${SNAME}_stdout.txt"     # never let a previous run's files stand in for this one
  for attempt in 1 2 3; do       # the board's sshd drops connections now and then; a run with no trace is retried
    timeout 2400 env CROSS="${CROSS:-}" bash ModelBlaster/scripts/run_xpurt_k1.sh \
      --schedule "$SCHED" --models "$MODELS" $STAGED \
      --backends "${BACKENDS:-rvv_x60,rvv_x60}" --quant int8 --out-root "tmp/${LABEL}_board" \
      > "$DEST/board_$TAG.log" 2>&1
    RC=$?
    # complete = one row per scheduled dispatch. The run's stdout (with the trace inside it) is
    # written on the board; the copy back is what the flaky link truncates, so a short trace is
    # re-pulled from the board's file first and the run itself only repeated if that fails too.
    WANT=$(python3 -c "import json,sys;print(len(json.load(open(sys.argv[1]))['dispatches']))" "$SCHED")
    GOT=$(( $(wc -l < "$GEN/${SNAME}_trace.csv" 2>/dev/null || echo 1) - 1 ))
    for pull in 1 2 3 4; do
      [ "$GOT" -ge "$WANT" ] && break
      sleep 2; scp -q "$HOST:/root/mb_k1/xpurt/${SNAME}_stdout.txt" "$GEN/${SNAME}_stdout.txt" 2>/dev/null
      awk '/MODELBLASTER_XPURT_TRACE_BEGIN/{f=1;next} /MODELBLASTER_XPURT_TRACE_END/{f=0} f' "$GEN/${SNAME}_stdout.txt" > "$GEN/${SNAME}_trace.csv"
      GOT=$(( $(wc -l < "$GEN/${SNAME}_trace.csv" 2>/dev/null || echo 1) - 1 ))
      echo "  re-pulled the board's stdout: $GOT of $WANT rows (pull $pull)"
    done
    [ "$GOT" -ge "$WANT" ] && break
    echo "  trace has $GOT of $WANT rows after attempt $attempt; re-running"; sleep 3
  done
  ssh "$HOST" "kill \$(cat /root/mb_k1/xpurt/cpu.pid); [ -f /root/mb_k1/xpurt/hog.pid ] && kill \$(cat /root/mb_k1/xpurt/hog.pid); sleep 0.3"
  cp "$GEN/${SNAME}_trace.csv" "$DEST/trace_$TAG.csv" 2>/dev/null
  cp "$GEN/${SNAME}_stdout.txt" "$DEST/stdout_$TAG.txt" 2>/dev/null
  scp -q "$HOST:/root/mb_k1/xpurt/cpu_$TAG.csv" "$DEST/cpu_$TAG.csv"
  .venv/bin/python - "$DEST" "$TAG" "$SCHED" "$RC" "$WALL0" <<'PY'
import sys, os, json, csv, hashlib, re
dest, tag, sched, rc, wall0 = sys.argv[1:6]
sys.path.insert(0, "ModelBlaster/scripts")
from parse_runtime_breakdown import parse_hart_acc
txt = open(f"{dest}/stdout_{tag}.txt").read() if os.path.exists(f"{dest}/stdout_{tag}.txt") else ""
acc = parse_hart_acc(txt) if txt else []
if acc:
    with open(f"{dest}/hart_acc_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(acc[0].keys())); w.writeheader(); w.writerows(acc)
m = re.search(r"run_t0_rdtime=(\d+)", txt); pol = re.search(r"sched_policy=(\S+)", txt)
ver = re.findall(r"MODELBLASTER_VERIFY[^\n]*", txt)
meta = json.load(open(sched)).get("metadata", {})
staged = "ModelBlaster/build/k1_xpurt/yolov8_nano_64x96/int8/.staged_from"
ir_sha = next((l.split("=", 1)[1].strip() for l in open(staged) if l.startswith("sha256=")), "") if os.path.exists(staged) else ""
def _cam_hz(meta):
    pn = meta.get("periodic_networks"); per = meta.get("camera_period_ms")
    if not per and isinstance(pn, dict):
        e = pn.get("yolov8_nano_64x96"); per = e if isinstance(e, (int, float)) else (e or {}).get("period") or (e or {}).get("period_ms")
    elif not per and isinstance(pn, list):
        e = next((x for x in pn if str(x.get("name", x.get("network", ""))).startswith("yolov8")), None); per = (e or {}).get("period") or (e or {}).get("period_ms")
    return round(1000.0 / float(per), 3) if per else None
def _span_s(path):
    try:
        cyc = [(int(r["actual_start_cycles"]), int(r["actual_end_cycles"])) for r in csv.DictReader(open(path)) if r.get("actual_end_cycles") not in (None, "", "0")]
        return round((max(e for _, e in cyc) - min(s for s, _ in cyc)) / 24e6, 3) if cyc else None
    except Exception:
        return None
sha_full = hashlib.sha256(open(sched, "rb").read()).hexdigest()
man = {"tag": tag, "schedule": sched, "schedule_relpath": os.path.relpath(sched, os.getcwd()), "schedule_sha256": sha_full, "schedule_sha256_16": sha_full[:16],
       "camera_hz": _cam_hz(meta), "seconds": _span_s(f"{dest}/trace_{tag}.csv"),
       "solver": meta.get("solver") or meta.get("scheduler") or os.path.basename(sched),
       "board_recost": meta.get("board_recost"), "ir_sha256_yolo": ir_sha,
       "sched_policy": pol.group(1) if pol else None, "run_t0_rdtime": int(m.group(1)) if m else None,
       "wall_start_epoch_ms": int(wall0), "exit_code": int(rc), "verify_lines": ver[:3],
       "rdtime_hz": 24000000, "note": "numeric verification covers instance 0 only"}
json.dump(man, open(f"{dest}/manifest_{tag}.json", "w"), indent=2)
print(f"  trace rows: {sum(1 for _ in open(f'{dest}/trace_{tag}.csv')) - 1 if os.path.exists(f'{dest}/trace_{tag}.csv') else 0}"
      f"  hart_acc: {len(acc)}  policy: {man['sched_policy']}  exit: {rc}  verify: {ver[:1]}")
PY
done
echo "XPURT_LONG_DONE $LABEL $(date +%H:%M:%S)"
