#!/usr/bin/env bash
# One board job = build ONE schedule with the stamps walker of a ModelBlaster checkout, check it runs, then
# run it N times at real-time priority 80 and bring the output back. Few logins (every login starts desktop
# services on the board):
#   1. board.sh acquire        board free check + our lock                                 (1 login)
#   2. ModelBlaster's runner   build, deploy, ONE run = the smoke run                      (a few logins)
#   3. one ssh session         a small loop ON the board runs the binary N times with the runner's own
#                              command line, detached from the session, holding the shared flock
#                              /tmp/k3_board.lock and the /tmp/K3_IN_USE notice so other users' scripts
#                              see the board as busy; packs the output                    (1 login)
#   4. copy of one archive (resumable), md5 checked against the board's                    (1 or more)
#   5. board.sh release                                                                    (1 login)
# Walker options: per-hart workers, warm-up pass, placement record, and (stamps=1) the two extra trace
# stamps got_cores and kernel_end.
#
#   build:  <ModelBlaster>/build/k3_timing_loop_<name>/
#   raw:    <raw dir>/   stdout_r<k>.txt trace_r<k>.csv placement_r<k>.csv smoke_*.txt|csv table.c
#                        walker_main.c schedule.json config.json ir/ golden/ config.txt runner.log
#                        board_acquire.txt board_release.txt batch.log batch.md5
#   board:  <board dir>/<name>/xpurt/   (our own work folder)
#
# usage (from anywhere): job.sh <name> <raw dir> <schedule.json> <config.json> <repeats> [stamps=1]
# Env: K3TL_HOST          ssh alias of the board on this machine (default k3w)
#      K3TL_MODELBLASTER  ModelBlaster checkout with the stamps walker (default <repo>/zephyr-chipyard-sw/modelblaster)
#      K3TL_MODELS        space list of the schedule's models (default "mlp_control dronet yolov8_nano")
#      K3TL_BOARD_DIR     our folder on the board (default <home on the board>/mb_k3_timing_loop)
#      K3TL_OWNER, K3TL_LOCK_FILE, K3TL_MAX_LOAD, K3TL_WAIT_S   see board.sh
#      K3TL_REUSE_BUILD=1 the build is on disk and on the board already: no rebuild, no smoke run
#      K3TL_FETCH_ONLY=1  the batch already ran on the board: only fetch its archive
#      STALL_S (150), WATCHDOG_S (600)   patience with a stalled copy / a silent deploy step
#      builder.sh's: K3TL_BUILDER_IMAGE (default: no container, build on this machine), K3TL_DOCKER, K3_SSH_KEY,
#      K3TL_BUILDER_BOARD_ALIAS
# Exit codes: 0 ok; 3 raw dir already has runs; 5 board not free; 9-12 set-up; 20-25 a step failed (see config.txt).
set -uo pipefail
JOB="$1"; RAW="$2"; SCHED="$3"; CFG="$4"; REPS="$5"; STAMPS="${6:-1}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
export K3TL_HOST="${K3TL_HOST:-k3w}"
HOST="$K3TL_HOST"; PRIO=80; WATCHDOG_S="${WATCHDOG_S:-600}"; STALL_S="${STALL_S:-150}"
export K3TL_OWNER="${K3TL_OWNER:-$(id -un), k3_timing_loop}"     # one name per loop that runs at the same time
OWNER="$K3TL_OWNER"
[[ "$JOB" =~ ^[A-Za-z0-9_]+$ ]] || { echo "name must be letters, digits, _" >&2; exit 2; }
[[ "$REPS" =~ ^[1-9][0-9]*$ && "$STAMPS" =~ ^[01]$ ]] || { echo "bad repeats/stamps" >&2; exit 2; }
[[ -f "$SCHED" && -f "$CFG" ]] || { echo "no schedule or config" >&2; exit 2; }
K3TL_MB="$(cd "${K3TL_MODELBLASTER:-$REPO/zephyr-chipyard-sw/modelblaster}" 2>/dev/null && pwd)" || { echo "no ModelBlaster checkout" >&2; exit 9; }
[[ -x "$K3TL_MB/scripts/run_xpurt_k3.sh" ]] || { echo "no scripts/run_xpurt_k3.sh in $K3TL_MB" >&2; exit 9; }
SCHED="$(cd "$(dirname "$SCHED")" && pwd)/$(basename "$SCHED")"; CFG="$(cd "$(dirname "$CFG")" && pwd)/$(basename "$CFG")"
NAME="$(basename "${SCHED%.json}")"
SSHO=(-o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=15 -o ServerAliveCountMax=4)
[[ -e "$RAW/trace_r1.csv" ]] && { echo "$RAW already has runs; refusing to overwrite" >&2; exit 3; }
mkdir -p "$RAW/ir" "$RAW/golden"; R="$(cd "$RAW" && pwd)"
. "$HERE/builder.sh"
note() { echo "$*" | tee -a "$R/config.txt"; }
md5of() { if command -v md5 >/dev/null 2>&1; then md5 -q "$1"; else md5sum "$1" | cut -d' ' -f1; fi; }
mtime() { stat -f %m "$1" 2>/dev/null || stat -c %Y "$1"; }

K3_IP="$(ssh -G "$HOST" 2>/dev/null | awk '$1 == "hostname" {print $2}')"
K3_USER="$(ssh -G "$HOST" 2>/dev/null | awk '$1 == "user" {print $2}')"
export K3_IP K3_USER
{
  echo "name=$JOB sched=$SCHED config=$CFG repeats=$REPS stamps=$STAMPS workers=hart warmup=1 warm_start=0 placement=1 kb_riscv=0 policy=SCHED_FIFO_$PRIO"
  echo "host_alias=$HOST"
  echo "xpurt=$(git -C "$REPO" rev-parse --short HEAD) branch=$(git -C "$REPO" branch --show-current)"
  echo "modelblaster=$K3TL_MB commit=$(git -C "$K3TL_MB" rev-parse --short HEAD) branch=$(git -C "$K3TL_MB" rev-parse --abbrev-ref HEAD) dirty=$(git -C "$K3TL_MB" status --porcelain --untracked-files=no | wc -l | tr -d ' ')"
  echo "started=$(date '+%F %T')"
} > "$R/config.txt"

# ---- 1. board free + our lock
"$HERE/board.sh" acquire "k3_timing_loop $JOB" > "$R/board_acquire.txt" 2>&1 || {
  tail -15 "$R/board_acquire.txt"; note "STOP: board not free / no lock"; exit 5; }
grep -E '^(kernel|rtprio|who|load|K3_IN_USE|flock|ACQUIRED)' "$R/board_acquire.txt" | tail -7 | sed 's/^/board: /' >> "$R/config.txt"
release() { "$HERE/board.sh" release >> "$R/board_release.txt" 2>&1; tail -1 "$R/board_release.txt"; }
grep -q PREEMPT_RT "$R/board_acquire.txt" || { note "STOP: board kernel is not PREEMPT_RT"; release; exit 11; }
rt="$(sed -n 's/^rtprio: //p' "$R/board_acquire.txt" | tail -1)"
[[ "$rt" =~ ^[0-9]+$ ]] && (( rt >= PRIO )) || { note "STOP: rtprio limit on the board is '$rt', need >= $PRIO"; release; exit 12; }
bhome="$(sed -n 's/^home: //p' "$R/board_acquire.txt" | tail -1)"
BDIR="${K3TL_BOARD_DIR:-$bhome/mb_k3_timing_loop}/$JOB"
[[ "$BDIR" == /* ]] || { note "STOP: could not work out our folder on the board ('$BDIR')"; release; exit 10; }

# ---- 2. build + deploy + one smoke run with the checkout's runner (watchdog on the deploy step)
OUT_ROOT="build/k3_timing_loop_$JOB"
B="$K3TL_MB/$OUT_ROOT"; G="$B/_gen/$NAME"
MODELS="${K3TL_MODELS:-mlp_control dronet yolov8_nano}"          # K3TL_MODELS="yolov8_nano": one model
if [[ "${K3TL_REUSE_BUILD:-0}" == 1 && -x "$B/_build/$NAME/xpurt_harness" && -f "$G/$NAME.c" ]]; then
  note "reusing the existing build and the binary already on the board (K3TL_REUSE_BUILD=1): no rebuild, no smoke run; the batch log's binary md5 is compared below"
else
IMPL="x100:rvv=rvv_k3,x100:ime=ime_k3_x100,a100:rvv=rvv_k3_a100,a100:ime=ime_k3_a100"
args=(--schedule "$SCHED" --models "${MODELS// /,}" --backends ime_k3_x100,ime_k3_a100
      --impl-backends "$IMPL" --out-root "$OUT_ROOT")
# Warm-up: ours only (MB_XPURT_WARMUP=1: one untimed pass of the whole schedule). MB_XPURT_WARM=0 switches off the
# walker's default warm start (one untimed frame of every network before it), which none of the measured loop runs had.
envs=(MODELBLASTER_K3_HOST="$(k3tl_board_alias "$HOST")" MODELBLASTER_K3_REMOTE_ROOT="$BDIR" MB_XPURT_WARM=0 MB_XPURT_WARMUP=1 MB_XPURT_PLACEMENT=1
      MB_XPURT_REPEATS=1 MB_XPURT_WORKERS=hart MB_KB_RISCV=0 MODELBLASTER_K3_RT_PRIORITY="$PRIO")
[[ "$STAMPS" == 1 ]] && envs+=(MB_XPURT_STAMPS=1)
qargs="$(printf '%q ' "${args[@]}")"
CNAME="k3tl_${JOB}_$$"
k3tl_builder "$CNAME" "" "$(dirname "$SCHED")" env "${envs[@]}" bash -c "cd $(printf '%q' "$K3TL_MB") && scripts/run_xpurt_k3.sh $qargs" > "$R/runner.log" 2>&1 </dev/null &
pid=$!
while kill -0 "$pid" 2>/dev/null; do
  sleep 15
  grep -q "5/5 deploy" "$R/runner.log" 2>/dev/null || continue
  age=$(( $(date +%s) - $(mtime "$R/runner.log") ))
  # The copy-BACK of the smoke output is not needed for the results. If it makes no progress for STALL_S (its local
  # file does not grow), stop only that copy so the runner's own retry takes over; the upload of the binary is never
  # touched. While the copy is growing, the watchdog clock is reset.
  out_size="$(cat "$G"/*_stdout*.txt 2>/dev/null | wc -c | tr -d ' ')"
  if [[ "$out_size" != "${last_size:-}" ]]; then last_size="$out_size"; last_change=$(date +%s); fi
  if k3tl_builder_exec "$CNAME" pgrep -f 'scp .*_stdout' >/dev/null 2>&1; then
    [[ -n "${copy_seen:-}" ]] || { copy_seen=1; last_change=$(date +%s); }
    if (( $(date +%s) - last_change > STALL_S )); then
      k3tl_builder_exec "$CNAME" bash -c 'pkill -f "scp .*_stdout"; pkill -f "PermitLocalCommand=no.*_stdout"' >/dev/null 2>&1
      note "smoke-output copy stopped after ${STALL_S}s without a new byte ($(date '+%T')); the runner retries it"
      copy_seen=""
    fi
    touch "$R/runner.log"; age=0
  else
    copy_seen=""
  fi
  if (( age > WATCHDOG_S )); then
    note "WATCHDOG $(date '+%F %T'): runner.log silent ${age}s in the deploy step; stopping the build $CNAME"
    k3tl_builder_stop "$CNAME" "$pid"
    break
  fi
done
wait "$pid"; rc=$?
note "runner (build + smoke run) exit $rc at $(date '+%F %T')"
if (( rc != 0 )); then
  tail -25 "$R/runner.log"; note "STOP: build or smoke run failed"; release; exit 20
fi
# exit 0 = the smoke run itself succeeded on the board (the runner returns the program's own exit code); its output
# not coming back over a bad link is no reason to stop: the N runs below come back in a checked archive
[[ -f "$G/${NAME}_trace.csv" ]] || note "the smoke run's output did not come back (link); continuing, it is not used for results"
fi
for kind in stdout trace placement; do
  for f in "$G/${NAME}_${kind}".*; do [[ -e "$f" ]] && cp -p "$f" "$R/smoke_${kind}.${f##*.}"; done
done
cp -p "$G/$NAME.c" "$R/table.c"; cp -p "$G/${NAME}_main.c" "$R/walker_main.c"
cp -p "$SCHED" "$R/schedule.json"; cp -p "$CFG" "$R/config.json"
for n in $MODELS; do
  cp -p "$B/$n/int8/graph.json" "$R/ir/$n.graph.json"; cp -p "$B/$n/int8/io.npz" "$R/golden/$n.io.npz"
  echo "build md5 $n graph weights: $(md5of "$B/$n/int8/graph.json") $(md5of "$B/$n/int8/weights.npz")" >> "$R/config.txt"
done
if [[ -f "$R/smoke_trace.csv" ]]; then
  note "smoke: $(grep -c . "$R/smoke_trace.csv") trace lines, header ends: $(head -1 "$R/smoke_trace.csv" | rev | cut -d, -f1-2 | rev), policy line: $(head -1 "$R/smoke_stdout.txt")"
  grep -q "FATAL" "$R/smoke_stdout.txt" && { note "STOP: FATAL in the smoke run"; release; exit 21; }
  # the runner copies the smoke output back without a checksum: say if it arrived whole (the N runs below do not
  # depend on it; they come back in one md5-checked archive)
  grep -q "MODELBLASTER_XPURT_TRACE_END" "$R/smoke_stdout.txt" && note "smoke output complete: yes" || note "smoke output complete: NO (cut off on the way back; not used for results)"
fi
BIN_MD5="$(md5of "$B/_build/$NAME/xpurt_harness")"
note "binary md5 (local build): $BIN_MD5"

# ---- 3. N runs in one session; the loop runs on the board, detached from the session
batch() {
  { printf 'OWNER=%q\nDIR=%q\nNAME=%q\nN=%q\nPRIO=%q\n' "$OWNER" "$BDIR/xpurt" "$NAME" "$REPS" "$PRIO"
    cat <<'EOS'
cd "$DIR" || { echo "NO_DIR"; exit 9; }
echo "STARTED"
rm -f batch.status batch.log batch.md5 batch.tgz
cat > run_batch.sh <<'EOB'
#!/usr/bin/env bash
# Batch loop written by k3_timing_loop/job.sh. The run line is the ModelBlaster runner's own:
# ulimit -n 8192, taskset -c 0-7, chrt -f <prio>; plus MB_XPURT_WARM=0 (only the walker's warm-up pass, as in the smoke run).
OWNER="$1"; NAME="$2"; N="$3"; PRIO="$4"
exec 7>>/tmp/k3_board.lock
if ! flock -n 7; then echo "FLOCK_BUSY" > batch.status; exit 3; fi
echo "IN USE: $OWNER since $(date). Lock: flock /tmp/k3_board.lock" > /tmp/K3_IN_USE
trap 'grep -q -F "$OWNER" /tmp/K3_IN_USE 2>/dev/null && rm -f /tmp/K3_IN_USE' EXIT
ulimit -n 8192
{ echo "batch start $(date -u '+%F %T UTC') runs 1-$N prio $PRIO kernel $(uname -rv)"
  echo "policy check: $(taskset -c 0-7 chrt -f "$PRIO" sh -c 'chrt -p $$' 2>&1 | tr '\n' ' ')"
  echo "binary md5 $(md5sum < "./$NAME" | cut -c1-32)"; } >> batch.log
files=""
for k in $(seq 1 "$N"); do
  log="${NAME}_stdout_r$k.txt"
  echo "r$k start $(date -u '+%T') load $(cut -d' ' -f1-3 /proc/loadavg) who $(who | wc -l)" >> batch.log
  printf '%s\n' "xpurt_runner: sched_policy=SCHED_FIFO priority=$PRIO" > "$log"
  MB_XPURT_WARM=0 taskset -c 0-7 chrt -f "$PRIO" "./$NAME" >> "$log" 2>&1
  echo $? > "$log.rc"
  echo "r$k rc $(cat "$log.rc") bytes $(stat -c %s "$log")" >> batch.log
  files="$files $log $log.rc"
  sleep 1
done
echo "batch end $(date -u '+%F %T UTC') load $(cut -d' ' -f1-3 /proc/loadavg)" >> batch.log
md5sum $files > batch.md5
# xz when the board has it: the archive is about 2.6x smaller than with gzip, which matters on a slow link
if command -v xz >/dev/null 2>&1; then
  echo "archive: tar + xz -6" >> batch.log; tar cf - $files batch.log batch.md5 | xz -6 > batch.tgz
else
  echo "archive: tar + gzip" >> batch.log; tar czf batch.tgz $files batch.log batch.md5
fi
echo "DONE $(md5sum < batch.tgz | cut -c1-32)" > batch.status
EOB
nohup setsid bash run_batch.sh "$OWNER" "$NAME" "$N" "$PRIO" > batch.out 2>&1 < /dev/null &
echo "LAUNCHED"
t=0
while [ ! -s batch.status ] && [ "$t" -lt 1500 ]; do sleep 5; t=$((t + 5)); done
echo "STATUS: $(cat batch.status 2>/dev/null || echo none)"
EOS
  } | ssh "${SSHO[@]}" "$HOST" bash -s 2>&1
}
status=""
repack() {   # K3TL_FETCH_ONLY=1: no new runs; make the existing archive smaller (xz) and report its status
  { printf 'DIR=%q\n' "$BDIR/xpurt"
    cat <<'EOS'
cd "$DIR" || { echo "NO_DIR"; exit 9; }
echo "STARTED"
if [ -s batch.status ] && grep -q '^DONE' batch.status && gzip -t batch.tgz 2>/dev/null && command -v xz >/dev/null 2>&1; then
  gzip -dc batch.tgz | xz -6 > batch.repack && mv batch.repack batch.tgz && echo "DONE $(md5sum < batch.tgz | cut -c1-32)" > batch.status && echo "REPACKED with xz"
fi
echo "LAUNCHED"
echo "STATUS: $(cat batch.status 2>/dev/null || echo none)"
EOS
  } | ssh "${SSHO[@]}" "$HOST" bash -s 2>&1
}
[[ "${K3TL_FETCH_ONLY:-0}" == 1 ]] && { note "fetch only (K3TL_FETCH_ONLY=1): the batch on the board is not run again"; batch() { repack; }; }
for try in 1 2 3 4 5 6 7 8; do            # a session that never reached the board (no STARTED line) is safe to repeat
  out="$(batch)"; echo "$out" | sed 's/^/batch: /' | tee -a "$R/config.txt"
  grep -q -e '^STARTED' -e '^LAUNCHED' -e '^NO_DIR' <<<"$out" && break
  note "batch session did not reach the board (try $try); again in 20 s"; sleep 20
done
status="$(sed -n 's/^STATUS: //p' <<<"$out")"
# STARTED without LAUNCHED: the session dropped inside the remote script; the batch may be running, so ask for its status below
if ! grep -q -e '^LAUNCHED' -e '^STARTED' <<<"$out"; then note "STOP: the batch did not start"; release; exit 22; fi
tries=0
while [[ "$status" != DONE* && "$status" != FLOCK_BUSY ]] && (( tries < 40 )); do   # session dropped: ask again
  sleep 45; tries=$((tries + 1))
  status="$(ssh -n "${SSHO[@]}" "$HOST" "cat $BDIR/xpurt/batch.status 2>/dev/null || echo none" 2>/dev/null | tail -1)"
  note "poll $tries: ${status:-unreachable}"
done
if [[ "$status" != DONE* ]]; then note "STOP: batch status '${status:-none}'"; release; exit 23; fi
want="${status#DONE }"

# ---- 4. fetch one archive, check it against the board's md5, unpack, check every file
ok=0
rm -f "$R/batch.tgz"                       # resumable copy: only the missing bytes are asked for on each attempt
"$HERE/fetch_resume.sh" "$HOST" "$BDIR/xpurt/batch.tgz" "$R/batch.tgz" "$want" 40 >> "$R/fetch.log" 2>&1 && ok=1
note "archive copy: $(tail -1 "$R/fetch.log")"
if (( ok == 0 )); then note "STOP: could not fetch a complete batch.tgz (want md5 $want)"; release; exit 24; fi
T="$(mktemp -d)"; tar xf "$R/batch.tgz" -C "$T"      # tar finds the compression itself (xz or gzip)
bad=0
while read -r sum f; do
  [[ "$(md5of "$T/$f")" == "$sum" ]] || { bad=$((bad + 1)); note "md5 MISMATCH after unpack: $f"; }
done < "$T/batch.md5"
cp -p "$T/batch.log" "$T/batch.md5" "$R/"
for k in $(seq 1 "$REPS"); do
  cp -p "$T/${NAME}_stdout_r$k.txt" "$R/stdout_r$k.txt"; cp -p "$T/${NAME}_stdout_r$k.txt.rc" "$R/stdout_r$k.rc"
  awk '/MODELBLASTER_XPURT_TRACE_BEGIN/{f=1;next} /MODELBLASTER_XPURT_TRACE_END/{f=0} f' "$R/stdout_r$k.txt" > "$R/trace_r$k.csv"
  awk '/MODELBLASTER_XPURT_PLACEMENT_BEGIN/{f=1;next} /MODELBLASTER_XPURT_PLACEMENT_END/{f=0} f' "$R/stdout_r$k.txt" > "$R/placement_r$k.csv"
done
rm -rf "$T" "$R/batch.tgz"
note "fetched $REPS runs; archive md5 $want matches the board; files with a wrong md5 after unpack: $bad"
note "exit codes: $(cat "$R"/stdout_r*.rc | sort | uniq -c | tr '\n' ' ')"
note "policy check on the board: $(grep 'policy check' "$R/batch.log")"
board_md5="$(sed -n 's/^binary md5 //p' "$R/batch.log" | head -1)"
if [[ "$board_md5" == "$BIN_MD5" ]]; then note "binary on the board == local build (md5 $BIN_MD5)"
else note "BINARY MISMATCH: board $board_md5 vs local build $BIN_MD5"; bad=$((bad + 1)); fi
note "trace rows per run: $(for k in $(seq 1 "$REPS"); do echo $(( $(wc -l < "$R/trace_r$k.csv") - 1 )); done | sort -n | uniq -c | tr '\n' ' ')"

# ---- 5. release
release
note "finished=$(date '+%F %T')"
(( bad == 0 )) || exit 25
