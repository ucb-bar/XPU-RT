#!/usr/bin/env bash
# run_loop_plan_on_qrb5165.sh
# ===========================
#
# On-board (QRB5165) measurement runner for the closed-loop scheduler's
# converged plan. Invokes qnn-net-run for each scheduled partition,
# captures per-lane wall time, and emits a measurement_record_board_v1
# JSON the loop can consume via xpu_rt_apply_measurement (host-side).
#
# This script is meant to be COPIED TO THE BOARD and run there. It does
# NOT ssh into the board itself — that step is the operator's
# responsibility (see scripts/board/README_board_loop.md).
#
# It also does NOT pre-build DLCs; it assumes plan.json's `dlc_path`
# entries already exist on the board.
#
# Usage:
#   bash run_loop_plan_on_qrb5165.sh \
#       --plan-json /data/local/tmp/plan.json \
#       --output-json /data/local/tmp/measurement.json \
#       [--qnn-sdk-root /root/qairt] \
#       [--input-list /data/local/tmp/input_list.txt] \
#       [--profiling-level basic|detailed|off]
#
# plan.json schema (see scripts/board/emit_loop_plan.py for the host-side
# generator):
#
#   {
#     "schema_version": "loop_plan_board_v1",
#     "workload_id": "yolov8n",
#     "target_id":   "qrb5165",
#     "iters":       10,
#     "partitions": [
#       {
#         "partition_id": "p0",
#         "backend":      "DSP",     # CPU | GPU | DSP
#         "dlc_path":     "/root/models/yolov8n/yolov8n_quantized.dlc",
#         "iters":        10,         # optional override
#         "input_list":   "/root/.../input_list.txt"   # optional override
#       },
#       ...
#     ]
#   }
#
# Output JSON shape (consumed by the host adapter in the README):
#
#   {
#     "schema_version":   "measurement_record_board_v1",
#     "target_id":        "qrb5165",
#     "captured_at":      "<ISO 8601 UTC>",
#     "workload_id":      "yolov8n",
#     "iters":            10,
#     "per_backend_mean_us": { "CPU": 134300.0, "DSP": 254800.0, ... },
#     "raw_per_partition_us": [
#       { "partition_id": "p0", "backend": "DSP",
#         "mean_us": 254800.0, "iters": 10, "ok": true },
#       ...
#     ]
#   }

set -euo pipefail

PROG="$(basename "$0")"

# ---------------------------------------------------------------------------
# Defaults (mirror xpu_rt/targets/backends/qnn/on_board_runner.py).
# ---------------------------------------------------------------------------
QNN_SDK_ROOT="${QNN_SDK_ROOT:-/root/qairt}"
INPUT_LIST_DEFAULT=""
PROFILING_LEVEL="basic"
PLAN_JSON=""
OUTPUT_JSON=""

# ---------------------------------------------------------------------------
# Backend → libQnn*.so mapping (matches BACKEND_LIB in on_board_runner.py).
# ---------------------------------------------------------------------------
backend_lib_for() {
    case "$1" in
        CPU) echo "libQnnCpu.so" ;;
        GPU) echo "libQnnGpu.so" ;;
        DSP|HTP) echo "libQnnHtp.so" ;;
        *) echo "" ;;
    esac
}

# ---------------------------------------------------------------------------
# CLI parsing.
# ---------------------------------------------------------------------------
print_usage() {
    cat <<EOF
$PROG — On-board (QRB5165) closed-loop plan runner.

Required:
  --plan-json    PATH    JSON describing partitions to measure (see header).
  --output-json  PATH    Where to write the measurement_record_board_v1 JSON.

Optional:
  --qnn-sdk-root PATH    QNN SDK root (default: \$QNN_SDK_ROOT or /root/qairt).
  --input-list   PATH    Default input_list.txt for partitions that do not
                         carry their own. Each partition may override via the
                         "input_list" field in plan.json.
  --profiling-level LVL  qnn-net-run --profiling_level (basic|detailed|off).
                         Default: basic. ("off" disables qnn profiling and
                         relies only on wall-clock timing.)
  -h, --help             Print this message and exit.

Notes:
  * This script must run ON the QRB5165 board (it shells out to
    /root/qairt/bin/target/qnn-net-run). Use --help to verify it parses.
  * The host-side companion is scripts/board/emit_loop_plan.py — it reads
    a converged LoopState and writes the plan.json this script consumes.
  * The host-side adapter that feeds output back to the loop is documented
    in scripts/board/README_board_loop.md.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --plan-json)        PLAN_JSON="$2";        shift 2 ;;
        --output-json)      OUTPUT_JSON="$2";      shift 2 ;;
        --qnn-sdk-root)     QNN_SDK_ROOT="$2";     shift 2 ;;
        --input-list)       INPUT_LIST_DEFAULT="$2"; shift 2 ;;
        --profiling-level)  PROFILING_LEVEL="$2";  shift 2 ;;
        -h|--help)          print_usage; exit 0 ;;
        *)                  echo "$PROG: unknown arg: $1" >&2; print_usage >&2; exit 2 ;;
    esac
done

if [[ -z "$PLAN_JSON" || -z "$OUTPUT_JSON" ]]; then
    echo "$PROG: --plan-json and --output-json are both required" >&2
    print_usage >&2
    exit 2
fi

if [[ ! -f "$PLAN_JSON" ]]; then
    echo "$PROG: plan json not found: $PLAN_JSON" >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# python3 (preferred) or python is required for JSON parsing.
# Falls back to a tiny built-in parser if neither is present (unusual on
# QRB5165 — the Yocto image ships python3 — but we degrade gracefully).
# ---------------------------------------------------------------------------
PYBIN=""
for cand in python3 python; do
    if command -v "$cand" >/dev/null 2>&1; then
        PYBIN="$cand"
        break
    fi
done

if [[ -z "$PYBIN" ]]; then
    echo "$PROG: neither python3 nor python found on PATH; aborting" >&2
    exit 3
fi

# ---------------------------------------------------------------------------
# Parse plan.json into shell-friendly TSV: one line per partition.
# Columns: partition_id <TAB> backend <TAB> dlc_path <TAB> iters <TAB> input_list
# ---------------------------------------------------------------------------
read_plan_tsv() {
    "$PYBIN" - "$PLAN_JSON" "$INPUT_LIST_DEFAULT" <<'PYEOF'
import json, sys
plan_path, default_input_list = sys.argv[1], sys.argv[2]
with open(plan_path) as fh:
    plan = json.load(fh)
default_iters = int(plan.get("iters", 10))
for part in plan.get("partitions", []):
    pid = str(part["partition_id"])
    backend = str(part["backend"])
    dlc = str(part["dlc_path"])
    iters = int(part.get("iters", default_iters))
    input_list = str(part.get("input_list") or default_input_list or "")
    print("\t".join([pid, backend, dlc, str(iters), input_list]))
PYEOF
}

PLAN_TSV="$(read_plan_tsv)"

if [[ -z "$PLAN_TSV" ]]; then
    echo "$PROG: plan.json has no partitions" >&2
    exit 2
fi

# Workload id and target id come from the plan header.
WORKLOAD_ID="$("$PYBIN" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("workload_id",""))' "$PLAN_JSON")"
TARGET_ID="$("$PYBIN" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("target_id","qrb5165"))' "$PLAN_JSON")"
DEFAULT_ITERS="$("$PYBIN" -c 'import json,sys; print(int(json.load(open(sys.argv[1])).get("iters",10)))' "$PLAN_JSON")"

# ---------------------------------------------------------------------------
# QNN environment (mirrors xpu_rt/targets/backends/qnn/on_board_runner.py).
# ---------------------------------------------------------------------------
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/target:${LD_LIBRARY_PATH:-}"
export ADSP_LIBRARY_PATH="$QNN_SDK_ROOT/lib/hexagon-v66;/dsp/cdsp;/dsp"

# ---------------------------------------------------------------------------
# Run one partition. Echoes "MEAN_US=<float>" and "OK=<true|false>"
# to stdout; stderr carries the qnn-net-run failure tail (if any).
# ---------------------------------------------------------------------------
run_partition() {
    local pid="$1" backend="$2" dlc="$3" iters="$4" input_list="$5"
    local lib
    lib="$(backend_lib_for "$backend")"
    if [[ -z "$lib" ]]; then
        echo "[partition $pid] unknown backend: $backend" >&2
        echo "MEAN_US=0"
        echo "OK=false"
        echo "ERROR=unknown_backend"
        return 0
    fi
    if [[ ! -f "$dlc" ]]; then
        echo "[partition $pid] DLC missing: $dlc" >&2
        echo "MEAN_US=0"
        echo "OK=false"
        echo "ERROR=dlc_missing"
        return 0
    fi
    if [[ -z "$input_list" || ! -f "$input_list" ]]; then
        echo "[partition $pid] input_list missing: $input_list" >&2
        echo "MEAN_US=0"
        echo "OK=false"
        echo "ERROR=input_list_missing"
        return 0
    fi

    local out_dir multi_input start_ns end_ns total_ns mean_us
    out_dir="/tmp/qnn_loop_$$_${pid}"
    rm -rf "$out_dir"; mkdir -p "$out_dir"
    multi_input="$(mktemp)"
    local line
    line="$(head -n1 "$input_list")"
    local i=0
    while [[ $i -lt $iters ]]; do echo "$line" >> "$multi_input"; i=$((i+1)); done

    local extra_flags=""
    if [[ "$PROFILING_LEVEL" != "off" ]]; then
        extra_flags="--profiling_level $PROFILING_LEVEL"
    fi

    start_ns="$(date +%s%N)"
    if ! "$QNN_SDK_ROOT/bin/target/qnn-net-run" \
            --dlc_path "$dlc" \
            --backend "$QNN_SDK_ROOT/lib/target/$lib" \
            --input_list "$multi_input" \
            --output_dir "$out_dir" \
            $extra_flags >"$out_dir/stdout.log" 2>"$out_dir/stderr.log"; then
        echo "[partition $pid] qnn-net-run failed (rc=$?)" >&2
        tail -n 20 "$out_dir/stderr.log" >&2 || true
        echo "MEAN_US=0"
        echo "OK=false"
        echo "ERROR=qnn_net_run_failed"
        rm -f "$multi_input"
        return 0
    fi
    end_ns="$(date +%s%N)"
    total_ns=$((end_ns - start_ns))
    mean_us="$("$PYBIN" -c "print(${total_ns}/1000.0/${iters})")"

    rm -f "$multi_input"
    echo "MEAN_US=$mean_us"
    echo "OK=true"
}

# ---------------------------------------------------------------------------
# Main loop.
# ---------------------------------------------------------------------------
TMP_RESULTS="$(mktemp)"
trap 'rm -f "$TMP_RESULTS"' EXIT

while IFS=$'\t' read -r pid backend dlc iters input_list; do
    [[ -z "$pid" ]] && continue
    echo "[plan] running partition=$pid backend=$backend dlc=$dlc iters=$iters" >&2
    out="$(run_partition "$pid" "$backend" "$dlc" "$iters" "$input_list" 2>&1 || true)"
    mean_us="$(echo "$out" | sed -n 's/^MEAN_US=//p' | head -n1)"
    ok="$(echo "$out" | sed -n 's/^OK=//p' | head -n1)"
    err="$(echo "$out" | sed -n 's/^ERROR=//p' | head -n1)"
    : "${mean_us:=0}"
    : "${ok:=false}"
    : "${err:=}"
    printf '%s\t%s\t%s\t%s\t%s\n' "$pid" "$backend" "$mean_us" "$iters" "$ok|$err" >> "$TMP_RESULTS"
done <<< "$PLAN_TSV"

# ---------------------------------------------------------------------------
# Aggregate per-backend means and write the final JSON.
# ---------------------------------------------------------------------------
"$PYBIN" - "$TMP_RESULTS" "$OUTPUT_JSON" "$WORKLOAD_ID" "$TARGET_ID" "$DEFAULT_ITERS" <<'PYEOF'
import json, sys, datetime
results_path, out_path, workload_id, target_id, default_iters = sys.argv[1:]
default_iters = int(default_iters)

raw = []
per_backend_sum: dict[str, float] = {}
per_backend_count: dict[str, int] = {}

with open(results_path) as fh:
    for line in fh:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 5:
            continue
        pid, backend, mean_us_s, iters_s, ok_err = parts
        mean_us = float(mean_us_s)
        iters = int(iters_s)
        ok_str, _, err = ok_err.partition("|")
        ok = ok_str == "true"
        raw.append({
            "partition_id": pid,
            "backend": backend,
            "mean_us": mean_us,
            "iters": iters,
            "ok": ok,
            "error": err,
        })
        if ok and mean_us > 0:
            per_backend_sum[backend] = per_backend_sum.get(backend, 0.0) + mean_us
            per_backend_count[backend] = per_backend_count.get(backend, 0) + 1

per_backend_mean = {
    b: per_backend_sum[b] / per_backend_count[b] for b in per_backend_sum
}

payload = {
    "schema_version": "measurement_record_board_v1",
    "target_id": target_id or "qrb5165",
    "captured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "workload_id": workload_id or "",
    "iters": default_iters,
    "per_backend_mean_us": per_backend_mean,
    "raw_per_partition_us": raw,
}

with open(out_path, "w") as fh:
    json.dump(payload, fh, indent=2)

print(f"[plan] wrote {out_path}", file=sys.stderr)
print(f"[plan] per_backend_mean_us = {per_backend_mean}", file=sys.stderr)
PYEOF

echo "[$PROG] done. Output: $OUTPUT_JSON" >&2
