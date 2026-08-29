#!/usr/bin/env bash
# Deprecated shim. This flow is no longer specific to the mlp_control +
# dronet + yolov8_nano spike workload: scripts/repro_workload.sh takes any
# workload spec and runs whatever that spec describes, on spike or FireSim.
#
# Forwards every argument through, defaulting to the spec this script used
# to hardcode, so existing commands (and docs/mlp_dronet_yolo_spike_
# reproduction.md's older sections) keep working.
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[repro] note: repro_mlp_dronet_yolo_spike.sh is now a shim for" \
     "repro_workload.sh -- call that directly with your spec" >&2

for arg in "$@"; do
    case "${arg}" in
        -h|--help) exec bash "${SCRIPT_DIR}/repro_workload.sh" --help ;;
        --networks-json) has_spec=1 ;;
        -*) ;;
        *) has_spec=1 ;;
    esac
done

if [[ -n "${has_spec:-}" ]]; then
    exec bash "${SCRIPT_DIR}/repro_workload.sh" "$@"
fi
exec bash "${SCRIPT_DIR}/repro_workload.sh" \
    "${SCRIPT_DIR}/../data/toplevel/networks_mlp_dronet_yolo_spike.json" "$@"
