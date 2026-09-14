# common.sh -- shared paths and environment for the arm-experiment repro scripts.
# SOURCED, never executed.  Every path here is absolute on purpose: these scripts
# are run from cron, from ssh, and from the wrong directory, and a relative path
# has already cost this project one silently-wrong sweep.
#
# usage:  source /scratch2/dima/misc_sw/octo_work/sim_eval/roselite/finegrain/repro/common.sh

set -u

# ---------------------------------------------------------------- local layout
CONDA_ROOT="${CONDA_ROOT:-/scratch2/dima/miniforge3}"
ENVNAME="${ENVNAME:-octo_sim}"
SIMROOT="${SIMROOT:-/scratch2/dima/misc_sw/octo_work/sim_eval}"
ROSE="$SIMROOT/roselite"
FINE="$ROSE/finegrain"
G5FINE="${G5FINE:-$FINE/g5fine}"
REPRO="$FINE/repro"

# ------------------------------------------------------------------ checkpoint
# PIN IT EXPLICITLY, ALWAYS.  run_eval.py DEFAULTS to octo-small-1.5; the fine
# harness defaults to octo-small (1.0); the published SIMPLER "octo-small" row is
# 1.0.  Mixing them invalidates every comparison in this study (1.0 vs 1.5 is
# 52.8% vs 26.4% on eggplant and 45.8% vs 4.2% on spoon -- see
# XPU-RT/qnn_models/octo/OCTO_INT8_QRB5165.md section 8).
CKPT="${CKPT:-hf://rail-berkeley/octo-small}"

# ------------------------------------------------------------- required env
# Sourced by every runner.  See REPRODUCE_ARM_EXPERIMENTS.md "Install traps".
octo_env() {
  # shellcheck disable=SC1091
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
  conda activate "$ENVNAME"
  # SAPIEN otherwise picks the llvmpipe CPU device -> ErrorExtensionNotPresent
  export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
  # JAX otherwise preallocates ~75% of the GPU, which this SHARED box does not own
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  export TOKENIZERS_PARALLELISM=false
  export DISPLAY=""
}

# --------------------------------------------------------------- the six arms
# name        latency_ms  cadence_ms   provenance (ALL MEASURED on the QRB5165;
#                                      see XPU-RT/qnn_models/octo/REPRODUCE_SCHEDULES.md)
#   lat0        0.0       auto         ideal reference, no compute latency
#   pipe110     117.7     117.6        pipelined, 110 ms cadence, 10 instances
#   pipe200     231.8     203.0        pipelined, 200 ms cadence, 5 instances
#   serial283   283.4     283.4        3-way serial CPU+DSP+HTA, one in flight
#   fp32_555    555.0     555.0        fp32 numerically-valid CPU path
#   cpu685      684.8     684.8        CPU-only int8 monolith
# This table is a COPY of the one in g5fine/job.sh, which is the table the
# published sweep ran.  Keep them identical.
arm_lat() { case "$1" in
    lat0) echo 0 ;; pipe110) echo 117.7 ;; pipe200) echo 231.8 ;;
    serial283) echo 283.4 ;; fp32_555) echo 555.0 ;; cpu685) echo 684.8 ;;
    *) echo "BADARM" ;; esac; }
arm_per() { case "$1" in
    lat0) echo auto ;; pipe110) echo 117.6 ;; pipe200) echo 203.0 ;;
    serial283) echo 283.4 ;; fp32_555) echo 555.0 ;; cpu685) echo 684.8 ;;
    *) echo "BADARM" ;; esac; }
ARMS_ALL="lat0 pipe110 pipe200 serial283 fp32_555 cpu685"

# task short name -> SimplerEnv task id (same mapping as g5fine/job.sh)
task_id() { case "$1" in
    spoon)  echo widowx_spoon_on_towel ;;
    egg)    echo widowx_put_eggplant_in_basket ;;
    drawer) echo google_robot_close_drawer ;;
    coke)   echo google_robot_pick_coke_can ;;
    *) echo "BADTASK" ;; esac; }

# ------------------------------------------------------------------- AWS
# The manager holds the credentials; this box has the aws CLI but not the creds.
# VERIFIED 2026-09-06: `awsq ec2 describe-instances ...` works from here.
MANAGER="${MANAGER:-ubuntu@3.88.218.39}"
KEY="${KEY:-$HOME/.ssh/firesim.pem}"
REGION="${REGION:-us-east-1}"
SSH="ssh -i $KEY -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=20"
WORKER_IDS=(i-043560448065532ff i-05289cfa068e00c74 i-09db19b86bffe6acb
            i-078a22ff12e6c885d i-039d68dfbf56a4f61 i-0f7e65981bc8d4d7c)
WORKER_NAMES=(w0 w1 w2 w3 w4 w5)

# Run one aws command on the manager.  Always time-bounded, never interactive.
awsq() { timeout -s KILL 300 $SSH "$MANAGER" "aws --region $REGION $*"; }
