#!/usr/bin/env bash
# usage: 09_analyse.sh [outfile]        (default g5fine/RESULTS_raw_repro.txt)
#
# Recompute every statistic in g5fine/RESULTS.txt from the fetched summaries.
# Runs, in g5fine/, the three analysis scripts that own the numbers:
#   analyze_fine.py   per-arm marginal rate with a NAIVE and a DESIGN-CORRECTED
#                     interval, plus the paired per-seed penalty vs lat0
#   compare_tasks.py  difference-in-differences between tasks (Welch df)
#   arm_timing.py     what each arm did to the COMMAND STREAM: observation age at
#                     actuation, dispatches/episode, new-result-per-actuation
#
# THE TWO STATISTICAL RULES, because results are wrong without them:
#
# * EPISODES CLUSTER BY CONFIG, so a naive binomial overstates precision.  A seed
#   (--init-rng) seeds ONLY the policy's JAX key; object placement comes from
#   obj_init_options.episode_id 0..23, identical in every run and every arm.  The
#   24 configs span only ~4x6 cm yet produce per-config success rates from 17.5%
#   to 93.7%.  analyze_fine.py therefore reports the design-corrected interval,
#   ICC with the CONFIG as cluster and DEFF = 1 + (m-1)*ICC.  Measured ICCs run
#   0.13 (egg lat0) to 0.58 (drawer serial283) -- DEFF up to 6.3.
#
# * PAIRED PER-SEED CONTRASTS USE t QUANTILES, NOT NORMAL.  With a handful of
#   seeds the normal quantile understates the interval badly: at 2 seeds
#   t(1) = 12.706 against z = 1.96, a 6.5x error.
#
# It does NOT matter that the harness is not run-to-run deterministic: the
# pairing is on the episode CONFIG set, which is deterministic.  Individual
# episodes are not reproducible; aggregate rates are.  See ../NONDETERMINISM.md.
#
# Time: ~20 s.  Produces: <outfile>, and prints it.
# It worked if the four task blocks print and the per-arm n is 240 per arm
# (10 seeds x 24 configs) for a full 6x10 ladder.

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/common.sh"
[ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ] && { sed -n '2,32p' "$0"; exit 0; }
OUT="${1:-$G5FINE/RESULTS_raw_repro.txt}"
octo_env
cd "$G5FINE"
{
  echo "Generated $(date -u +%Y-%m-%dT%H:%M:%SZ) from $(find runs -name summary.json | wc -l) summaries"
  echo "harness md5 $(md5sum "$FINE/finegrain_eval.py" | cut -c1-12)   checkpoint $CKPT"
  echo; echo "################ analyze_fine.py ################";  python analyze_fine.py
  echo; echo "################ compare_tasks.py ################"; python compare_tasks.py
  echo; echo "################ arm_timing.py ################";    python arm_timing.py
} > "$OUT" 2>&1
cat "$OUT"
echo; echo "wrote $OUT ($(wc -l < "$OUT") lines)"
echo "(g5fine/make_raw_google.sh is the original of this, writing RESULTS_raw_google.txt)"
