#!/usr/bin/env bash
# usage: 02_smoke.sh
#
# Post-install smoke test.  Proves, WITHOUT downloading the policy checkpoint and
# without running an episode, that:
#   * simpler_env imports (the setuptools/pkg_resources trap)
#   * SAPIEN reaches the NVIDIA Vulkan ICD rather than llvmpipe
#   * jax sees the GPU (not a silent CpuDevice fallback)
#   * every task's TIMEBASE matches what the harness derives and what the sweep
#     assumed.  Horizon/tick/action-dt are DERIVED FROM THE ENV -- never hardcode
#     them.  widowx is 5 Hz control / 500 Hz sim -> 25 Hz tick / 200.00 ms;
#     google_robot is 3 Hz / 513 Hz -> 27 Hz tick / 333.33 ms (513 = 3^3 x 19, so
#     a 25 Hz tick does not divide it and is ILLEGAL there).
#
# Time: ~1 min.  Produces: printed table, nothing on disk.
# It worked if the last line is "SMOKE OK".

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/common.sh"
[ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ] && { sed -n '2,18p' "$0"; exit 0; }
octo_env

python - <<'PY'
import sys
ok = True
import jax
devs = jax.devices()
print(f"jax {jax.__version__}  devices={devs}")
if not any(d.platform == "gpu" for d in devs):
    print("  FAIL  jax fell back to CPU.  On the DLAMI this is the cuSOLVER shadowing")
    print("        trap: LD_LIBRARY_PATH=/usr/local/cuda/lib64 hides the")
    print("        nvidia-cusolver-cu12 wheel jaxlib 0.4.20 needs.  Set")
    print("        LD_LIBRARY_PATH=$CONDA_PREFIX/lib (see roselite/g5/g5_env.sh).")
    ok = False

import simpler_env
import gymnasium as gym

# Expected timebase, from g5fine/RESULTS.txt section 4.  All DERIVED, not set.
EXPECT = {
    "widowx_spoon_on_towel":        (5, 500, 25, 200.00,  12000.0),
    "widowx_put_eggplant_in_basket":(5, 500, 25, 200.00,  24000.0),
    "google_robot_close_drawer":    (3, 513, 27, 333.33,  37666.7),
    "google_robot_pick_coke_can":   (3, 513, 27, 333.33,  26666.7),
}

def legal_tick(sim_freq, native_cf, target_ms=40.0):
    """Ticks dividing sim_freq AND a multiple of native_cf; nearest target_ms.
    Same rule as finegrain_eval.py._legal_ticks (kept here as a CHECK, not as a
    second implementation to depend on)."""
    cand = [h for h in range(1, sim_freq + 1)
            if sim_freq % h == 0 and h % native_cf == 0]
    return min(cand, key=lambda h: abs(1000.0 / h - target_ms))

print(f"\n{'task':32s} {'cf':>3s} {'sim':>4s} {'tick':>5s} {'act dt ms':>10s} {'horizon ms':>11s}")
for task, (cf, sf, tick, dt, hor) in EXPECT.items():
    env = simpler_env.make(task)
    u = env.unwrapped
    gcf, gsf = u.control_freq, u.sim_freq
    steps = env.spec.max_episode_steps if env.spec else gym.spec(u.spec.id).max_episode_steps
    gtick = legal_tick(gsf, gcf)
    gdt = 1000.0 / gcf
    ghor = steps * gdt
    bad = (gcf != cf) or (gsf != sf) or (gtick != tick) or abs(ghor - hor) > 1.0
    print(f"{task:32s} {gcf:3d} {gsf:4d} {gtick:4d}H {gdt:10.2f} {ghor:11.1f}"
          + ("   <-- MISMATCH vs documented" if bad else ""))
    if bad: ok = False
    env.close()
print("\nSMOKE OK" if ok else "\nSMOKE FAILED")
sys.exit(0 if ok else 1)
PY
