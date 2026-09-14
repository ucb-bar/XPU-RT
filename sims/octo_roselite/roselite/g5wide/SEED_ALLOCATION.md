# Global seed allocation — Octo latency sweep (roselite)

A "seed" is `--init-rng`, which seeds **only the policy's JAX sampling key**.
`latency_eval.py` and `run_eval.py` both reset with
`options={"obj_init_options": {"episode_id": ep_id}}`, `ep_id = 0..23`, in every
run — so the 24 official visual-matching episode *configurations* are identical
across every seed and every arm. A seed is therefore a policy-noise replicate
over a fixed episode set, not a fresh draw of episodes.

Two consequences:
* Arms sharing a seed are **paired** (same configs, same policy seed) — the
  per-seed and per-episode-config paired analyses are valid.
* Pooled episode-level intervals (Fisher / Newcombe) assume independence the
  design does not have. `analyze_wide.py` reports a **seed-clustered** interval
  alongside them.

Seeds must be globally disjoint *within an arm* so results pool cleanly.

## Allocation

| range | task / purpose | status |
|---|---|---|
| 0-4   | local reference box (TITAN RTX), eggplant arms A/B/C | USED (prior) |
| 0,2,4 | local reference box, spoon arm A | USED (prior) |
| 10-19 | AWS g5 `i-043560448065532ff`, eggplant arms A/B (`g5/`) | USED (prior) |
| **20**    | reserved, not issued | — |
| **21-25** | this sweep: per-worker **validation gate**, `widowx_spoon_on_towel` via `run_eval.py` (one seed per new worker). Same configuration as spoon arm A, so also pooled as spoon arm A. | this sweep |
| **30-79** | this sweep: eggplant arms **A** and **B** (50 seeds x 2 arms = 100 runs) | this sweep |
| **30-39** | this sweep: eggplant arm **C** (684.8 ms serial), 10 seeds | this sweep |
| **30-49** | this sweep: spoon arms **A** and **B**, 20 seeds (spoon A:30 run as the probe) | this sweep |
| **30-32** | this sweep: carrot and stack_cube arm **A** floor check, 3 seeds each | this sweep |
| **80-84** | FINE-GRAIN sweep: `google_robot_close_drawer` baseline probe (zero-latency arm only), 5 seeds | fine sweep |
| **80-84** | FINE-GRAIN sweep: `google_robot_pick_coke_can` baseline probe (zero-latency arm only), 5 seeds | fine sweep |
| **90-99** | FINE-GRAIN sweep: `widowx_spoon_on_towel`, all 6 arms x 10 seeds = 60 runs | fine sweep |
| **100-109** | FINE-GRAIN sweep: `widowx_put_eggplant_in_basket`, all 6 arms x 10 seeds = 60 runs. Mirrors the spoon design exactly so the task-dependence contrast is symmetric; the prior fine eggplant runs at seeds 0/2/4 are only 3 seeds. | fine sweep |
| **110-119** | FINE-GRAIN GOOGLE sweep: `google_robot_close_drawer`, all 6 arms x 10 seeds = 60 runs, `--actuation native`. | google sweep |
| **120-129** | FINE-GRAIN GOOGLE sweep: `google_robot_pick_coke_can`, all 6 arms x 10 seeds = 60 runs, `--actuation native`. | google sweep |

Seeds 110-129 use the FIXED fine harness (`--actuation native`: the env runs at
its own 3 Hz control_freq, the 27 Hz grid only decides which result is current at
each native step). They are NOT poolable with the seeds 80-84 google probe, which
used the broken 27 Hz-actuation model and scored 0/24 at zero latency; those
probe runs are quarantined in `finegrain/g5fine/runs_prefix_broken/`.

Seeds 80+ belong to the FINE-GRAIN harness (`finegrain/finegrain_eval.py`, 40 ms /
37 ms tick with zero-order hold). They are NOT poolable with the coarse
`latency_eval.py` runs at seeds 0-79 -- different timing model, different
numbers. The two probe tasks reuse 80-84 because they are different tasks; the
disjointness rule is per (task, arm).

## Worker assignment

Instance inventory, standing cost and the prune procedure live in `WORKERS.md`.

Seed `s` goes to worker `s mod 6`. **Both arms of a seed run on the same
worker**, so a box effect cannot confound the A-vs-B contrast — it can only add
variance to the paired differences.

| worker | instance | AZ | host |
|---|---|---|---|
| w0 | i-043560448065532ff (pre-existing) | us-east-1a | 3.93.66.216 |
| w1 | i-05289cfa068e00c74 | us-east-1b | 18.234.29.157 |
| w2 | i-09db19b86bffe6acb | us-east-1c | 3.232.133.234 |
| w3 | i-078a22ff12e6c885d | us-east-1d | 3.87.210.147 |
| w4 | i-039d68dfbf56a4f61 | us-east-1f | 32.198.80.184 |
| w5 | i-0f7e65981bc8d4d7c | us-east-1b | 204.236.245.40 |

The given subnet `subnet-061e6fb07d6ea44da` (us-east-1a) had **no g5.xlarge
capacity**, so the new workers were placed in the same VPC's other-AZ subnets.
Everything else matches the specified launch parameters.

## Resulting per-arm n (eggplant)

| arm | prior | added | total |
|---|---|---|---|
| A 0 ms baseline        | 312 (13 seeds) | 1200 (50 seeds) | **1512** |
| B 283.4 ms pipelined   | 312 (13 seeds) | 1200 (50 seeds) | **1512** |
| C 684.8 ms serial      | 120 (5 seeds)  |  240 (10 seeds) | **360**  |

1512/arm is the sample size the power calculation called for (detecting 5 points
around a 55% base rate at 80% power / alpha 0.05 two-sided needs ~1565).

`684.8 ms pipelined` is deliberately NOT extended: at D=4 no chunk entry is ever
valid, `sum|applied action| = 0.000` on every episode, so more n refines nothing.
