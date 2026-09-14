# AWS sweep workers — what they are for, and how to prune them

**Status as of 2026-09-05: all six are `stopped`, none terminated.** Kept
deliberately (user decision) for further sweeps; prune at the end of the study.
Verified stopped via `aws ec2 describe-instances` on the manager.

## Why they are being kept

The boxes hold no unique *results* — every `summary.json` from the 4,752-episode
sweep is already fetched to `g5wide/runs/<worker>/` (218 files, 5.2 MB) and
`g5wide/validation/<worker>/`. What they hold is the **working environment**, and
that is the whole reason not to terminate:

* Miniforge3 + the `simpler` conda env, pinned `setuptools==80.10.2` (see the
  setup traps in `RESULTS.txt` §7 — a fresh box dies at `import simpler_env`
  without that pin)
* SimplerEnv `06accaca`, ManiSkill2 `ef7a4d4f`, octo `241fb351`, and the
  octo-small-1.5 checkpoint
* the harness, md5-identical to the local tree (`latency_eval.py`,
  `octo15_inference.py`, `run_eval.py`)
* a *passed* per-worker port-validation gate (`VALIDATION.txt`) — a rebuilt box
  has to re-earn that before its numbers may be pooled

Rebuild cost per box is roughly 40 min of `bootstrap.sh` + `setup_simpler_g5.sh`
plus the validation-gate run. Restart cost from `stopped` is ~2 min.

## The workers

Seed `s` runs on worker `s mod 6`; **both arms of a seed run on the same worker**,
so a box effect can add variance to the paired difference but cannot confound it.

| worker | instance | AZ | public IP (stale once stopped) | root vol | sweep runs fetched |
|---|---|---|---|---|---|
| w0 | `i-043560448065532ff` | us-east-1a | 3.93.66.216 | `vol-01d13369a1aaf1faa` | 60 |
| w1 | `i-05289cfa068e00c74` | us-east-1b | 18.234.29.157 | `vol-0a2ed0d2323ee1cf6` | 34 |
| w2 | `i-09db19b86bffe6acb` | us-east-1c | 3.232.133.234 | `vol-042397005bfada834` | 30 |
| w3 | `i-078a22ff12e6c885d` | us-east-1d | 3.87.210.147 | `vol-01dc3d0ea45baae5f` | 32 |
| w4 | `i-039d68dfbf56a4f61` | us-east-1f | 32.198.80.184 | `vol-0b6cd19607791e3b9` | 31 |
| w5 | `i-0f7e65981bc8d4d7c` | us-east-1b | 204.236.245.40 | `vol-011d53557a8a4f4e9` | 31 |

w0 is the **original** box (pre-dates this sweep; carries the `g5/` eggplant arms
at seeds 10-19, hence its 60 runs). w1-w5 were built for the wide sweep by
`bootstrap.sh`. All are `g5.xlarge` (A10G 24 GB, 4 vCPU, 16 GB), key `firesim`,
SG `sg-0cb78aaa92a9912fd`, AMI `ami-012ba162b9cd2729c`.

Tagged `Owner=dima`, `Purpose=octo-latency-sweep`, `Name=dima-octo-sweep-{1..5}`.

**Public IPs above are not stable across a stop/start.** Re-read them after
starting and update the `HOSTS` map in `fetch_wide.sh` before fetching.

## What they have been used for

1. **Coarse wide sweep** (200 ms model, `latency_eval.py`) -- 4,752 episodes across
   eggplant/spoon/carrot/stack_cube. Results in `RESULTS.txt`.
2. **Fine-grain sweep** (40 ms tick + ZOH, `finegrain_eval.py`) -- 70 runs / 1,680
   episodes: the full 6-arm spoon ladder at 10 seeds, plus a google_robot baseline
   probe. Results in `../finegrain/g5fine/RESULTS.txt`. This sweep also found that
   the fine harness does not work on google_robot tasks
   (`../finegrain/GOOGLE_ROBOT_PORT.md`).
3. **Fine-grain GOOGLE sweep** (2026-09-06) -- the google_robot port, now FIXED,
   then 120 runs / 2,880 episodes: the full 6-arm ladder x 10 seeds x
   `google_robot_close_drawer` (seeds 110-119) and `google_robot_pick_coke_can`
   (seeds 120-129). Harness md5 `b9be655f74a6`, identical on all six boxes,
   `--actuation native` for google_robot (widowx unchanged, verified bit-exact).
   Results in `../finegrain/g5fine/RESULTS_GOOGLE.txt`; the port write-up and the
   validation gate are in `../finegrain/GOOGLE_ROBOT_PORT.md`. The pre-fix
   google probe runs (seeds 80-84, 0/24 at zero latency) are quarantined in
   `../finegrain/g5fine/runs_prefix_broken/` and are rejected by both analysis
   scripts.

   Two things learned about these boxes on that sweep:
   * **Same box + same seed is bit-reproducible** (a widowx run re-run alone on
     w0 reproduced its exact success set, though the original ran 3-concurrent),
     but **different GPU is not**: `coke lat0 rng80` is 9/24 on the local TITAN
     RTX and 14/24 on an A10G (measured twice, on two different A10G boxes). A
     google_robot marginal rate is only comparable to one from the same GPU.
     Paired per-seed contrasts are unaffected -- every arm of a seed runs on the
     same worker.
   * Job order within a worker is now **shuffled** (fixed seed) before launch.
     The tail of an `xargs -P3` queue runs at lower concurrency than the body;
     with the arms in a fixed order the same arm always landed in that tail on
     every box, correlating an arm with its machine load.

## What they are wanted for next

`RESULTS.txt` §6 names three follow-ups, in the order they buy the most:

1. **More tasks.** The single largest effect found: the same 283.4 ms costs
   **-7.1** points on eggplant and **-22.9** on spoon. Task choice moved the
   answer 3x — more than any sample-size decision. carrot and stack_cube sit at
   a 3-7% floor and resolve nothing; new tasks need to be off the floor.
2. **More episode *configs*.** Marginal per-arm rates are capped at ~±9 points
   by K=24 configs no matter how many seeds run (`n_eff -> K/ICC = 119`).
   Only more `episode_id` configurations can lift that ceiling.
0. ~~**Eggplant fine-grain at 10 seeds.**~~ **DONE** (seeds 100-109). It was the
   binding limit on the task-dependence question; with 10 seeds against spoon's
   10, the coarse sweep's "task choice moves the answer 3x" headline is retired
   -- the two widowx tasks differ by +1.3 [-11.8,+14.3] at 283.4 ms.
1b. ~~**Fix the google_robot fine port**~~ **DONE 2026-09-06** -- two bugs, fixed
   and gate-validated, and the second embodiment's full 6-arm ladder is measured
   at 10 seeds on both tasks. See `../finegrain/GOOGLE_ROBOT_PORT.md` and
   `../finegrain/g5fine/RESULTS_GOOGLE.txt`.
3. **More seeds**, for the paired A-vs-B difference only — and with a hard floor
   at ~±2.6 pts (eggplant) / ±4.6 pts (spoon) from the real config-dependent
   component of the penalty.

Seeds 30-79 are consumed; 21-25 were validation gates; **20 is reserved and
unissued**. Any new work must take seeds from 80 up and record them in
`SEED_ALLOCATION.md`, or the pooled analyses stop being valid.

## Standing cost

6 x 200 GB gp3 = 1.2 TB. At $0.08/GB-month that is **~$96/month** while stopped
(a stopped instance is charged for its EBS, not its compute). Per box: ~$16/month.
The sweep compute itself was 11.01 instance-hours = **$11.08** for 4,752 episodes.

So the storage bill overtakes the entire sweep's compute in under four days of
idling. If the follow-ups above are not going to run soon, prune early.

## Pruning, when the study is done

**Before terminating anything**, pull the per-job stdout logs — `fetch_wide.sh`
copies only `summary.json`, so `/home/ubuntu/simpler/logs/gw_*.log` (per-episode
progress, the `SUCCESS RATE` lines, any warnings) exists **only on the boxes**:

```bash
# start the boxes, re-read IPs, then per worker:
rsync -a -e "ssh -i ~/.ssh/firesim.pem" \
  ubuntu@<ip>:/home/ubuntu/simpler/logs/ g5wide/logs/<worker>/
```

Then, from the manager (`ubuntu@3.88.218.39`, creds in `~/.aws`):

```bash
aws ec2 terminate-instances --instance-ids <ids...>
```

Terminating deletes the root volumes (`DeleteOnTermination` is the AMI default —
**verify** with `describe-instances --query '...Ebs.DeleteOnTermination'` before
relying on it, and check for orphaned volumes afterwards with
`describe-volumes --filters Name=status,Values=available`).

Suggested end-state: keep **w0** alone (it carries the original environment and
the prior `g5/` arms) and terminate w1-w5, cutting the standing cost from ~$96 to
~$16/month while leaving one validated box able to seed a rebuild.
