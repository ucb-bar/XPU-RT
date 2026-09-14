# RoSE-lite large-n replication on an AWS g5.xlarge

Second, independent execution of the RoSE-lite headline comparison on a
different GPU (A10G) from the local reference box (TITAN RTX), at ~3.3x the
episode count per arm, to tighten the "accelerated board latency preserves task
success" claim.

Everything here is MEASURED unless a line says MODELLED. The only MODELLED
quantity is *when* a policy output becomes available to the environment:
`D = ceil(latency_ms / 200 ms)` steps, from the MEASURED QRB5165 board latency.
`control_freq = 5` is verified on the env, so one env step = 200 ms.

## Files

| file | what it is |
|---|---|
| `setup_simpler_g5.sh` | rebuilds the whole box-side install from scratch; documents every trap hit |
| `requirements_g5.txt`  | `pip freeze` of the validated local `octo_sim`, editables stripped — the pin source of truth |
| `g5_env.sh`            | the environment every run sources (deployed to the box as `sim_eval/roselite/env.sh`) |
| `validate.sh`          | port-validation run (`widowx_spoon_on_towel`, octo-small-1.0), one seed per invocation |
| `sweep_job.sh`         | one sweep job, `ARM:SEED`; run as `xargs -a joblist.txt -n1 -P3 sweep_job.sh` |
| `joblist.txt`          | the 20 jobs actually run (arms A/B x seeds 10-19) |
| `fetch_g5.sh`          | rsyncs `summary.json` files off the box into `runs/` |
| `analyze_g5.py`        | per-seed + pooled rates, Clopper-Pearson CIs, Fisher exact, paired sign test |
| `VALIDATION.txt`       | the port-validation gate result |
| `RESULTS.txt`          | the sweep result and the verdict |
| `RESULTS_raw.txt`      | raw `analyze_g5.py` output |
| `setup.log`            | the actual install transcript from the box |
| `validation/`, `runs/` | raw `summary.json` files pulled back from the box (3 + 20) |

## Box-side layout (all under `/home/ubuntu/simpler/`)

```
simpler/
  SimplerEnv/            06accaca93535902d408da4855f21cece12bceb7
    ManiSkill2_real2sim/ ef7a4d4fdf4b69f2c2154db5b15b9ac8dfe10682  (assets are git-tracked, 265 MB)
  octo/                  241fb3514b7c40957a86d869fecb7c7fc353f540
  sim_eval/              octo15_inference.py, run_eval.py       (md5-identical to the local tree)
    roselite/            latency_eval.py, env.sh                 (md5-identical to the local tree)
      runs/              sweep output
  logs/                  setup.log, val_rng*.log, sweep_*.log
```
Conda env `octo_sim` in `/home/ubuntu/miniforge3/envs`. Nothing else on the box
was touched; the pre-existing `fastdepth`/`xpurt` envs and the running
`train_fastdepth.py` job were left alone (and were competing for the box's 4
vCPUs throughout — see the throughput note in `VALIDATION.txt`).

## Reproducing

```bash
scp setup_simpler_g5.sh requirements_g5.txt ubuntu@<host>:/home/ubuntu/simpler/
ssh ubuntu@<host> 'bash /home/ubuntu/simpler/setup_simpler_g5.sh'   # ~8 min
# port validation gate -- do not run the sweep until this passes
ssh ubuntu@<host> '/home/ubuntu/simpler/validate.sh 0'
# sweep
ssh ubuntu@<host> 'xargs -a joblist.txt -n1 -P3 /home/ubuntu/simpler/sweep_job.sh'
./fetch_g5.sh && python analyze_g5.py
```

## The correctness fix that is carried over

`octo15_inference.py` does the masked unnormalization through octo's own
`unnormalization_statistics=` rather than the stock wrapper's blanket
`a*std + mean`. SimplerEnv's stock Octo wrapper is correct for octo-1.0 and
silently broken for octo-1.5: 1.5's bridge action stats carry
`mask = [T,T,T,T,T,T,False]`, so the gripper dim is emitted raw in [0,1], and
applying `a*std+mean` with `mean=0.588, std=0.488` maps commanded-close 0.0 to
0.588 and commanded-open 1.0 to 1.076 — both above the 0.5 close threshold, so
the binarizer returns OPEN forever, the gripper never closes, and any pick task
scores 0%. octo-1.0 has gripper `mean=0.0, std=1.0`, which makes the stock
formula an identity, so this run (octo-small-1.0) is unaffected by the bug —
but the corrected path is what is being exercised, not the stock one.
