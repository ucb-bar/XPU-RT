# The eval harness is not run-to-run deterministic

MEASURED, 2026-09-06. Three byte-identical invocations of the same configuration:

```
python finegrain_eval_videosnap.py --task widowx_put_eggplant_in_basket \
    --latency-ms 283.4 --issue-period-ms 283.4 --init-rng 100 --n 1 --ep-start 0
```

| repeat | episode 0 outcome |
|---|---|
| 1 | success=False |
| 2 | success=True  |
| 3 | success=False |

Same machine, same GPU, same seed, same flags. A separate 24-episode run of the
same configuration reported episode 0 as success=True, and a `--n 1` no-video run
reported False -- consistent with the same cause, not with any dependence on `n`
or on video rendering (both were checked; nothing in the harness reads `args.n`
except the loop bound, and `Octo15Inference.reset()` does not touch `self.rng`).

Almost certainly non-deterministic GPU reductions in XLA/JAX. Untested remedy:
`XLA_FLAGS=--xla_gpu_deterministic_ops=true` plus `TF_DETERMINISTIC_OPS=1`, at
some throughput cost. Nobody has verified that it makes this harness reproducible.

## What this does NOT invalidate

The sweep conclusions stand. The paired analysis pairs arms on the **episode
config set**, which IS deterministic (`obj_init_options.episode_id` 0..23 rebuilds
the identical scene every time). Config difficulty is the dominant variance
component -- MEASURED ICC 0.13 to 0.27 depending on task and arm -- and that is
exactly what the pairing cancels. Marginal rates, paired penalties, the
difference-in-differences across tasks: all are aggregate quantities over 240
episodes per arm and are unaffected.

## What it DOES invalidate

1. **The seed ledger's framing.** `g5wide/SEED_ALLOCATION.md` describes a seed as
   "a policy-noise replicate over a fixed episode set", implying `--init-rng s`
   pins the policy noise so that two arms at seed s share a noise realisation.
   It does not. Each run is an independent draw. The runs are still valid
   replicates and the pairing still works (via the configs, not the seed), but
   "same policy seed" should be read as "same replicate index", nothing more.

2. **Any claim about an individual episode.** No single episode's outcome from
   this harness is reproducible, so a specific (seed, episode_id) success cannot
   be cited, replayed, or held up as a case study of a configuration. Only
   aggregate rates over the full config set mean anything.

3. **Video labelling.** A rendered video cannot be presented as a replay of a
   specific swept episode. `render_ladder_videos.sh` therefore retries until a run
   produces the target outcome and keeps THAT run -- each video is internally
   consistent (its frames and its `ep*_action_age_ms.npy` come from the same run),
   and the arm-level success rate quoted on it is the AWS sweep's aggregate, not a
   property of the clip.

## How this was found

Rendering contrast videos locally for episodes the AWS sweep recorded as
successes produced failures instead. The first hypothesis -- that `--save-video-every`
perturbs the simulation -- was tested and REJECTED: a no-video run of the same
config also failed, with identical timing (600 ticks, 85 inferences, mean
observation age 405.3 ms) but a different trajectory. The second hypothesis --
GPU nondeterminism between the AWS A10G and the local TITAN RTX -- was only half
right: the repeat test above shows it is nondeterministic on a single machine.
