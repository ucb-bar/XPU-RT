# Does the divergence survive a different consumption policy?

The plan lists as an open limitation that `latest_completed` is one consumption policy
among several, and that release-matched and versioned alternatives were never evaluated.
This closes it. **The answer is yes for the divergence and no for one specific framing of
it**, and the second half is a real qualification of the headline.

All three re-scorings run over **identical schedules** — `solver_wall_s` is 0.0 in every
cell, because consumption policy is an evaluator setting that cannot change a schedule,
only how one is scored. `static_nominal`, all 25 (B, φ) cells.

Reproduce:

```bash
for p in latest_completed newest_version release_matched; do
  python -m benchmarks.freshness_eval.run \
    --config data/toplevel/freshness_canon_300ms.json \
    --output-dir results/freshness_consumption/$p \
    --seeds 0 --bursts 0,1,2,3,4 --policies static_nominal \
    --consumption-policy $p --reuse-fixtures
done
```

## The three policies

| policy | which readable sample is "the input" |
|---|---|
| `latest_completed` | the most recently **written** sample (max `end_time`) |
| `newest_version` | the freshest **sample** present (max instance index) |
| `release_matched` | strictly the **current frame**; no substitution |

## Result at φ = A0 + 20

| B | `latest_completed` | `newest_version` | `release_matched` |
|---|---|---|---|
| | valid / stale / no-input | valid / stale / no-input | valid / stale / no-input |
| 0 | .933 / .000 / .067 | .933 / .000 / .067 | **.600 / .000 / .400** |
| 1 | .633 / .067 / .300 | .633 / .067 / .300 | .367 / .000 / .633 |
| 2 | .400 / .300 / .300 | .400 / .300 / .300 | .133 / .000 / .867 |
| 3! | .220 / .561 / .220 | .220 / .561 / .220 | .000 / .000 / 1.000 |
| 4! | .000 / .750 / .250 | .000 / .750 / .250 | .000 / .000 / 1.000 |

`!` = the schedule overruns the 300 ms epoch, so that row is not comparable to a fitting
one. Deadline success is 1.000 in every cell of all three.

## Finding 1 — `newest_version` is bit-identical to `latest_completed`, in all 25 cells

So the headline is **not** an artifact of preferring "most recently written" over "newest
sample". The two can only differ when producers complete out of release order, and on
this workload they never do: DroNet's six instances complete in order at every burst.
The distinction is real and is exercised by `test_consumption_policies.py` on a
hand-built out-of-order trace — it simply does not arise here.

That is worth stating as a property of the workload rather than a general result. A
schedule that placed successive perception instances on the fast and slow clusters would
separate them, and this grid never does.

## Finding 2 — `release_matched` never reports stale input at all, and that is structural

Its stale column is **0.000 in every cell**, and not by accident. Hand-computed, no
evaluator involved: a matched frame's worst-case age is `T_p − T_c + L_p` =
40 + 18.614 = **58.61 ms**, which is below every φ in the sweep (the smallest is 65.5).
So a release-matched consumer either has the current frame — necessarily fresh — or
nothing. Staleness is unreachable by construction.

The 40% loss at **zero contention** is likewise structural: with a 10 ms consumer and a
50 ms/18.6 ms producer, the first two consumers of each producer period fire before the
current frame exists. 2 of every 5, i.e. 12 of 30 → 0.600 valid. The measured value is
exactly 0.600.

## What this does to the headline

**Survives:** the divergence itself. Deadline success is 1.000 while output validity
falls to .400 (B=2, `latest_completed`) or .133 (B=2, `release_matched`) — under every
policy, local deadline compliance fails to imply valid output. That is the paper's claim
and it is policy-independent.

**Qualified:** the phrase "the controller acts on stale perception". That failure mode
exists only under a *substituting* policy. Under release-matched semantics the controller
never acts on stale input; it is starved instead. Same schedules, same deadline
compliance, same invalidity — different failure. Any claim about staleness specifically
has to name the consumption policy, and the Gate A summary's decomposition into
stale-vs-no-input is a statement about `latest_completed`, not about the pipeline.

## Why `latest_completed` is nonetheless the right default here

Not a free choice — it is the only one this system can physically realise. All K
instances of a network share **one output buffer** (F5: buffers are named
`buf_<model_id>_<tensor>`, mangled by model id only, with no instance, version or slot,
and `harness_xpurt/CMakeLists.txt` enforces one buffers translation unit per model).
A consumer reading that buffer gets whatever was written most recently, which is exactly
`latest_completed`. `newest_version` needs a version word that does not exist;
`release_matched` needs to identify a specific frame, which the shared buffer cannot
express.

So the ordering is: `latest_completed` is what the hardware does today, and the stricter
policies are what it *could* do given per-instance buffers or a sequence word — the same
missing mechanism as F5, and the subject of the next task. That connects the two:

**adding a version word would not only make freshness observable, it would make the
safer failure mode available.** Under release-matched semantics a late producer starves
the controller (detectable, hold-last-command) instead of feeding it a stale sample that
looks valid (undetectable at the consumer). That is a systems argument for the mechanism,
independent of the measurement argument.

## Limitations

- One policy per run, `static_nominal` only. The candidate ladder was not re-scored under
  the alternatives; the point here is whether the *phenomenon* is policy-dependent, not
  to restate the ladder three ways.
- `release_matched` as implemented does not wait: a consumer whose current frame is
  incomplete gets nothing, rather than blocking. A blocking variant would convert the
  no-input cases into deadline misses instead, which is a third failure attribution and
  is not evaluated.
- The B=3 and B=4 rows overrun the epoch and are not comparable to the fitting rows, as
  everywhere else in this study.
